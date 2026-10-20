"""Client-side preprocessing of T2-weighted bladder MRI (Section 3.2).

Per-volume pipeline:

1. load the NIfTI volume and reorient it to RAS;
2. optional N4 bias-field correction (disabled by default);
3. in-plane resampling to a uniform spacing (B-spline for the image,
   nearest neighbour for the mask);
4. foreground percentile clipping and Z-score normalisation, with the
   normalised values clipped to a symmetric range;
5. 2.5D stacking: for every target slice, the two slices above and the two
   below are stacked into a five-channel tensor;
6. tumour-centred ROI crop and resize to the fixed model input size.

Every step is driven by :class:`PreprocessConfig`, so the values can be set
from ``configs/preprocessing.yaml`` without editing this file. The defaults
are the values used for the reported experiments; where those differ from the
prose of the paper, the difference is recorded in
``docs/PAPER_CODE_AUDIT.md``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk

__all__ = ["PreprocessConfig", "preprocess_volume", "preprocess_patient"]


@dataclass
class PreprocessConfig:
    """Preprocessing parameters.

    Attributes:
        target_spacing: in-plane spacing in mm. The through-plane spacing is
            left untouched.
        input_size: side length of the square model input in pixels.
        apply_n4: run N4 bias-field correction. Off by default: the final
            paper's pipeline and the archived experiments do not use it.
        n4_shrink_factor: downsampling factor used to fit the N4 field (the
            estimated field is applied at full resolution).
        n4_iterations: N4 iterations per resolution level.
        clip_percentiles: foreground percentiles used for intensity clipping.
        z_clip: symmetric clip applied after Z-score normalisation.
        n_context_slices: slices taken above and below the target slice; two
            on each side gives the five-channel 2.5D stack of the paper.
        tumour_padding: padding in pixels added around the tumour bounding box
            before cropping.
        n_nontumour_slices: tumour-free slices sampled per patient to provide
            background context.
        seed: seed of the tumour-free slice sampling.
    """

    target_spacing: tuple = (0.8, 0.8)
    input_size: int = 224
    apply_n4: bool = False
    n4_shrink_factor: int = 4
    n4_iterations: list = field(default_factory=lambda: [50, 50, 50, 50])
    clip_percentiles: tuple = (0.5, 99.5)
    z_clip: float = 5.0
    n_context_slices: int = 2
    tumour_padding: int = 40
    n_nontumour_slices: int = 2
    seed: int = 42

    @property
    def in_channels(self) -> int:
        """Channels of the 2.5D stack (five for two context slices a side)."""
        return 2 * self.n_context_slices + 1


# --------------------------------------------------------------------------
# Volume-level steps
# --------------------------------------------------------------------------


def load_and_orient(path: str | Path) -> sitk.Image:
    """Read a NIfTI volume and reorient it to the RAS convention."""
    return sitk.DICOMOrient(sitk.ReadImage(str(path)), "RAS")


def n4_bias_field_correction(
    image: sitk.Image, shrink_factor: int = 4, iterations: list | None = None
) -> sitk.Image:
    """Apply N4 bias-field correction.

    The field is estimated on a downsampled copy for speed, then evaluated at
    full resolution and divided out, so no spatial detail is lost.
    """
    iterations = iterations or [50, 50, 50, 50]

    image = sitk.Cast(image, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)

    if shrink_factor > 1:
        shrink = [shrink_factor] * image.GetDimension()
        small_image = sitk.Shrink(image, shrink)
        small_mask = sitk.Shrink(mask, shrink)
    else:
        small_image, small_mask = image, mask

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(iterations)
    corrector.Execute(small_image, small_mask)

    log_field = corrector.GetLogBiasFieldAsImage(image)
    return image / sitk.Exp(log_field)


def resample_inplane(
    image: sitk.Image, spacing_xy: tuple, is_mask: bool = False
) -> sitk.Image:
    """Resample in-plane to ``spacing_xy``, keeping the native slice spacing."""
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()

    new_size = [
        int(round(orig_size[0] * orig_spacing[0] / spacing_xy[0])),
        int(round(orig_size[1] * orig_spacing[1] / spacing_xy[1])),
        orig_size[2],
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([spacing_xy[0], spacing_xy[1], orig_spacing[2]])
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline
    )
    return resampler.Execute(image)


def normalize_intensity(volume: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Clip foreground percentiles, then Z-score normalise the volume.

    Args:
        volume: array of shape ``(H, W, Z)``.
        cfg: preprocessing configuration.

    Returns:
        Normalised volume, clipped to ``[-cfg.z_clip, cfg.z_clip]``.
    """
    fg_threshold = np.percentile(volume, 5)
    foreground = volume[volume > fg_threshold]
    if foreground.size < 100:  # degenerate / near-empty volume
        return volume.astype(np.float32)

    p_lo, p_hi = np.percentile(foreground, list(cfg.clip_percentiles))
    volume = np.clip(volume, p_lo, p_hi)

    foreground = volume[volume > p_lo]
    mean = foreground.mean()
    std = foreground.std() + 1e-8
    volume = (volume - mean) / std

    return np.clip(volume, -cfg.z_clip, cfg.z_clip).astype(np.float32)


# --------------------------------------------------------------------------
# Slice-level steps
# --------------------------------------------------------------------------


def build_slice_stack(volume: np.ndarray, index: int, n_context: int = 2) -> np.ndarray:
    """Stack a target slice with its neighbours into a 2.5D tensor.

    Args:
        volume: array of shape ``(H, W, Z)``.
        index: index of the target slice along z.
        n_context: slices taken on each side; indices are clamped at the
            volume borders, which replicates the edge slice.

    Returns:
        Array of shape ``(2 * n_context + 1, H, W)``.
    """
    depth = volume.shape[2]
    indices = [
        min(max(index + offset, 0), depth - 1)
        for offset in range(-n_context, n_context + 1)
    ]
    return np.stack([volume[:, :, i] for i in indices], axis=0).astype(np.float32)


def tumour_centred_crop(
    stack: np.ndarray, mask: np.ndarray, padding: int, target: int
) -> tuple:
    """Crop a square ROI around the tumour and resize it to ``target``.

    Bladder tumours sit on the bladder wall and are rarely centred in the
    field of view, so the crop is centred on the tumour bounding box rather
    than on the image. Slices with an empty mask fall back to a centre crop.

    The mask is used only during preprocessing, never as a model input.

    Returns:
        ``(image, mask, centroid)`` with the image of shape
        ``(C, target, target)``, the mask ``(target, target)`` and the crop
        centroid in the coordinates of the resampled slice.
    """
    from PIL import Image as PILImage

    channels, height, width = stack.shape
    nonzero = np.argwhere(mask > 0)

    if nonzero.size > 0:
        r_min, c_min = nonzero.min(axis=0)
        r_max, c_max = nonzero.max(axis=0)
        side = max(r_max - r_min + 1, c_max - c_min + 1) + 2 * padding
        centre_r = (r_min + r_max) // 2
        centre_c = (c_min + c_max) // 2
    else:
        side = min(height, width)
        centre_r, centre_c = height // 2, width // 2

    half = side // 2
    r0 = max(0, centre_r - half)
    c0 = max(0, centre_c - half)
    r1 = min(height, r0 + side)
    c1 = min(width, c0 + side)
    r0 = max(0, r1 - side)
    c0 = max(0, c1 - side)

    out_img = np.zeros((channels, target, target), dtype=np.float32)
    for ch in range(channels):
        patch = PILImage.fromarray(stack[ch, r0:r1, c0:c1], mode="F")
        out_img[ch] = np.array(
            patch.resize((target, target), PILImage.BILINEAR), dtype=np.float32
        )

    patch_mask = PILImage.fromarray(mask[r0:r1, c0:c1].astype(np.uint8), mode="L")
    out_mask = np.array(
        patch_mask.resize((target, target), PILImage.NEAREST), dtype=np.uint8
    )

    return out_img, out_mask, (int(centre_r), int(centre_c))


# --------------------------------------------------------------------------
# Patient-level pipeline
# --------------------------------------------------------------------------


def preprocess_volume(
    image_path: str | Path, mask_path: str | Path, cfg: PreprocessConfig
) -> tuple:
    """Load, bias-correct, resample and normalise one patient volume.

    Returns:
        ``(image, mask)`` arrays of shape ``(H, W, Z)``.
    """
    image = load_and_orient(image_path)
    mask = load_and_orient(mask_path)

    if cfg.apply_n4:
        image = n4_bias_field_correction(
            image, shrink_factor=cfg.n4_shrink_factor, iterations=list(cfg.n4_iterations)
        )

    image = resample_inplane(image, cfg.target_spacing, is_mask=False)
    mask = resample_inplane(mask, cfg.target_spacing, is_mask=True)

    # SimpleITK arrays are (Z, Y, X); transpose to (H, W, Z).
    image_np = sitk.GetArrayFromImage(image).transpose(1, 2, 0).astype(np.float32)
    mask_np = sitk.GetArrayFromImage(mask).transpose(1, 2, 0).astype(np.uint8)
    mask_np = (mask_np > 0).astype(np.uint8)

    return normalize_intensity(image_np, cfg), mask_np


def preprocess_patient(
    image_path: str | Path,
    mask_path: str | Path,
    label: int,
    patient_id: str,
    center: int,
    output_dir: str | Path,
    cfg: PreprocessConfig,
    rng: random.Random | None = None,
) -> list:
    """Preprocess one patient and write its slice files.

    ``rng`` is the generator that draws the tumour-free context slices. The
    preprocessing script passes one generator shared by all patients, as in the
    recorded runs; if omitted, a fresh generator seeded with ``cfg.seed`` is used.

    Two files are written per retained slice, ``<fname>_img.npy`` of shape
    ``(C, S, S)`` and ``<fname>_mask.npy`` of shape ``(S, S)``.

    Returns:
        One index record per written slice, carrying the patient id, the
        centre, the MIBC/NMIBC label and the tumour flags used downstream.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_np, mask_np = preprocess_volume(image_path, mask_path, cfg)
    depth = mask_np.shape[2]

    per_slice_area = mask_np.sum(axis=(0, 1))
    tumour_slices = np.where(per_slice_area > 0)[0].tolist()
    best_slice = int(np.argmax(per_slice_area)) if tumour_slices else depth // 2

    # Every tumour slice plus its context neighbours, the largest-area slice,
    # and a few tumour-free slices for background context.
    selected = set()
    for s in tumour_slices:
        for offset in range(-cfg.n_context_slices, cfg.n_context_slices + 1):
            if 0 <= s + offset < depth:
                selected.add(s + offset)
    selected.add(best_slice)

    tumour_free = [z for z in range(depth) if z not in tumour_slices]
    if tumour_free:
        rng = rng or random.Random(cfg.seed)
        selected.update(
            rng.sample(tumour_free, min(cfg.n_nontumour_slices, len(tumour_free)))
        )

    records = []
    for s_idx in sorted(selected):
        mask_slice = mask_np[:, :, s_idx]
        stack = build_slice_stack(image_np, s_idx, cfg.n_context_slices)

        img_out, mask_out, centroid = tumour_centred_crop(
            stack, mask_slice, cfg.tumour_padding, cfg.input_size
        )

        fname = f"{patient_id}_s{s_idx:03d}"
        np.save(output_dir / f"{fname}_img.npy", img_out)
        np.save(output_dir / f"{fname}_mask.npy", mask_out)

        records.append(
            {
                "patient_id": patient_id,
                "center": int(center),
                "slice_idx": int(s_idx),
                "has_tumor": int(mask_slice.sum() > 0),
                "is_best_slice": int(s_idx == best_slice),
                "label": int(label),
                "centroid_r": centroid[0],
                "centroid_c": centroid[1],
                "fname": fname,
            }
        )

    return records
