#!/usr/bin/env python3
"""Preprocess the FedBCa dataset into 2.5D slice tensors.

Example::

    export FEDBCA_ROOT=/data/FedBCa
    python scripts/preprocess.py --config configs/preprocessing.yaml

The expected layout of ``FEDBCA_ROOT`` is documented in ``docs/DATASET.md``.
The script writes ``<fname>_img.npy`` / ``<fname>_mask.npy`` per retained
slice plus ``slice_index.csv``, which ``scripts/make_splits.py`` consumes.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

from tqdm import tqdm
import pandas as pd

# Make the repository root importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import PreprocessConfig, preprocess_patient
from src.utils import load_config, resolve_path

logger = logging.getLogger("preprocess")

#: Candidate directory names per centre, in order of preference.
IMAGE_DIRS = ("Image", "T2WI", "image", "images")
MASK_DIRS = ("Annotation", "annotation", "Mask", "mask")

#: Candidate column names in the per-centre label spreadsheet. The mask
#: candidates are ordered so that a column holding file names is preferred
#: over one holding bare numeric identifiers.
COLUMN_CANDIDATES = {
    "image": ("image", "Image", "image_name", "filename", "FileName", "file"),
    "mask": ("mask_new", "annotation", "Annotation", "mask", "Mask", "mask_name", "seg"),
    "label": ("MIBC", "mibc", "label", "Label", "class"),
}

LABEL_FILES = (
    "Center_{i}_label.xlsx",
    "center{i}_label.xlsx",
    "label_{i}.xlsx",
    "label.xlsx",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/preprocessing.yaml", help="YAML configuration"
    )
    parser.add_argument("--dataset-root", default=None, help="override dataset_root")
    parser.add_argument("--output-dir", default=None, help="override output_dir")
    parser.add_argument(
        "--centers",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="centre indices to process",
    )
    return parser.parse_args()


def find_dir(base, names):
    """Return the first existing sub-directory among ``names``."""
    for name in names:
        if (base / name).is_dir():
            return base / name
    return None


def find_label_file(base, index: int):
    """Return the first existing per-centre label spreadsheet."""
    for pattern in LABEL_FILES:
        candidate = base / pattern.format(i=index)
        if candidate.is_file():
            return candidate
    return None


def find_column(df: pd.DataFrame, candidates) -> str:
    """Return the first candidate column present in ``df``."""
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(
        f"None of {list(candidates)} found. Available columns: {list(df.columns)}"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    config = load_config(args.config)
    dataset_root = resolve_path(
        args.dataset_root or config.get("dataset_root"), "dataset_root"
    )
    output_dir = resolve_path(args.output_dir or config.get("output_dir"), "output_dir")

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}. See docs/DATASET.md for how "
            "to obtain FedBCa and how the directory should be laid out."
        )

    cfg = PreprocessConfig(
        target_spacing=tuple(config.get("target_spacing", (0.8, 0.8))),
        input_size=int(config.get("input_size", 224)),
        apply_n4=bool(config.get("apply_n4", False)),
        n4_shrink_factor=int(config.get("n4_shrink_factor", 4)),
        n4_iterations=list(config.get("n4_iterations", [50, 50, 50, 50])),
        clip_percentiles=tuple(config.get("clip_percentiles", (0.5, 99.5))),
        z_clip=float(config.get("z_clip", 5.0)),
        n_context_slices=int(config.get("n_context_slices", 2)),
        tumour_padding=int(config.get("tumour_padding", 40)),
        n_nontumour_slices=int(config.get("n_nontumour_slices", 2)),
        seed=int(config.get("seed", 42)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Dataset root : %s", dataset_root)
    logger.info("Output dir   : %s", output_dir)
    logger.info(
        "N4=%s spacing=%s input=%d channels=%d",
        cfg.apply_n4,
        cfg.target_spacing,
        cfg.input_size,
        cfg.in_channels,
    )

    all_records: list[dict] = []
    rng = random.Random(cfg.seed)  # shared by all patients (background-slice sampling)

    for center in args.centers:
        center_dir = dataset_root / f"Center{center}"
        image_dir = find_dir(center_dir, IMAGE_DIRS) if center_dir.is_dir() else None
        mask_dir = find_dir(center_dir, MASK_DIRS) if center_dir.is_dir() else None
        label_file = find_label_file(center_dir, center) if center_dir.is_dir() else None

        if not (image_dir and mask_dir and label_file):
            logger.warning("Center %d: missing image/mask directory or label file, skipping", center)
            continue

        labels = pd.read_excel(label_file)
        image_col = find_column(labels, COLUMN_CANDIDATES["image"])
        mask_col = find_column(labels, COLUMN_CANDIDATES["mask"])
        label_col = find_column(labels, COLUMN_CANDIDATES["label"])

        logger.info("Center %d: %d rows in %s", center, len(labels), label_file.name)

        for row_idx, row in tqdm(
            labels.iterrows(), total=len(labels), desc=f"Center {center}"
        ):
            image_path = image_dir / str(row[image_col])
            mask_path = mask_dir / str(row[mask_col])

            if not image_path.is_file() or not mask_path.is_file():
                logger.warning(
                    "Center %d row %s: missing volume or mask, skipping",
                    center,
                    row_idx,
                )
                continue

            patient_id = f"C{center}_" + image_path.name.replace(".nii.gz", "").replace(".nii", "")

            try:
                records = preprocess_patient(
                    image_path=image_path,
                    mask_path=mask_path,
                    label=int(row[label_col]),
                    patient_id=patient_id,
                    center=center,
                    output_dir=output_dir,
                    cfg=cfg,
                    rng=rng,
                )
            except Exception as exc:  # one unreadable volume must not stop the run
                logger.warning("Center %d patient %s failed: %s", center, patient_id, exc)
                continue

            all_records.extend(records)

    if not all_records:
        raise RuntimeError(
            "No slices were written. Check that the dataset layout matches "
            "docs/DATASET.md."
        )

    index_path = output_dir / "slice_index.csv"
    pd.DataFrame(all_records).to_csv(index_path, index=False)

    logger.info(
        "Wrote %d slices from %d patients to %s",
        len(all_records),
        len({r["patient_id"] for r in all_records}),
        output_dir,
    )
    logger.info("Slice index: %s", index_path)


if __name__ == "__main__":
    main()
