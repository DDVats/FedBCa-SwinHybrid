"""Augmentation strategies evaluated in the paper.

The selected final strategy is **Geo+Elastic** (Section 3.4):

* random horizontal flip, p = 0.5
* rotation within +/- 15 degrees, p = 0.70
* elastic deformation, sigma = 5.0, alpha = 18.0, p = 0.20

The remaining strategies (``none``, ``gin_only``, ``stack``, ``geo_bias``,
``geo_gamma``, ``geo_noise``) are kept because they constitute the
augmentation study of Table 3. Build any of them by name with
:func:`build_augmentation`.

Every pipeline takes ``(image, mask)`` with ``image`` of shape ``(C, H, W)``
float32 and ``mask`` of shape ``(H, W)`` uint8, and returns the same shapes.
Intensity-only transforms leave the mask untouched.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

__all__ = [
    "GeometricAugmentation",
    "ElasticDeformation",
    "GINAugmentation",
    "ResNetGINAugmentation",
    "BiasFieldAugmentation",
    "GammaAugmentation",
    "GaussianNoise",
    "NoAugmentation",
    "GeoElasticPipeline",
    "GeoBiasPipeline",
    "GeoGammaPipeline",
    "GeoNoisePipeline",
    "GINOnlyPipeline",
    "AugmentationStackPipeline",
    "AUGMENTATION_STRATEGIES",
    "build_augmentation",
]


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


class GeometricAugmentation:
    """Random horizontal flip and in-plane rotation.

    Args:
        p_flip: probability of the horizontal flip.
        max_angle: maximum absolute rotation in degrees.
        p_rotate: probability of applying a rotation.
    """

    def __init__(
        self, p_flip: float = 0.5, max_angle: float = 15.0, p_rotate: float = 0.70
    ) -> None:
        self.p_flip = p_flip
        self.max_angle = max_angle
        self.p_rotate = p_rotate

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        image = np.ascontiguousarray(image)

        if np.random.random() < self.p_flip:
            image = image[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()

        if np.random.random() < self.p_rotate and self.max_angle > 0:
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            h, w = mask.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

            for ch in range(image.shape[0]):
                image[ch] = cv2.warpAffine(
                    image[ch],
                    matrix,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )

            mask = cv2.warpAffine(
                mask.astype(np.uint8),
                matrix,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REFLECT,
            )

        return image.astype(np.float32), mask.astype(np.uint8)


class ElasticDeformation:
    """Random elastic deformation of the image and its mask.

    A smooth random displacement field is produced by filtering Gaussian noise
    with ``sigma`` and scaling it by ``alpha``. The mask is warped with
    nearest-neighbour interpolation so that it stays binary.
    """

    def __init__(self, p: float = 0.20, alpha: float = 18.0, sigma: float = 5.0) -> None:
        self.p = p
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        if np.random.random() > self.p:
            return image, mask

        h, w = mask.shape
        dx = gaussian_filter(np.random.randn(h, w), self.sigma) * self.alpha
        dy = gaussian_filter(np.random.randn(h, w), self.sigma) * self.alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        coords = [
            np.clip(y + dy, 0, h - 1).ravel(),
            np.clip(x + dx, 0, w - 1).ravel(),
        ]

        out = image.copy()
        for ch in range(image.shape[0]):
            out[ch] = map_coordinates(
                image[ch], coords, order=1, mode="reflect"
            ).reshape(h, w)

        out_mask = map_coordinates(
            mask.astype(np.float32), coords, order=0, mode="reflect"
        ).reshape(h, w)

        return out.astype(np.float32), out_mask.astype(np.uint8)


class GINAugmentation:
    """Global Intensity Non-linear augmentation (FedGIN, Nagaraju et al.).

    A randomly initialised 1x1 convolutional network applies a non-linear
    intensity mapping to each slice. The result is blended with the original
    slice by a random factor and rescaled to the original L2 norm, simulating
    scanner appearance variation without changing the overall signal energy.
    """

    def __init__(self, n_layers: int = 4, p: float = 0.70) -> None:
        self.n_layers = n_layers
        self.p = p

    def _build_net(self):
        import torch.nn as nn

        layers: list = []
        in_ch = 1
        for _ in range(self.n_layers):
            out_ch = int(np.random.randint(2, 8))
            layers += [nn.Conv2d(in_ch, out_ch, 1), nn.LeakyReLU(0.2, inplace=True)]
            in_ch = out_ch
        layers.append(nn.Conv2d(in_ch, 1, 1))

        net = nn.Sequential(*layers)
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1)
                nn.init.zeros_(m.bias)
        return net

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() > self.p:
            return image

        import torch

        alpha = np.random.uniform(0.20, 0.60)
        net = self._build_net()
        result = image.copy()

        with torch.no_grad():
            for ch in range(image.shape[0]):
                x = torch.from_numpy(np.ascontiguousarray(image[ch])).float()[None, None]
                out = net(x)[0, 0].numpy()

                blended = alpha * out + (1.0 - alpha) * image[ch]

                orig_norm = float(np.linalg.norm(image[ch]))
                blend_norm = float(np.linalg.norm(blended)) + 1e-8
                if orig_norm > 1e-6:
                    result[ch] = blended * orig_norm / blend_norm
                else:
                    result[ch] = blended

        return np.nan_to_num(result, nan=0.0).astype(np.float32)


class ResNetGINAugmentation:
    """Milder GIN variant used by the "GIN Only" strategy in the recorded runs.

    Same construction as :class:`GINAugmentation` with narrower layers
    (2-5 channels), smaller weight initialisation (std 0.5) and a smaller
    blending factor (0.08-0.22).
    """

    def __init__(self, n_layers: int = 3, p: float = 0.30) -> None:
        self.n_layers = n_layers
        self.p = p

    def _build_net(self):
        import torch.nn as nn

        layers: list = []
        in_ch = 1
        for _ in range(self.n_layers):
            out_ch = int(np.random.randint(2, 6))
            layers += [nn.Conv2d(in_ch, out_ch, 1), nn.LeakyReLU(0.2, inplace=True)]
            in_ch = out_ch
        layers.append(nn.Conv2d(in_ch, 1, 1))

        net = nn.Sequential(*layers)
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.5)
                nn.init.zeros_(m.bias)
        return net

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() > self.p:
            return image

        import torch

        alpha = np.random.uniform(0.08, 0.22)
        net = self._build_net()
        result = image.copy()

        with torch.no_grad():
            for ch in range(image.shape[0]):
                x = torch.from_numpy(np.ascontiguousarray(image[ch])).float()[None, None]
                out = net(x)[0, 0].numpy()

                blended = alpha * out + (1.0 - alpha) * image[ch]

                orig_norm = float(np.linalg.norm(image[ch]))
                blend_norm = float(np.linalg.norm(blended)) + 1e-8
                if orig_norm > 1e-6:
                    result[ch] = blended * orig_norm / blend_norm
                else:
                    result[ch] = blended

        return np.nan_to_num(result, nan=0.0).astype(np.float32)


class BiasFieldAugmentation:
    """Multiplicative low-frequency bias field, simulating B1 inhomogeneity."""

    def __init__(self, p: float = 0.30, max_strength: float = 0.12) -> None:
        self.p = p
        self.max_strength = max_strength

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() > self.p:
            return image

        h, w = image.shape[1], image.shape[2]
        ctrl = np.random.uniform(-1, 1, (4, 4)).astype(np.float32)
        bias = cv2.resize(ctrl, (w, h), interpolation=cv2.INTER_CUBIC)
        bias *= self.max_strength

        result = image.copy()
        for ch in range(image.shape[0]):
            result[ch] = image[ch] * (1.0 + bias)
        return result.astype(np.float32)


class GammaAugmentation:
    """Gamma/contrast variation applied per slice on a min-max normalised copy."""

    def __init__(self, p: float = 0.35, gamma_range: tuple = (0.85, 1.18)) -> None:
        self.p = p
        self.gamma_range = gamma_range

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() > self.p:
            return image

        gamma = np.random.uniform(*self.gamma_range)
        result = image.copy()

        for ch in range(image.shape[0]):
            img = image[ch]
            mn, mx = img.min(), img.max()
            if mx - mn < 1e-8:
                continue
            normed = np.clip((img - mn) / (mx - mn), 0, 1)
            result[ch] = np.power(normed, gamma) * (mx - mn) + mn

        return result.astype(np.float32)


class GaussianNoise:
    """Additive Gaussian sensor noise."""

    def __init__(self, p: float = 0.25, sigma_range: tuple = (0.0, 0.04)) -> None:
        self.p = p
        self.sigma_range = sigma_range

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() > self.p:
            return image
        sigma = np.random.uniform(*self.sigma_range)
        noise = np.random.randn(*image.shape).astype(np.float32) * sigma
        return (image + noise).astype(np.float32)


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


class NoAugmentation:
    """The ``none`` baseline: identity transform."""

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        return image.astype(np.float32), mask.astype(np.uint8)


class GeoElasticPipeline:
    """Final strategy of the paper: geometric transforms + elastic deformation."""

    def __init__(self) -> None:
        self.geometric = GeometricAugmentation(p_flip=0.5, max_angle=15.0, p_rotate=0.70)
        self.elastic = ElasticDeformation(p=0.20, alpha=18.0, sigma=5.0)

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        image, mask = self.geometric(image, mask)
        image, mask = self.elastic(image, mask)
        return image.astype(np.float32), mask.astype(np.uint8)


class GeoBiasPipeline:
    """Geometric transforms + simulated bias field."""

    def __init__(self) -> None:
        self.geometric = GeometricAugmentation()
        self.bias = BiasFieldAugmentation(p=0.30, max_strength=0.12)

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        image, mask = self.geometric(image, mask)
        return self.bias(image).astype(np.float32), mask.astype(np.uint8)


class GeoGammaPipeline:
    """Geometric transforms + gamma variation."""

    def __init__(self) -> None:
        self.geometric = GeometricAugmentation()
        self.gamma = GammaAugmentation(p=0.35, gamma_range=(0.85, 1.18))

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        image, mask = self.geometric(image, mask)
        return self.gamma(image).astype(np.float32), mask.astype(np.uint8)


class GeoNoisePipeline:
    """Geometric transforms + additive Gaussian noise."""

    def __init__(self) -> None:
        self.geometric = GeometricAugmentation()
        self.noise = GaussianNoise(p=0.25, sigma_range=(0.0, 0.04))

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        image, mask = self.geometric(image, mask)
        return self.noise(image).astype(np.float32), mask.astype(np.uint8)


class GINOnlyPipeline:
    """GIN intensity augmentation alone, with no geometric transform.

    This is the "GIN Only" strategy of Table 3. As in the recorded Swin Hybrid
    runs it uses the milder :class:`ResNetGINAugmentation` (3 layers, p=0.30);
    the stronger :class:`GINAugmentation` appears only inside the Augmentation
    Stack.
    """

    def __init__(self) -> None:
        self.gin = ResNetGINAugmentation(n_layers=3, p=0.30)

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        return self.gin(image).astype(np.float32), mask.astype(np.uint8)


class AugmentationStackPipeline:
    """The full stack: geometric + GIN + elastic + bias + gamma + noise."""

    def __init__(self) -> None:
        self.geometric = GeometricAugmentation()
        self.gin = GINAugmentation(n_layers=4, p=0.40)
        self.elastic = ElasticDeformation(p=0.20, alpha=18.0, sigma=5.0)
        self.bias = BiasFieldAugmentation(p=0.30, max_strength=0.12)
        self.gamma = GammaAugmentation(p=0.35, gamma_range=(0.85, 1.18))
        self.noise = GaussianNoise(p=0.25, sigma_range=(0.0, 0.04))

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        image, mask = self.geometric(image, mask)
        image, mask = self.elastic(image, mask)
        image = self.gin(image)
        image = self.bias(image)
        image = self.gamma(image)
        image = self.noise(image)
        return image.astype(np.float32), mask.astype(np.uint8)


#: Strategy name -> factory. Names match the rows of Table 3 of the paper.
AUGMENTATION_STRATEGIES = {
    "none": NoAugmentation,
    "gin_only": GINOnlyPipeline,
    "stack": AugmentationStackPipeline,
    "geo_elastic": GeoElasticPipeline,
    "geo_bias": GeoBiasPipeline,
    "geo_gamma": GeoGammaPipeline,
    "geo_noise": GeoNoisePipeline,
}


def build_augmentation(name: str):
    """Instantiate an augmentation strategy by name.

    Args:
        name: one of the keys of :data:`AUGMENTATION_STRATEGIES`. The final
            strategy reported in the paper is ``"geo_elastic"``.

    Raises:
        ValueError: if ``name`` is not a known strategy.
    """
    key = str(name).lower()
    if key not in AUGMENTATION_STRATEGIES:
        raise ValueError(
            f"Unknown augmentation strategy {name!r}. "
            f"Available: {sorted(AUGMENTATION_STRATEGIES)}"
        )
    return AUGMENTATION_STRATEGIES[key]()
