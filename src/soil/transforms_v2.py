"""Phase 6 V2 augmentation pipeline + Test-Time Augmentation (TTA).

V2's training transform is deliberately heavy compared to V1's modest
augmentation (``src/soil/transforms.py``) — strong rotation, flips, two
elastic-style spatial distortions, stronger colour jitter, camera noise
simulation, and small CoarseDropout holes. Goal: push the texture head
from V1's 67.86% test top-1 toward the 75-82% range without changing
the locked EfficientNet-B0 backbone or the 224x224 input size.

V2 does NOT modify ``src/soil/transforms.py`` — V1's eval transform is
imported and reused for both the V1-style validation loaders and as the
**base** for the TTA views built here.

Albumentations 2.x note: the installed version (``albumentations>=2.0``)
moved several APIs:

- ``RandomResizedCrop`` now takes ``size=(h, w)`` instead of
  ``height/width``.
- ``CoarseDropout`` uses ``num_holes_range`` /
  ``hole_height_range`` / ``hole_width_range`` instead of
  ``max_holes`` / ``max_height`` / ``max_width``.
- ``ElasticTransform`` removed ``alpha_affine``.
- ``GaussNoise`` switched from ``var_limit`` (in [0, 255]) to
  ``std_range`` (in [0, 1]).

The spec's example block in ``PHASE6_V2_AUGMENTATION_PROMPT.md`` was
written against the older API. The code below uses the modern 2.x form
with equivalent magnitudes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from albumentations import Compose


def build_soil_train_aug_v2(
    img_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> "Compose":
    """V2 strong-augmentation training pipeline (albumentations 2.x)."""
    import albumentations as A  # noqa: PLC0415
    import cv2  # noqa: PLC0415
    from albumentations.pytorch import ToTensorV2  # noqa: PLC0415

    # CoarseDropout hole-size budget in pixels
    hole_max = max(1, int(img_size * 0.10))
    hole_min = max(1, int(img_size * 0.03))

    # GaussNoise std_range in 0-1 scale; the spec's var=(5, 30) on 0-255
    # corresponds to std ~= sqrt(var)/255  -> ~(0.0088, 0.0215).
    gauss_std_range = (0.0088, 0.0215)

    return A.Compose(
        [
            # ----- Geometric: wider scale + orientation variation
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.7),

            # Small shift + tiny zoom + ±15° rotation, separately from
            # the bigger Rotate above. Black borders fill the empty
            # corners after rotation.
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT, p=0.3,
            ),

            # Texture-aware spatial distortion. Choose ONE per call.
            A.OneOf(
                [
                    A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
                    A.ElasticTransform(alpha=15, sigma=4, p=1.0),
                ],
                p=0.3,
            ),

            # ----- Colour
            A.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.6,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.4,
            ),

            # ----- Camera-noise simulation. One of blur OR Gaussian noise.
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(std_range=gauss_std_range, p=1.0),
                ],
                p=0.3,
            ),

            # ----- Cutout-style: small black holes.
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(hole_min, hole_max),
                hole_width_range=(hole_min, hole_max),
                fill=0,
                p=0.3,
            ),

            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def build_tta_views(
    img_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> list["Compose"]:
    """5 deterministic albumentations transforms for test-time augmentation.

    Returns
    -------
    list[Compose]
        ``[original, hflip, vflip, rot90, rot270]`` — applied to one
        image each at eval time; averaging the resulting logits gives
        the final per-task prediction. All transforms share the same
        Resize-256 -> CenterCrop-``img_size`` base so the spatial
        framing matches V1's eval pipeline exactly.
    """
    import albumentations as A  # noqa: PLC0415
    from albumentations.pytorch import ToTensorV2  # noqa: PLC0415

    # Base resize stays at 256 (matches V1 build_soil_eval_aug semantics)
    base_size = int(round(img_size * 256 / 224))

    base = [
        A.Resize(base_size, base_size),
        A.CenterCrop(img_size, img_size),
    ]
    norm = [A.Normalize(mean=mean, std=std), ToTensorV2()]

    return [
        A.Compose(base + norm),                                # original
        A.Compose(base + [A.HorizontalFlip(p=1.0)] + norm),    # hflip
        A.Compose(base + [A.VerticalFlip(p=1.0)] + norm),      # vflip
        A.Compose(base + [A.Rotate(limit=(90, 90), p=1.0)] + norm),   # rot90
        A.Compose(base + [A.Rotate(limit=(270, 270), p=1.0)] + norm), # rot270
    ]


__all__ = ["build_soil_train_aug_v2", "build_tta_views"]
