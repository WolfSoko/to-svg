from __future__ import annotations

import logging
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

__all__ = ["quantize_image", "quantize_with_palette"]

ColorArray = NDArray[np.uint8]
LabelArray = NDArray[np.int32]


def quantize_image(
    image: NDArray[np.uint8], n_colors: int = 8
) -> tuple[ColorArray, NDArray[np.int32], ColorArray]:
    """Quantize image to up to n_colors using KMeans.

    Returns: (quantized_rgb, label_2d, palette_array)
    """
    data: NDArray[np.uint8] = image.reshape((-1, 3))
    # Distinct color estimation with sampling guard for very large images
    if data.shape[0] > 200_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(data.shape[0], 200_000, replace=False)
        # Explicit cast to satisfy numpy stubs (advanced indexing type narrowing)
        sample: NDArray[np.uint8] = cast(NDArray[np.uint8], data[idx])
    else:
        sample = data
    # Use axis=0 unique to get distinct RGB triplets; yields 2D array (k,3)
    # Cast to int64 for stable hashing regardless of original dtype.
    sample_int: NDArray[np.int64] = sample.astype(np.int64, copy=False)
    uniq: NDArray[np.int64] = np.unique(sample_int, axis=0)
    distinct: int = int(uniq.shape[0])
    k = max(1, min(n_colors, distinct))
    if k < n_colors:
        logger.info(
            "Reduced requested clusters from %d to %d (distinct colors=%d)", n_colors, k, distinct
        )
    else:
        logger.debug("Using %d clusters (distinct colors=%d)", k, distinct)
    kmeans = KMeans(n_clusters=k, n_init=4, random_state=42)
    labels_1d: NDArray[np.int32] = kmeans.fit_predict(data).astype(np.int32, copy=False)  # type: ignore[assignment]
    centers: NDArray[np.float64] = kmeans.cluster_centers_
    palette: ColorArray = centers.astype(np.uint8)
    quant: ColorArray = palette[labels_1d].reshape(image.shape)  # type: ignore[index]
    logger.debug("Quantization finished: palette_size=%d", int(palette.shape[0]))
    return quant, labels_1d.reshape(image.shape[:2]), palette


def quantize_with_palette(
    image: NDArray[np.uint8], palette_list
) -> tuple[ColorArray, NDArray[np.int32], ColorArray]:
    palette: ColorArray = np.array(palette_list, dtype=np.uint8)
    logger.info("Using fixed palette of size %d", int(palette.shape[0]))
    img_flat = image.reshape(-1, 3).astype(np.int16)
    pal = palette.astype(np.int16)
    dists = np.sum((img_flat[:, None, :] - pal[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1).astype(np.int32)
    quant = palette[labels].reshape(image.shape)  # type: ignore[index]
    logger.debug("Fixed palette assignment done")
    return quant, labels.reshape(image.shape[:2]), palette
