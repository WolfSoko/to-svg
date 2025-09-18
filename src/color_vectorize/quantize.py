from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

__all__ = ["quantize_image", "quantize_with_palette"]

def quantize_image(image, n_colors: int = 8):
    """Quantize image to up to n_colors using KMeans.

    If the image contains fewer distinct colors than requested, the cluster
    count is reduced to avoid convergence warnings.
    Returns: (quantized_rgb, label_2d, palette_array)
    """
    data = image.reshape((-1, 3))
    # Distinct color estimation with sampling guard for very large images
    if data.shape[0] > 200_000:
        rng = np.random.default_rng(42)
        sample = data[rng.choice(data.shape[0], 200_000, replace=False)]
    else:
        sample = data
    uniq = np.unique(sample.view([("r", sample.dtype), ("g", sample.dtype), ("b", sample.dtype)]))
    distinct = len(uniq)
    k = max(1, min(n_colors, distinct))
    if k < n_colors:
        logger.info(
            "Reduced requested clusters from %d to %d (distinct colors=%d)",
            n_colors, k, distinct
        )
    else:
        logger.debug("Using %d clusters (distinct colors=%d)", k, distinct)
    kmeans = KMeans(n_clusters=k, n_init=4, random_state=42)
    labels = kmeans.fit_predict(data)
    palette = np.uint8(kmeans.cluster_centers_)
    quant = palette[labels].reshape(image.shape)
    logger.debug("Quantization finished: palette_size=%d", len(palette))
    return quant, labels.reshape(image.shape[:2]), palette


def quantize_with_palette(image, palette_list):
    palette = np.array(palette_list, dtype=np.uint8)
    logger.info("Using fixed palette of size %d", len(palette))
    img_flat = image.reshape(-1, 3).astype(np.int16)
    pal = palette.astype(np.int16)
    dists = np.sum((img_flat[:, None, :] - pal[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)
    quant = palette[labels].reshape(image.shape)
    logger.debug("Fixed palette assignment done")
    return quant, labels.reshape(image.shape[:2]), palette
