from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans

__all__ = ["quantize_image"]

def quantize_image(image, n_colors: int = 8):
    data = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, n_init=4, random_state=42)
    labels = kmeans.fit_predict(data)
    palette = np.uint8(kmeans.cluster_centers_)
    quant = palette[labels].reshape(image.shape)
    return quant, labels.reshape(image.shape[:2]), palette

