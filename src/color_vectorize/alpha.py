from __future__ import annotations

import cv2
import numpy as np

from .utils import parse_hex_color

__all__ = ["load_image_rgba_handled"]


# Helper kept un-annotated internally to avoid numpy typing conflicts
def _flatten_rgba(
    bgr: np.ndarray, alpha: np.ndarray, bg_rgb: tuple[int, int, int], threshold: int
) -> np.ndarray:
    a = alpha.astype(np.float32) / 255.0
    if threshold > 0:
        # Zero out below / equal threshold
        mask = alpha <= threshold
        if mask.any():
            a = a.copy()
            a[mask] = 0.0
    # Composite over background
    inv = 1.0 - a
    rgb_src = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    bg_vec = np.array(bg_rgb, dtype=np.float32)
    comp = rgb_src * a[..., None] + bg_vec[None, None, :] * inv[..., None]
    return comp.astype(np.uint8)


def load_image_rgba_handled(
    path: str, bg_color: str, alpha_mode: str = "ignore", alpha_threshold: int = 0
) -> np.ndarray:
    """Load image and resolve alpha according to mode (ignore | flatten | binary)."""
    flags = cv2.IMREAD_UNCHANGED if alpha_mode != "ignore" else cv2.IMREAD_COLOR
    raw = cv2.imread(path, flags)
    if raw is None:
        raise FileNotFoundError(path)
    bg = parse_hex_color(bg_color)

    if raw.ndim == 3 and raw.shape[2] == 4 and alpha_mode != "ignore":
        bgr = raw[:, :, :3]
        alpha = raw[:, :, 3]
        if alpha_mode == "flatten":
            return _flatten_rgba(bgr, alpha, bg, alpha_threshold)
        if alpha_mode == "binary":
            mask = alpha > alpha_threshold
            rgb_src = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_src[~mask] = bg
            return rgb_src
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    if raw.ndim == 3 and raw.shape[2] == 4:
        raw = raw[:, :, :3]
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
