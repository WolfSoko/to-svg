from __future__ import annotations
import cv2
import numpy as np
from .utils import parse_hex_color

__all__ = ["load_image_rgba_handled"]

def load_image_rgba_handled(path: str, bg_color: str, alpha_mode: str = "ignore", alpha_threshold: int = 0):
    """Load image and resolve alpha according to mode.

    alpha_mode: ignore | flatten | binary
    Returns RGB ndarray (H,W,3) uint8
    """
    flags = cv2.IMREAD_UNCHANGED if alpha_mode != 'ignore' else cv2.IMREAD_COLOR
    raw = cv2.imread(path, flags)
    if raw is None:
        raise FileNotFoundError(path)
    bg = parse_hex_color(bg_color)

    if raw.ndim == 3 and raw.shape[2] == 4 and alpha_mode != 'ignore':
        bgr = raw[:, :, :3]
        alpha = raw[:, :, 3]
        if alpha_mode == 'flatten':
            a = (alpha.astype(np.float32) / 255.0)
            a = np.clip(a, 0, 1)
            if alpha_threshold > 0:
                a = np.where(alpha <= alpha_threshold, 0.0, a)
            bg_arr = np.array(bg, dtype=np.float32)[None, None, :]
            rgb_src = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            comp = (rgb_src * a[..., None] + bg_arr * (1.0 - a[..., None]))
            return comp.astype(np.uint8)
        if alpha_mode == 'binary':
            mask = alpha > alpha_threshold
            rgb_src = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_src[~mask] = bg
            return rgb_src
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # No alpha processing
    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    if raw.ndim == 3 and raw.shape[2] == 4:
        raw = raw[:, :, :3]
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

