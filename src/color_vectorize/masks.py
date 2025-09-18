from __future__ import annotations
import cv2
import numpy as np
from .geometry import prepare_points, contour_points_to_path

__all__ = [
    "mask_for_color",
    "dilate_mask",
    "build_compound_paths",
]

def mask_for_color(label_img: np.ndarray, color_idx: int):
    return (label_img == color_idx).astype(np.uint8) * 255

def dilate_mask(mask: np.ndarray, overlap: float):
    if overlap <= 0:
        return mask
    radius = int(round(overlap))
    if radius < 1:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask, k)

def build_compound_paths(contours, hierarchy, min_area, min_hole_area, epsilon, smooth, bezier, precision):
    """Return list of {'d': path_string, 'area': float} including holes (evenodd)."""
    results = []
    if hierarchy is None:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= min_area:
                continue
            pts = prepare_points(cnt, epsilon, smooth)
            if pts is None or len(pts) < 3:
                continue
            d = contour_points_to_path(pts, bezier, precision)
            results.append({'d': d, 'area': area})
        return results
    hier = hierarchy[0]
    n = len(contours)
    for i in range(n):
        if hier[i][3] != -1:
            continue
        outer_area = cv2.contourArea(contours[i])
        if outer_area <= min_area:
            continue
        outer_pts = prepare_points(contours[i], epsilon, smooth)
        if outer_pts is None or len(outer_pts) < 3:
            continue
        path_d = contour_points_to_path(outer_pts, bezier, precision)
        child = hier[i][2]
        while child != -1:
            hole_area = cv2.contourArea(contours[child])
            if hole_area > min_hole_area:
                hole_pts = prepare_points(contours[child], epsilon, smooth)
                if hole_pts is not None and len(hole_pts) >= 3:
                    path_d += " " + contour_points_to_path(hole_pts, bezier, precision)
            child = hier[child][0]
        results.append({'d': path_d, 'area': outer_area})
    return results

