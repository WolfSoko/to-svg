from __future__ import annotations
import cv2
import numpy as np

__all__ = [
    "chaikin",
    "fmt_point",
    "poly_to_cubic_beziers",
    "contour_points_to_path",
    "prepare_points",
]

def chaikin(points: np.ndarray, iterations: int):
    pts = points.astype(float)
    for _ in range(int(max(0, iterations))):
        new_pts = []
        for i in range(len(pts)):
            p0 = pts[i]
            p1 = pts[(i + 1) % len(pts)]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_pts.extend([q, r])
        pts = np.array(new_pts)
    return pts

def fmt_point(p, precision: int):
    return f"{p[0]:.{precision}f} {p[1]:.{precision}f}"

def poly_to_cubic_beziers(points: np.ndarray, precision: int):
    path = f"M {fmt_point(points[0], precision)} "
    for i in range(1, len(points)):
        p0 = points[i - 1]
        p1 = points[i]
        c1 = p0 + (p1 - p0) / 3.0
        c2 = p0 + 2 * (p1 - p0) / 3.0
        path += ("C "
                 f"{fmt_point(c1, precision)} "
                 f"{fmt_point(c2, precision)} "
                 f"{fmt_point(p1, precision)} ")
    path += "Z"
    return path

def contour_points_to_path(points: np.ndarray, bezier: bool = False, precision: int = 2):
    if bezier:
        return poly_to_cubic_beziers(points, precision)
    d = f"M {fmt_point(points[0], precision)} "
    for p in points[1:]:
        d += f"L {fmt_point(p, precision)} "
    d += "Z"
    return d

def prepare_points(cnt: np.ndarray, epsilon: float = 0.0, smooth: int = 0):
    from .geometry import chaikin  # local import for clarity
    pts = cnt.squeeze()
    if pts.ndim != 2:
        return None
    if epsilon > 0:
        approx = cv2.approxPolyDP(pts, epsilon, True).squeeze()
        if approx.ndim == 1 or len(approx) < 3:
            return None
        pts = approx
    if smooth > 0:
        pts = chaikin(pts, smooth)
    return pts

