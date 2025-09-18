import cv2
import numpy as np
from sklearn.cluster import KMeans
import svgwrite
import argparse
import xml.etree.ElementTree as ET
import math

# Helper: parse hex color to RGB tuple
def parse_hex_color(s):
    s = s.strip()
    if s.startswith('#'):
        s = s[1:]
    if len(s) == 3:
        s = ''.join(c*2 for c in s)
    if len(s) != 6:
        raise ValueError("Invalid hex color: expected 3 or 6 digits")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (r, g, b)

# Color quantization via KMeans
def quantize_image(image, n_colors=8):
    data = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, n_init=4, random_state=42)
    labels = kmeans.fit_predict(data)
    palette = np.uint8(kmeans.cluster_centers_)
    quant = palette[labels].reshape(image.shape)
    return quant, labels.reshape(image.shape[:2]), palette

# Binary mask for one color cluster
def mask_for_color(label_img, color_idx):
    return (label_img == color_idx).astype(np.uint8) * 255

# Chaikin corner cutting for smoothing polygons
def chaikin(points, iterations):
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

# Formatting helper for coordinate precision
def fmt_point(p, precision):
    return f"{p[0]:.{precision}f} {p[1]:.{precision}f}"

# Convert polygon edges to cubic Bézier segments (simple linear subdivision)
def poly_to_cubic_beziers(points, precision):
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

# Build path string from points
def contour_points_to_path(points, bezier=False, precision=2):
    if bezier:
        return poly_to_cubic_beziers(points, precision)
    d = f"M {fmt_point(points[0], precision)} "
    for p in points[1:]:
        d += f"L {fmt_point(p, precision)} "
    d += "Z"
    return d

# Simplify + smooth contour points
def prepare_points(cnt, epsilon=0.0, smooth=0):
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

# Darken color for auto stroke
def darken_rgb(rgb, factor=0.6):
    r, g, b = rgb
    return (max(0, int(r * factor)), max(0, int(g * factor)), max(0, int(b * factor)))

# Add external super contour paths to drawing
def add_supercontour(dwg, svg_path, stroke_color="black", stroke_width=2):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    for elem in root.findall('.//svg:path', ns):
        d_attr = elem.attrib.get('d')
        if d_attr:
            dwg.add(dwg.path(d=d_attr, fill='none', stroke=stroke_color, stroke_width=stroke_width))

# Collect outer contours plus holes (compound path via evenodd)
def build_compound_paths(contours, hierarchy, min_area, min_hole_area, epsilon, smooth, bezier, precision):
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
        parent = hier[i][3]
        if parent != -1:
            continue
        area_outer = cv2.contourArea(contours[i])
        if area_outer <= min_area:
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
        results.append({'d': path_d, 'area': area_outer})
    return results

# Dilate mask to create overlap and avoid gaps
def dilate_mask(mask, overlap):
    if overlap <= 0:
        return mask
    radius = int(round(overlap))
    if radius < 1:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask, k)

# Main conversion routine
def image_to_svg(input_path, output_path, n_colors=8, min_area=50, bg_color='#ffffff',
                 supercontour=None, contour_color='black', contour_width=2,
                 smooth=0, epsilon=0.0, bezier=False,
                 outline=False, outline_color='auto', outline_width=1.5,
                 outline_join='round', outline_cap='round', min_hole_area=5,
                 overlap=0.0, precision=2, order='area-desc',
                 alpha_mode='ignore', alpha_threshold=0):
    # Decide loading flags (keep alpha if we want to process it)
    flags = cv2.IMREAD_UNCHANGED if alpha_mode != 'ignore' else cv2.IMREAD_COLOR
    raw = cv2.imread(str(input_path), flags)
    if raw is None:
        raise FileNotFoundError(f"Image not found: {input_path}")

    bg_rgb = parse_hex_color(bg_color)

    # Alpha handling
    if raw.ndim == 3 and raw.shape[2] == 4 and alpha_mode != 'ignore':
        bgr = raw[:, :, :3]
        alpha = raw[:, :, 3]
        if alpha_mode == 'flatten':
            a = (alpha.astype(np.float32) / 255.0)
            a = np.clip(a, 0, 1)
            if alpha_threshold > 0:
                a = np.where(alpha <= alpha_threshold, 0.0, a)
            bg_arr = np.array(bg_rgb, dtype=np.float32)[None, None, :]
            rgb_src = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            comp = (rgb_src * a[..., None] + bg_arr * (1.0 - a[..., None]))
            img_rgb = comp.astype(np.uint8)
        elif alpha_mode == 'binary':
            mask = alpha > alpha_threshold
            rgb_src = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img_rgb = rgb_src.copy()
            img_rgb[~mask] = bg_rgb
        else:
            img_rgb = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        if raw.ndim == 2:
            img_rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        else:
            if raw.shape[2] == 4:
                raw = raw[:, :, :3]
            img_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    h, w = img_rgb.shape[:2]

    _, label_img, palette = quantize_image(img_rgb, n_colors)

    dwg = svgwrite.Drawing(str(output_path), size=(w, h))
    dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill=bg_color))

    path_records = []
    for idx, color in enumerate(palette):
        mask = mask_for_color(label_img, idx)
        mask = dilate_mask(mask, overlap)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        paths = build_compound_paths(contours, hierarchy, min_area, min_hole_area, epsilon, smooth, bezier, precision)
        if not paths:
            continue
        fill_hex = svgwrite.utils.rgb(int(color[0]), int(color[1]), int(color[2]))
        if outline_color == 'auto':
            stroke_rgb = darken_rgb(color, 0.55)
            stroke_hex = svgwrite.utils.rgb(*stroke_rgb)
        else:
            stroke_hex = outline_color
        for p in paths:
            path_records.append({
                'd': p['d'],
                'area': p['area'],
                'fill': fill_hex,
                'stroke': stroke_hex if outline else 'none'
            })

    if order == 'area-desc':
        path_records.sort(key=lambda x: x['area'], reverse=True)
    elif order == 'area-asc':
        path_records.sort(key=lambda x: x['area'])

    for rec in path_records:
        if outline and rec['stroke'] != 'none':
            dwg.add(dwg.path(d=rec['d'], fill=rec['fill'], stroke=rec['stroke'],
                              stroke_width=outline_width, fill_rule='evenodd',
                              stroke_linejoin=outline_join, stroke_linecap=outline_cap))
        else:
            dwg.add(dwg.path(d=rec['d'], fill=rec['fill'], stroke='none', fill_rule='evenodd'))

    if supercontour:
        add_supercontour(dwg, supercontour, stroke_color=contour_color, stroke_width=contour_width)

    dwg.save()
    print(f"SVG written to {output_path}")

# CLI entry point
def main():
    parser = argparse.ArgumentParser(description="Vectorize a raster image into a multi-color SVG with smoothing, holes, overlap, strokes, and alpha handling.")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--colors", type=int, default=8, help="Number of color clusters")
    parser.add_argument("--min-area", type=float, default=50, help="Min area of outer contours")
    parser.add_argument("--min-hole-area", type=float, default=5, help="Min area of holes")
    parser.add_argument("--bg", default="#ffffff", help="Background color hex")
    parser.add_argument("--supercontour", help="SVG file whose paths are added as an outline layer")
    parser.add_argument("--contour-color", default="black", help="Stroke color for super contour")
    parser.add_argument("--contour-width", type=float, default=2, help="Stroke width for super contour")
    parser.add_argument("--smooth", type=int, default=0, help="Chaikin smoothing iterations")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Douglas-Peucker tolerance (0 disables)")
    parser.add_argument("--bezier", action="store_true", help="Convert line segments to cubic Béziers")
    parser.add_argument("--outline", action="store_true", help="Draw each filled shape with a stroke")
    parser.add_argument("--outline-color", default='auto', help="Stroke color or 'auto' for darkened fill")
    parser.add_argument("--outline-width", type=float, default=1.5, help="Stroke width for shape outlines")
    parser.add_argument("--outline-join", default='round', choices=['miter','round','bevel'], help="Stroke line join")
    parser.add_argument("--outline-cap", default='round', choices=['butt','round','square'], help="Stroke line cap")
    parser.add_argument("--overlap", type=float, default=0.0, help="Mask dilation in pixels to eliminate gaps")
    parser.add_argument("--precision", type=int, default=3, help="Decimal places for coordinates (>=2)")
    parser.add_argument("--order", default='area-desc', choices=['area-desc','area-asc','orig'], help="Drawing order")
    parser.add_argument("--alpha-mode", choices=['ignore','flatten','binary'], default='ignore', help="Alpha handling mode")
    parser.add_argument("--alpha-threshold", type=int, default=0, help="Alpha threshold (0-255)")
    args = parser.parse_args()

    image_to_svg(
        args.input,
        args.output,
        n_colors=args.colors,
        min_area=args.min_area,
        bg_color=args.bg,
        supercontour=args.supercontour,
        contour_color=args.contour_color,
        contour_width=args.contour_width,
        smooth=args.smooth,
        epsilon=args.epsilon,
        bezier=args.bezier,
        outline=args.outline,
        outline_color=args.outline_color,
        outline_width=args.outline_width,
        outline_join=args.outline_join,
        outline_cap=args.outline_cap,
        min_hole_area=args.min_hole_area,
        overlap=args.overlap,
        precision=max(2, args.precision),
        order=args.order,
        alpha_mode=args.alpha_mode,
        alpha_threshold=max(0, min(255, args.alpha_threshold))
    )

if __name__ == "__main__":
    main()
