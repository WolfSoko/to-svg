import cv2
import numpy as np
from sklearn.cluster import KMeans
import svgwrite
import argparse
import xml.etree.ElementTree as ET
import math

# Neu: Hilfsfunktionen für Alpha / Farben
def parse_hex_color(s):
    s = s.strip()
    if s.startswith('#'):
        s = s[1:]
    if len(s) == 3:
        s = ''.join(c*2 for c in s)
    if len(s) != 6:
        raise ValueError("Ungültige Farb-Hex: expected 3 oder 6 Stellen")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (r, g, b)


def quantize_image(image, n_colors=8):
    data = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, n_init=4, random_state=42)
    labels = kmeans.fit_predict(data)
    palette = np.uint8(kmeans.cluster_centers_)
    quant = palette[labels].reshape(image.shape)
    return quant, labels.reshape(image.shape[:2]), palette


def mask_for_color(label_img, color_idx):
    return (label_img == color_idx).astype(np.uint8) * 255


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

# Neue Utility für dynamische Präzision
def fmt_point(p, precision):
    return f"{p[0]:.{precision}f} {p[1]:.{precision}f}"


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


def contour_points_to_path(points, bezier=False, precision=2):
    if bezier:
        return poly_to_cubic_beziers(points, precision)
    d = f"M {fmt_point(points[0], precision)} "
    for p in points[1:]:
        d += f"L {fmt_point(p, precision)} "
    d += "Z"
    return d


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


def darken_rgb(rgb, factor=0.6):
    r, g, b = rgb
    return (max(0, int(r * factor)), max(0, int(g * factor)), max(0, int(b * factor)))


def add_supercontour(dwg, svg_path, stroke_color="black", stroke_width=2):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    for elem in root.findall('.//svg:path', ns):
        d_attr = elem.attrib.get('d')
        if d_attr:
            dwg.add(dwg.path(d=d_attr, fill='none', stroke=stroke_color, stroke_width=stroke_width))


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

# NEU: Masken-Dilation für Überlappung
def dilate_mask(mask, overlap):
    if overlap <= 0:
        return mask
    radius = int(round(overlap))
    if radius < 1:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask, k)


def image_to_svg(input_path, output_path, n_colors=8, min_area=50, bg_color='#ffffff',
                 supercontour=None, contour_color='black', contour_width=2,
                 smooth=0, epsilon=0.0, bezier=False,
                 outline=False, outline_color='auto', outline_width=1.5,
                 outline_join='round', outline_cap='round', min_hole_area=5,
                 overlap=0.0, precision=2, order='area-desc',
                 alpha_mode='ignore', alpha_threshold=0):
    # Eingabe lesen (ggf. mit Alpha)
    flags = cv2.IMREAD_UNCHANGED if alpha_mode != 'ignore' else cv2.IMREAD_COLOR
    raw = cv2.imread(str(input_path), flags)
    if raw is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {input_path}")

    bg_rgb = parse_hex_color(bg_color)

    if raw.ndim == 3 and raw.shape[2] == 4 and alpha_mode != 'ignore':
        bgr = raw[:, :, :3]
        alpha = raw[:, :, 3]
        if alpha_mode == 'flatten':
            a = (alpha.astype(np.float32) / 255.0)
            a = np.clip(a, 0, 1)
            # Schwellwert: alles <= threshold als 0 behandeln (voll transparent)
            if alpha_threshold > 0:
                a = np.where(alpha <= alpha_threshold, 0.0, a)
            bg_arr = np.array(bg_rgb, dtype=np.float32)[None, None, :]
            # BGR -> RGB nach cv2-Konventionen anpassen später
            rgb_src = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            comp = (rgb_src * a[..., None] + bg_arr * (1.0 - a[..., None]))
            img_rgb = comp.astype(np.uint8)
        elif alpha_mode == 'binary':
            mask = alpha > alpha_threshold
            rgb_src = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img_rgb = rgb_src.copy()
            img_rgb[~mask] = bg_rgb
        else:  # sollte nicht eintreten
            img_rgb = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        # Kein Alpha oder ignoriert
        if raw.ndim == 2:
            img_rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        else:
            # raw BGR
            if raw.shape[2] == 4:  # Alpha vorhanden aber ignoriert
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
    print(f"SVG gespeichert unter {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Farbbild zu farbigem SVG vektorisieren (Alpha-Optionen, Löcher, Strokes, Overlap)")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--colors", type=int, default=8)
    parser.add_argument("--min-area", type=float, default=50, help="Min Fläche äußerer Konturen")
    parser.add_argument("--min-hole-area", type=float, default=5, help="Min Fläche von Löchern (Augen)")
    parser.add_argument("--bg", default="#ffffff")
    parser.add_argument("--supercontour")
    parser.add_argument("--contour-color", default="black")
    parser.add_argument("--contour-width", type=float, default=2)
    parser.add_argument("--smooth", type=int, default=0, help="Chaikin-Iterationen")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Douglas-Peucker Toleranz")
    parser.add_argument("--bezier", action="store_true", help="Polylinien in kubische Bézier umwandeln")
    parser.add_argument("--outline", action="store_true", help="Farbflächen mit Stroke zeichnen")
    parser.add_argument("--outline-color", default='auto', help="Stroke-Farbe oder 'auto'")
    parser.add_argument("--outline-width", type=float, default=1.5)
    parser.add_argument("--outline-join", default='round', choices=['miter','round','bevel'])
    parser.add_argument("--outline-cap", default='round', choices=['butt','round','square'])
    parser.add_argument("--overlap", type=float, default=0.0, help="Masken-Dilation in Pixeln zur Überlappung")
    parser.add_argument("--precision", type=int, default=3, help="Dezimalstellen für Koordinaten (>=2)")
    parser.add_argument("--order", default='area-desc', choices=['area-desc','area-asc','orig'], help="Zeichenreihenfolge")
    # Neu: Alpha Parameter
    parser.add_argument("--alpha-mode", choices=['ignore','flatten','binary'], default='ignore', help="Alpha-Verarbeitung")
    parser.add_argument("--alpha-threshold", type=int, default=0, help="Schwellwert für Alpha (0-255)")
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
