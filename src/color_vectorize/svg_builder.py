from __future__ import annotations
import svgwrite
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from .quantize import quantize_image, quantize_with_palette
from .masks import mask_for_color, dilate_mask, build_compound_paths
from .utils import darken_rgb, parse_hex_color, parse_palette
from .alpha import load_image_rgba_handled

__all__ = ["image_to_svg"]

def add_supercontour(dwg: svgwrite.Drawing, svg_path: str, stroke_color: str = "black", stroke_width: float = 2.0):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    for elem in root.findall('.//svg:path', ns):
        d_attr = elem.attrib.get('d')
        if d_attr:
            dwg.add(dwg.path(d=d_attr, fill='none', stroke=stroke_color, stroke_width=stroke_width))


def image_to_svg(input_path: str, output_path: str, n_colors: int = 8, min_area: float = 50, bg_color: str = '#ffffff',
                 supercontour: str | None = None, contour_color: str = 'black', contour_width: float = 2,
                 smooth: int = 0, epsilon: float = 0.0, bezier: bool = False,
                 outline: bool = False, outline_color: str = 'auto', outline_width: float = 1.5,
                 outline_join: str = 'round', outline_cap: str = 'round', min_hole_area: float = 5,
                 overlap: float = 0.0, precision: int = 2, order: str = 'area-desc',
                 alpha_mode: str = 'ignore', alpha_threshold: int = 0,
                 palette: str | None = None):
    rgb = load_image_rgba_handled(input_path, bg_color, alpha_mode=alpha_mode, alpha_threshold=alpha_threshold)
    h, w = rgb.shape[:2]

    if palette:
        palette_list = parse_palette(palette)
        _, label_img, palette_arr = quantize_with_palette(rgb, palette_list)
    else:
        _, label_img, palette_arr = quantize_image(rgb, n_colors)

    dwg = svgwrite.Drawing(str(output_path), size=(w, h))
    dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill=bg_color))

    path_records: list[dict] = []
    for idx, color in enumerate(palette_arr):
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
    return output_path
