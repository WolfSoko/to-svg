from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import cv2
import svgwrite

from .alpha import load_image_rgba_handled
from .masks import build_compound_paths, dilate_mask, mask_for_color
from .quantize import quantize_image, quantize_with_palette
from .utils import darken_rgb, parse_palette

logger = logging.getLogger(__name__)

__all__ = ["image_to_svg"]


def add_supercontour(dwg: svgwrite.Drawing, svg_path: str, stroke_color: str = "black", stroke_width: float = 2.0):
    logger.info("Adding super contour layer from %s", svg_path)
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    added = 0
    for elem in root.findall('.//svg:path', ns):
        d_attr = elem.attrib.get('d')
        if d_attr:
            dwg.add(dwg.path(d=d_attr, fill='none', stroke=stroke_color, stroke_width=stroke_width))
            added += 1
    logger.debug("Added %d super contour paths", added)


def image_to_svg(
    input_path: str,
    output_path: str,
    n_colors: int = 8,
    min_area: float = 50,
    bg_color: str = '#ffffff',
    supercontour: str | None = None,
    contour_color: str = 'black',
    contour_width: float = 2,
    smooth: int = 0,
    epsilon: float = 0.0,
    bezier: bool = False,
    outline: bool = False,
    outline_color: str = 'auto',
    outline_width: float = 1.5,
    outline_join: str = 'round',
    outline_cap: str = 'round',
    min_hole_area: float = 5,
    overlap: float = 0.0,
    precision: int = 2,
    order: str = 'area-desc',
    alpha_mode: str = 'ignore',
    alpha_threshold: int = 0,
    palette: str | None = None,
):
    logger.info("Vectorizing %s -> %s", input_path, output_path)
    logger.debug(
        "Params: n_colors=%d smooth=%d eps=%.3f bezier=%s overlap=%.2f outline=%s alpha_mode=%s palette=%s",
        n_colors, smooth, epsilon, bezier, overlap, outline, alpha_mode, palette,
    )
    rgb = load_image_rgba_handled(input_path, bg_color, alpha_mode=alpha_mode, alpha_threshold=alpha_threshold)
    h, w = rgb.shape[:2]
    logger.debug("Loaded image size=%dx%d", w, h)

    if palette:
        palette_list = parse_palette(palette)
        logger.info("Using fixed palette (%d colors)", len(palette_list))
        _, label_img, palette_arr = quantize_with_palette(rgb, palette_list)
    else:
        _, label_img, palette_arr = quantize_image(rgb, n_colors)
        logger.info("Effective palette size after quantization: %d", len(palette_arr))

    dwg = svgwrite.Drawing(str(output_path), size=(w, h))
    dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill=bg_color))

    path_records: list[dict] = []
    for idx, color in enumerate(palette_arr):
        mask = mask_for_color(label_img, idx)
        if overlap > 0:
            mask = dilate_mask(mask, overlap)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        paths = build_compound_paths(contours, hierarchy, min_area, min_hole_area, epsilon, smooth, bezier, precision)
        if not paths:
            logger.debug("Color #%d produced no paths (area filter?)", idx)
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
        logger.debug("Color #%d -> %d path(s)", idx, len(paths))

    if order == 'area-desc':
        path_records.sort(key=lambda x: x['area'], reverse=True)
    elif order == 'area-asc':
        path_records.sort(key=lambda x: x['area'])
    logger.debug("Sorted %d paths with order=%s", len(path_records), order)

    for rec in path_records:
        if outline and rec['stroke'] != 'none':
            dwg.add(dwg.path(
                d=rec['d'],
                fill=rec['fill'],
                stroke=rec['stroke'],
                stroke_width=outline_width,
                fill_rule='evenodd',
                stroke_linejoin=outline_join,
                stroke_linecap=outline_cap,
            ))
        else:
            dwg.add(dwg.path(d=rec['d'], fill=rec['fill'], stroke='none', fill_rule='evenodd'))

    if supercontour:
        add_supercontour(dwg, supercontour, stroke_color=contour_color, stroke_width=contour_width)

    dwg.save()
    logger.info("Saved SVG (%d paths) to %s", len(path_records), output_path)
    return output_path
