from __future__ import annotations

import argparse
import logging

from .svg_builder import image_to_svg

PRESETS = {
    "smooth-contours": {
        "smooth": 2,
        "epsilon": 1.0,
        "bezier": True,
        "overlap": 1.0,
        "outline": True,
        "outline_width": 2.0,
        "min_hole_area": 2.0,
    },
    "high-fidelity": {
        "smooth": 0,
        "epsilon": 0.0,
        "bezier": False,
        "overlap": 0.5,
        "outline": False,
        "min_hole_area": 1.0,
    },
}


def apply_preset(ns: argparse.Namespace):
    if not ns.preset:
        return
    cfg = PRESETS.get(ns.preset)
    if not cfg:
        return
    # Overwrite parameters unconditionally (explicit preset wins)
    for k, v in cfg.items():
        if hasattr(ns, k):
            setattr(ns, k, v)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Vectorize raster image into multi-color SVG (smoothing, holes, overlap, strokes, alpha)."
    )
    add = p.add_argument
    add("input", help="Input raster image (PNG/JPG)")
    add("output", help="Output SVG path")
    add(
        "--colors",
        type=int,
        default=8,
        help="Number of color clusters (ignored if --palette is set)",
    )
    add("--palette", help="Comma separated fixed palette hex colors, e.g. #112233,#445566,#aabbcc")
    add("--min-area", type=float, default=50, help="Min outer contour area")
    add("--min-hole-area", type=float, default=5, help="Min hole (inner contour) area")
    add("--bg", default="#ffffff", help="Background color hex")
    add("--supercontour", help="SVG whose paths are added as outline layer")
    add("--contour-color", default="black", help="Stroke color for super contour layer")
    add("--contour-width", type=float, default=2, help="Stroke width for super contour layer")
    add("--smooth", type=int, default=0, help="Chaikin smoothing iterations (0-3)")
    add("--epsilon", type=float, default=0.0, help="Douglas-Peucker tolerance (0 = off)")
    add("--bezier", action="store_true", help="Convert polylines to cubic Bézier")
    add("--outline", action="store_true", help="Add stroke around each filled region")
    add("--outline-color", default="auto", help="Stroke color or 'auto'")
    add("--outline-width", type=float, default=1.5, help="Stroke width for region outlines")
    add(
        "--outline-join",
        default="round",
        choices=["miter", "round", "bevel"],
        help="Stroke line join style",
    )
    add(
        "--outline-cap",
        default="round",
        choices=["butt", "round", "square"],
        help="Stroke line cap style",
    )
    add("--overlap", type=float, default=0.0, help="Mask dilation in px to remove hairline gaps")
    add("--precision", type=int, default=3, help="Decimal precision for coordinates")
    add(
        "--order",
        default="area-desc",
        choices=["area-desc", "area-asc", "orig"],
        help="Drawing order",
    )
    add(
        "--alpha-mode",
        choices=["ignore", "flatten", "binary"],
        default="ignore",
        help="Alpha handling mode",
    )
    add("--alpha-threshold", type=int, default=0, help="Alpha threshold (flatten/binary modes)")
    add(
        "--preset",
        choices=list(PRESETS.keys()),
        help="Apply a predefined parameter set for convenience",
    )
    add(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity",
    )
    return p


def configure_logging(level_name: str):
    level = getattr(logging, level_name.upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    else:
        root.setLevel(level)


def run_from_args(ns: argparse.Namespace):
    configure_logging(ns.log_level)
    apply_preset(ns)
    image_to_svg(
        ns.input,
        ns.output,
        n_colors=ns.colors,
        min_area=ns.min_area,
        bg_color=ns.bg,
        supercontour=ns.supercontour,
        contour_color=ns.contour_color,
        contour_width=ns.contour_width,
        smooth=ns.smooth,
        epsilon=ns.epsilon,
        bezier=ns.bezier,
        outline=ns.outline,
        outline_color=ns.outline_color,
        outline_width=ns.outline_width,
        outline_join=ns.outline_join,
        outline_cap=ns.outline_cap,
        min_hole_area=ns.min_hole_area,
        overlap=ns.overlap,
        precision=max(2, ns.precision),
        order=ns.order,
        alpha_mode=ns.alpha_mode,
        alpha_threshold=max(0, min(255, ns.alpha_threshold)),
        palette=ns.palette,
    )


def main(argv: list[str] | None = None):
    parser = build_parser()
    ns = parser.parse_args(argv)
    run_from_args(ns)


if __name__ == "__main__":  # pragma: no cover
    main()
