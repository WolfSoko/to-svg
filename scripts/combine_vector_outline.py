#!/usr/bin/env python
"""Convenience wrapper to generate colored SVG plus optional high quality outline layer.

Usage (only color fill):
    python scripts/combine_vector_outline.py input.png output.svg --colors 8

Usage (with existing outline SVG to overlay):
    python scripts/combine_vector_outline.py input.png output.svg --colors 8 --super eprivacy_logo.svg

Typical smooth pleasant result:
    python scripts/combine_vector_outline.py input.png output.svg --preset smooth-contours --colors 8 --overlap 1

This is a thin wrapper around color_vectorize.image_to_svg adding sane defaults and existence checks.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# Make sure package import works when running from repo root
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from color_vectorize import image_to_svg  # type: ignore  # noqa: E402

PRESET_DEFAULTS = {
    "smooth-contours": dict(smooth=2, epsilon=1.0, bezier=True, overlap=1.0, outline=True, outline_width=2.0, min_hole_area=2.0),
    "high-fidelity": dict(smooth=0, epsilon=0.0, bezier=False, overlap=0.5, outline=False, min_hole_area=1.0),
}

def parse_args():
    ap = argparse.ArgumentParser(description="Combine color vectorization with optional external outline layer.")
    ap.add_argument("input", help="Input raster (PNG/JPG)")
    ap.add_argument("output", help="Output SVG")
    ap.add_argument("--colors", type=int, default=8, help="Number of color clusters")
    ap.add_argument("--palette", help="Fixed hex palette list e.g. #112233,#aabbcc")
    ap.add_argument("--super", dest="supercontour", help="External outline SVG to overlay")
    ap.add_argument("--bg", default="#ffffff", help="Background color")
    ap.add_argument("--smooth", type=int, default=0)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--bezier", action="store_true")
    ap.add_argument("--outline", action="store_true")
    ap.add_argument("--outline-color", default="auto")
    ap.add_argument("--outline-width", type=float, default=1.5)
    ap.add_argument("--overlap", type=float, default=0.0)
    ap.add_argument("--min-area", type=float, default=50)
    ap.add_argument("--min-hole-area", type=float, default=5)
    ap.add_argument("--precision", type=int, default=3)
    ap.add_argument("--alpha-mode", choices=["ignore","flatten","binary"], default="ignore")
    ap.add_argument("--alpha-threshold", type=int, default=0)
    ap.add_argument("--preset", choices=list(PRESET_DEFAULTS.keys()))
    return ap.parse_args()


def apply_preset(ns):
    if ns.preset:
        for k, v in PRESET_DEFAULTS[ns.preset].items():
            setattr(ns, k, v)


def main():
    ns = parse_args()
    apply_preset(ns)
    in_path = pathlib.Path(ns.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    super_path = None
    if ns.supercontour:
        p = pathlib.Path(ns.supercontour)
        if p.exists():
            super_path = str(p)
        else:
            print(f"[warn] super contour SVG not found: {p} (ignored)")
    image_to_svg(
        str(in_path),
        ns.output,
        n_colors=ns.colors,
        min_area=ns.min_area,
        bg_color=ns.bg,
        supercontour=super_path,
        contour_color="black",
        contour_width=2.0,
        smooth=ns.smooth,
        epsilon=ns.epsilon,
        bezier=ns.bezier,
        outline=ns.outline,
        outline_color=ns.outline_color,
        outline_width=ns.outline_width,
        outline_join='round',
        outline_cap='round',
        min_hole_area=ns.min_hole_area,
        overlap=ns.overlap,
        precision=ns.precision,
        order='area-desc',
        alpha_mode=ns.alpha_mode,
        alpha_threshold=ns.alpha_threshold,
        palette=ns.palette,
    )
    print(f"Generated {ns.output}")

if __name__ == "__main__":
    main()
