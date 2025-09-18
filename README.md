# color_vectorize

Multi-color raster → SVG vectorization featuring:
- KMeans / fixed palette color quantization
- Chaikin smoothing + optional Douglas-Peucker simplification
- Optional cubic Bézier conversion for smoother organic curves
- Hole support (RETR_CCOMP + evenodd fill rule)
- Per-shape strokes with automatic darkened outline option
- Overlap (mask dilation) to eliminate hairline gaps between regions
- Alpha handling (ignore / flatten / binary)
- Optional import of a pre-drawn high quality super contour layer
- Preset profiles for quick usage

## Quick Start
```bash
python -m venv .venv
.venv\\Scripts\\activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Run via module (dev mode):
python -m color_vectorize.cli examples/assets/eprivacy_logo.png out.svg --preset smooth-contours
# Or after editable install:
color-vectorize examples/assets/eprivacy_logo.png out.svg --preset smooth-contours
```

## CLI Highlights
```
--colors N                  number of color clusters (ignored if --palette)
--palette C1,C2,...         fixed hex palette (#RRGGBB)
--min-area PX               minimum outer contour area
--min-hole-area PX          minimum hole (inner contour) area
--smooth N                  Chaikin iterations (0–3 typical)
--epsilon F                 Douglas-Peucker tolerance (0 = off)
--bezier                    convert polylines to cubic Bézier
--outline                   add stroke around each filled region
--outline-color C|auto      stroke color or auto (darkened fill)
--outline-width F           region outline stroke width
--overlap PX                dilate mask (overlap) to hide gaps
--precision N               decimal places for coordinates
--order area-desc|...       drawing order strategy
--alpha-mode ignore|...     alpha processing mode
--alpha-threshold T         threshold for flatten/binary alpha
--supercontour file.svg     add external high-quality outline layer
--contour-width F           stroke width for super contour layer
--preset NAME               apply predefined parameter set
```

## Presets
| Name | Purpose | Settings |
|------|---------|----------|
| smooth-contours | Attractive smooth shapes with Bézier + overlap + strokes | `smooth=2, epsilon=1.0, bezier, overlap=1, outline=on, outline-width=2` |
| high-fidelity | Max faithfulness (no smoothing) | `smooth=0, epsilon=0, no bezier, minimal overlap` |

Use a preset then optionally override parameters:
```bash
python -m color_vectorize.cli examples/assets/eprivacy_logo.png styled.svg --preset smooth-contours --colors 8 --outline-color auto
```

## Combining Color Fill + High Quality Outline
If you already produced a precise monocolor outline (e.g. `eprivacy_logo.svg`), you can stack it over the multi-color result:
```bash
python -m color_vectorize.cli examples/assets/eprivacy_logo.png color.svg --colors 8 --smooth 2 --epsilon 1.0 --bezier --overlap 1 --supercontour examples/assets/eprivacy_logo.svg --contour-width 2
```
This yields smooth colored regions plus the crisp outline paths.

## Avoiding Hairline Gaps
1. Add slight overlap: `--overlap 1` (increase to 2 if still visible).
2. Increase coordinate precision: `--precision 4`.
3. Draw large shapes first: default `--order area-desc`.
4. Handle alpha fringe first: `--alpha-mode flatten --alpha-threshold 12`.

## Capturing Small Holes (e.g. Eyes)
If tiny interior holes (eyes) disappear, reduce `--min-hole-area`, e.g.:
```bash
python -m color_vectorize.cli examples/assets/eprivacy_logo.png out.svg --min-hole-area 2 --smooth 1 --epsilon 0.8
```

## Typical Recipes
Smooth pleasant illustration:
```bash
python -m color_vectorize.cli examples/assets/eprivacy_logo.png smooth.svg --preset smooth-contours --colors 8
```
High precision technical logo:
```bash
python -m color_vectorize.cli examples/assets/eprivacy_logo.png tech.svg --colors 10 --smooth 0 --epsilon 0 --bezier --overlap 0.5 --precision 5
```
Dark auto-strokes:
```bash
python -m color_vectorize.cli examples/assets/eprivacy_logo.png outlined.svg --outline --outline-color auto --outline-width 2 --overlap 1
```
Fixed palette (brand colors only):
```bash
python -m color_vectorize.cli examples/assets/eprivacy_logo.png brand.svg --palette #003366,#ffffff,#111111 --smooth 1 --epsilon 0.8
```

## Logging
Use the `--log-level` flag for verbosity control:
```
--log-level INFO    # default concise progress
--log-level DEBUG   # detailed internal steps (paths, palette, reductions)
```
Example:
```bash
python -m color_vectorize.cli input.png out.svg --colors 20 --log-level DEBUG
```
Typical log events:
- Quantization cluster reduction (INFO)
- Fixed palette usage (INFO)
- Per-color path counts (DEBUG)
- Final saved SVG path count (INFO)

For programmatic use:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
from color_vectorize import image_to_svg
image_to_svg('input.png', 'out.svg', n_colors=12)
```

## Architecture Overview
```
color_vectorize/
  cli.py            CLI parsing + presets
  svg_builder.py    Main image_to_svg orchestration
  quantize.py       KMeans / fixed palette quantization
  masks.py          Mask construction, contour combination, dilation
  geometry.py       Smoothing, simplification, Bézier conversion
  alpha.py          Alpha loading & flatten/binary handling
  utils.py          Helpers (color ops, parsing)
```
Flow: load RGBA → (optional alpha flatten/binary) → quantize → per-color mask → optional dilation (overlap) → contours (RETR_CCOMP) → smoothing/simplification/Bézier → SVG path build (evenodd) → optional per-shape stroke → optional super contour overlay → save.

## Alpha Handling
- ignore: Keep RGB channels; transparent pixels may fringe if semi-transparent.
- flatten: Composite over background (recommended to avoid halos) before quantizing.
- binary: Fully keep or drop by `--alpha-threshold`.

## Development
Run tests:
```bash
pytest -v
```
Type checking (optional):
```bash
mypy src/color_vectorize
```
Run CLI without install:
```bash
python -m color_vectorize.cli -h
```

## Performance Tips
- Downscale huge sources first.
- Reduce color count for speed.
- Limit smoothing + Bézier on very large images.

## License
PolyForm Noncommercial License 1.0.0 (non-commercial use only). For commercial licensing please contact the author.

## Disclaimer
Vector output is heuristic; manual post-edit in an SVG editor may further improve results.

## Note about removed wrapper
The previous top-level `color_vectorize.py` wrapper file was removed to avoid namespace conflicts. Use module invocation (`python -m color_vectorize.cli`) or the installed entry point `color-vectorize`.

Note: The bundled sample image now resides under `examples/assets/eprivacy_logo.png`. You can remove it or replace it with your own assets; it is included for quick experimentation.

## Continuous Integration
GitHub Actions run tests, type checking, linting and build on pushes & PRs to `main` across Python 3.9–3.13.

Badges (replace OWNER/REPO):
```
![CI](https://github.com/WolfSoko/to-svg/actions/workflows/ci.yml/badge.svg)
![Lint](https://github.com/WolfSoko/to-svg/actions/workflows/lint.yml/badge.svg)
```
Replace OWNER/REPO with your GitHub namespace.
