# color_vectorize

Tool for color vectorization (PNG/JPG → SVG) featuring:
- Color quantization (KMeans)
- Smoothing (Chaikin / optional Bézier conversion)
- Polygon simplification (Douglas-Peucker)
- Hole support (RETR_CCOMP + evenodd fill rule)
- Optional strokes per shape
- Overlap dilation to eliminate hairline gaps
- Alpha handling (flatten / binary)

## Installation
```bash
python -m venv .venv
.venv/Scripts/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Minimal usage
```bash
python color_vectorize.py eprivacy_logo.png out.svg
```

## Key parameters
- `--colors N`              number of color clusters
- `--bg #RRGGBB`            background color (hex)
- `--min-area PX`           minimum area of outer contours
- `--min-hole-area PX`      minimum area of holes
- `--smooth N`              Chaikin iterations (0–3 typical)
- `--epsilon F`             Douglas-Peucker tolerance (0 = off)
- `--bezier`                convert polylines to cubic segments
- `--outline`               draw strokes around each filled shape
- `--outline-color C|auto`  stroke color or auto (darkened fill)
- `--outline-width F`       stroke width
- `--overlap PX`            dilate mask to overlap adjacent shapes
- `--precision N`           decimal places for coordinates
- `--order area-desc|...`   drawing order
- `--alpha-mode ignore|flatten|binary` alpha processing mode
- `--alpha-threshold T`     threshold for alpha (0–255)

## Typical commands
Balanced smooth shapes with slight overlap:
```bash
python color_vectorize.py eprivacy_logo.png out.svg --colors 8 --bg "#ffffff" --smooth 2 --epsilon 1.0 --bezier --overlap 1 --precision 4
```
Add strokes:
```bash
python color_vectorize.py eprivacy_logo.png out.svg --outline --outline-width 2 --overlap 1
```
Remove alpha halo (soft flatten):
```bash
python color_vectorize.py logo.png out.svg --alpha-mode flatten --alpha-threshold 12 --overlap 1
```
Hard edges from RGBA:
```bash
python color_vectorize.py logo.png out.svg --alpha-mode binary --alpha-threshold 0
```
Add a pre-drawn super contour:
```bash
python color_vectorize.py eprivacy_logo.png out.svg --supercontour eprivacy_logo.svg --contour-width 2
```

## Avoiding hairline gaps
1. `--overlap 1` (try 2 if still visible)  
2. `--precision 4` or `5`  
3. Ensure large shapes are drawn first (default `area-desc`)  
4. Use alpha flatten before quantization when original has semi‑transparent edges.

## Finer contours
- Fewer jaggies: `--smooth 1..2` + moderate `--epsilon (0.8–1.5)`
- Maximum fidelity: `--smooth 0 --epsilon 0`
- Bézier gives smoother organic look but more path data.

## Performance notes
- Downscale huge images beforehand.
- Fewer colors = faster.
- High overlap + many colors = more nodes.

## Post optimization
Use svgo / scour if needed:
```bash
npx svgo out.svg -o out.min.svg
```

## Troubleshooting
| Issue | Fix |
|-------|-----|
| Empty SVG | Check input path |
| Missing holes (eyes) | Lower `--min-hole-area` (e.g. 2) |
| Fringe / halos | `--alpha-mode flatten` + overlap |
| Too many micro shapes | Raise `--min-area` |
| Shapes too blocky | Lower `--epsilon`, add `--smooth 1` |

## License
Internal / adapt as needed.
