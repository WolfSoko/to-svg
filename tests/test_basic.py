# Basic tests for color_vectorize using synthetic images (no external asset dependency)
import sys, pathlib, subprocess, re
import numpy as np
import cv2

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from color_vectorize import image_to_svg  # type: ignore


def make_test_image(path: pathlib.Path):
    """Create a small synthetic RGB image with multiple color regions and a hole."""
    w, h = 160, 120
    img = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)  # white bg
    cv2.rectangle(img, (10, 10), (70, 90), (255, 0, 0), -1)      # Blue (BGR)
    cv2.rectangle(img, (50, 30), (130, 110), (0, 0, 255), -1)    # Red
    cv2.rectangle(img, (80, 15), (95, 30), (0, 255, 0), -1)      # Green
    cv2.circle(img, (40, 50), 10, (255, 255, 255), -1)           # Hole
    cv2.imwrite(str(path), img)
    return path


def test_sanity_math():
    assert 2 * 2 == 4


def test_image_to_svg_synthetic(tmp_path):
    png = make_test_image(tmp_path / "synthetic.png")
    out_svg = tmp_path / "out.svg"
    # Use n_colors=4 to match distinct synthetic colors and avoid convergence warning
    image_to_svg(str(png), str(out_svg), n_colors=4, smooth=1, epsilon=0.8, outline=True, overlap=1)
    assert out_svg.exists(), "SVG not created"
    txt = out_svg.read_text(encoding="utf-8")
    assert txt.count("<path") >= 3, "Expected multiple path elements"
    assert "svg" in txt.lower()


def _has_color(txt: str, hex_code: str, rgb_tuple):
    r, g, b = rgb_tuple
    patterns = [
        re.escape(hex_code.lower()),
        fr"rgb\({r},{g},{b}\)",
        fr"rgb\({r}, {g}, {b}\)",
    ]
    return any(re.search(p, txt) for p in patterns)


def test_palette_fixed(tmp_path):
    png = make_test_image(tmp_path / "pal.png")
    out_svg = tmp_path / "pal.svg"
    palette = "#0000ff,#ff0000,#ffffff"  # blue, red, white
    image_to_svg(str(png), str(out_svg), palette=palette, outline=False)
    txt = out_svg.read_text(encoding="utf-8").lower()
    assert _has_color(txt, "#0000ff", (0, 0, 255)), "Blue palette color missing"
    assert _has_color(txt, "#ff0000", (255, 0, 0)), "Red palette color missing"


def test_cli_help():
    proc = subprocess.run([sys.executable, "-m", "color_vectorize.cli", "-h"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "vectorize" in proc.stdout.lower()
