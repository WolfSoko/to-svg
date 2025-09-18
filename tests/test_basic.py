import sys, pathlib, re, subprocess

# Ensure src/ is importable
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from color_vectorize import image_to_svg  # type: ignore

def test_image_to_svg(tmp_path):
    input_img = ROOT / "eprivacy_logo.png"
    out_file = tmp_path / "out.svg"
    image_to_svg(str(input_img), str(out_file), n_colors=5, smooth=1, epsilon=0.5, outline=True)
    assert out_file.exists(), "SVG not created"
    txt = out_file.read_text(encoding="utf-8")
    assert "<path" in txt, "No path elements in SVG"


def test_palette(tmp_path):
    input_img = ROOT / "eprivacy_logo.png"
    out_file = tmp_path / "pal.svg"
    # Limit to 3 fixed colors
    image_to_svg(str(input_img), str(out_file), palette="#003366,#ffffff,#111111", outline=False)
    txt = out_file.read_text(encoding="utf-8")
    # Expect only these three rgb values (allow case-insensitive search)
    assert "#003366" in txt.lower() or "rgb(0, 51, 102)" in txt.lower()


def test_cli_help():
    # Use wrapper script to ensure CLI works without installation
    proc = subprocess.run([sys.executable, str(ROOT / "color_vectorize.py"), "-h"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Vectorize" in proc.stdout or "vectorize" in proc.stdout
