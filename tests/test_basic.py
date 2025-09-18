# Basic tests for color_vectorize
import sys, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from color_vectorize import image_to_svg  # type: ignore

def _input_logo() -> pathlib.Path:
    p = ROOT / "eprivacy_logo.png"
    if not p.exists():
        raise RuntimeError("Missing test asset: eprivacy_logo.png")
    return p


def test_sanity_math():
    # Ensure pytest actually runs
    assert 1 + 1 == 2


def test_image_to_svg(tmp_path):
    input_img = _input_logo()
    out_file = tmp_path / "out.svg"
    image_to_svg(str(input_img), str(out_file), n_colors=5, smooth=1, epsilon=0.5, outline=True)
    assert out_file.exists(), "SVG not created"
    txt = out_file.read_text(encoding="utf-8")
    assert "<path" in txt.lower(), "No path elements in SVG"


def test_palette(tmp_path):
    input_img = _input_logo()
    out_file = tmp_path / "pal.svg"
    image_to_svg(str(input_img), str(out_file), palette="#003366,#ffffff,#111111", outline=False)
    txt = out_file.read_text(encoding="utf-8")
    assert "#003366" in txt.lower() or "rgb(0, 51, 102)" in txt.lower()


def test_cli_help():
    # Aufruf über Modul statt über entfernte Wrapper-Datei
    proc = subprocess.run([sys.executable, "-m", "color_vectorize.cli", "-h"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "vectorize" in proc.stdout.lower()
