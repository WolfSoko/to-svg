import logging
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(SRC))  # noqa: E402

from color_vectorize import image_to_svg  # type: ignore  # noqa: E402


def _two_color_image(path: pathlib.Path):
    h, w = 40, 80
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Left half red, right half green
    img[:, : w // 2] = (0, 0, 255)  # BGR red
    img[:, w // 2 :] = (0, 255, 0)  # BGR green
    cv2.imwrite(str(path), img)
    return path


def test_logging_cluster_reduction(caplog, tmp_path):
    png = _two_color_image(tmp_path / "twocolor.png")
    out_svg = tmp_path / "out.svg"
    # Request more clusters than distinct colors
    caplog.set_level(logging.INFO, logger="color_vectorize")
    image_to_svg(str(png), str(out_svg), n_colors=12, smooth=0, epsilon=0)
    assert out_svg.exists()
    txt = caplog.text
    # Expect reduction info OR at least effective palette size 2
    assert ("Reduced requested clusters" in txt) or ("Effective palette size" in txt)
    assert "Saved SVG" in txt


def test_logging_fixed_palette(caplog, tmp_path):
    png = _two_color_image(tmp_path / "pal.png")
    out_svg = tmp_path / "pal.svg"
    caplog.set_level(logging.INFO, logger="color_vectorize")
    image_to_svg(str(png), str(out_svg), palette="#ff0000,#00ff00,#0000ff", outline=False)
    assert out_svg.exists()
    txt = caplog.text
    assert "Using fixed palette" in txt
    assert "Saved SVG" in txt
