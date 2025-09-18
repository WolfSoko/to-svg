#!/usr/bin/env python
import sys, pathlib
# Ensure src/ is before project root so that 'color_vectorize' package is found
SRC_DIR = pathlib.Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from color_vectorize.cli import main

if __name__ == "__main__":
    main()
