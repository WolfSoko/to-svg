# Changelog

All notable changes to this project will be documented in this file.
The format loosely follows Keep a Changelog.

## [Unreleased]
### Added
- (planned) `--legend` option to embed color legend
- (planned) `--max-size` automatic downscale before quantization
- (planned) `--no-bg-rect` transparent background mode

### Improvements
- (planned) Additional Bezier path tests
- (planned) JSON metadata export (palette + path stats)

## [0.2.1] - 2025-09-18
### Fixed
- Stabilisierte Alpha-Verarbeitung: Refaktorierter Flatten-Pfad in `alpha.py` zur Vermeidung von NumPy/Mypy Typkonflikten durch Auslagerung in `_flatten_rgba` (klarer float32 Pfad, keine gemischten `np.where`-Typen).

### Internal
- Typing-Härtung (mypy jetzt clean für `alpha.py`).

## [0.2.0] - 2025-09-18
### Added
- Presets (`smooth-contours`, `high-fidelity`)
- Bézier conversion flag `--bezier`
- Outline controls (`--outline`, `--outline-color`, `--outline-width`, join/cap)
- Overlap handling `--overlap` to eliminate hairline gaps
- Alpha handling modes (`--alpha-mode`, `--alpha-threshold`)
- Fixed palette support via `--palette`
- Logging system with `--log-level`
- Cluster auto-reduction to avoid convergence warnings
- Synthetic test suite (no external image dependency)
- Logging tests (`test_logging.py`)

### Changed
- Removed top-level proxy wrapper `color_vectorize.py` (use module invocation)
- Improved CLI help & README (architecture, logging, presets)

### Fixed
- Palette test stability (robust rgb/hex detection)
- Gap issues via overlap and precision guidance in README

## [0.1.0] - Initial
### Added
- Basic multi-color vectorization pipeline
- KMeans quantization
- Chaikin smoothing & Douglas-Peucker simplification
- Hole support with evenodd fill
- Basic SVG output

[0.2.1]: https://example.com/compare/v0.2.0...v0.2.1
[0.2.0]: https://example.com/compare/v0.1.0...v0.2.0
