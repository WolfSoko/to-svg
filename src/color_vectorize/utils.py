from __future__ import annotations

__all__ = ["parse_hex_color", "darken_rgb", "parse_palette"]

def parse_hex_color(s: str):
    s = s.strip()
    if s.startswith('#'):
        s = s[1:]
    if len(s) == 3:
        s = ''.join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError("Invalid hex color: expected 3 or 6 hex digits")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (r, g, b)

def darken_rgb(rgb, factor: float = 0.6):
    r, g, b = rgb
    return (max(0, int(r * factor)), max(0, int(g * factor)), max(0, int(b * factor)))


def parse_palette(p: str):
    parts = [c.strip() for c in p.split(',') if c.strip()]
    return [parse_hex_color(x) for x in parts]
