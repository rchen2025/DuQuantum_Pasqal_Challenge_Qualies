#!/usr/bin/env python
"""Compose the three report panels into one page-width figure."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "report" / "figures"


def main() -> None:
    names = ("register.png", "couplings.png", "pulse_scan.png")
    panels = [Image.open(FIGURES / name).convert("RGB") for name in names]
    cell_w, cell_h, gap, margin = 510, 510, 35, 35
    canvas = Image.new("RGB", (2 * margin + 3 * cell_w + 2 * gap, 550), "white")
    for index, panel in enumerate(panels):
        panel.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        x = margin + index * (cell_w + gap) + (cell_w - panel.width) // 2
        y = (cell_h - panel.height) // 2
        canvas.paste(panel, (x, y))
    destination = FIGURES / "overview.png"
    canvas.save(destination, optimize=True)
    print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
