"""Split the 4-up comparison panels in vis_cm_test/panels/ into four separate
images per sample: GT | damaged(input) | predicted sketch | restored.

The panel layout is fixed by make_panel() in vis_comparemodel_test.py:
  width  = w*4 + 5*8      (5 gaps of 8px)
  height = h + 46         (26px title row + 20px caption row)
  cell c (0..3) is pasted at (8 + c*(w+8), 26), size (w, h)
so we crop each cell region. Runs purely on CPU / PIL, no model needed.

Usage:
  python split_panels.py [--in-dir vis_cm_test/panels] [--out-dir vis_cm_test/split]
"""
from __future__ import annotations
import argparse
from pathlib import Path

from PIL import Image, ImageOps

# column order matches make_panel(): titles = ["ground truth", "damaged (input)",
# "predicted sketch", "restored"]
SUFFIXES = ["gt", "damaged", "sketch", "restored"]


def split_one(panel: Image.Image):
    W, H = panel.size
    w = (W - 40) / 4          # 5 gaps * 8px
    h = H - 46
    assert abs(w - round(w)) < 1e-3 and abs(h - round(h)) < 1e-3, (W, H, w, h)
    w, h = int(round(w)), int(round(h))
    cells = []
    for c in range(4):
        x0 = 8 + c * (w + 8)
        y0 = 26
        cell = panel.crop((x0, y0, x0 + w, y0 + h))
        cells.append(cell)
    return cells


def stem_of(name: str) -> str:
    # panel filename is {id}.png where id may contain underscores
    return name[:-4] if name.endswith(".png") else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="vis_cm_test/panels")
    ap.add_argument("--out-dir", default="vis_cm_test/split")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = sorted(in_dir.glob("*.png"))
    print(f"[split] {len(panels)} panels -> {out_dir}")
    for i, p in enumerate(panels):
        panel = Image.open(p).convert("RGB")
        stem = stem_of(p.name)
        cells = split_one(panel)
        for suf, cell in zip(SUFFIXES, cells):
            cell.save(out_dir / f"{stem}_{suf}.png")
        if (i + 1) % 50 == 0 or (i + 1) == len(panels):
            print(f"[split] {i+1}/{len(panels)}", flush=True)
    print(f"[split] done. files in {out_dir}: {len(list(out_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
