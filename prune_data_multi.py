"""Prune data_multi to the test-matched mask-coverage range, then add our
original (test-matched) masks as an extra variant.

Why: the DhMurals benchmark's 5000 train masks average 24.8% coverage, while
this project's own 1564 masks average 17.3% and the TEST masks average 16.9%
(p90 = 24.7%). Training on a systematically larger-coverage distribution would
shift the model away from the test regime, so we keep only benchmark masks whose
coverage falls inside the range our own masks span, and mix in all 1564 of the
originals. Result: ~2.4x the (image, mask) pairs with a matched coverage
distribution.
"""
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data_multi"
SRC = ROOT / "data"
MAX_COV = 0.27          # ~ our own masks' p90 (0.266); test p90 = 0.247

mask_files = sorted((OUT / "masks").glob("v*/*.png"))
print(f"pruning {len(mask_files)} pairs (threshold coverage <= {MAX_COV:.2f})", flush=True)

kept, dropped = 0, 0
covs = []
for mp in mask_files:
    a = np.asarray(Image.open(mp).convert("L").resize((96, 96), Image.BILINEAR))
    cov = (a > 127).mean()
    covs.append(cov)
    if cov > MAX_COV:
        rel = mp.relative_to(OUT / "masks")          # v000/stem.png
        stem = mp.stem
        for sub, ext in (("images", None), ("sketches", None), ("masks", ".png")):
            d = OUT / sub / rel.parent
            if ext is None:
                for cand in d.glob(stem + ".*"):
                    cand.unlink()
            else:
                p = d / (stem + ext)
                if p.exists():
                    p.unlink()
        dropped += 1
    else:
        kept += 1
    if (kept + dropped) % 1000 == 0:
        print(f"  {kept+dropped}/{len(mask_files)} kept={kept} dropped={dropped}", flush=True)

covs = np.array(covs)
print(f"kept={kept} dropped={dropped}; coverage of kept ~{covs[covs <= MAX_COV].mean():.3f}")

# ---- add our original 1564 (test-matched) masks as variant v900 ----
src_imgs = sorted(p for p in (SRC / "images").glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
added = 0
for ip in src_imgs:
    stem = ip.stem
    msk = None
    for ext in (".png", ".jpg", ".jpeg"):
        if (SRC / "masks" / (stem + ext)).exists():
            msk = SRC / "masks" / (stem + ext)
            break
    sk = None
    for ext in (".jpg", ".jpeg", ".png"):
        if (SRC / "sketches" / (stem + ext)).exists():
            sk = SRC / "sketches" / (stem + ext)
            break
    if msk is None or sk is None:
        continue
    for sub in ("images", "sketches", "masks"):
        (OUT / sub / "v900").mkdir(parents=True, exist_ok=True)
    for src, sub in ((ip, "images"), (sk, "sketches"), (msk, "masks")):
        dst = OUT / sub / "v900" / src.name
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    added += 1
print(f"added {added} original (test-matched) pairs as variant v900")

total = len(list((OUT / "images").glob("v*/*")))
print(f"TOTAL training pairs: {total}")
