"""Build an expanded training set from the DhMurals inpainting benchmark.

The benchmark ships 1564 train images but **5000** damage masks (~3.2 masks per
image); this project's original `data/` only ever used 1564 of them (one per
image). This script materialises all 5000 (image, mask) pairs using the
dataset loader's supported "nested variant" layout, so no loader change is
needed:

    data_multi/
      images/v000/<stem>.jpg  -> hardlink to benchmark train/images/<stem>.jpg
      sketches/v000/<stem>.jpg-> hardlink to benchmark train/edges/<stem>.jpg
      masks/v000/<stem>.png   -> copy of benchmark train/masks/<02NNN>.png

Pairing: benchmark mask j (02000 + j) is paired with image (j % n_images), and
the variant index is j // n_images. Every mask is a plausible irregular damage
blob (10-20% coverage, same family as the test masks), so any (image, mask)
combination is a valid inpainting training pair.

Images/sketches are hardlinked (no extra disk); masks are copied (they are
512x512 and get resized to the image size by the loader).
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
BM = Path("F:/data/DhMurals-inpainting-dataset/train")
OUT = ROOT / "data_multi"

images = sorted(p.name for p in (BM / "images").glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
masks = sorted(p.name for p in (BM / "masks").glob("*.png"))
n_img = len(images)
print(f"benchmark: {n_img} images, {len(masks)} masks", flush=True)
if not images or not masks:
    sys.exit("benchmark not found")

for sub in ("images", "sketches", "masks"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

made = 0
cov = []
for j, mname in enumerate(masks):
    stem = Path(images[j % n_img]).stem
    variant = f"v{j // n_img:03d}"
    for sub in ("images", "sketches", "masks"):
        (OUT / sub / variant).mkdir(parents=True, exist_ok=True)
    src_img = BM / "images" / images[j % n_img]
    src_sk = BM / "edges" / images[j % n_img]
    dst_img = OUT / "images" / variant / (stem + src_img.suffix)
    dst_sk = OUT / "sketches" / variant / (stem + src_sk.suffix)
    dst_mk = OUT / "masks" / variant / (stem + ".png")
    if not dst_img.exists():
        try:
            os.link(src_img, dst_img)          # hardlink: zero extra disk
        except OSError:
            shutil.copy2(src_img, dst_img)
    if not dst_sk.exists():
        try:
            os.link(src_sk, dst_sk)
        except OSError:
            shutil.copy2(src_sk, dst_sk)
    if not dst_mk.exists():
        shutil.copy2(BM / "masks" / mname, dst_mk)
    made += 1
    if j % 500 == 0:
        a = np.asarray(Image.open(dst_mk).convert("L").resize((64, 64), Image.BILINEAR))
        cov.append((a > 127).mean())
        print(f"  {made}/{len(masks)} pairs (variant {variant}) cov~{cov[-1]*100:.1f}%", flush=True)

total = sum(len(list((OUT / s).glob("v*/*"))) for s in ("images", "sketches", "masks"))
print(f"done: {made} (image,mask) pairs; files per split = {total // 3}")
print(f"mean mask coverage ~{np.mean(cov)*100:.1f}%")
