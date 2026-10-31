"""Build the final training set by HARDLINKING only (never moving/deleting).

Rationale: the DhMurals benchmark's 5000 train masks average 24.8% coverage,
while this project's own 1564 masks average 17.3% and the TEST masks 16.9%
(p90 = 24.7%). Training on a systematically larger-coverage distribution would
shift the model off the test regime. So we keep only the benchmark masks whose
coverage falls inside the range our own masks span, and mix in all 1564 of the
originals.

Everything is created with os.link() (metadata-only, zero extra disk, and fully
reversible by deleting data_final). No source file is ever moved or removed.
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
MULTI = ROOT / "data_multi"
SRC = ROOT / "data"
OUT = ROOT / "data_final"
MAX_COV = 0.27          # ~ our own masks' p90 (0.266); test p90 = 0.247


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            import shutil
            shutil.copy2(src, dst)


# ---- pass 1: measure coverage of every benchmark mask (local copies) ----
pairs = []
for mp in sorted((MULTI / "masks").glob("v*/*.png")):
    a = np.asarray(Image.open(mp).convert("L").resize((96, 96), Image.BILINEAR))
    cov = float((a > 127).mean())
    pairs.append((mp, cov))
print(f"measured {len(pairs)} benchmark masks", flush=True)

kept = [(mp, c) for mp, c in pairs if c <= MAX_COV]
print(f"kept {len(kept)} (coverage <= {MAX_COV}), mean coverage "
      f"{np.mean([c for _, c in kept]):.3f}", flush=True)

# ---- pass 2: hardlink the kept benchmark triples ----
n = 0
for mp, cov in kept:
    rel = mp.relative_to(MULTI / "masks")     # v000/stem.png
    stem = mp.stem
    for sub in ("images", "sketches"):
        for cand in (MULTI / sub / rel.parent).glob(stem + ".*"):
            link(cand, OUT / sub / rel.parent / cand.name)
    link(mp, OUT / "masks" / rel.parent / mp.name)
    n += 1
    if n % 1000 == 0:
        print(f"  linked {n}/{len(kept)}", flush=True)
print(f"linked {n} benchmark pairs", flush=True)

# ---- pass 3: hardlink our original 1564 test-matched masks as v900 ----
added = 0
for ip in sorted(p for p in (SRC / "images").glob("*")
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png")):
    stem = ip.stem
    msk = next((SRC / "masks" / (stem + e) for e in (".png", ".jpg", ".jpeg")
                if (SRC / "masks" / (stem + e)).exists()), None)
    sk = next((SRC / "sketches" / (stem + e) for e in (".jpg", ".jpeg", ".png")
               if (SRC / "sketches" / (stem + e)).exists()), None)
    if msk is None or sk is None:
        continue
    link(ip, OUT / "images" / "v900" / ip.name)
    link(sk, OUT / "sketches" / "v900" / sk.name)
    link(msk, OUT / "masks" / "v900" / msk.name)
    added += 1
print(f"linked {added} original (test-matched) pairs as v900", flush=True)

total = len(list((OUT / "images").glob("v*/*")))
print(f"TOTAL training pairs in data_final: {total}")
