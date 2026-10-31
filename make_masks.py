#!/usr/bin/env python
"""
Generate training damage masks for DiffSketch-Joint.

The `MuralDataset` (data/dataset.py) requires, for every file in
`data/images/<stem>.<ext>`, a paired single-channel mask at
`data/masks/<stem>.<ext>` where 1 (white) = damaged region. The model
synthesizes the training input as `damaged = image * (1 - mask)` and learns
to reconstruct the complete `image` inside the masked area (self-supervised
inpainting). Because the provided `data/` only contains complete murals, we
synthesize free-form damage masks.

Masks are saved as lossless PNG (single channel 'L'), white = damaged.
Generation is fully reproducible: each mask is seeded from a hash of its
filename stem plus a global BASE_SEED, so re-running reproduces identical
masks unless `--force`/different seed is given.

Usage:
    python make_masks.py                 # generate missing masks
    python make_masks.py --force         # regenerate all masks
    python make_masks.py --seed 12345    # change global seed
"""
import argparse
import glob
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BASE_SEED = 20260601          # change to regenerate a different but fixed set
SIZE = 720                   # all images are 720x720
MIN_COV, MAX_COV = 0.08, 0.40  # target damaged-area ratio per image


def stem_seed(stem: str, base_seed: int) -> int:
    """Deterministic 31-bit seed derived from the filename stem + base seed."""
    h = hashlib.md5(stem.encode("utf-8")).digest()
    return (int.from_bytes(h[:4], "big") ^ base_seed) & 0x7FFFFFFF


def draw_shapes(rng: np.random.Generator, scale: float) -> Image.Image:
    """Draw a free-form damage mask: eroded blobs + crack-like strokes."""
    img = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(img)

    # Large/medium blobs (missing plaster / flaked-off regions)
    n_blobs = int(rng.integers(3, 8))
    for _ in range(n_blobs):
        cx, cy = rng.integers(0, SIZE, 2)
        r = int(rng.uniform(0.08, 0.30) * SIZE * scale)
        k = int(rng.integers(5, 9))
        pts = []
        for a in np.linspace(0, 2 * np.pi, k, endpoint=False):
            rr = r * rng.uniform(0.5, 1.2)
            pts.append((cx + rr * np.cos(a), cy + rr * np.sin(a)))
        d.polygon(pts, fill=255)

    # Thin/long strokes (cracks, veins of erosion)
    n_strokes = int(rng.integers(2, 6))
    for _ in range(n_strokes):
        x, y = rng.integers(0, SIZE, 2)
        pts = [(int(x), int(y))]
        w = int(rng.uniform(4, 18) * scale)
        steps = int(rng.integers(1, 4))
        for _ in range(steps):
            x += rng.integers(-SIZE // 4, SIZE // 4)
            y += rng.integers(-SIZE // 4, SIZE // 4)
            x = min(max(int(x), 0), SIZE - 1)
            y = min(max(int(y), 0), SIZE - 1)
            pts.append((x, y))
        d.line(pts, fill=255, width=w, joint="curve")

    # Soften polygon edges -> organic boundaries, then keep it binary.
    img = img.filter(ImageFilter.GaussianBlur(radius=2.0))
    return img


def gen_mask(seed: int):
    """Return (binary_mask_uint8, coverage) as close to [MIN_COV, MAX_COV] as possible."""
    scale = 1.0
    last_arr = None
    last_cov = None
    for attempt in range(14):
        rng = np.random.default_rng((seed * 97 + attempt * 131 + 7) & 0x7FFFFFFF)
        arr = np.array(draw_shapes(rng, scale), dtype=np.float32)
        cov = float((arr > 128).mean())
        last_arr, last_cov = arr, cov
        if MIN_COV <= cov <= MAX_COV:
            return (arr > 128).astype(np.uint8) * 255, cov
        scale *= 1.25 if cov < MIN_COV else 0.78

    # Fallback: clip coverage by raising threshold if we overshot.
    arr, cov = last_arr, last_cov
    t = 128
    if cov > MAX_COV:
        for tt in (140, 160, 180, 200, 215, 230):
            if (arr > tt).mean() <= MAX_COV:
                t = tt
                break
    mask = (arr > t).astype(np.uint8) * 255
    return mask, float((mask > 0).mean())


def process_one(args):
    img_path, mask_dir, force, base_seed = args
    stem = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(mask_dir, stem + ".png")
    if (not force) and os.path.exists(out_path):
        return stem, None  # already present, skip
    mask, cov = gen_mask(stem_seed(stem, base_seed))
    Image.fromarray(mask, mode="L").save(out_path)
    return stem, cov


def main():
    ap = argparse.ArgumentParser(description="Generate DiffSketch-Joint damage masks")
    ap.add_argument("--data", default="data", help="Dataset root (contains images/, sketches/, masks/)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--force", action="store_true", help="Regenerate all masks")
    ap.add_argument("--seed", type=int, default=BASE_SEED, help="Override global base seed")
    args = ap.parse_args()

    image_dir = os.path.join(args.data, "images")
    mask_dir = os.path.join(args.data, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                    glob.glob(os.path.join(image_dir, "*.png")) +
                    glob.glob(os.path.join(image_dir, "*.jpeg")))
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    jobs = [(p, mask_dir, args.force, args.seed) for p in images]
    coverages = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for stem, cov in ex.map(process_one, jobs):
            done += 1
            if cov is not None:
                coverages.append(cov)
            if done % 200 == 0 or done == len(jobs):
                print(f"  processed {done}/{len(jobs)}")

    # Report
    n_mask = len(glob.glob(os.path.join(mask_dir, "*.png")))
    print(f"Images: {len(images)} | Masks on disk: {n_mask}")
    if coverages:
        cv = np.array(coverages)
        print(f"Newly generated: {len(coverages)} | coverage min/mean/max: "
              f"{cv.min():.3f}/{cv.mean():.3f}/{cv.max():.3f}")


if __name__ == "__main__":
    main()
