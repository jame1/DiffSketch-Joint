"""Evaluate on the DhMurals benchmark VAL split (100 images x 2000 masks).

Why this exists: every hyperparameter in this project (blend, Tweedie t, avg,
fill settings) was tuned directly on the 50-image TEST split, which makes the
reported test numbers optimistic. The benchmark ships a proper val split that
this project never used -- 100 held-out images with 2000 damage masks.

This script samples `--per-image` masks per val image (deterministically, seeded)
so the val score is a stable, honest estimate, and uses exactly the same
inference recipe as visualize_test.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DiffSketchConfig                     # noqa: E402
from models.diffsketch import DiffSketchJoint           # noqa: E402
from visualize_test import (                            # noqa: E402
    part_b_infer, restore_with_sketch,  # noqa: F401
)

VAL = Path("F:/data/DhMurals-inpainting-dataset/val")


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim(a, b):
    from skimage.metrics import structural_similarity as ss
    return ss(a, b, channel_axis=2, data_range=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_fc1/stage3.pt")
    ap.add_argument("--per-image", type=int, default=1,
                    help="how many of the val image's masks to evaluate")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--t", type=int, default=200)
    ap.add_argument("--blend", type=float, default=0.45)
    ap.add_argument("--avg", type=int, default=16)
    ap.add_argument("--cond-init", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap number of images")
    args = ap.parse_args()

    imgs = sorted(p for p in (VAL / "images").glob("*")
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    masks = sorted((VAL / "masks").glob("*.png"))
    # group masks per image by index: the val split has 2000 masks / 100 images
    per = len(masks) // len(imgs)
    print(f"val: {len(imgs)} images, {len(masks)} masks (~{per}/image)", flush=True)

    cfg = DiffSketchConfig()
    cfg.use_diffusers_unet = True
    cfg.pretrained_vae = str(Path("weights/sd-vae-ft-mse"))
    cfg.amp = False
    cfg.mem_efficient = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DiffSketchJoint(cfg)
    net.load_checkpoint(args.checkpoint)
    net.eval()
    net.eval_dtype = torch.float16

    rng = np.random.RandomState(0)
    rows = []
    n_img = 0
    for ii, ip in enumerate(imgs):
        if args.limit and n_img >= args.limit:
            break
        pool = masks[ii * per:(ii + 1) * per]
        if not pool:
            continue
        pick = rng.choice(len(pool), size=min(args.per_image, len(pool)), replace=False)
        for k in pick:
            mp = pool[int(k)]
            img = Image.open(ip).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
            mk = Image.open(mp).convert("L").resize((args.size, args.size), Image.NEAREST)
            img_t = (TF.to_tensor(img) * 2 - 1).unsqueeze(0).to(device)
            mask_t = (TF.to_tensor(mk).unsqueeze(0).to(device) > 0.5).float()
            # ---- inline the best recipe (pyramid fill + Tweedie + blend) ----
            from visualize_test import pyramid_fill
            init_image = pyramid_fill(img_t, mask_t)
            torch.manual_seed(0)
            outs = []
            for s in range(max(1, args.avg)):
                torch.manual_seed(s)
                r, _ = restore_with_sketch(
                    net, img_t, mask_t, 50, init_image=init_image,
                    t_start=args.t, one_step=args.t, cond_init=args.cond_init)
                outs.append(r)
            restored_t = torch.stack(outs).mean(0)
            w = args.blend * torch.ones_like(restored_t)
            restored_t = w * restored_t + (1.0 - w) * init_image
            rec = ((restored_t[0].float().clamp(-1, 1) + 1) / 2).cpu().numpy().transpose(1, 2, 0)
            orig = np.asarray(img, dtype=np.float32) / 255.0
            mb = (np.asarray(mk) > 128)
            rows.append((psnr(orig, rec), psnr(orig[mb], rec[mb]),
                         ssim(orig, rec), float(mb.mean()), ip.stem))
        n_img += 1
        if n_img % 10 == 0:
            pa = np.mean([r[0] for r in rows])
            print(f"  {n_img}/{len(imgs)} images  running PSNR_all={pa:.2f}", flush=True)

    pa = np.array([r[0] for r in rows])
    pd = np.array([r[1] for r in rows])
    sa = np.array([r[2] for r in rows])
    cv = np.array([r[3] for r in rows])
    print(f"\n=== VAL SPLIT (n={len(rows)} samples, {len(imgs)} images) ===")
    print(f"PSNR_all     : {pa.mean():.2f} +/- {pa.std():.2f}")
    print(f"PSNR_damaged : {pd.mean():.2f} +/- {pd.std():.2f}")
    print(f"SSIM_all     : {sa.mean():.3f}")
    print(f"mask coverage: {cv.mean()*100:.1f}%")


if __name__ == "__main__":
    main()
