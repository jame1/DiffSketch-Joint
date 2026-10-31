"""Visualize DiffSketch-Joint restoration on the CompareModel benchmark test set.

Reference masks come from E:/DunhuangMural/CompareModel/data/test/
  {id}_gt.png      ground-truth (clean) mural
  {id}_mask.png    1=hole mask (L)
  {id}_masked.png  masked input (RGB, hole zeroed/garbled)

We run THIS model (DiffSketch-Joint) on those exact masks so the outputs are
directly comparable to the other models under CompareModel/results/
(brushnet, lama, mat, moebius, pixelhacker). The sketch is derived internally
by the sketch branch, so no external sketch files are needed.

Recipe = best known (2026-10-27): pyramid fill init + one-step Tweedie at
t=200 + 16-seed averaging + 0.65 blend toward the fill + cond-init latent.

Outputs (under --out-dir, default vis_cm_test):
  panels/{id}.png    4-up comparison: GT | damaged | sketch | restored
  restored/{id}.png  the bare restored image (matches CompareModel's layout)
  eval_summary.csv   per-sample PSNR/SSIM/consistency + mean row

Usage:
  python vis_comparemodel_test.py [--split test|val] [--limit N] [--gpu 0]
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
CM = Path(r"E:/DunhuangMural/CompareModel")
CKPT = Path(r"C:/wb_diffsketch/checkpoints_fc1/stage3.pt")  # fast SSD copy (identical)
sys.path.insert(0, str(ROOT))

# Reuse the established helpers (panel layout, metrics, fills, sampler).
from visualize_test import (  # noqa: E402
    pyramid_fill, restore_with_sketch,
    psnr, ssim, ssim_masked,
    color_consistency, texture_consistency, edge_consistency,
)


def load_model(checkpoint, device, args=None):
    from config import DiffSketchConfig
    from infer import DiffSketchJoint
    # Build with the SD diffusers UNet so the ~1B-param inpaint branch loads
    # cleanly. ckpt_fc1 was trained under exactly this switch.
    cfg = DiffSketchConfig(use_diffusers_unet=True)
    # Build the HF refiner module when requested (or when a weight path is given).
    if args is not None:
        cfg.hf_refiner = bool(getattr(args, "hf_refiner", False)) or bool(
            getattr(args, "hf_refiner_path", ""))
        cfg.hf_refiner_base = getattr(args, "hf_refiner_base", 32)
        cfg.hf_refiner_blocks = getattr(args, "hf_refiner_blocks", 4)
    net = DiffSketchJoint(cfg).to(device)
    # Full sequential CPU read (NOT mmap) -- mmap random reads hang for tens of
    # minutes on the slow E: drive.
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for name, module, sd in (
        ("encoder", net.encoder, ck["encoder"]),
        ("sketch", net.sketch_branch, ck["sketch"]),
        ("inpaint", net.inpaint_branch, ck["inpaint"]),
    ):
        missing, unexpected = module.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[load:{name}] missing={len(missing)} unexpected={len(unexpected)}",
                  flush=True)
    # Optional: load a standalone trained HighFreqRefiner (train_hf_refiner.py).
    if args is not None and getattr(args, "hf_refiner_path", ""):
        sd = torch.load(args.hf_refiner_path, map_location="cpu", weights_only=True)
        ref = net.inpaint_branch.hf_refiner
        ref.load_state_dict(sd)
        ref.out_scale = float(getattr(args, "hf_refiner_out_scale", 0.25))
        ref.to(net.device)
        print(f"[refiner] loaded {args.hf_refiner_path} (out_scale={ref.out_scale})",
              flush=True)
    # Route-C: swap in a fine-tuned VAE decoder.
    if args is not None and getattr(args, "vae_decoder_path", ""):
        from diffusers import AutoencoderKL
        new_vae = AutoencoderKL.from_pretrained(args.vae_decoder_path).to(net.device).float()
        new_vae.eval()
        net.inpaint_branch.vae = new_vae
        print(f"[route-C] VAE decoder swapped from {args.vae_decoder_path}", flush=True)
    # Route-C latent refiner (train_latent_refiner.py): applied to x0_hat before decode.
    net.latent_refiner = None
    if args is not None and getattr(args, "latent_refiner_path", ""):
        from latent_refiner_model import LatentRefiner
        ref = LatentRefiner().to(net.device).float().eval()
        ref.load_state_dict(torch.load(args.latent_refiner_path,
                                       map_location="cpu", weights_only=True))
        net.latent_refiner = ref
        # The latent refiner + VAE decoder trigger a cuDNN crash on this
        # Windows/WDDM box; native kernels are stable (just slower).
        torch.backends.cudnn.enabled = False
        print(f"[route-C] latent refiner loaded from {args.latent_refiner_path} (cuDNN disabled)", flush=True)
    net.eval()
    return net


@torch.no_grad()
def restore_pair(net, masked_pil, mask_pil, args):
    device = net.device
    # Parse here (not in main) so the helper is self-contained.
    rts = [int(v) for v in getattr(args, "refine_ts", "").split(",") if v.strip()] or None
    img_t = (TF.to_tensor(masked_pil.convert("RGB")) * 2.0 - 1.0).unsqueeze(0).to(device)
    mask_t = (TF.to_tensor(mask_pil.convert("L")).unsqueeze(0).to(device) > 0.5).float()

    if args.init == "pyramid":
        init_image = pyramid_fill(img_t, mask_t, reps=6, scale=1.0)
    elif args.init == "smooth":
        from visualize_test import smooth_fill
        init_image = smooth_fill(img_t, mask_t)
    else:
        init_image = None

    outs = []
    for k in range(max(1, args.avg)):
        torch.manual_seed(args.seed + k)
        r, _ = restore_with_sketch(
            net, img_t, mask_t, args.steps,
            sampler=args.sampler, init_image=init_image,
            t_start=args.t_start, one_step=args.one_step,
            cond_init=args.cond_init,
            refine_ts=rts, latent_refiner=getattr(net, "latent_refiner", None))
        outs.append(r)
    restored_t = torch.stack(outs).mean(0)

    if args.blend < 1.0 and init_image is not None:
        w = args.blend * torch.ones_like(restored_t)
        restored_t = w * restored_t + (1.0 - w) * init_image
    # Paste back the original (masked) pixels outside the hole.
    final_t = restored_t * mask_t + img_t * (1.0 - mask_t)

    restored_01 = ((final_t[0].float().clamp(-1, 1) + 1.0) * 0.5).cpu()
    return TF.to_pil_image(restored_01), mask_t


def make_panel(gt_pil, masked_pil, mask_pil, restored_pil, sketch_pil, stem, caption):
    damaged = np.array(masked_pil.convert("RGB")).copy()
    mb = np.asarray(mask_pil.convert("L")) > 128
    damaged[mb] = (216, 32, 216)  # magenta hole, matches visualize_test.py
    cells = [np.array(gt_pil.convert("RGB")), damaged,
             np.array(sketch_pil.convert("RGB")), np.array(restored_pil.convert("RGB"))]
    titles = ["ground truth", "damaged (input)", "predicted sketch", "restored"]
    h, w = cells[0].shape[:2]
    panel = Image.new("RGB", (w * 4 + 5 * 8, h + 46), (255, 255, 255))
    d = ImageDraw.Draw(panel)
    for c, t in enumerate(titles):
        d.text((8 + c * (w + 8), 8), t, fill=(0, 0, 0))
    d.text((8, h + 28), caption, fill=(0, 0, 0))
    for c, cell in enumerate(cells):
        panel.paste(Image.fromarray(cell), (8 + c * (w + 8), 26))
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CKPT),
                    help="checkpoint to evaluate (default: production stage3.pt)")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", default="vis_cm_test")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    ap.add_argument("--init", default="pyramid", choices=["none", "smooth", "pyramid"])
    ap.add_argument("--t-start", type=int, default=200)
    ap.add_argument("--one-step", type=int, default=200)
    ap.add_argument("--avg", type=int, default=16)
    ap.add_argument("--blend", type=float, default=0.65)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cond-init", action="store_true", default=True)
    ap.add_argument("--refine-ts", default="",
                    help="descending Tweedie refinement ladder, e.g. '150,100,60'")
    ap.add_argument("--hf-refiner", action="store_true",
                    help="build the learned image-space HF refiner module")
    ap.add_argument("--hf-refiner-path", default="",
                    help="standalone trained HighFreqRefiner state_dict (train_hf_refiner.py)")
    ap.add_argument("--hf-refiner-base", type=int, default=32)
    ap.add_argument("--hf-refiner-blocks", type=int, default=4)
    ap.add_argument("--hf-refiner-out-scale", type=float, default=0.25)
    ap.add_argument("--vae-decoder-path", default="",
                    help="Route-C: fine-tuned AutoencoderKL dir (train_vae_decoder_c.py). "
                         "Swapped in for the default VAE decoder at inference.")
    ap.add_argument("--latent-refiner-path", default="",
                    help="Route-C latent refiner: LatentRefiner state_dict "
                         "(train_latent_refiner.py); applied to x0_hat before decode.")
    args = ap.parse_args()

    data_dir = CM / "data" / args.split
    out_dir = ROOT / args.out_dir
    (out_dir / "panels").mkdir(parents=True, exist_ok=True)
    (out_dir / "restored").mkdir(parents=True, exist_ok=True)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[cm-vis] device={device} split={args.split} ckpt={args.checkpoint}", flush=True)
    net = load_model(args.checkpoint, device, args)
    print("[cm-vis] model loaded", flush=True)

    ids = sorted(p.name[:-len("_masked.png")] for p in data_dir.glob("*_masked.png"))
    if args.limit:
        ids = ids[:args.limit]
    print(f"[cm-vis] {len(ids)} pairs", flush=True)

    rows = []
    t_all = time.time()
    for i, sid in enumerate(ids):
        gt_pil = Image.open(data_dir / f"{sid}_gt.png").convert("RGB")
        masked_pil = Image.open(data_dir / f"{sid}_masked.png").convert("RGB")
        mask_pil = Image.open(data_dir / f"{sid}_mask.png").convert("L")

        t0 = time.time()
        restored_pil, mask_t = restore_pair(net, masked_pil, mask_pil, args)
        # Need the sketch for the panel: recompute once (cheap relative to the
        # 16-seed average). Re-run a single seed to grab sketch_prob.
        img_t = (TF.to_tensor(masked_pil.convert("RGB")) * 2.0 - 1.0).unsqueeze(0).to(device)
        m_t = (TF.to_tensor(mask_pil.convert("L")).unsqueeze(0).to(device) > 0.5).float()
        init_image = pyramid_fill(img_t, m_t, reps=6, scale=1.0) if args.init == "pyramid" else None
        _, sketch_prob = restore_with_sketch(
            net, img_t, m_t, args.steps, sampler=args.sampler,
            init_image=init_image, t_start=args.t_start,
            one_step=args.one_step, cond_init=args.cond_init)
        sketch_pil = TF.to_pil_image(sketch_prob[0].float().cpu().clamp(0, 1))
        dt = time.time() - t0

        # Metrics vs ground truth.
        orig = np.asarray(gt_pil, dtype=np.float32) / 255.0
        rec = np.asarray(restored_pil, dtype=np.float32) / 255.0
        mb = np.asarray(mask_pil.convert("L")) > 128
        p_all = psnr(orig, rec)
        p_dmg = psnr(orig[mb], rec[mb]) if mb.any() else float("nan")
        s_all = ssim(orig, rec)
        s_dmg = ssim_masked(orig, rec, mb)
        ccon = tcon = ecom = float("nan")
        if mb.any():
            ccon = color_consistency(rec[mb], orig[~mb])
            rec_g = rec[..., 0] * 0.299 + rec[..., 1] * 0.587 + rec[..., 2] * 0.114
            orig_g = orig[..., 0] * 0.299 + orig[..., 1] * 0.587 + orig[..., 2] * 0.114
            tcon = texture_consistency(rec_g, orig_g, mb)
            ecom = edge_consistency(rec, orig, mb)

        cap = (f"{sid} | PSNR dmg={p_dmg:.2f} SSIM dmg={s_dmg:.3f} | "
               f"CCON={ccon:.3f} TCON={tcon:.3f} ECON={ecom:.3f} | {dt:.1f}s")
        panel = make_panel(gt_pil, masked_pil, mask_pil, restored_pil, sketch_pil, sid, cap)
        panel.save(out_dir / "panels" / f"{sid}.png")
        restored_pil.save(out_dir / "restored" / f"{sid}.png")

        rows.append(dict(sample_id=sid, psnr_all=round(p_all, 4), psnr_damaged=round(p_dmg, 4),
                         ssim_all=round(s_all, 4), ssim_damaged=round(s_dmg, 4),
                         ccon=round(ccon, 4), tcon=round(tcon, 4), econ=round(ecom, 4),
                         seconds=round(dt, 3)))
        if (i + 1) % 10 == 0 or (i + 1) == len(ids):
            print(f"[cm-vis] {i+1}/{len(ids)} {sid} "
                  f"PSNR_all={p_all:.2f} PSNR_dmg={p_dmg:.2f} TCON={tcon:.3f} "
                  f"({time.time()-t_all:.0f}s)", flush=True)

    # Summary CSV (with mean row).
    keys = ["sample_id", "psnr_all", "psnr_damaged", "ssim_all", "ssim_damaged",
            "ccon", "tcon", "econ", "seconds"]
    with open(out_dir / "eval_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
        mean = {k: round(float(np.mean([r[k] for r in rows if k != "sample_id"])), 4)
                for k in keys if k not in ("sample_id", "seconds")}
        mean.update(sample_id="MEAN", seconds=round(float(np.mean([r["seconds"] for r in rows])), 3))
        wr.writerow(mean)
    print(f"[cm-vis] done {len(rows)} pairs in {time.time()-t_all:.0f}s -> {out_dir}", flush=True)
    print(f"[cm-vis] MEAN PSNR_all={mean['psnr_all']:.4f} PSNR_dmg={mean['psnr_damaged']:.4f} "
          f"SSIM_dmg={mean['ssim_damaged']:.4f} TCON={mean['tcon']:.4f}", flush=True)


if __name__ == "__main__":
    main()
