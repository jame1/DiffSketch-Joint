#!/usr/bin/env python
"""
Visualization + quantitative evaluation for DiffSketch-Joint.

Part A -- Data pipeline: grid of original / sketch / mask / damaged
          for a few dataset samples.
Part B -- Model inference: load a trained checkpoint (must be a model trained
          with use_diffusers_unet=True), run the full restoration pipeline, and
          render per-sample panels:
          original | damaged(+mask) | predicted sketch | restored.
          Reports PSNR and SSIM over the whole image and inside the damaged
          region, averaged over N samples, and writes a CSV summary.

          Also reports three *consistency* metrics that measure how seamlessly
          the inpainted (repair) region blends with the surrounding healthy
          region (all "lower = better"):
            CCON - color consistency: chi-square distance between the color
                   histograms of the repair region and the healthy region.
            TCON - texture consistency: chi-square distance between LBP
                   (local binary pattern) histograms of the two regions.
            ECON - edge consistency: absolute difference of the mean edge
                   (Sobel) gradient magnitude between the two regions.

Runs on CUDA if available (fast for the 859M SD UNet), else CPU.

Usage:
    python visualize_test.py --checkpoint checkpoints/stage3.pt \
        --num-samples 12 --size 512 --steps 25
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

from config import DiffSketchConfig
from infer import DiffSketchJoint
from models.tt_prior import TTGuidance, TTConfig

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "vis_test"


def _find(directory: Path, stem: str) -> Image.Image:
    """Open <stem>.(png|jpg|jpeg) from a paired-data directory."""
    for ext in (".png", ".jpg", ".jpeg"):
        p = directory / (stem + ext)
        if p.exists():
            return Image.open(p)
    raise FileNotFoundError(f"No paired file for '{stem}' in {directory}")


def pick_samples(data_dir: Path, n: int, size: int, use_all: bool = False):
    """Return (image, sketch, mask, stem) tuples, resized to size x size.

    `data_dir` must contain `images/`, `sketches/`, `masks/` subdirectories
    (or be the project `data/` dir which has the same layout). When `use_all`
    is True, every image in `images/` is evaluated (n is ignored); otherwise a
    linspace subset of `n` images is chosen.
    """
    img_dir = data_dir / "images"
    sk_dir = data_dir / "sketches"
    mk_dir = data_dir / "masks"
    files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if use_all:
        idxs = list(range(len(files)))
    else:
        n = min(n, len(files))
        idxs = np.linspace(0, len(files) - 1, n).astype(int).tolist()
    out = []
    for i in sorted(set(idxs)):
        p = files[i]
        stem = p.stem
        img = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
        sk = _find(sk_dir, stem).convert("L").resize((size, size), Image.LANCZOS)
        mk = _find(mk_dir, stem).convert("L").resize((size, size), Image.NEAREST)
        out.append((img, sk, mk, stem))
    return out


def part_a_grid(samples, path: Path):
    """original / sketch / mask / damaged grid (mask drawn magenta on image)."""
    rows = []
    for img, sk, mk, _ in samples:
        mask_bin = np.array(mk) > 128
        damaged = np.array(img).copy()
        damaged[mask_bin] = (216, 32, 216)  # magenta = damaged
        rows.append([
            np.array(img),
            np.array(sk.convert("RGB")),
            np.array(mk.convert("RGB")),
            damaged,
        ])
    h, w = rows[0][0].shape[:2]
    titles = ["image", "sketch", "mask (white=damaged)", "damaged"]
    grid = np.ones((h * len(rows) + 30, w * 4 + 5 * 8, 3), dtype=np.uint8) * 255
    canvas = Image.fromarray(grid)
    d = ImageDraw.Draw(canvas)
    for c, t in enumerate(titles):
        d.text((8 + c * (w + 8), 8), t, fill=(0, 0, 0))
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            canvas.paste(Image.fromarray(cell), (8 + c * (w + 8), 30 + r * (h + 8)))
    canvas.save(path)
    print(f"[Part A] saved {path}")


# ---------- metrics ----------
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def _box_mean(x: np.ndarray, r: int) -> np.ndarray:
    """Uniform window mean via integral image (dependency-free). Window (2r+1)."""
    x = x.astype(np.float64)
    h, w = x.shape
    p = np.pad(x, r, mode="reflect")
    I = np.zeros((h + 2 * r + 1, w + 2 * r + 1), dtype=np.float64)
    I[1:, 1:] = np.cumsum(np.cumsum(p, axis=0), axis=1)
    s = (I[2 * r + 1:2 * r + 1 + h, 2 * r + 1:2 * r + 1 + w]
         - I[0:h, 2 * r + 1:2 * r + 1 + w]
         - I[2 * r + 1:2 * r + 1 + h, 0:w]
         + I[0:h, 0:w])
    return s / ((2 * r + 1) ** 2)


def ssim_map(a: np.ndarray, b: np.ndarray, data_range=1.0, r=5) -> np.ndarray:
    """Per-pixel SSIM map (11x11 box window). a,b: [H,W] float in [0,1]."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ma = _box_mean(a, r)
    mb = _box_mean(b, r)
    ma2 = _box_mean(a * a, r)
    mb2 = _box_mean(b * b, r)
    mab = _box_mean(a * b, r)
    sa2 = ma2 - ma * ma
    sb2 = mb2 - mb * mb
    sab = mab - ma * mb
    num = (2 * ma * mb + c1) * (2 * sab + c2)
    den = (ma * ma + mb * mb + c1) * (sa2 + sb2 + c2)
    return num / np.maximum(den, 1e-12)


def ssim(a: np.ndarray, b: np.ndarray, data_range=1.0) -> float:
    """Mean SSIM over all pixels, averaged across channels. a,b: [H,W,3]."""
    tot = 0.0
    for c in range(a.shape[2]):
        tot += ssim_map(a[..., c], b[..., c], data_range).mean()
    return float(tot / a.shape[2])


def ssim_masked(a: np.ndarray, b: np.ndarray, mask_bin: np.ndarray,
                data_range=1.0) -> float:
    """Mean SSIM restricted to damaged pixels, averaged across channels."""
    vals = []
    for c in range(a.shape[2]):
        m = ssim_map(a[..., c], b[..., c], data_range)
        if mask_bin.any():
            vals.append(m[mask_bin].mean())
    return float(np.mean(vals)) if vals else float("nan")


def _hist_norm(values, bins, vmin=0.0, vmax=1.0):
    """Return a normalized histogram (sum=1) of `values` in [vmin, vmax]."""
    counts, _ = np.histogram(values, bins=bins, range=(vmin, vmax))
    total = counts.sum()
    if total == 0:
        return np.full(bins, 1.0 / bins)
    return counts.astype(np.float64) / total


def _chi_square(p, q):
    """Chi-square histogram distance; p,q are normalized histograms."""
    eps = 1e-12
    return float(np.sum((p - q) ** 2 / (p + q + eps)))


def color_consistency(repair_rgb, healthy_rgb, bins=64):
    """
    CCON: chi-square distance between the color histograms of the repair
    region and the healthy (surrounding) region. Per-channel, then averaged.
    repair_rgb / healthy_rgb: flat arrays of shape [N,3] in [0,1].
    """
    ch = []
    for c in range(3):
        p = _hist_norm(repair_rgb[:, c], bins)
        q = _hist_norm(healthy_rgb[:, c], bins)
        ch.append(_chi_square(p, q))
    return float(np.mean(ch))


def _lbp(gray):
    """Basic 8-neighborhood LBP on a [H,W] float image in [0,1].
    Returns a [H,W] uint8 code map (uniform/non-uniform merged)."""
    h, w = gray.shape
    g = np.pad(gray, 1, mode="edge")
    center = gray
    out = np.zeros((h, w), dtype=np.uint8)
    # clockwise from top-left: (dy, dx) for the 8 surrounding pixels
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, 1), (1, 1), (1, 0),
                 (1, -1), (0, -1)]
    for k, (dy, dx) in enumerate(neighbors):
        neigh = g[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        out |= ((neigh >= center).astype(np.uint8) << k)
    return out


def texture_consistency(repair_gray_full, healthy_gray_full, mask_bin, bins=256):
    """
    TCON: chi-square distance between the LBP histograms of the repair region
    and the healthy region. Inputs are full grayscale [H,W] maps in [0,1];
    the mask selects the respective pixels.
    """
    p = _hist_norm(_lbp(repair_gray_full)[mask_bin].astype(np.float64), bins, 0, 255)
    q = _hist_norm(_lbp(healthy_gray_full)[~mask_bin].astype(np.float64), bins, 0, 255)
    return _chi_square(p, q)


def _grad_mag_map(rgb):
    """Sobel gradient magnitude on the luminance of an [H,W,3] image in [0,1]."""
    lum = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    gp = np.pad(lum, 1, mode="edge")
    gx = gp[1:-1, 2:] - gp[1:-1, :-2]
    gy = gp[2:, 1:-1] - gp[:-2, 1:-1]
    return np.sqrt(gx ** 2 + gy ** 2)


def edge_consistency(repair_rgb_full, healthy_rgb_full, mask_bin):
    """
    ECON: absolute difference between the mean edge (Sobel) gradient magnitude
    of the repair region and that of the surrounding healthy region.
    Lower = the inpainted region has edge strength matching its surroundings.
    """
    gr = _grad_mag_map(repair_rgb_full)
    gh = _grad_mag_map(healthy_rgb_full)
    return float(abs(gr[mask_bin].mean() - gh[~mask_bin].mean()))


@torch.no_grad()
def restore_with_sketch(net, img_t, mask_t, num_steps, tt=None, sampler="ddim",
                        init_image=None, t_start=0, one_step=0, one_step_multi=None,
                        cascade=False, cond_init=False, sketch_init=False,
                        refine_ts=None, return_latent=False, latent_refiner=None):
    cfg = net.cfg
    device = net.device
    amp = bool(cfg.amp and device.type == "cuda")
    with torch.amp.autocast(device_type=device.type, dtype=getattr(net, "eval_dtype", torch.float16), enabled=amp):
        damaged_t = img_t * (1.0 - mask_t)
        if sketch_init and init_image is not None:
            # The sketch branch normally sees a GRAY hole and must hallucinate
            # all line structure inside it. Feeding the classical fill instead
            # gives it a complete (if smooth) image, so predicted lines can
            # continue across the hole from real surrounding evidence.
            enc_in = init_image
        else:
            enc_in = damaged_t
        shared_feats = net.encoder(torch.cat([enc_in, mask_t], dim=1))
        sketch_prob = net.sketch_branch(shared_feats, mode="infer", infer_steps=num_steps)
        restored = net.inpaint_branch.generate(img_t, mask_t, sketch_prob, num_steps=num_steps,
                                               tt=tt, sampler=sampler,
                                               init_image=init_image, t_start=t_start,
                                               one_step=one_step,
                                               one_step_multi=one_step_multi,
                                               cascade=cascade,
                                               cond_init=cond_init,
                                               refine_ts=refine_ts,
                                               return_latent=return_latent,
                                               latent_refiner=latent_refiner)
    if return_latent:
        restored_img, x0_hat = restored
        return restored_img, sketch_prob, x0_hat
    return restored, sketch_prob




def smooth_fill(img_t: torch.Tensor, mask_t: torch.Tensor,
                iters: int = 12, k: int = 31) -> torch.Tensor:
    """Classical normalized-convolution color fill: propagate border colors into
    the hole. img_t [B,3,H,W] in [-1,1], mask_t [B,1,H,W] with 1 = hole.
    Used as the SDEdit init anchor (color-correct -> high-PSNR starting point)."""
    w = (1.0 - mask_t)
    x = img_t * w
    for _ in range(iters):
        wb = F.avg_pool2d(w, k, stride=1, padding=k // 2)
        xb = F.avg_pool2d(x * w, k, stride=1, padding=k // 2)
        est = xb / wb.clamp_min(1e-4)
        x = x * (1.0 - mask_t) + est * mask_t
        w = torch.ones_like(w)  # allow propagation in later passes
    return img_t * (1.0 - mask_t) + x * mask_t


def pyramid_fill(img_t: torch.Tensor, mask_t: torch.Tensor,
                 ks=(63, 47, 31, 23, 15, 11, 7, 5, 3), reps: int = 6,
                 tail: int = 16, scale: float = 1.0) -> torch.Tensor:
    """Coarse-to-fine normalized-convolution fill: start with a huge kernel so
    color floods deep holes globally, then refine with decreasing kernels, and
    finish with a long k=3 tail that converges toward the harmonic
    (Laplace) interpolation -- measurably the best classical PSNR fill
    (+2.9 dB PSNR_all over single-scale smooth fill on the mural test set).

    `scale` multiplies the kernel sizes so the fill's *relative* smoothing is
    resolution-independent (use size/512 when evaluating below 512px)."""
    if scale != 1.0:
        ks = tuple(max(3, int(round(k * scale)) // 2 * 2 + 1) for k in ks)
    w = (1.0 - mask_t)
    x = img_t.clone()
    for k in list(ks) + [3] * tail:
        for _ in range(reps if k in ks else 1):
            wb = F.avg_pool2d(w, k, stride=1, padding=k // 2)
            xb = F.avg_pool2d(x * w, k, stride=1, padding=k // 2)
            est = xb / wb.clamp_min(1e-4)
            x = x * (1.0 - mask_t) + est * mask_t
            w = torch.ones_like(w)
    return img_t * (1.0 - mask_t) + x * mask_t


def erode_depth(mask_t: torch.Tensor, iters: int) -> torch.Tensor:
    """Chebyshev distance-to-boundary inside the hole, in pixels.
    Binary erosion by a 3x3 min-filter, accumulating how many erosions each
    pixel survives. Pixels at the hole rim get 0, deep interior gets higher."""
    d = torch.zeros_like(mask_t)
    m = mask_t.clone()
    for _ in range(iters):
        m = -F.max_pool2d(-m, 3, stride=1, padding=1)  # binary erosion
        d = d + m
    return d


def part_b_infer(sample, net, num_steps: int, path: Path, tt=None, sampler="ddim",
                 init="none", t_start=0, avg=1, seed=0, one_step=0, blend=1.0,
                 fill_reps=6, one_step_multi=None, cascade=False, dist_blend=0,
                 cond_init=False, fill_scale=1.0, freq_blend=0.0, sketch_init=False,
                 refine_ts=None, latent_refiner=None):
    img, sk, mk, stem = sample
    device = net.device
    img_t = (TF.to_tensor(img) * 2.0 - 1.0).unsqueeze(0).to(device)
    mask_t = (TF.to_tensor(mk).unsqueeze(0).to(device) > 0.5).float()

    if init == "smooth":
        init_image = smooth_fill(img_t, mask_t)
    elif init == "pyramid":
        init_image = pyramid_fill(img_t, mask_t, reps=fill_reps, scale=fill_scale)
    else:
        init_image = None

    t0 = time.time()
    outs, sketch_prob = [], None
    for k in range(max(1, avg)):
        torch.manual_seed(seed + k)  # paired A/B: same seed -> same init noise
        r, sketch_prob = restore_with_sketch(net, img_t, mask_t, num_steps, tt=tt,
                                             sampler=sampler,
                                             init_image=init_image, t_start=t_start,
                                             one_step=one_step,
                                             one_step_multi=one_step_multi,
                                             cascade=cascade,
                                             cond_init=cond_init,
                                             sketch_init=sketch_init,
                                             refine_ts=refine_ts,
                                             latent_refiner=latent_refiner)
        outs.append(r)
    restored_t = torch.stack(outs).mean(0)  # seed-averaging approximates E[x|obs] (PSNR-optimal)
    if blend < 1.0 and init_image is not None:
        if dist_blend > 0:
            # Spatially varying shrinkage: interpolation is reliable at the hole
            # rim, so trust the fill there; trust the model deep inside where
            # structure matters. w(d) = blend * min(1, d/dist_blend).
            d = erode_depth(mask_t, max(2 * dist_blend, 8))
            w = torch.clamp(d / float(dist_blend), 0.0, 1.0)
            w = torch.where(mask_t > 0.5, blend * w, torch.zeros_like(w))
            restored_t = w * restored_t + (1.0 - w) * init_image
        elif freq_blend > 0:
            # Frequency-split shrinkage. The classical fill is a harmonic
            # interpolant, so its error is almost entirely HIGH frequency (it
            # cannot reproduce thin outlines); the diffusion estimate has the
            # better texture statistics (far lower TCON) but noisier pixels.
            # Take the LOW frequencies from the fill and scale only the model's
            # HIGH-frequency residual:
            #   out = blur(fill) + w * (model - blur(model))
            # w=1 -> pure model, w=0 -> blurred fill. Used together with
            # --blend 1.0 so `restored_t` is the pure model estimate.
            def _lp(t, s=2.0):
                k = int(2 * round(2 * s) + 1)
                r = torch.arange(k, device=t.device, dtype=t.dtype) - k // 2
                g = torch.exp(-(r ** 2) / (2 * s * s))
                g = (g / g.sum()).view(1, 1, 1, -1).expand(t.shape[1], 1, 1, -1)
                gv = g.reshape(t.shape[1], 1, -1, 1)
                t = F.conv2d(F.pad(t, (k // 2,) * 4, mode="reflect"), g, padding=0, groups=t.shape[1])
                return F.conv2d(F.pad(t, (k // 2,) * 4, mode="reflect"), gv, padding=0, groups=t.shape[1])
            restored_t = _lp(init_image) + freq_blend * (restored_t - _lp(restored_t))
        else:
            # Global shrinkage toward the color-correct classical fill (both
            # equal the original image outside the mask).
            w = blend * torch.ones_like(restored_t)
            restored_t = w * restored_t + (1.0 - w) * init_image
    dt = time.time() - t0

    restored_01 = ((restored_t[0].float().clamp(-1, 1) + 1.0) * 0.5).cpu()
    restored = TF.to_pil_image(restored_01)
    sketch_pred = TF.to_pil_image(sketch_prob[0].float().cpu().clamp(0, 1))

    orig = np.asarray(img, dtype=np.float32) / 255.0
    rec = np.asarray(restored, dtype=np.float32) / 255.0
    mask_bin = (np.asarray(mk) > 128)
    p_all = psnr(orig, rec)
    p_dmg = psnr(orig[mask_bin], rec[mask_bin]) if mask_bin.any() else float("nan")
    s_all = ssim(orig, rec)
    s_dmg = ssim_masked(orig, rec, mask_bin)

    # ---- consistency metrics: repair region vs surrounding healthy region ----
    ccon = tcon = ecom = float("nan")
    if mask_bin.any():
        ccon = color_consistency(rec[mask_bin], orig[~mask_bin])
        rec_gray = rec[..., 0] * 0.299 + rec[..., 1] * 0.587 + rec[..., 2] * 0.114
        orig_gray = orig[..., 0] * 0.299 + orig[..., 1] * 0.587 + orig[..., 2] * 0.114
        tcon = texture_consistency(rec_gray, orig_gray, mask_bin)
        ecom = edge_consistency(rec, orig, mask_bin)

    damaged = np.array(img).copy()
    damaged[mask_bin] = (216, 32, 216)
    cells = [np.array(img), damaged, np.array(sketch_pred.convert("RGB")), np.array(restored)]
    titles = ["original", "damaged (input)", f"predicted sketch ({num_steps} steps)",
              "restored"]
    h, w = cells[0].shape[:2]
    panel = Image.new("RGB", (w * 4 + 5 * 8, h + 46), (255, 255, 255))
    d = ImageDraw.Draw(panel)
    for c, t in enumerate(titles):
        d.text((8 + c * (w + 8), 8), t, fill=(0, 0, 0))
    d.text((8, h + 28),
           f"{stem} | PSNR dmg={p_dmg:.2f} dB | SSIM dmg={s_dmg:.3f} | "
           f"CCON={ccon:.3f} TCON={tcon:.3f} ECON={ecom:.3f} | {dt:.1f}s",
           fill=(0, 0, 0))
    for c, cell in enumerate(cells):
        panel.paste(Image.fromarray(cell), (8 + c * (w + 8), 26))
    panel.save(path)
    print(f"[Part B] {stem}: PSNR_all={p_all:.2f} dB, PSNR_dmg={p_dmg:.2f} dB, "
          f"SSIM_all={s_all:.3f}, SSIM_dmg={s_dmg:.3f}, "
          f"CCON={ccon:.3f}, TCON={tcon:.3f}, ECON={ecom:.3f}, time={dt:.1f}s -> {path}")
    return p_all, p_dmg, s_all, s_dmg, ccon, tcon, ecom, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "stage3.pt"))
    ap.add_argument("--data", default=str(ROOT / "data"),
                   help="base dir containing images/ sketches/ masks/ subdirs "
                        "(e.g. F:/bihua/test or F:/bihua/val)")
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--all", action="store_true",
                   help="evaluate every image in --data/images (ignores --num-samples)")
    ap.add_argument("--size", type=int, default=512, help="working resolution (multiple of 64)")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"],
                    help="ddim (default): deterministic DDIM sampling -- clean fills at "
                         "20-50 steps. ddpm: strided DDPM (legacy behaviour; leaves "
                         "residual noise speckles inside holes at low step counts).")
    ap.add_argument("--no-fdp-eval", action="store_true",
                    help="disable the FDP frequency-sharpening module at inference "
                         "(A/B ablation; training default is FDP on)")
    ap.add_argument("--hf-refiner", action="store_true",
                    help="build the learned image-space HIGH-FREQUENCY generator so its "
                         "weights are loaded from a checkpoint trained with --hf-refiner "
                         "(must match how the checkpoint was trained; ignored otherwise)")
    ap.add_argument("--hf-refiner-path", default="",
                    help="path to a standalone trained HighFreqRefiner state_dict "
                         "(produced by train_hf_refiner.py). Loaded after the main "
                         "checkpoint; the model is auto-built with --hf-refiner when given.")
    ap.add_argument("--hf-refiner-base", type=int, default=32,
                    help="HighFreqRefiner base width; must match the trained refiner.")
    ap.add_argument("--hf-refiner-blocks", type=int, default=4,
                    help="HighFreqRefiner depth; must match the trained refiner.")
    ap.add_argument("--hf-refiner-out-scale", type=float, default=0.25,
                    help="HighFreqRefiner residual scale; must match the trained refiner.")
    ap.add_argument("--no-hf-refiner", action="store_true",
                    help="inference-time ablation: drop the learned HF generator even if "
                         "the checkpoint contains one")
    ap.add_argument("--vae-decoder-path", default="",
                    help="Route-C: path to a fine-tuned AutoencoderKL directory "
                         "(train_vae_decoder_c.py). Swapped in for the default VAE decoder "
                         "so decode(z_pred) uses the trained latent->image mapping.")
    ap.add_argument("--latent-refiner-path", default="",
                    help="Route-C latent refiner: path to a LatentRefiner state_dict "
                         "(train_latent_refiner.py). Applied to x0_hat BEFORE decode, "
                         "correcting z_pred toward z_gt in latent space.")
    # ---- SDEdit init + seed averaging (PSNR-oriented inference) ----
    ap.add_argument("--init", default="none", choices=["none", "smooth", "pyramid"],
                    help="smooth = normalized-conv color fill; pyramid = coarse-to-fine "
                         "fill (better on large scattered masks). Both serve as the "
                         "SDEdit/MMSE init anchor.")
    ap.add_argument("--t-start", type=int, default=0,
                    help="noise level to start sampling from when --init smooth "
                         "(e.g. 300; 0 = full-noise generation)")
    ap.add_argument("--avg", type=int, default=1,
                    help="average N seeded samples (approximates the conditional "
                         "mean, which maximizes PSNR)")
    ap.add_argument("--one-step", type=int, default=0,
                    help="if >0, do a single Tweedie denoise step at this timestep "
                         "(MMSE / conditional-mean inpainting) instead of iterative "
                         "sampling. Combine with --init smooth --t-start T for "
                         "init-anchored one-step denoising.")
    ap.add_argument("--one-step-multi", default="",
                    help="comma-separated timesteps, e.g. '150,300,450': average "
                         "Tweedie x0 estimates across several noise levels (lower "
                         "variance than any single t). Overrides --one-step.")
    ap.add_argument("--cascade", action="store_true",
                    help="with --one-step-multi: cascaded refinement -- each step "
                         "re-noises the PREVIOUS estimate (300->150->75) instead of "
                         "always re-noising the original fill")
    ap.add_argument("--cond-init", action="store_true",
                    help="condition the UNet's 9th channel on the classical fill latent "
                         "instead of the gray-hole damaged image (matches --fill-consistent "
                         "training; only use with a fill-consistent checkpoint)")
    ap.add_argument("--dist-blend", type=int, default=0,
                    help="if >0, spatially varying blend: trust the classical fill at "
                         "the hole rim and the model deep inside; w(d)=blend*min(1,d/D) "
                         "with d = pixels of distance to the boundary")
    ap.add_argument("--blend", type=float, default=1.0,
                    help="blend model output toward the smooth fill: out = blend*model "
                         "+ (1-blend)*smooth (1.0 = model only; e.g. 0.5 shrinks "
                         "halfway toward the color-correct fill)")
    ap.add_argument("--sketch-init", action="store_true",
                    help="feed the classical fill (instead of the gray-hole damaged "
                         "image) to the sketch branch, so predicted lines continue "
                         "across the hole from real surrounding evidence")
    ap.add_argument("--freq-blend", type=float, default=0.0,
                    help="if >0, frequency-split shrinkage: low frequencies from the "
                         "classical fill, high frequencies from the model scaled by this "
                         "weight. Use together with --blend 1.0.")
    ap.add_argument("--fill-scale", type=float, default=1.0,
                    help="multiply pyramid-fill kernel sizes by this factor; use "
                         "--size/512 to keep the fill's relative smoothing "
                         "resolution-independent when evaluating below 512px")
    ap.add_argument("--fill-reps", type=int, default=6,
                    help="refinement passes per pyramid level of the classical fill "
                         "(higher = smoother harmonic-like interpolation)")
    ap.add_argument("--bf16", action="store_true",
                   help="Opt-in bf16 inference. Default is fp32, which is REQUIRED for "
                        "fp32-trained checkpoints: casting their weights to bf16 collapses "
                        "whole-image PSNR by ~15 dB (the SD UNet/additive heads are sensitive "
                        "to bf16 rounding). Only use --bf16 for models trained with train_bf16.")
    # ---- Low-rank (Tensor-Train) structure guidance (OFF by default) ----
    ap.add_argument("--tt", action="store_true", help="enable TT structure guidance")
    ap.add_argument("--tt-mode", default="dynamic", choices=["dynamic", "static"],
                   help="dynamic = self low-rank projection; static = fitted global prior")
    ap.add_argument("--tt-rank", type=int, default=4)
    ap.add_argument("--tt-lambda", type=float, default=0.5, help="peak guidance strength")
    ap.add_argument("--tt-schedule", default="early",
                   choices=["early", "mid", "late", "constant"],
                   help="early = structure first (recommended); late = ablation")
    ap.add_argument("--tt-t-lo", type=float, default=0.15)
    ap.add_argument("--tt-t-hi", type=float, default=1.0)
    ap.add_argument("--tt-p", type=float, default=1.0)
    ap.add_argument("--tt-fit-iters", type=int, default=200)
    ap.add_argument("--tt-fit-space", default="latent", choices=["latent", "image"])
    ap.add_argument("--tt-scale", default="signal", choices=["signal", "boost"],
                   help="signal = z += sqrt(alpha_bar_{t-1})*delta (late-weighted); "
                        "boost = un-attenuated, so an early schedule really acts")
    ap.add_argument("--tt-out", default="", help="suffix appended to the output CSV name")
    ap.add_argument("--out-dir", default=None,
                   help="output directory for images + eval_summary.csv. "
                        "Defaults to ROOT/vis_test. Set a distinct dir per run so that "
                        "concurrent evaluations cannot overwrite each other's results.")
    ap.add_argument("--seed", type=int, default=0,
                   help="seed per sample (seed + sample index) so that two runs "
                        "see the identical sampler noise -> paired A/B comparison")
    ap.add_argument("--refine-ts", default="",
                   help="descending Tweedie refinement ladder, e.g. '150,100,60'. "
                        "Keeps the FINAL (lowest-t) estimate; off by default.")
    args = ap.parse_args()
    args.size = max(64, (args.size // 64) * 64)
    osp = [int(v) for v in args.one_step_multi.split(",") if v.strip()] or None
    rts = [int(v) for v in args.refine_ts.split(",") if v.strip()] or None
    data_dir = Path(args.data).resolve()

    global OUT_DIR
    if args.out_dir:
        OUT_DIR = Path(args.out_dir).resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Part A: data pipeline ----
    samples = pick_samples(data_dir, min(args.num_samples, 6), 360)
    part_a_grid(samples, OUT_DIR / "a_data_pipeline.png")

    # ---- Part B: model inference ----
    cfg = DiffSketchConfig()
    # CRITICAL: the evaluated checkpoint was trained with the full SD UNet.
    cfg.use_diffusers_unet = True
    cfg.pretrained_vae = str(ROOT / "weights" / "sd-vae-ft-mse")
    cfg.amp = args.bf16                   # fp32 by default (reliable for any checkpoint)
    cfg.mem_efficient = True               # attention + VAE slicing on by default
    # Build the HF refiner module when explicitly requested OR when a standalone
    # refiner weight path is supplied (so --hf-refiner-path alone is enough).
    cfg.hf_refiner = bool(args.hf_refiner) or bool(args.hf_refiner_path)
    cfg.hf_refiner_base = args.hf_refiner_base
    cfg.hf_refiner_blocks = args.hf_refiner_blocks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {args.checkpoint} on {device} ...")
    t0 = time.time()
    net = DiffSketchJoint(cfg)
    net.load_checkpoint(args.checkpoint)
    net.eval()
    # Default fp32 inference. Only --bf16 casts weights to bf16, and ONLY for
    # checkpoints actually trained with train_bf16 (casting fp32-trained weights
    # to bf16 collapses whole-image PSNR by ~15 dB). The SD VAE is kept fp32
    # regardless (forced in encode/decode) because it is numerically fragile.
    if args.bf16:
        net = net.to(torch.bfloat16)
        if getattr(getattr(net, "inpaint_branch", None), "vae", None) is not None:
            net.inpaint_branch.vae = net.inpaint_branch.vae.float()
    net.eval_dtype = torch.bfloat16 if args.bf16 else torch.float16
    if getattr(args, "no_fdp_eval", False) and getattr(net.inpaint_branch, "fdp", None) is not None:
        net.inpaint_branch.fdp = None  # inference-time ablation of FDP sharpening
        print("[ablation] FDP disabled at inference")
    if getattr(args, "no_hf_refiner", False) and getattr(net.inpaint_branch, "hf_refiner", None) is not None:
        net.inpaint_branch.hf_refiner = None
        print("[ablation] learned HF refiner disabled at inference")
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Optional: load a standalone trained HighFreqRefiner (train_hf_refiner.py).
    if args.hf_refiner_path:
        sd = torch.load(args.hf_refiner_path, map_location="cpu", weights_only=True)
        ref = net.inpaint_branch.hf_refiner
        assert ref is not None, "--hf-refiner-path needs the refiner module built"
        ref.load_state_dict(sd)
        ref.out_scale = float(args.hf_refiner_out_scale)
        ref.to(net.device)
        print(f"[refiner] loaded {args.hf_refiner_path} (out_scale={ref.out_scale})")

    # Route-C: swap in a fine-tuned VAE decoder (train_vae_decoder_c.py).
    if args.vae_decoder_path:
        from diffusers import AutoencoderKL
        new_vae = AutoencoderKL.from_pretrained(args.vae_decoder_path).to(net.device).float()
        net.inpaint_branch.vae = new_vae
        net.inpaint_branch.vae.eval()
        print(f"[route-C] VAE decoder swapped from {args.vae_decoder_path}")

    # Route-C latent refiner (train_latent_refiner.py): applied to x0_hat BEFORE decode.
    latent_refiner = None
    if args.latent_refiner_path:
        from latent_refiner_model import LatentRefiner
        ref = LatentRefiner().to(net.device).float().eval()
        ref.load_state_dict(torch.load(args.latent_refiner_path,
                                       map_location="cpu", weights_only=True))
        latent_refiner = ref
        # The latent refiner + VAE decoder trigger a cuDNN crash on this
        # Windows/WDDM box; native kernels are stable (just slower).
        torch.backends.cudnn.enabled = False
        print(f"[route-C] latent refiner loaded from {args.latent_refiner_path} (cuDNN disabled)")

    tt = None
    if args.tt:
        tt = TTGuidance(TTConfig(
            enabled=True, mode=args.tt_mode, rank=args.tt_rank, lam=args.tt_lambda,
            schedule=args.tt_schedule, t_lo=args.tt_t_lo, t_hi=args.tt_t_hi,
            p=args.tt_p, fit_iters=args.tt_fit_iters, fit_space=args.tt_fit_space,
            scale_mode=args.tt_scale,
        ), timesteps=cfg.diffusion_steps)
        print(f"[TT guidance] mode={args.tt_mode} rank={args.tt_rank} "
              f"lambda={args.tt_lambda} schedule={args.tt_schedule} "
              f"band=[{args.tt_t_lo},{args.tt_t_hi}] fit_space={args.tt_fit_space}")

    samples_b = pick_samples(data_dir, args.num_samples, args.size, use_all=args.all)
    print(f"Evaluating {len(samples_b)} samples from {data_dir}")
    summary = []
    for i, s in enumerate(samples_b):
        out_path = OUT_DIR / f"b_restore_{s[3]}.png"
        # Paired comparison: identical sampler noise for every configuration.
        torch.manual_seed(args.seed + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + i)
        summary.append(part_b_infer(s, net, args.steps, out_path, tt=tt, sampler=args.sampler,
                                    init=args.init, t_start=args.t_start,
                                    avg=args.avg, seed=args.seed,
                                    one_step=args.one_step, blend=args.blend,
                                    fill_reps=args.fill_reps, fill_scale=args.fill_scale,
                                    freq_blend=args.freq_blend,
                                    sketch_init=args.sketch_init,
                                    one_step_multi=osp, cascade=args.cascade,
                                    dist_blend=args.dist_blend,
                                    cond_init=args.cond_init,
                                    refine_ts=rts,
                                    latent_refiner=latent_refiner))

    # ---- aggregate ----
    p_all = np.array([x[0] for x in summary])
    p_dmg = np.array([x[1] for x in summary])
    s_all = np.array([x[2] for x in summary])
    s_dmg = np.array([x[3] for x in summary])
    ccon_a = np.array([x[4] for x in summary])
    tcon_a = np.array([x[5] for x in summary])
    ecom_a = np.array([x[6] for x in summary])
    dts = np.array([x[7] for x in summary])

    # Use nan-aware aggregation so a single degenerate sample (e.g. an empty
    # mask, which makes the damaged-region metrics undefined) cannot poison the
    # whole mean/variance of the run.
    n_valid = int(np.isfinite(s_dmg).sum())
    print("\n=== Summary ===")
    print(f"PSNR_all     : {np.nanmean(p_all):.2f} +/- {np.nanstd(p_all):.2f} dB  (n={len(p_all)})")
    print(f"PSNR_damaged : {np.nanmean(p_dmg):.2f} +/- {np.nanstd(p_dmg):.2f} dB  (valid n={n_valid})")
    print(f"SSIM_all     : {np.nanmean(s_all):.3f} +/- {np.nanstd(s_all):.3f}")
    print(f"SSIM_damaged : {np.nanmean(s_dmg):.3f} +/- {np.nanstd(s_dmg):.3f}")
    print(f"CCON (color χ², lower=better)   : {np.nanmean(ccon_a):.4f} +/- {np.nanstd(ccon_a):.4f}")
    print(f"TCON (LBP texture χ², lower=better): {np.nanmean(tcon_a):.4f} +/- {np.nanstd(tcon_a):.4f}")
    print(f"ECON (edge |Δgrad|, lower=better): {np.nanmean(ecom_a):.4f} +/- {np.nanstd(ecom_a):.4f}")
    print(f"avg time/sample: {np.nanmean(dts):.1f}s")
    print(f"Outputs in {OUT_DIR}")

    tag = args.tt_out or ("tt" if args.tt else "")
    csv_path = OUT_DIR / (f"eval_summary_{tag}.csv" if tag else "eval_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "psnr_all", "psnr_damaged", "ssim_all", "ssim_damaged",
                    "ccon", "tcon", "econ", "time_s"])
        for (stem, (pa, pd, sa, sd, cc, tc, ec, dt)) in zip([s[3] for s in samples_b], summary):
            w.writerow([stem, f"{pa:.4f}", f"{pd:.4f}", f"{sa:.4f}", f"{sd:.4f}",
                        f"{cc:.4f}", f"{tc:.4f}", f"{ec:.4f}", f"{dt:.2f}"])
        w.writerow(["MEAN", f"{np.nanmean(p_all):.4f}", f"{np.nanmean(p_dmg):.4f}",
                    f"{np.nanmean(s_all):.4f}", f"{np.nanmean(s_dmg):.4f}",
                    f"{np.nanmean(ccon_a):.4f}", f"{np.nanmean(tcon_a):.4f}", f"{np.nanmean(ecom_a):.4f}",
                    f"{np.nanmean(dts):.2f}"])
        w.writerow(["STD", f"{np.nanstd(p_all):.4f}", f"{np.nanstd(p_dmg):.4f}",
                    f"{np.nanstd(s_all):.4f}", f"{np.nanstd(s_dmg):.4f}",
                    f"{np.nanstd(ccon_a):.4f}", f"{np.nanstd(tcon_a):.4f}", f"{np.nanstd(ecom_a):.4f}", ""])
    print(f"[CSV] saved {csv_path}")


if __name__ == "__main__":
    main()
