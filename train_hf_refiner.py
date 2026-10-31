#!/usr/bin/env python
"""
train_hf_refiner.py -- VAE decode-side refinement (the winnable form of
"VAE decoder fine-tuning" for DiffSketch-Joint).

WHY THIS, NOT decoder-weight retraining
---------------------------------------
vae_ceiling.py shows that encoding GT and decoding it back yields at most
~25.37 dB PSNR inside the damaged hole. That 25.37 dB is the VAE *latent*
information ceiling: the 8x-downsampled latent simply cannot carry more hole
detail, so re-training the VAE decoder weights CANNOT beat it. The current
20.68 dB hole PSNR is the *diffusion latent* being imperfect, not the decoder
being weak.

The only way to push hole fidelity past what the latent carries is to inject
the detail the latent dropped -- and that detail lives in the MASK (hole
boundary / surrounding colour) and the SKETCH (line layout). That is exactly
what HighFreqRefiner does: an image-domain, mask+sketch-conditioned residual
added after the VAE decode (and after FDP sharpening). It is zero-init, so
loading it on top of a converged checkpoint is the identity until trained.

This trainer never touches the UNet / VAE / sketch branch -- only the refiner
conv weights get gradients. That sidesteps the whole-model "resume loses ~0.4
dB then band-loss only recovers 0.18" failure that killed the FDDM attempt,
and directly attacks the diagnosed "HF error = 4.6x LF error" deficit.

Training signal (streaming, RAM-safe)
-------------------------------------
For every training image we compute on the fly  I0 = decode(encode(GT))  under
the frozen VAE+FDP (the exact tensor the refiner sees as its input at
inference; the refiner itself is toggled OFF during this step so it does not
contaminate its own training input). The refiner is then trained to map
(I0, mask, sketch) -> GT, supervised only inside the hole. At inference the
same refiner refines the *real* diffusion decode, which is also a VAE decode
of some latent -- so the "VAE-decode -> sharp hole" mapping transfers.

We recompute I0 per epoch (VAE is frozen + fast) instead of caching ~13 GB of
tensors, which matters on this box where free system RAM is ~0.

Usage:
    python train_hf_refiner.py \
        --data C:/wb_diffsketch/data \
        --checkpoint C:/wb_diffsketch/checkpoints_fc1/stage3.pt \
        --out C:/wb_diffsketch/checkpoints_fc1/hf_refiner.pt \
        --epochs 40 --batch 8 --lr 2e-4
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from config import DiffSketchConfig
from infer import DiffSketchJoint
from losses.joint_loss import VGGPerceptual, sobel_edges


def build_model(cfg, checkpoint, device):
    net = DiffSketchJoint(cfg).to(device)
    net.load_checkpoint(checkpoint)
    net.eval()
    # Freeze EVERYTHING except the HF refiner. We only ever backprop through
    # the refiner's conv stack; the VAE / FDP / UNet stay frozen.
    for p in net.parameters():
        p.requires_grad_(False)
    ref = net.inpaint_branch.hf_refiner
    assert ref is not None, "model built without hf_refiner"
    for p in ref.parameters():
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in ref.parameters() if p.requires_grad)
    print(f"[model] trainable refiner params: {n_train:,}")
    return net, ref


def gather_pairs(data_dir):
    img_dir = Path(data_dir) / "images"
    sk_dir = Path(data_dir) / "sketches"
    mk_dir = Path(data_dir) / "masks"
    files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    pairs = []
    for p in files:
        stem = p.stem
        skp = sk_dir / (stem + ".jpg")
        if not skp.exists():
            skp = sk_dir / (stem + ".png")
        mkp = mk_dir / (stem + ".png")
        if skp.exists() and mkp.exists():
            pairs.append((p, skp, mkp))
    print(f"[data] {len(pairs)} paired samples in {img_dir}")
    return pairs


def load_sample(p, skp, mkp, size, device):
    img = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
    sk = Image.open(skp).convert("L").resize((size, size), Image.LANCZOS)
    mk = Image.open(mkp).convert("L").resize((size, size), Image.NEAREST)
    # TF.to_tensor returns [C,H,W]; add a batch dim so torch.cat stacks on batch.
    gt = (TF.to_tensor(img) * 2.0 - 1.0).to(device).unsqueeze(0)        # [1,3,H,W]
    mask = (TF.to_tensor(mk).to(device) > 0.5).float().unsqueeze(0)     # [1,1,H,W]
    sketch = (TF.to_tensor(sk).to(device)).unsqueeze(0)                 # [1,1,H,W]
    return gt, mask, sketch


@torch.no_grad()
def vae_io(net, ref, gtb):
    """I0 = VAE+FDP decode of GT batch, with the (trained) refiner toggled OFF so
    the refiner never sees its own output as its training input."""
    net.inpaint_branch.hf_refiner = None
    z = net.inpaint_branch.encode(gtb)                  # [B,4,h,w]
    I0 = net.inpaint_branch.decode(z).clamp(-1.0, 1.0)  # [B,3,H,W]
    net.inpaint_branch.hf_refiner = ref
    return I0


def weighted_perceptual(vgg, pred, gt, mask, device):
    """Per-sample perceptual L1, weighted by the (hole) mask at each feature res."""
    with torch.amp.autocast(device_type=device.type, enabled=False):
        fp = vgg(pred.float())
        with torch.no_grad():
            ft = vgg(gt.float())
        per_sample = []
        for a, b in zip(fp, ft):
            w = F.interpolate(mask.float(), size=a.shape[-2:], mode="nearest")
            d = (a - b).abs().mean(dim=1, keepdim=True)        # [B,1,h,w]
            per = (d * w).sum(dim=(1, 2, 3)) / w.sum(dim=(1, 2, 3)).clamp_min(1e-6)
            per_sample.append(per)
        return torch.stack(per_sample, 0).mean(0)              # [B]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="C:/wb_diffsketch/data")
    ap.add_argument("--checkpoint", default="C:/wb_diffsketch/checkpoints_fc1/stage3.pt")
    ap.add_argument("--out", default="C:/wb_diffsketch/checkpoints_fc1/hf_refiner.pt")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hf-base", type=int, default=64)
    ap.add_argument("--hf-blocks", type=int, default=6)
    ap.add_argument("--hf-out-scale", type=float, default=0.3)
    ap.add_argument("--lambda-perc", type=float, default=0.5)
    ap.add_argument("--lambda-scc", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of training pairs (0 = use all); for smoke tests")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = DiffSketchConfig()
    cfg.use_diffusers_unet = True
    cfg.pretrained_vae = str(Path(args.checkpoint).resolve().parents[1] / "weights" / "sd-vae-ft-mse")
    cfg.amp = False
    cfg.mem_efficient = True
    cfg.hf_refiner = True
    cfg.hf_refiner_base = args.hf_base
    cfg.hf_refiner_blocks = args.hf_blocks
    net, ref = build_model(cfg, args.checkpoint, device)
    ref.out_scale = float(args.hf_out_scale)
    print(f"[refiner] base={args.hf_base} blocks={args.hf_blocks} "
          f"out_scale={args.hf_out_scale}")

    pairs = gather_pairs(args.data)
    if args.limit:
        pairs = pairs[:args.limit]
    N = len(pairs)
    vgg = VGGPerceptual((3, 8, 15)).to(device).eval()
    for p in vgg.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(ref.parameters(), lr=args.lr, weight_decay=1e-4)

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for ep in range(args.epochs):
        idx = rng.permutation(N)
        ep_l1 = ep_perc = ep_scc = 0.0
        nb = 0
        for s in range(0, N, args.batch):
            b = idx[s:s + args.batch]
            samples = [load_sample(*pairs[i], args.size, device) for i in b]
            gtb = torch.cat([x[0] for x in samples], 0)
            mb = torch.cat([x[1] for x in samples], 0)
            skb = torch.cat([x[2] for x in samples], 0)

            # I0 = VAE+FDP decode of GT (refiner toggled off during this step).
            I0 = vae_io(net, ref, gtb)

            refined = ref(I0.detach(), mb, skb)            # grad flows in refiner only
            m3 = mb.expand_as(refined)
            l1 = ((refined - gtb).abs() * m3).sum() / m3.sum().clamp_min(1.0)

            perc = (weighted_perceptual(vgg, refined, gtb, mb, device).mean()
                    if args.lambda_perc > 0 else torch.zeros((), device=device))

            scc = torch.zeros((), device=device)
            if args.lambda_scc > 0:
                pe = sobel_edges(refined.float())
                se = sobel_edges(skb.float())
                if pe.shape[-2:] != se.shape[-2:]:
                    pe = F.interpolate(pe, size=se.shape[-2:], mode="bilinear",
                                       align_corners=False)
                line = (se > 0.05).float()
                focus = line * mb.float()
                scc = ((pe - se).abs() * focus).sum() / focus.sum().clamp_min(1.0)

            loss = l1 + args.lambda_perc * perc + args.lambda_scc * scc
            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_l1 += l1.item(); ep_perc += perc.item(); ep_scc += scc.item(); nb += 1

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[ep {ep+1:03d}/{args.epochs}] "
                  f"l1={ep_l1/nb:.4f} perc={ep_perc/nb:.4f} scc={ep_scc/nb:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    torch.save(ref.state_dict(), args.out)
    print(f"[done] saved refiner -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
