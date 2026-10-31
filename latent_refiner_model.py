"""LatentRefiner -- a tiny residual network that maps a (slightly off) diffusion
latent z_pred -> a corrected latent closer to z_gt, in latent space.

Why latent space (not image space):
  The ~4.8 dB in-hole gap lives entirely in z_pred != z_gt (same VAE decoder
  gives 25.5 dB on z_gt but only 20.6 dB on z_pred). A latent correction is the
  only intervention that can shrink that gap; an image-space refiner/decoder
  fine-tune just re-attempts the same (already-overfit) idea after decode.

Training is latent-only: we supervise toward z_gt = encode(GT) and never backprop
through the VAE decoder, so it is immune to the cuDNN decoder-backward deadlock
that blocks VAE-decoder fine-tuning on this Windows/WDDM box.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, ch)
        self.gn2 = nn.GroupNorm(8, ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        h = self.act(self.gn1(self.c1(x)))
        h = self.gn2(self.c2(h))
        return x + self.act(h)


class LatentRefiner(nn.Module):
    """Residual correction:  z_refined = z + net(cat([z, mask_latent]))."""

    def __init__(self, in_ch: int = 4, cond_ch: int = 1, base: int = 64,
                 blocks: int = 4, out_ch: int = 4):
        super().__init__()
        self.stem = nn.Conv2d(in_ch + cond_ch, base, 3, padding=1, bias=False)
        self.body = nn.Sequential(*[_ResBlock(base) for _ in range(blocks)])
        self.head = nn.Conv2d(base, out_ch, 3, padding=1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        # zero-init the head so the refiner starts as an exact identity (no-op)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @torch.no_grad()
    def _mask(self, m, z):
        if m.dim() == 3:
            m = m.unsqueeze(0)
        if m.shape[-2:] != z.shape[-2:]:
            m = F.interpolate(m, size=z.shape[-2:], mode="nearest")
        return m.to(z.dtype)

    def forward(self, z, m):
        if z.dim() == 3:
            z = z.unsqueeze(0)
        m = self._mask(m, z)
        x = torch.cat([z.float(), m.float()], dim=1)
        h = self.act(self.stem(x))
        h = self.body(h)
        return z + self.head(h)  # residual; identity at init
