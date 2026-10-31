"""Frequency-domain decomposition for DiffSketch-Joint.

Design rationale (evidence-driven, see probe_freqband.py)
--------------------------------------------------------
The premise we set out to verify was: "the classical pyramid fill already
solves the low frequencies (tone / composition); the deficit is in the high
frequencies (lines / texture / edges), therefore split the restoration target
into an LF and an HF component and supervise each separately."

The probe measured this directly on 24 samples at 512px:

  * In the LATENT domain (where eps is predicted) the eps target under
    ``--fill-consistent`` is ``eps = noise + c*r`` with ``c = sqrt(ab/(1-ab))``.
    At t=200 the learnable signal ``c*r`` is only **15.4%** of the target
    variance -- confirming the dilution bottleneck.  However, band-splitting
    that target buys nothing: SNR_H / SNR_overall stays flat at 0.90-0.95x for
    every split tested (Gaussian sigma 0.5..8, FFT radius 0.125..0.75).
    Reason: DDPM noise is white, so its energy distributes over the bands in
    almost exactly the same shape as the fill residual does.  There is NO
    latent-space band where the restoration signal beats the noise by more
    than average -> "upweight HF because its SNR is better" is NOT supported.

  * What IS strongly band-dependent is the *image-domain* target.  The LF part
    (tone / composition) is large-amplitude; the HF part (lines / texture /
    edges) is roughly an order of magnitude smaller in amplitude while being
    decisive for SSIM_dmg.  A single whole-image L1/L2 is dominated by the LF
    error, so the HF deficit -- the visible one -- receives almost no gradient.
    This module therefore splits the *decoded restoration target* per band and
    normalises each band by its own statistics, so both bands contribute.

  * A second, independent argument for doing the HF work in IMAGE space: the SD
    VAE latent is 8x downsampled, so pixel-scale line detail is simply not
    representable there -- it must be produced after the decoder.  That is why
    ``L_hf`` (Laplacian, computed on the decoded image) was measured to be
    nearly zero inside the hole: the latent path cannot express HF at all.
    :class:`HighFreqRefiner` gives the model a parameterised route to generate
    exactly what the latent is structurally unable to carry.

Everything here is OFF by default and zero-initialised, so enabling it on top
of a converged checkpoint starts from the identity (resume-safe).
"""
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------- low-pass --

def gaussian_lowpass(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur (reflect padding, depthwise). sigma in pixels.

    Padding is applied only along the axis currently being filtered; padding
    both axes on both passes makes the result come back oversized.
    """
    if sigma <= 0:
        return x
    k = int(2 * round(3 * sigma) + 1)
    if k < 3:
        k = 3
    r = torch.arange(k, device=x.device, dtype=torch.float32) - k // 2
    g = torch.exp(-(r ** 2) / (2 * sigma * sigma))
    g = (g / g.sum()).to(x.dtype)  # [k]
    # IMPORTANT: build the kernels with .repeat (contiguous), NOT .expand.  A
    # *broadcast* (stride-0) weight fed to F.conv2d with groups>1 is unsupported
    # by cuDNN and triggers a backward hang on Windows/WDDM (regressed after a
    # CUDA/cuDNN update -- the Oct-27 run worked, a later one deadlocks).
    kern_h = g.view(1, 1, 1, -1).repeat(x.shape[1], 1, 1, 1).contiguous()  # [C,1,1,k]
    kern_v = g.view(1, 1, -1, 1).repeat(x.shape[1], 1, 1, 1).contiguous()  # [C,1,k,1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), kern_h,
                 padding=0, groups=x.shape[1])
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), kern_v,
                    padding=0, groups=x.shape[1])


def fft_lowpass(x: torch.Tensor, radius: float) -> torch.Tensor:
    """Ideal circular low-pass. `radius` is a fraction of Nyquist in (0, 1]."""
    B, C, H, W = x.shape
    fy = torch.fft.fftfreq(H, device=x.device).view(H, 1).expand(H, W)
    fx = torch.fft.fftfreq(W, device=x.device).view(1, W).expand(H, W)
    rr = torch.sqrt(fy ** 2 + fx ** 2) / 0.5          # 1.0 == Nyquist
    m = (rr <= radius).to(x.dtype).view(1, 1, H, W)
    X = torch.fft.fft2(x.float(), norm="ortho")
    return torch.fft.ifft2(X * m, norm="ortho").real.to(x.dtype)


# --------------------------------------------------------------- band split --

class BandSplit(nn.Module):
    """Lossless low/high decomposition:  ``low = LPF(x)``, ``high = x - low``.

    Guarantees ``low + high == x`` exactly (up to float round-off), so nothing
    is discarded -- the two bands are a complete description of the signal and
    supervising them jointly is equivalent to supervising x, only re-weighted.

    The cutoff is defined RELATIVE to the spatial size
    (``sigma_frac * H`` pixels) so that "low frequency = tone / composition"
    keeps the same meaning whether we train at 256px or evaluate at 512px --
    a fixed pixel cutoff would correspond to a different image scale at each
    resolution.
    """

    def __init__(self, mode: str = "gaussian", sigma_frac: float = 0.01,
                 radius: float = 0.25, base_size: int = 256):
        super().__init__()
        self.mode = mode
        self.sigma_frac = float(sigma_frac)
        self.radius = float(radius)
        self.base_size = int(base_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "fft":
            low = fft_lowpass(x, self.radius)
        else:
            sigma = max(self.sigma_frac * x.shape[-2], 1e-3)
            low = gaussian_lowpass(x.float(), sigma).to(x.dtype)
        return low, x - low

    def low(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]


def band_reference(gt_band: torch.Tensor, floor: float = 0.02) -> torch.Tensor:
    """Typical spatial contrast of a GT band, used to normalise its loss.

    ``mean |gt_b - local_mean(gt_b)|``  over the batch.  Detached: this is a
    scale estimate, not a target, so no gradient flows through it.  Using each
    band's OWN reference is the whole point -- it is what stops the
    large-amplitude LF band from swamping the small-amplitude HF band.

    The result is clamped to ``floor``.  A band whose contrast is below 0.02 on
    a [-1,1] image contains essentially no structure, and dividing by anything
    smaller would turn measurement noise into an enormous loss.  Without this
    floor a nearly-flat training patch produces an unbounded normalised error.
    """
    ref = (gt_band.detach() - gt_band.detach().mean(dim=(-1, -2), keepdim=True)).abs()
    return ref.mean().clamp_min(floor)


# ------------------------------------------------- learnable HF generation --

class HighFreqRefiner(nn.Module):
    """Image-space, content-conditioned HIGH-FREQUENCY generator.

    Takes the decoded clean estimate together with the damage mask and the
    structure (sketch) map and produces a residual that is ADDED to the
    decoded image::

        out = x + net(cat([x, mask, sketch]))

    Why image space and not latent space: the SD VAE latent is 8x downsampled,
    so pixel-scale line / texture detail is not representable there -- the
    decoder has to invent it, and a latent-space objective (including the
    Laplacian L_hf loss) can only reward it after the fact.  This module gives
    the model a direct, parameterised route to generate that band, conditioned
    on the surrounding evidence (the known pixels) and on the line layout (the
    sketch).

    Why adding to x rather than replacing x: the diffusion branch already
    produces a good LF solution (that is what the pyramid-fill consistency
    training optimised).  The refiner only has to supply what is MISSING.

    Zero-init (ControlNet zero-conv): the final conv starts at 0, so at
    initialisation the module is the exact identity and a converged checkpoint
    is untouched until it actually trains.
    """

    def __init__(self, in_ch: int = 5, base: int = 32, blocks: int = 4,
                 out_ch: int = 3, out_scale: float = 0.25):
        super().__init__()
        self.out_scale = float(out_scale)
        layers: list = [
            nn.Conv2d(in_ch, base, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        for _ in range(blocks):
            layers += [
                nn.Conv2d(base, base, kernel_size=3, padding=1),
                nn.GELU(),
            ]
        layers.append(nn.Conv2d(base, out_ch, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                sketch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None or sketch is None:
            # No conditioning available -> at init this is the identity anyway.
            return x
        H, W = x.shape[-2:]
        m = mask.float()
        if m.shape[-2:] != (H, W):
            m = F.interpolate(m, size=(H, W), mode="nearest")
        s = sketch.float()
        if s.shape[-2:] != (H, W):
            s = F.interpolate(s, size=(H, W), mode="bilinear", align_corners=False)
        x_in = torch.cat([x.float(), m, s], dim=1)
        # tanh + fixed scale: the residual stays within +-out_scale no matter how
        # far the weights drift, so a diverging refiner can damage image quality
        # but cannot blow the loss up. Identity at init either way (tanh(0)=0).
        return x + self.out_scale * torch.tanh(self.net(x_in)).to(x.dtype)
