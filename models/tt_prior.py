"""Low-rank Tensor-Train (TT) structure guidance for the reverse diffusion process.

Motivation
----------
Convolutions and (cross-)attention are *local / pairwise*: with a large hole the
denoiser has no mechanism to enforce that the left half of a mural agrees with
the right half. A Tensor-Train decomposition is one way to obtain a *global*
coupling: a rank-R TT of an image/latent tensor spans all spatial modes at once,
so every core "sees" the whole field.

Two guidance flavours are implemented:

  dynamic  (default, safest)
      x0_hat  ->  TT_rank_R(x0_hat)  (self low-rank projection)
      x0_g    =  x0_hat + w(t) * m * (TT_R(x0_hat) - x0_hat)
      i.e. we remove a fraction w(t) of everything that is NOT global/low-rank
      inside the hole. No external content is imposed, only the high-frequency
      part is damped -- and only while w(t) > 0.

  static
      Fit the TT completion of the *damaged latent* on its observed (healthy)
      pixels ONCE, giving a global structure prior P; then
      x0_g = x0_hat + w(t) * m * (P - x0_hat).
      This one DOES impose content (a smooth global layout). Use a smaller
      weight and only in the mid-timestep band.

Both are applied in x0 space and then propagated back to z with the exact
"signal-component" update (keep the predicted noise, move only sqrt(alpha_bar)
part):

      z_prev_guided = z_prev + sqrt(alpha_bar_{t-1}) * (x0_g - x0_hat)

so the intervention is a clean, invertible, noise-preserving shift and, with
w(t) == 0 or mask == 0, is mathematically a no-op.

Everything is OFF by default (TTConfig.enabled = False), so existing behaviour
and checkpoints are bit-identical.

Cost: TT-SVD is a chain of small batched SVDs (~2-4 ms per step at 64x64x4 on a
4090). The static fit is done once per sample.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


# --------------------------------------------------------------------------- #
# Tensor-Train primitives (batched, dependency-free)
# --------------------------------------------------------------------------- #
def factorize(n: int, base: int = 4) -> list:
    """256 -> [4,4,4,4]; 128 -> [4,4,4,2]; 64 -> [4,4,4]; 32 -> [4,4,2]."""
    out = []
    while n > 1 and n % base == 0:
        out.append(base)
        n //= base
    while n > 1:
        if n % 2 == 0:
            out.append(2)
            n //= 2
        else:
            out.append(base)
            n = max(1, n // base)
    return out or [1]


def latent_dims(shape: Tuple[int, ...]) -> list:
    """[B, C, H, W] -> TT dims [C, *factorize(H), *factorize(W)]."""
    return [shape[1]] + factorize(shape[2]) + factorize(shape[3])


def tt_svd(x: torch.Tensor, dims: list, rank: int, eps: float = 0.0) -> List[torch.Tensor]:
    """
    Batched sequential-SVD TT decomposition (Oseledets).
    x:     [B, prod(dims)] (flattened, C-order)
    cores: list of [B, r_{k-1}, d_k, r_k], r_0 = r_K = 1.
    """
    B = x.shape[0]
    cores: List[torch.Tensor] = []
    r_prev = 1
    M = x.reshape(B, dims[0], -1)
    for i in range(len(dims) - 1):
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        r = min(rank, U.shape[1], U.shape[2])
        if eps > 0:
            energy = (S ** 2).sum(dim=1, keepdim=True)
            tail = torch.cumsum((S ** 2).flip(1), 1).flip(1)
            keep = int(((tail / energy.clamp_min(1e-12)) > eps).sum(dim=1).max().item())
            r = min(r, max(1, keep))
        U, S, Vh = U[:, :, :r], S[:, :r], Vh[:, :r, :]
        cores.append(U.reshape(B, r_prev, dims[i], r))
        M = (S.unsqueeze(-1) * Vh).reshape(B, r * dims[i + 1], -1)
        r_prev = r
    cores.append(M.reshape(B, r_prev, dims[-1], 1))
    return cores


def tt_dense(cores: List[torch.Tensor], shape: Tuple[int, ...]) -> torch.Tensor:
    """Contract batched TT cores back into a dense tensor of `shape`."""
    M = cores[0][:, 0]                       # [B, d1, r1]
    for c in cores[1:]:
        M = torch.einsum("bnr,brjs->bnjs", M, c).reshape(M.shape[0], -1, c.shape[3])
    return M.reshape(shape)


def tt_rank_r(x: torch.Tensor, dims: list, rank: int) -> torch.Tensor:
    """Best rank-R TT approximation of x (shape preserved)."""
    return tt_dense(tt_svd(x.reshape(x.shape[0], -1), dims, rank), tuple(x.shape))


def tt_completion_fit(
    x_obs: torch.Tensor,
    obs_weight: torch.Tensor,
    dims: list,
    rank: int,
    iters: int = 200,
    lr: float = 0.05,
) -> torch.Tensor:
    """
    Fit TT cores to the OBSERVED entries only (analogous to matrix completion).
    x_obs:       [B, C, H, W] field with the hole filled with anything
    obs_weight:  [B, 1, H, W] broadcastable weight (1 = observed, 0 = hole)
    returns the completed field (same shape).
    """
    shape = tuple(x_obs.shape)
    flat = x_obs.reshape(x_obs.shape[0], -1)
    w = obs_weight.expand_as(x_obs).reshape(x_obs.shape[0], -1)
    denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)
    cores = [c.clone().requires_grad_(True) for c in tt_svd(flat, dims, rank)]
    opt = torch.optim.Adam(cores, lr=lr)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        rec = tt_dense(cores, shape).reshape(flat.shape)
        (((rec - flat) ** 2) * w).sum(dim=1, keepdim=True).div(denom).mean().backward()
        opt.step()
    with torch.no_grad():
        return tt_dense(cores, shape)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class TTConfig:
    """All TT-guidance knobs. Defaults = disabled (no-op)."""
    enabled: bool = False
    mode: str = "dynamic"        # "dynamic" | "static"
    rank: int = 4                # TT bond rank R (probe: 4-8 is the sweet spot)
    lam: float = 0.5             # peak guidance strength
    schedule: str = "early"      # "early" | "mid" | "constant" | "late"
    t_lo: float = 0.15           # guidance is 0 below this fraction of T
    t_hi: float = 1.00           # guidance reaches `lam` at this fraction of T
    p: float = 1.0               # exponent of the ramp
    fit_iters: int = 200         # static mode: Adam steps for the completion fit
    fit_lr: float = 0.05
    fit_space: str = "latent"    # "latent" | "image"
    # How the x0-space correction is transferred back to z.
    #   "signal" (default, mathematically consistent):
    #       z_prev += sqrt(alpha_bar_{t-1}) * delta
    #       -> the noise realisation is untouched and only the signal part moves.
    #       BUT sqrt(alpha_bar) is ~0.007 at t=T, so early-step guidance is
    #       automatically (and almost completely) attenuated: this mode is
    #       intrinsically late-weighted.
    #   "boost":
    #       z_prev += delta / max(sqrt(alpha_bar_{t-1}), boost_floor)
    #       -> un-attenuated correction, so an EARLY schedule actually acts.
    #       Needs the delta clamp (noisy x0 estimates at high t).
    scale_mode: str = "signal"
    boost_floor: float = 0.2
    delta_clamp: float = 3.0


# --------------------------------------------------------------------------- #
# The guidance operator
# --------------------------------------------------------------------------- #
class TTGuidance:
    """
    Usage inside the sampler loop (see InpaintDiffusionBranch.generate):

        tt = TTGuidance(cfg.tt, timesteps=T)
        tt.prepare(z_masked, mask_latent)          # static mode only
        ...
        x0_hat = (z_in - sqrt(1-ab_t) * eps) / sqrt(ab_t)
        delta  = tt.delta(x0_hat, t_frac, mask_latent)
        z      = z_prev + sqrt(ab_prev) * delta
    """

    def __init__(self, cfg: TTConfig, timesteps: int = 1000):
        self.cfg = cfg
        self.T = float(timesteps)
        self._dims_cache = {}
        self.prior: Optional[torch.Tensor] = None

    # ---------------- schedule ----------------
    def weight(self, t_frac: float) -> float:
        """Guidance strength for a normalised timestep (1 = start, 0 = end)."""
        c = self.cfg
        if not c.enabled or c.lam <= 0:
            return 0.0
        x = min(max(float(t_frac), 0.0), 1.0)
        if c.schedule == "constant":
            w = 1.0
        elif c.schedule == "early":        # structure first, texture later
            w = (x - c.t_lo) / max(c.t_hi - c.t_lo, 1e-6)
            w = min(max(w, 0.0), 1.0) ** c.p
        elif c.schedule == "late":         # ablation: structure last
            w = (c.t_hi - x) / max(c.t_hi - c.t_lo, 1e-6)
            w = min(max(w, 0.0), 1.0) ** c.p
        elif c.schedule == "mid":          # band-pass around mid timesteps
            sigma = max((c.t_hi - c.t_lo) * 0.5, 1e-3)
            mu = 0.5 * (c.t_hi + c.t_lo)
            w = float(torch.exp(torch.tensor(-0.5 * ((x - mu) / sigma) ** 2)).item())
        else:
            raise ValueError(f"unknown schedule {c.schedule}")
        return float(c.lam * w)

    # ---------------- dims ----------------
    def _dims(self, shape: Tuple[int, ...]) -> list:
        key = tuple(shape[1:])
        if key not in self._dims_cache:
            self._dims_cache[key] = latent_dims(shape)
        return self._dims_cache[key]

    # ---------------- static prior ----------------
    @torch.no_grad()
    def prepare(self, z_obs: torch.Tensor, mask_latent: torch.Tensor,
                vae_encode=None, image: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> None:
        """
        Build the static global-structure prior.
        fit_space="latent": complete the damaged latent directly (cheap).
        fit_space="image":  complete the damaged IMAGE then encode it (the prior
                            is then expressed in the same space the loss lives
                            in, but costs one extra VAE encode).
        """
        if not self.cfg.enabled or self.cfg.mode != "static":
            self.prior = None
            return
        dev, dt = z_obs.device, torch.float32
        # tt_completion_fit optimises the cores with Adam, so gradients must be
        # enabled locally even though this method is called under no_grad().
        with torch.enable_grad():
            if self.cfg.fit_space == "image" and image is not None and vae_encode is not None:
                dims = self._dims(image.shape)
                comp = tt_completion_fit(
                    image.detach().to(dt), (1.0 - mask).detach().to(dt),
                    dims, self.cfg.rank, self.cfg.fit_iters, self.cfg.fit_lr)
                self.prior = vae_encode(comp.to(image.dtype)).to(dev).float()
            else:
                dims = self._dims(z_obs.shape)
                self.prior = tt_completion_fit(
                    z_obs.detach().to(dt), (1.0 - mask_latent).detach().to(dt),
                    dims, self.cfg.rank, self.cfg.fit_iters, self.cfg.fit_lr
                ).to(z_obs.dtype)

    # ---------------- the intervention ----------------
    @torch.no_grad()
    def delta(self, x0_hat: torch.Tensor, t_frac: float,
              mask_latent: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Return  w(t) * m * (target - x0_hat)  or None when inactive.
        Caller applies:  z_prev += sqrt(alpha_bar_{t-1}) * delta
        """
        w = self.weight(t_frac)
        if w <= 0.0:
            return None
        cfg = self.cfg
        with torch.amp.autocast(device_type=x0_hat.device.type, enabled=False):
            x = x0_hat.float()
            if cfg.mode == "static":
                if self.prior is None:
                    return None
                target = self.prior.to(x.device, dtype=torch.float32)
                if target.shape[-2:] != x.shape[-2:]:
                    target = torch.nn.functional.interpolate(
                        target, size=x.shape[-2:], mode="bilinear", align_corners=False)
            else:  # dynamic
                target = tt_rank_r(x, self._dims(x.shape), cfg.rank)
            m = mask_latent.float()
            if m.shape[-2:] != x.shape[-2:]:
                m = torch.nn.functional.interpolate(m, size=x.shape[-2:], mode="nearest")
            d = w * m * (target - x)
            if not torch.isfinite(d).all():
                return None
            return d.to(x0_hat.dtype)

    # ---------------- optional training-time loss ----------------
    def structure_loss(self, x0_hat: torch.Tensor, mask_latent: torch.Tensor,
                       t_frac: float) -> torch.Tensor:
        """
        Training-time version of the same constraint (NOT wired by default):
        L_tt = w(t) * || m * (x0_hat - TT_R(x0_hat)) ||^2 / ||m||_1
        Gradient flows through the contraction but NOT through the SVD (the
        projection target is detached) -- this keeps the SVD backward (which is
        numerically fragile near degenerate singular values) out of the graph.
        """
        w = self.weight(t_frac)
        if w <= 0.0:
            return x0_hat.new_zeros(())
        with torch.amp.autocast(device_type=x0_hat.device.type, enabled=False):
            x = x0_hat.float()
            target = tt_rank_r(x.detach(), self._dims(x.shape), self.cfg.rank)
            m = mask_latent.float()
            if m.shape[-2:] != x.shape[-2:]:
                m = torch.nn.functional.interpolate(m, size=x.shape[-2:], mode="nearest")
            diff = (x - target) * m
            return w * (diff ** 2).sum() / m.sum().clamp_min(1.0)
