"""Branch 1: Sketch conditional diffusion branch (CDiffu-DL style)."""
import warnings
from typing import List, Tuple, Optional, Union
import torch
from torch import nn
import torch.nn.functional as F
from .blocks import ConvBlock, TinyDenoiser


class GlobalAttentionBlock(nn.Module):
    """Global self-attention residual block over a low-resolution token grid.

    The CFE cond map can be large (e.g. 128x128 = 16k tokens); an all-pairs MHSA
    there is intractable (~4GB of attention per sample). We therefore compute the
    global context on a fixed low-res grid (default 32x32 -> 1024 tokens, ~16MB of
    attention), then bilinearly upsample the residual back to full resolution and
    add it. The output projection is zero-initialized so the block is the identity
    at initialization -- a pretrained CFE (stage1.pt) is therefore untouched until
    the new weights are actually trained.

    This is the "correct form" for giving the sketch branch global / symmetric
    composition context without blowing up memory or disturbing the already
    converged CFE features.
    """

    def __init__(self, dim: int = 64, heads: int = 4, tok: int = 32):
        super().__init__()
        self.tok = tok
        # Pre-norm on the conv feature map (GroupNorm is the natural choice for
        # [B, C, H, W] features and keeps the attention input well-conditioned).
        self.norm = nn.GroupNorm(8, dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        # Zero-init the output projection -> identity at init (protects pretrained weights).
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        low = F.adaptive_avg_pool2d(x, (self.tok, self.tok))   # [B, C, tok, tok]
        h = self.norm(low).flatten(2).transpose(1, 2)          # [B, tok*tok, C]
        a, _ = self.attn(h, h, h)                              # [B, tok*tok, C]
        a = a.transpose(1, 2).reshape(B, C, self.tok, self.tok)
        a = F.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
        return x + a                                           # residual add


class CFE(nn.Module):
    """
    Condition Feature Extractor (CFE).
    Fuses multi-scale features from SharedEncoder into a unified condition map.

    Input:  feats: List of tensors at [1/2, 1/4, 1/8] resolutions
    Output: cond_map [B, out_channels, H/2, W/2]
    """

    def __init__(
        self,
        feat_channels: Tuple[int, ...] = (32, 64, 128),
        out_channels: int = 64,
        global_attn: bool = True,
    ):
        super().__init__()
        self.projs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=1) for c in feat_channels]
        )
        self.fuse = ConvBlock(out_channels * len(feat_channels), out_channels, stride=1)
        # Global self-attention over a low-res token grid, applied right after the
        # fuse so the sketch branch can reference long-range / symmetric composition
        # and surrounding figure semantics when predicting large holes.
        self.attn = GlobalAttentionBlock(out_channels, heads=4, tok=32) if global_attn else None

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        target_size = feats[0].shape[-2:]
        aligned = [
            F.interpolate(proj(f), size=target_size, mode="bilinear", align_corners=False)
            for proj, f in zip(self.projs, feats)
        ]
        x = self.fuse(torch.cat(aligned, dim=1))
        if self.attn is not None:
            x = self.attn(x)
        return x


class SketchDiffusionBranch(nn.Module):
    """
    Conditional DDPM branch for mural line-drawing (sketch) prediction.

    Interfaces:
        - load_pretrained_cfe(path): loads CFE weights or warns on empty path.
        - forward(shared_feats, sketch_gt=None, mode='train'|'infer'):
            * mode == 'train': returns (pred_noise [B,1,h,w], S [B,1,H,W])
            * mode == 'infer': returns S [B,1,H,W]
          where S is strictly differentiable via smooth tanh + temperature-scaled sigmoid in [0, 1].
    """

    def __init__(
        self,
        feat_channels: Tuple[int, ...] = (32, 64, 128),
        cond_channels: int = 64,
        unet_channels: Tuple[int, ...] = (16, 32, 64),
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        sigmoid_temp: float = 5.0,
    ):
        super().__init__()
        self.cfe = CFE(feat_channels, cond_channels)
        self.mle = TinyDenoiser(
            in_ch=1,
            cond_ch=cond_channels,
            channels=unet_channels,
            out_ch=1,
        )
        self.timesteps = timesteps
        self.sigmoid_temp = sigmoid_temp

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.last_target_noise: Optional[torch.Tensor] = None

    def load_pretrained_cfe(self, path: str) -> bool:
        """
        Load pretrained CFE weights.

        Accepts three source formats so the slot is robust:
          * {"cfe": <CFE state_dict>}  (our exported standalone file)
          * a bare CFE state_dict      (keys like "projs.0.weight")
          * a full sketch-branch checkpoint (keys prefixed "cfe.")
        Returns True if weights were loaded, False if skipped (empty path).
        """
        if not path:
            warnings.warn(
                "No pretrained CFE path provided; CFE remains randomly initialized.",
                UserWarning,
            )
            return False

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and isinstance(ckpt.get("cfe"), dict):
            state_dict = ckpt["cfe"]
        elif isinstance(ckpt, dict) and isinstance(ckpt.get("state_dict"), dict):
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # Strip a leading "cfe." prefix if a full sketch/branch checkpoint was passed.
        if state_dict and all(k.startswith("cfe.") for k in state_dict.keys()):
            state_dict = {k[len("cfe.") :]: v for k, v in state_dict.items()}

        if not state_dict:
            warnings.warn(
                "Pretrained CFE file contained no CFE weights; CFE stays random.",
                UserWarning,
            )
            return False

        missing, unexpected = self.cfe.load_state_dict(state_dict, strict=False)
        n_loaded = len(state_dict) - len(missing)
        print(
            f"[CFE] Loaded pretrained weights from '{path}' "
            f"({n_loaded}/{len(state_dict)} tensors matched; "
            f"missing={len(missing)}, unexpected={len(unexpected)})."
        )
        return True

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward DDPM diffusion: q(x_t | x_0)."""
        ab = self.alpha_bars[t].view(-1, 1, 1, 1).to(x0.dtype)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise

    def forward(
        self,
        shared_feats: List[torch.Tensor],
        sketch_gt: Optional[torch.Tensor] = None,
        mode: str = "train",
        infer_steps: int = 50,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        cond = self.cfe(shared_feats)
        feat_h, feat_w = shared_feats[0].shape[-2:]
        full_size = (feat_h * 2, feat_w * 2)

        if mode == "train":
            if sketch_gt is None:
                raise ValueError("sketch_gt must be provided when mode='train'.")
            full_size = sketch_gt.shape[-2:]
            x0 = (
                F.interpolate(
                    sketch_gt, size=(feat_h, feat_w), mode="bilinear", align_corners=False
                )
                * 2.0
                - 1.0
            )
            t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device).long()
            noise = torch.randn_like(x0)
            self.last_target_noise = noise

            xt = self.q_sample(x0, t, noise)
            pred_noise = self.mle(xt, t, cond)

            # Differentiable one-step clean estimate with smooth tanh bounding in [-3, 3]
            ab = self.alpha_bars[t].view(-1, 1, 1, 1).to(x0.dtype)
            x0_raw = (xt - torch.sqrt(1.0 - ab) * pred_noise) / torch.sqrt(ab).clamp_min(1e-3)
            x0_pred = 3.0 * torch.tanh(x0_raw / 3.0)

            # Temperature-scaled sigmoid in [0, 1] (everywhere differentiable)
            sketch_prob = torch.sigmoid(x0_pred * self.sigmoid_temp)
            sketch_prob = F.interpolate(
                sketch_prob, size=full_size, mode="bilinear", align_corners=False
            )
            return pred_noise, sketch_prob

        if mode == "infer":
            bsz = cond.shape[0]
            x = torch.randn((bsz, 1, feat_h, feat_w), device=cond.device, dtype=cond.dtype)
            steps = torch.linspace(
                self.timesteps - 1, 0, min(infer_steps, self.timesteps), device=cond.device
            ).long()

            for idx, ti in enumerate(steps):
                t = torch.full((bsz,), int(ti.item()), device=cond.device, dtype=torch.long)
                eps = self.mle(x, t, cond)
                ab_t = self.alpha_bars[t[0]].to(x.dtype)
                x0_hat = (x - torch.sqrt(1.0 - ab_t) * eps) / torch.sqrt(ab_t).clamp_min(1e-3)
                x0_hat = 3.0 * torch.tanh(x0_hat / 3.0)

                if idx < len(steps) - 1:
                    t_prev = steps[idx + 1]
                    ab_prev = self.alpha_bars[int(t_prev.item())].to(x.dtype)
                    x = torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1.0 - ab_prev) * eps
                else:
                    x = x0_hat

            sketch_prob = torch.sigmoid(x * self.sigmoid_temp)
            return F.interpolate(
                sketch_prob, size=full_size, mode="bilinear", align_corners=False
            )

        raise ValueError(f"Unsupported mode '{mode}'. Expected 'train' or 'infer'.")