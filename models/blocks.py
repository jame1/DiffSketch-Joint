"""Reusable convolutional, residual, time-embedding, and modulation blocks."""
import math
from typing import Tuple, List, Optional
import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embeddings.
    Input:  timesteps [B]
    Output: embeddings [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ConvBlock(nn.Module):
    """
    2D convolution block with GroupNorm and SiLU.
    Input:  [B, C_in, H, W]
    Output: [B, C_out, H/stride, W/stride]
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1, groups: int = 8):
        super().__init__()
        g = min(groups, c_out)
        while c_out % g != 0 and g > 1:
            g -= 1
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(g, c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(g, c_out),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    """
    Time-conditioned residual block.
    Input:  x [B, C_in, H, W], t_emb [B, time_dim]
    Output: [B, C_out, H, W]
    """

    def __init__(self, c_in: int, c_out: int, time_dim: int):
        super().__init__()
        g1 = min(8, c_in)
        while c_in % g1 != 0 and g1 > 1:
            g1 -= 1
        g2 = min(8, c_out)
        while c_out % g2 != 0 and g2 > 1:
            g2 -= 1

        self.norm1 = nn.GroupNorm(g1, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, c_out)
        self.norm2 = nn.GroupNorm(g2, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionModulator(nn.Module):
    """
    Element-wise residual gating modulator:
        F_out = F_in * (1 + beta * sigmoid(Proj(F_s)))

    Input:  f_in [B, C, H, W], f_s [B, C_s, H_s, W_s]
    Output: f_out [B, C, H, W]
    """

    def __init__(
        self,
        feature_channels: int,
        cond_channels: Optional[int] = None,
        beta_init: float = 0.5,
    ):
        super().__init__()
        c_s = cond_channels if cond_channels is not None else feature_channels
        self.beta = nn.Parameter(torch.tensor(float(beta_init), dtype=torch.float32))
        self.proj = nn.Conv2d(c_s, feature_channels, kernel_size=1)

    def forward(self, f_in: torch.Tensor, f_s: torch.Tensor) -> torch.Tensor:
        if f_s.shape[-2:] != f_in.shape[-2:]:
            f_s = F.interpolate(f_s, size=f_in.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.proj(f_s))
        return f_in * (1.0 + self.beta.to(f_in.dtype) * gate)


class TinyDenoiser(nn.Module):
    """
    Lightweight conditional U-Net denoiser (MLE for sketch branch).
    Input:  x [B, C_in, H, W], t [B], cond [B, C_cond, H_c, W_c]
    Output: pred_noise [B, C_out, H, W]
    """

    def __init__(
        self,
        in_ch: int,
        cond_ch: int,
        channels: Tuple[int, ...] = (16, 32, 64),
        time_dim: int = 128,
        out_ch: Optional[int] = None,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.in_conv = nn.Conv2d(in_ch + cond_ch, channels[0], kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.downs.append(ResBlock(channels[i], channels[i + 1], time_dim))

        self.mid = ResBlock(channels[-1], channels[-1], time_dim)

        self.ups = nn.ModuleList()
        for i in reversed(range(len(channels) - 1)):
            self.ups.append(ResBlock(channels[i + 1] + channels[i], channels[i], time_dim))

        self.out_conv = nn.Conv2d(channels[0], out_ch or in_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        te = self.time_mlp(timestep_embedding(t, self.time_dim))
        if cond.shape[-2:] != x.shape[-2:]:
            cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)

        h = self.in_conv(torch.cat([x, cond], dim=1))
        skips: List[torch.Tensor] = [h]

        for down in self.downs:
            h = down(F.avg_pool2d(h, kernel_size=2), te)
            skips.append(h)

        h = self.mid(h, te)
        skips.pop()

        for up in self.ups:
            skip = skips.pop()
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(torch.cat([h, skip], dim=1), te)

        return self.out_conv(h)


class ModulatedUNet(nn.Module):
    """
    LDM U-Net with per-layer ConditionModulator injection at every encoder,
    bottleneck, and decoder stage.

    Input:
        x:          [B, C_in, h, w]
        t:          [B]
        cond_feats: List of tensors [F_s^0, F_s^1, ...] matching `channels`
    Output:
        pred_noise: [B, C_out, h, w]
    """

    def __init__(
        self,
        in_ch: int = 9,
        out_ch: int = 4,
        channels: Tuple[int, ...] = (64, 128, 256),
        time_dim: int = 256,
        beta_init: float = 0.5,
    ):
        super().__init__()
        self.channels = channels
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.in_conv = nn.Conv2d(in_ch, channels[0], kernel_size=3, padding=1)
        self.in_mod = ConditionModulator(channels[0], channels[0], beta_init=beta_init)

        self.downs = nn.ModuleList()
        self.down_mods = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.downs.append(ResBlock(channels[i], channels[i + 1], time_dim))
            self.down_mods.append(
                ConditionModulator(channels[i + 1], channels[i + 1], beta_init=beta_init)
            )

        self.mid = ResBlock(channels[-1], channels[-1], time_dim)
        self.mid_mod = ConditionModulator(channels[-1], channels[-1], beta_init=beta_init)

        self.ups = nn.ModuleList()
        self.up_mods = nn.ModuleList()
        for i in reversed(range(len(channels) - 1)):
            self.ups.append(ResBlock(channels[i + 1] + channels[i], channels[i], time_dim))
            self.up_mods.append(
                ConditionModulator(channels[i], channels[i], beta_init=beta_init)
            )

        self.out_conv = nn.Conv2d(channels[0], out_ch, kernel_size=3, padding=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond_feats: List[torch.Tensor]
    ) -> torch.Tensor:
        te = self.time_mlp(timestep_embedding(t, self.time_dim))

        h = self.in_conv(x)
        h = self.in_mod(h, cond_feats[0])
        skips: List[torch.Tensor] = [h]

        for idx, (down, mod) in enumerate(zip(self.downs, self.down_mods)):
            h = down(F.avg_pool2d(h, kernel_size=2), te)
            h = mod(h, cond_feats[idx + 1])
            skips.append(h)

        h = self.mid(h, te)
        h = self.mid_mod(h, cond_feats[-1])
        skips.pop()

        for idx, (up, mod) in enumerate(zip(self.ups, self.up_mods)):
            scale_idx = len(self.channels) - 2 - idx
            skip = skips.pop()
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(torch.cat([h, skip], dim=1), te)
            h = mod(h, cond_feats[scale_idx])

        return self.out_conv(h)