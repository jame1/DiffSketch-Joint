"""
Bilateral contrastive loss for DiffSketch-Joint (Dunhuang mural inpainting).

Goal: in feature space, pull the restored image's representation close to the
true (GT) "positive" and push it away from the wrong-restoration "negative".
This encodes *semantic* correctness (hand gestures, drapery direction, ornament
motifs) that pixel-level losses (L_diff / L_perc) cannot express.

Design (bilateral):
  - L_pos = -log sim(z_pred, z_pos)   : anchor->positive attraction
  - L_neg = -log (1 - sim(z_pred, z_neg))  : anchor->negative repulsion
  combined as a symmetric InfoNCE over {pos, neg}:
      L = -log( exp(s(a,p)/tau) / ( exp(s(a,p)/tau) + exp(s(a,n)/tau) ) )

Feature space: VGG19 perceptual features (early/mid ReLU layers), pooled + an
optional trainable MLP projection head. The projector is the ONLY new trainable
component; the VGG backbone stays frozen (reuses the existing VGGPerceptual).

All gradients flow from the decoded image back through the VAE decode, i.e. this
is a *training-time* objective that shapes the diffusion/decoder prediction --
it is NOT a post-hoc refiner.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.joint_loss import VGGPerceptual


def _l2_norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)


class ContrastiveHead(nn.Module):
    """Small trainable MLP projection head on top of pooled VGG features."""

    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BilateralContrastiveLoss(nn.Module):
    def __init__(
        self,
        perc_layers: Tuple[int, ...] = (3, 8, 15),
        temperature: float = 0.07,
        head_hidden: int = 512,
        head_out: int = 128,
        use_head: bool = True,
    ):
        super().__init__()
        self.tau = temperature
        self.use_head = use_head
        self._vgg: Optional[VGGPerceptual] = None
        self._head: Optional[ContrastiveHead] = None
        self._head_cfg = (head_hidden, head_out)
        self._perc_layers = perc_layers
        self._vgg_dim = None  # resolved lazily

    def _get_vgg(self, device):
        if self._vgg is None:
            self._vgg = VGGPerceptual(self._perc_layers).to(device)
            self._vgg.eval()
        return self._vgg

    def _vgg_feature_dim(self, x: torch.Tensor) -> int:
        # resolve by a dummy forward if not cached
        if self._vgg_dim is None:
            vgg = self._get_vgg(x.device)
            with torch.no_grad():
                outs = vgg(x[:1])
            self._vgg_dim = sum(o.shape[1] for o in outs)
        return self._vgg_dim

    def _pool_feature(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pooled concatenation of VGG layer features -> [B, D]."""
        vgg = self._get_vgg(x.device)
        with torch.no_grad():
            outs = vgg(x)  # list of [B, C, h, w]
        pooled = []
        for o in outs:
            if mask is not None and mask.shape[-2:] == o.shape[-2:]:
                m = mask.float()
                denom = m.sum(dim=(2, 3)).clamp_min(1e-6)
                g = (o * m).sum(dim=(2, 3)) / denom   # in-hole weighted mean
            else:
                g = o.mean(dim=(2, 3))
            pooled.append(g)
        return torch.cat(pooled, dim=1)  # [B, D]

    def project(self, feat: torch.Tensor) -> torch.Tensor:
        if not self.use_head:
            return feat
        if self._head is None:
            h, o = self._head_cfg
            self._head = ContrastiveHead(feat.shape[1], h, o).to(feat.device)
        return self._head(feat)

    def forward(
        self,
        anchor: torch.Tensor,   # decoded restored image, [-1,1], [B,3,H,W]
        positive: torch.Tensor, # GT image, [-1,1], [B,3,H,W]
        negative: torch.Tensor, # wrong restoration, [-1,1], [B,3,H,W]
        mask: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,  # [B] SNR weights
    ) -> torch.Tensor:
        # Pooled VGG features
        fa = self._pool_feature(anchor, mask)
        fp = self._pool_feature(positive, mask)
        fn = self._pool_feature(negative, mask)
        # project
        za = _l2_norm(self.project(fa))
        zp = _l2_norm(self.project(fp.detach()))
        zn = _l2_norm(self.project(fn.detach()))

        sim_p = (za * zp).sum(dim=1) / self.tau   # [B]
        sim_n = (za * zn).sum(dim=1) / self.tau   # [B]
        # InfoNCE: -log( exp(s_p) / (exp(s_p) + exp(s_n)) )
        logit = torch.stack([sim_p, sim_n], dim=1)  # [B,2]
        loss = -torch.log_softmax(logit, dim=1)[:, 0]  # [B]

        if sample_weights is not None:
            w = sample_weights.view(-1).float().clamp_min(1e-6)
            return (loss * w).mean()
        return loss.mean()
