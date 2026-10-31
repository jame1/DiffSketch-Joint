"""Scan classical anchor-fill variants on the test set (no model involved).

The pyramid fill is the PSNR anchor of the whole pipeline (pure fill = 27.34 dB
whole-image, and the final output shrinks toward it at blend 0.45), so improving
the fill directly raises the ceiling of the blended result.

Variants tested:
  * reps / tail length (how far the k=3 harmonic tail converges)
  * Gaussian-weighted normalized convolution vs. box (avg_pool) weights
  * anisotropic, structure-tensor-guided weights (diffuse ALONG edges instead
    of across them -- murals are dominated by thin dark outlines, so
    isotropic smoothing destroys exactly the structure that carries PSNR)
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torchvision.transforms import functional as TF

SIZE = 512
KS = (63, 47, 31, 23, 15, 11, 7, 5, 3)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def _box(w, x, m, k, reps):
    for _ in range(reps):
        wb = F.avg_pool2d(w, k, stride=1, padding=k // 2)
        xb = F.avg_pool2d(x * w, k, stride=1, padding=k // 2)
        est = xb / wb.clamp_min(1e-4)
        x = x * (1.0 - m) + est * m
        w = torch.ones_like(w)
    return x


def _sep(v, g1, r):
    """Separable conv that works for any channel count (groups=channels)."""
    c = v.shape[1]
    kh = g1.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    kv = g1.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    v = F.conv2d(v, kh, padding=(0, r), groups=c)
    return F.conv2d(v, kv, padding=(r, 0), groups=c)


def _gauss(w, x, m, k, reps):
    """Normalized convolution with a separable Gaussian kernel."""
    sig = max(k / 3.0, 1.0)
    r = k // 2
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    g1 = torch.exp(-(t ** 2) / (2 * sig * sig))
    g1 = g1 / g1.sum()
    for _ in range(reps):
        ww = _sep(w, g1, r)
        xx = _sep(x * w, g1, r)
        est = xx / ww.clamp_min(1e-5)
        x = x * (1.0 - m) + est * m
        w = torch.ones_like(w)
    return x


def _structure(img):
    """Smoothed structure tensor (J = [a b; b c]) of a luminance image."""
    g = (img[:, :1] * 0.299 + img[:, 1:2] * 0.587 + img[:, 2:3] * 0.114)
    gx = F.pad(g, (1, 0, 0, 0))[:, :, :, 1:] - F.pad(g, (0, 1, 0, 0))[:, :, :, :-1]
    gy = F.pad(g, (0, 0, 1, 0))[:, :, 1:, :] - F.pad(g, (0, 0, 0, 1))[:, :, :-1, :]
    k = 9
    sm = lambda v: F.avg_pool2d(F.pad(v, (k // 2,) * 4, mode="reflect"), k, 1)
    a, b, c = sm(gx * gx), sm(gx * gy), sm(gy * gy)
    return a, b, c


def _aniso(w, x, m, img, k, reps, eps=1e-3):
    """Edge-aware fill: rotate the smoothing kernel to follow local orientation.

    Inside the hole there is no image evidence, so the structure tensor of the
    *known* region is extrapolated by normalised convolution itself."""
    a, b, c = _structure(img * (1.0 - m))
    # extrapolate the tensor into the hole with a large box
    big = 63
    a = _box(w, a, m, big, 2)
    b = _box(w, b, m, big, 2)
    c = _box(w, c, m, big, 2)
    # coherence: 0 = isotropic, 1 = perfectly oriented
    tr = a + c
    coh = torch.sqrt(((a - c) ** 2 + 4 * b * b).clamp_min(1e-8)) / tr.clamp_min(eps)
    coh = coh.clamp(0, 1)
    # orientation angle of the dominant direction
    theta = 0.5 * torch.atan2(2 * b, (a - c).clamp_min(1e-6) * torch.sign(a - c + 1e-9))
    for _ in range(reps):
        est = _oriented_blur(x * w, w, k, theta, coh)
        x = x * (1.0 - m) + est * m
        w = torch.ones_like(w)
    return x


def _oriented_blur(xw, w, k, theta, coh):
    """Blur with a kernel elongated along the local edge direction."""
    c = xw.shape[1]
    r = k // 2
    t = torch.arange(-r, r + 1, device=xw.device, dtype=xw.dtype)
    u, v = torch.meshgrid(t, t, indexing="ij")
    ct, st = torch.cos(theta), torch.sin(theta)
    pu = (u * ct + v * st).unsqueeze(1)
    pv = (-u * st + v * ct).unsqueeze(1)
    su = (1.0 - coh) * r * 0.5 + coh * r * 1.0
    sv = (1.0 - coh) * r * 0.5 + coh * r * 0.12
    ker = torch.exp(-(pu ** 2) / (2 * su ** 2 + 1e-6) - (pv ** 2) / (2 * sv ** 2 + 1e-6))
    ker = ker / ker.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    pad = (r, r, r, r)
    xx = F.conv2d(F.pad(xw, pad, mode="reflect"), ker, padding=0, groups=c)
    ww = F.conv2d(F.pad(w, pad, mode="reflect"), ker, padding=0, groups=c)
    return xx / ww.clamp_min(1e-5)


def fill(img_t, mask_t, kind="box", reps=6, tail=16, ks=KS):
    m = mask_t
    w = (1.0 - m)
    x = img_t.clone()
    fn = {"box": _box, "gauss": _gauss}.get(kind, _box)
    for k in list(ks) + [3] * tail:
        r = reps if k in ks else 1
        if kind == "aniso":
            x = _aniso(w, x, m, img_t, k, r)
        else:
            x = fn(w, x, m, k, r)
        w = torch.ones_like(w)
    return img_t * (1.0 - m) + x * m


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(Path("F:/bihua/test/images").glob("*"))
    data = []
    for ip in files:
        mp = Path("F:/bihua/test/masks") / (ip.stem + ".png")
        if not mp.exists():
            continue
        img = TF.to_tensor(Image.open(ip).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS))
        m = torch.from_numpy(
            np.asarray(Image.open(mp).convert("L").resize((SIZE, SIZE), Image.NEAREST)) > 128
        ).float().unsqueeze(0)
        data.append(((img * 2 - 1).unsqueeze(0).to(device), m.unsqueeze(0).to(device),
                     img.numpy().transpose(1, 2, 0),
                     np.asarray(Image.open(mp).convert("L").resize((SIZE, SIZE), Image.NEAREST)) > 128))

    cfgs = [
        ("box r6 tail16 (current)", dict(kind="box", reps=6, tail=16)),
        ("box r6 tail32", dict(kind="box", reps=6, tail=32)),
        ("box r12 tail16", dict(kind="box", reps=12, tail=16)),
        ("box r3 tail16", dict(kind="box", reps=3, tail=16)),
        ("gauss r6 tail16", dict(kind="gauss", reps=6, tail=16)),
        ("gauss r12 tail16", dict(kind="gauss", reps=12, tail=16)),
        ("aniso r6 tail16", dict(kind="aniso", reps=6, tail=16)),
        ("aniso r6 tail8", dict(kind="aniso", reps=6, tail=8)),
    ]
    for name, kw in cfgs:
        pa, pd = [], []
        with torch.no_grad():
            for x, m, o, mb in data:
                r = fill(x, m, **kw)
                rec = ((r[0].float().clamp(-1, 1) + 1) / 2).cpu().numpy().transpose(1, 2, 0)
                pa.append(psnr(o, rec))
                pd.append(psnr(o[mb], rec[mb]))
        print(f"{name:26s} PSNR_all={np.mean(pa):6.2f}  PSNR_dmg={np.mean(pd):6.2f}", flush=True)


if __name__ == "__main__":
    main()
