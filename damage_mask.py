"""
敦煌壁画病害掩码生成 (Dunhuang mural damage mask generation)
============================================================

在无人工标注的前提下，用多线索传统视觉方法估计病害区域，输出：

  1. mask/        与原图同尺寸的二值掩码 (255=病害)  —— 直接用于 inpainting
  2. labels/      多类别标签图 (0=完好,1=裂隙,2=剥落/露地仗,3=发黑/烟熏,4=白斑/盐析)
  3. viz/         原图 | 病害热图 | 掩码 | 叠加图 四联可视化

病害语义分两类，对应不同修复任务：
  - 信息缺失型 (剥落、露地仗、白斑/盐析) -> 图像补全/inpainting
  - 信息失真型 (发黑、烟熏、变色)         -> 色彩复原/风格迁移
  - 结构型 (裂隙)                          -> 结构修复，保守检测

作者说明：传统方法精度有限，阈值可在 Params 中调整。
"""
import os
import cv2
import numpy as np
from dataclasses import dataclass, asdict

DET_MAX = 1400  # 检测时最长边，加速；掩码最后会放大回原图尺寸


# ----------------------------- IO (unicode-safe) -----------------------------
def imread_u(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


# ------------------------------- parameters ---------------------------------
@dataclass
class Params:
    # 白边/书页裁剪
    paper_v: int = 175          # 纸张亮度阈值
    paper_s: int = 45           # 纸张饱和度阈值
    max_trim: float = 0.35      # 单边最多裁掉比例

    # 裂隙
    crack_len: int = 13         # 线状核长度(检测尺度)
    crack_thick: int = 3
    crack_min_resp: float = 16
    crack_pct: float = 99.2     # 取响应最高的百分位
    crack_min_area: int = 24

    # 剥落 / 露地仗 / 白化
    loss_s_max: int = 58        # 低饱和上限
    loss_v_min: int = 155       # 高亮下限
    loss_tex_max: float = 13.0  # 局部纹理(标准差)上限
    loss_ed_max: float = 0.05   # 局部边缘密度上限(区分空白地仗 vs 有线描的白衣)
    loss_min_area: int = 90

    # 发黑 / 烟熏
    dark_v_max: int = 60
    dark_s_max: int = 70        # 低饱和: 中性发黑 vs 深蓝/深红彩绘
    dark_min_area: int = 400

    # 白斑 / 盐析 (小而亮的点)
    spot_kernel: int = 9
    spot_v_min: int = 195
    spot_pct: float = 99.3
    spot_max_area: int = 900


# ------------------------------ helpers -------------------------------------
def line_kernel(length, thick, angle_deg):
    k = np.zeros((length, length), np.uint8)
    c = length // 2
    dx = int(round(np.cos(np.deg2rad(angle_deg)) * (length // 2)))
    dy = int(round(np.sin(np.deg2rad(angle_deg)) * (length // 2)))
    cv2.line(k, (c - dx, c - dy), (c + dx, c + dy), 1, thick)
    return k


def paper_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    return ((V > 175) & (S < 45)).astype(np.uint8)


def content_bbox(bgr, p: Params):
    """裁掉扫描件的白纸边框/图注区。若判定无白边则返回整幅。"""
    H, W = bgr.shape[:2]
    P = paper_mask(bgr)
    rf = P.mean(axis=1)
    cf = P.mean(axis=0)
    t = 0
    while t < int(p.max_trim * H) and rf[t] > 0.5:
        t += 1
    b = H - 1
    while b > int((1 - p.max_trim) * H) and rf[b] > 0.5:
        b -= 1
    l = 0
    while l < int(p.max_trim * W) and cf[l] > 0.5:
        l += 1
    r = W - 1
    while r > int((1 - p.max_trim) * W) and cf[r] > 0.5:
        r -= 1
    # 仅当最外侧确实是纸才承认裁剪
    if rf[:5].mean() < 0.35:
        t = 0
    if rf[-5:].mean() < 0.35:
        b = H - 1
    if cf[:5].mean() < 0.35:
        l = 0
    if cf[-5:].mean() < 0.35:
        r = W - 1
    if (b - t) < 0.4 * H:
        t, b = 0, H - 1
    if (r - l) < 0.4 * W:
        l, r = 0, W - 1
    return int(l), int(t), int(r + 1), int(b + 1)


def crop_resize(bgr, box, det_max=DET_MAX):
    l, t, r, b = box
    roi = bgr[t:b, l:r]
    H, W = roi.shape[:2]
    s = min(1.0, det_max / max(H, W))
    if s < 1:
        roi = cv2.resize(roi, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
    return roi, s


def drop_thin(mask, max_min_dim=6):
    """去掉近似轴对齐的细线(书页边框线/扫描线)，保留斜向细裂隙。"""
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask, np.uint8)
    for i in range(1, n):
        w = st[i, cv2.CC_STAT_WIDTH]
        h = st[i, cv2.CC_STAT_HEIGHT]
        if min(w, h) > max_min_dim:
            out[lab == i] = 1
    return out


def _clean(mask, min_area, close_k=3):
    mask = (mask > 0).astype(np.uint8)
    if close_k:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return out


# ------------------------------ detectors -----------------------------------
def detect_all(roi, p: Params):
    """在检测尺度上返回各线索的二值图与响应热图。"""
    H, W = roi.shape[:2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[..., 0], hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_u = gray.astype(np.uint8)

    # --- 局部纹理强度 ---
    m = cv2.blur(gray, (7, 7))
    sq = cv2.blur(gray * gray, (7, 7))
    std = np.sqrt(np.maximum(sq - m * m, 0))
    tex = cv2.blur(std, (15, 15))
    # 局部边缘密度: 空白地仗低, 有线描/衣纹的白衣高
    edges = cv2.Canny(gray_u, 40, 120)
    ed = cv2.blur(edges.astype(np.float32) / 255.0, (21, 21))

    # --- 1) 裂隙: 多方向 line black-hat ---
    resp = np.zeros_like(gray, np.float32)
    for ang in range(0, 180, 30):
        k = line_kernel(p.crack_len, p.crack_thick, ang)
        bh = cv2.morphologyEx(gray_u, cv2.MORPH_BLACKHAT, k).astype(np.float32)
        resp = np.maximum(resp, bh)
    thr = max(p.crack_min_resp, np.percentile(resp, p.crack_pct))
    crack = (resp > thr).astype(np.uint8)
    crack = _clean(crack, p.crack_min_area, close_k=3)
    # 去掉近似方块的团块(裂隙应细长)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(crack, 8)
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        fill = stats[i, cv2.CC_STAT_AREA] / max(1, w * h)
        if max(w, h) < 10 or (fill > 0.6 and min(w, h) > 6):
            crack[lab == i] = 0
    crack = drop_thin(crack, 6)

    # --- 2) 剥落/露地仗/白化: 低饱和 + 高亮 + 低纹理 ---
    loss = ((S < p.loss_s_max) & (V > p.loss_v_min) &
            (tex < p.loss_tex_max) & (ed < p.loss_ed_max)).astype(np.uint8)
    loss = cv2.morphologyEx(loss, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    loss = _clean(loss, p.loss_min_area, close_k=5)

    # --- 3) 发黑/烟熏: 大面积低亮度 ---
    dark = ((V < p.dark_v_max) & (S < p.dark_s_max)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    dark = _clean(dark, p.dark_min_area, close_k=7)

    # --- 4) 白斑/盐析: 小尺度亮点 top-hat ---
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.spot_kernel, p.spot_kernel))
    th = cv2.morphologyEx(gray_u, cv2.MORPH_TOPHAT, k_small).astype(np.float32)
    vthr = max(28, np.percentile(th, p.spot_pct))
    spot = ((th > vthr) & (V > p.spot_v_min) & (S < 80)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(spot, 8)
    spot2 = np.zeros_like(spot)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        roundness = a / max(1, w * h)
        if 6 <= a <= p.spot_max_area and roundness >= 0.42:
            spot2[lab == i] = 1
    spot = spot2

    # 裂隙优先，避免与剥落重叠时类别混乱
    loss = loss & (1 - crack)
    spot = spot & (1 - crack) & (1 - loss)

    labels = np.zeros((H, W), np.uint8)
    labels[crack > 0] = 1
    labels[loss > 0] = 2
    labels[dark > 0] = 3
    labels[spot > 0] = 4

    # 边框保护带: 抑制扫描/截图边缘 1~2px 的伪病害
    g = max(3, int(0.004 * min(H, W)))
    labels[:g, :] = 0
    labels[-g:, :] = 0
    labels[:, :g] = 0
    labels[:, -g:] = 0
    crack = (labels == 1).astype(np.uint8)
    loss = (labels == 2).astype(np.uint8)
    dark = (labels == 3).astype(np.uint8)
    spot = (labels == 4).astype(np.uint8)

    # 归一化热图 (0-1)
    heat = np.zeros((H, W), np.float32)
    heat += (labels > 0).astype(np.float32)
    heat += 0.6 * (labels == 2)
    heat += 0.4 * (labels == 1)
    heat = np.clip(heat, 0, 1)

    return dict(labels=labels, heat=heat, crack=crack, loss=loss, dark=dark, spot=spot,
                tex=tex, S=S, V=V)


# ------------------------------ full pipeline -------------------------------
def analyze(bgr, p: Params = None):
    p = p or Params()
    H, W = bgr.shape[:2]
    box = content_bbox(bgr, p)
    roi, s = crop_resize(bgr, box)
    det = detect_all(roi, p)

    l, t, r, b = box
    cw, ch = r - l, b - t

    def to_full(m):
        m8 = cv2.resize(m.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST)
        full = np.zeros((H, W), np.uint8)
        full[t:b, l:r] = m8
        return full

    labels_full = to_full(det["labels"])

    # 二值掩码: 主掩码不含裂隙(裂隙线索与线描混淆，单列输出)
    binary = np.isin(labels_full, [2, 3, 4]).astype(np.uint8) * 255
    # 语义拆分
    inpaint = np.isin(labels_full, [2, 4]).astype(np.uint8) * 255   # 缺信息: 剥落+白斑
    degrade = np.isin(labels_full, [3]).astype(np.uint8) * 255      # 色彩失真: 发黑
    crack_only = np.isin(labels_full, [1]).astype(np.uint8) * 255   # 裂隙(低置信，单独)

    # 只在内容区域内统计
    area = cw * ch
    ratios = {
        "crack": float(((labels_full == 1)[t:b, l:r]).sum()) / area,
        "loss": float(((labels_full == 2)[t:b, l:r]).sum()) / area,
        "dark": float(((labels_full == 3)[t:b, l:r]).sum()) / area,
        "spot": float(((labels_full == 4)[t:b, l:r]).sum()) / area,
        "damage": float((binary[t:b, l:r] > 0).sum()) / area,
    }
    return {
        "image": bgr, "box": box, "labels": labels_full, "binary": binary,
        "inpaint": inpaint, "degrade": degrade, "crack": crack_only,
        "roi": roi, "det": det, "ratios": ratios, "scale": s,
    }


# ------------------------------ visualization -------------------------------
def make_viz(res, p: Params = None, max_side=900):
    bgr = res["image"].copy()
    labels = res["labels"]
    H, W = labels.shape

    overlay = bgr.copy()
    colors = {1: (0, 0, 255), 2: (0, 200, 255), 3: (255, 0, 120), 4: (0, 255, 0)}  # BGR
    for k, c in colors.items():
        overlay[labels == k] = c
    overlay = cv2.addWeighted(bgr, 0.55, overlay, 0.45, 0)
    # 描边
    edge = cv2.Canny((labels > 0).astype(np.uint8) * 255, 50, 150)
    overlay[edge > 0] = (255, 255, 255)

    heat = res["det"]["heat"]
    heat_full = np.zeros((H, W), np.float32)
    l, t, r, b = res["box"]
    heat_full[t:b, l:r] = cv2.resize(heat, (r - l, b - t), interpolation=cv2.INTER_LINEAR)
    hm = cv2.applyColorMap((heat_full * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm = cv2.addWeighted(bgr, 0.45, hm, 0.55, 0)

    binary3 = cv2.cvtColor(res["binary"], cv2.COLOR_GRAY2BGR)

    # 统一缩放拼图
    def rs(im):
        return cv2.resize(im, (max_side, int(im.shape[0] * max_side / im.shape[1])),
                          interpolation=cv2.INTER_AREA) if im.shape[1] != max_side else im
    panels = [rs(bgr), rs(overlay), rs(hm), rs(binary3)]
    hmin = min(x.shape[0] for x in panels)
    panels = [x[:hmin] for x in panels]
    gap = np.full((hmin, 8, 3), 255, np.uint8)
    out = panels[0]
    for x in panels[1:]:
        out = np.hstack([out, gap, x])
    return out


# ------------------------------ batch ---------------------------------------
def process_dir(indir, outdir, p: Params = None, limit=None, save_labels=True):
    p = p or Params()
    os.makedirs(os.path.join(outdir, "mask"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "inpaint"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "degrade"), exist_ok=True)
    if save_labels:
        os.makedirs(os.path.join(outdir, "label"), exist_ok=True)
    files = sorted([f for f in os.listdir(indir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if limit:
        files = files[:limit]
    rows = []
    for i, f in enumerate(files):
        bgr = imread_u(os.path.join(indir, f))
        if bgr is None:
            continue
        res = analyze(bgr, p)
        stem = os.path.splitext(f)[0]
        imwrite_u(os.path.join(outdir, "mask", stem + ".png"), res["binary"])
        imwrite_u(os.path.join(outdir, "inpaint", stem + ".png"), res["inpaint"])
        imwrite_u(os.path.join(outdir, "degrade", stem + ".png"), res["degrade"])
        if save_labels:
            imwrite_u(os.path.join(outdir, "label", stem + ".png"), res["labels"])
        r = res["ratios"]
        rows.append({"file": f, **{k: round(v, 5) for k, v in r.items()}})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)} ...", flush=True)
    return rows


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", nargs="*", help="生成四联可视化的一组图片路径")
    ap.add_argument("--out", default="viz_out")
    args = ap.parse_args()
    if args.viz:
        os.makedirs(args.out, exist_ok=True)
        for path in args.viz:
            bgr = imread_u(path)
            res = analyze(bgr)
            stem = os.path.splitext(os.path.basename(path))[0]
            imwrite_u(os.path.join(args.out, stem + "_viz.png"), make_viz(res))
            print(stem, {k: round(v, 4) for k, v in res["ratios"].items()})
