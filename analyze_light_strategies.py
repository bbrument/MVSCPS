"""
Compute view-light selection strategies (gray/Sobel) for DiLiGenT-MV and LucesMV datasets.
Outputs a summary table + one plot per object per dataset.

Usage:
    python analyze_light_strategies.py
"""

import os
import re
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Dataset definitions ────────────────────────────────────────────────────────
DATASETS = {
    "DiLiGenT-MV": {
        "root": Path("/home/babrument/dev/MVSCPS/data/DiLiGenT-MV"),
        "objects": ["bear", "buddha", "cow", "pot2", "reading"],
    },
    "LucesMV": {
        "root": Path("/home/babrument/dev/MVSCPS/data/LucesMV_processed"),
        "objects": ["Bowl", "Buddha", "Bunny", "Cup", "Die", "Hippo", "House", "Owl", "Queen", "Squirrel"],
    },
}

OUT_ROOT = Path("/tmp/light_strategy_analysis")
OUT_ROOT.mkdir(exist_ok=True)


def parse_image_list(img_dir):
    """Return sorted list of (vi, li) from V{VV}L{LL}.png filenames."""
    pairs = []
    for f in sorted(os.listdir(img_dir)):
        m = re.match(r"V(\d+)L(\d+)\.png", f)
        if m:
            pairs.append((int(m.group(1)), int(m.group(2))))
    return pairs


def compute_scores(img_dir, mask_dir, n_views, n_lights):
    """Build (n_views, n_lights) gray and sobel score matrices."""
    gray_mat = np.full((n_views, n_lights), np.nan)
    sobel_mat = np.full((n_views, n_lights), np.nan)

    # Load masks
    masks = {}
    for vi in range(n_views):
        mp = os.path.join(mask_dir, f"V{vi:02d}.png")
        if os.path.exists(mp):
            m = cv.imread(mp, cv.IMREAD_GRAYSCALE)
            if m is not None:
                masks[vi] = m > 127

    for vi in range(n_views):
        if vi not in masks:
            continue
        mask = masks[vi]
        n_pix = mask.sum()
        if n_pix == 0:
            continue
        for li in range(n_lights):
            ip = os.path.join(img_dir, f"V{vi:02d}L{li:02d}.png")
            if not os.path.exists(ip):
                continue
            img = cv.imread(ip)
            if img is None:
                continue
            img_f = img.astype(np.float64) / 255.0
            gray = 0.299*img_f[:,:,2] + 0.587*img_f[:,:,1] + 0.114*img_f[:,:,0]
            gray_mat[vi, li] = gray[mask].mean()

            gray_u8 = np.clip(gray * 255, 0, 255).astype(np.uint8)
            sx = cv.Sobel(gray_u8, cv.CV_64F, 1, 0, ksize=3)
            sy = cv.Sobel(gray_u8, cv.CV_64F, 0, 1, ksize=3)
            sobel_mat[vi, li] = np.sqrt(sx**2 + sy**2)[mask].mean()

    return gray_mat, sobel_mat


def apply_strategies(gray_mat, sobel_mat):
    """Apply all 6 strategies, return dict of (n_views,) arrays of selected light indices."""
    n_views, n_lights = gray_mat.shape

    def nanargmax(arr):
        arr = np.where(np.isnan(arr), -np.inf, arr)
        return np.argmax(arr, axis=1)

    def nanargmin(arr):
        arr = np.where(np.isnan(arr), np.inf, arr)
        return np.argmin(arr, axis=1)

    sel_maxgray  = nanargmax(gray_mat)
    sel_mingray  = nanargmin(gray_mat)
    sel_medgray  = np.array([
        np.nanargmin(np.abs(gray_mat[vi] - np.nanmedian(gray_mat[vi])))
        for vi in range(n_views)
    ])
    sel_maxsobel = nanargmax(sobel_mat)

    sel_1l_front = np.zeros(n_views, dtype=int)

    gray_std = np.nanstd(gray_mat, axis=0)
    best_light = int(np.nanargmax(gray_std))
    sel_1l_maxvar = np.full(n_views, best_light, dtype=int)

    return {
        "maxgray":   sel_maxgray,
        "mingray":   sel_mingray,
        "medgray":   sel_medgray,
        "maxsobel":  sel_maxsobel,
        "1l_front":  sel_1l_front,
        "1l_maxvar": sel_1l_maxvar,
    }, best_light, float(np.nanmax(gray_std))


def plot_strategy_heatmap(strategies, gray_mat, sobel_mat, obj_name, dataset_name, out_path):
    """Heatmap: rows=strategies, cols=views, color=chosen light index.
    Also a side panel showing the gray/sobel score of the chosen light per view.
    """
    n_views = gray_mat.shape[0]
    strat_names = list(strategies.keys())
    n_strats = len(strat_names)

    fig, axes = plt.subplots(n_strats + 1, 1, figsize=(max(12, n_views * 0.35), 3 * (n_strats + 1)),
                              gridspec_kw={'height_ratios': [1]*n_strats + [1.5]})

    cmap = plt.cm.tab20
    n_lights = gray_mat.shape[1]

    for i, (sname, sel) in enumerate(strategies.items()):
        ax = axes[i]
        data = sel[np.newaxis, :]  # (1, n_views)
        im = ax.imshow(data, aspect='auto', cmap=cmap,
                       vmin=0, vmax=n_lights - 1, interpolation='nearest')
        ax.set_yticks([0])
        ax.set_yticklabels([sname], fontsize=9)
        ax.set_xticks([])
        # Annotate light index
        for vi in range(n_views):
            ax.text(vi, 0, str(sel[vi]), ha='center', va='center', fontsize=5.5,
                    color='white' if sel[vi] > n_lights * 0.5 else 'black')
        if i == 0:
            ax.set_title(f"{dataset_name} / {obj_name}  —  Selected light per view per strategy\n"
                         f"({n_views} views × {n_lights} lights)", fontsize=11)

    # Bottom panel: gray score curves per strategy
    ax_scores = axes[-1]
    colors_s = plt.cm.Set2(np.linspace(0, 1, n_strats))
    for i, (sname, sel) in enumerate(strategies.items()):
        scores = np.array([gray_mat[vi, sel[vi]] if not np.isnan(gray_mat[vi, sel[vi]]) else 0
                           for vi in range(n_views)])
        ax_scores.plot(scores, label=sname, color=colors_s[i], linewidth=1.5)
    ax_scores.set_xlabel('View index')
    ax_scores.set_ylabel('Mean gray (selected light)')
    ax_scores.legend(fontsize=8, ncol=3)
    ax_scores.grid(alpha=0.3)
    ax_scores.set_xlim(0, n_views - 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


def print_summary_table(strategies, best_light, best_std, gray_mat, sobel_mat, obj_name):
    n_views, n_lights = gray_mat.shape
    print(f"\n  {'Strategy':<12} | {'Light dist (top 5)':<40} | mean_gray(sel) | mean_sobel(sel)")
    print(f"  {'-'*12}-+-{'-'*40}-+-{'-'*14}-+-{'-'*15}")
    for sname, sel in strategies.items():
        uniq, cnt = np.unique(sel, return_counts=True)
        top = sorted(zip(cnt, uniq), reverse=True)[:5]
        dist_str = '  '.join([f"L{l:02d}×{c}" for c, l in top])
        mg = np.nanmean([gray_mat[vi, sel[vi]] for vi in range(n_views)])
        ms = np.nanmean([sobel_mat[vi, sel[vi]] for vi in range(n_views)])
        flag = "← best_maxvar" if sname == "1l_maxvar" else ""
        print(f"  {sname:<12} | {dist_str:<40} | {mg:.4f}         | {ms:.2f}  {flag}")
    print(f"  → 1l_maxvar best light = L{best_light:02d}  (cross-view gray std={best_std:.4f})")


# ── Main ───────────────────────────────────────────────────────────────────────
for ds_name, ds_cfg in DATASETS.items():
    print(f"\n{'='*70}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*70}")
    ds_out = OUT_ROOT / ds_name
    ds_out.mkdir(exist_ok=True)

    for obj in ds_cfg["objects"]:
        obj_dir = ds_cfg["root"] / obj
        img_dir = obj_dir / "image"
        mask_dir = obj_dir / "mask"

        if not img_dir.exists():
            print(f"  [{obj}] image dir not found, skipping")
            continue

        pairs = parse_image_list(img_dir)
        if not pairs:
            print(f"  [{obj}] no images found, skipping")
            continue

        n_views = max(p[0] for p in pairs) + 1
        n_lights = max(p[1] for p in pairs) + 1
        print(f"\n  [{obj}]  {n_views} views × {n_lights} lights  ({len(pairs)} images)")

        gray_mat, sobel_mat = compute_scores(str(img_dir), str(mask_dir), n_views, n_lights)

        strategies, best_light, best_std = apply_strategies(gray_mat, sobel_mat)
        print_summary_table(strategies, best_light, best_std, gray_mat, sobel_mat, obj)

        out_path = ds_out / f"{obj}_strategies.png"
        plot_strategy_heatmap(strategies, gray_mat, sobel_mat, obj, ds_name, str(out_path))

print(f"\nAll done. Plots in {OUT_ROOT}")
