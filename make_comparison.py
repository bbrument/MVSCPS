"""
Script to generate a comparison summary for suzanne_point_colored_shadows/suzanne evaluations.
Creates /exp/suzanne_point_colored_shadows/suzanne/comparison/
"""

import json
import shutil
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

EVAL_ROOT = Path("/home/babrument/dev/MVSCPS/eval/suzanne_point_colored_shadows/suzanne")
OUT_ROOT = Path("/home/babrument/dev/MVSCPS/exp/suzanne_point_colored_shadows/suzanne/comparison")

# Ordered runs (nicer display order)
RUNS_ORDER = [
    "20v1l_learnlight_point_lambertian_shadow",
    "20v1l_learnlight_point_neuralbrdf_shadow",
    "20v1l_learnlight_dir_neuralbrdf_shadow",
    "20v20l_learnlight_alblearn_shadow",
    "20v20l_learnlight_neuralbrdf_shadow",
    "20v20l_learnlight_dir_neuralbrdf_shadow",
]

# Short labels for plots
SHORT_LABELS = {
    "20v1l_learnlight_point_lambertian_shadow":  "1L-pt-lamb",
    "20v1l_learnlight_point_neuralbrdf_shadow":  "1L-pt-nbrdf",
    "20v1l_learnlight_dir_neuralbrdf_shadow":    "1L-dir-nbrdf",
    "20v20l_learnlight_alblearn_shadow":         "20L-pt-lamb",
    "20v20l_learnlight_neuralbrdf_shadow":       "20L-pt-nbrdf",
    "20v20l_learnlight_dir_neuralbrdf_shadow":   "20L-dir-nbrdf",
}

VIEWS = ["view_000", "view_015", "view_030"]
VIZ_TYPES = ["accuracy", "completeness", "uniform"]
THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]


def load_metrics(run_name):
    p = EVAL_ROOT / run_name / "eval_results" / "metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def load_curves(run_name):
    base = EVAL_ROOT / run_name / "eval_results" / "curves"
    data = {}
    for k in ["fscore", "precision", "recall", "thresholds"]:
        fp = base / f"{k}.npy"
        if fp.exists():
            data[k] = np.load(fp)
    return data if data else None


# ─── 1. Collect all metrics ───────────────────────────────────────────────────
all_metrics = {}
for run in RUNS_ORDER:
    m = load_metrics(run)
    if m:
        all_metrics[run] = m

print(f"Loaded metrics for {len(all_metrics)} runs.")

# ─── 2. Prepare output dirs ───────────────────────────────────────────────────
OUT_ROOT.mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "meshes").mkdir(exist_ok=True)
(OUT_ROOT / "visualizations").mkdir(exist_ok=True)
for vt in VIZ_TYPES:
    (OUT_ROOT / "visualizations" / vt).mkdir(exist_ok=True)

# ─── 3. Copy meshes ───────────────────────────────────────────────────────────
for run in RUNS_ORDER:
    src = EVAL_ROOT / run / "results_cleaned" / "mesh.ply"
    if src.exists():
        dst = OUT_ROOT / "meshes" / f"{run}.ply"
        shutil.copy2(src, dst)

# Copy GT mesh
gt_src = EVAL_ROOT / "Groundtruth" / "gt_cleaned.ply"
if gt_src.exists():
    shutil.copy2(gt_src, OUT_ROOT / "meshes" / "Groundtruth.ply")

print("Meshes copied.")

# ─── 4. Copy visualizations per view ─────────────────────────────────────────
# We make a grid image per (viz_type, view): rows=runs, cols are just that view
for view in VIEWS:
    for vt in VIZ_TYPES:
        imgs = []
        labels = []
        for run in RUNS_ORDER:
            p = EVAL_ROOT / run / "visualizations" / vt / f"{view}.png"
            if p.exists():
                img = plt.imread(str(p))
                imgs.append(img)
                labels.append(SHORT_LABELS.get(run, run))
        if not imgs:
            continue
        n = len(imgs)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, img, label in zip(axes, imgs, labels):
            ax.imshow(img)
            ax.set_title(label, fontsize=8)
            ax.axis('off')
        fig.suptitle(f"{vt} – {view}", fontsize=10)
        plt.tight_layout()
        out_p = OUT_ROOT / "visualizations" / vt / f"{view}.png"
        plt.savefig(str(out_p), dpi=120, bbox_inches='tight')
        plt.close(fig)

# Also copy GT uniform views
for view in VIEWS:
    p = EVAL_ROOT / "Groundtruth" / "visualizations" / "uniform" / f"{view}.png"
    if p.exists():
        shutil.copy2(p, OUT_ROOT / "visualizations" / "uniform" / f"GT_{view}.png")

print("Visualization grids created.")

# ─── 5. Bar chart: Chamfer distance ───────────────────────────────────────────
runs_with_m = [r for r in RUNS_ORDER if r in all_metrics]
labels = [SHORT_LABELS.get(r, r) for r in runs_with_m]
chamfer_vals = [all_metrics[r]["chamfer"] for r in runs_with_m]
chamfer_a2b = [all_metrics[r]["chamfer_a2b"] for r in runs_with_m]
chamfer_b2a = [all_metrics[r]["chamfer_b2a"] for r in runs_with_m]

x = np.arange(len(runs_with_m))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 5))
bars1 = ax.bar(x - width, chamfer_a2b, width, label='Chamfer pred→GT (accuracy↓)', color='#e07070')
bars2 = ax.bar(x, chamfer_b2a, width, label='Chamfer GT→pred (completeness↓)', color='#70a0e0')
bars3 = ax.bar(x + width, chamfer_vals, width, label='Chamfer mean↓', color='#70c070')

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
ax.set_ylabel('Chamfer distance (scene units)')
ax.set_title('Chamfer distances – suzanne_point_colored_shadows')
ax.legend()
ax.grid(axis='y', alpha=0.4)

# annotate mean values
for bar in bars3:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.0002, f'{h:.4f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(str(OUT_ROOT / "chamfer_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Chamfer chart saved.")

# ─── 6. F-score @ key thresholds bar chart ───────────────────────────────────
KEY_THRESH_IDX = [0, 1, 2, 3]  # 1cm, 2cm, 5cm, 10cm

colors = plt.cm.tab10(np.linspace(0, 1, len(KEY_THRESH_IDX)))
fig, axes = plt.subplots(1, len(KEY_THRESH_IDX), figsize=(5 * len(KEY_THRESH_IDX), 5), sharey=True)

for i, ti in enumerate(KEY_THRESH_IDX):
    ax = axes[i]
    thr = THRESHOLDS[ti]
    fscores = [all_metrics[r]["fscore"][ti] for r in runs_with_m]
    bars = ax.bar(labels, fscores, color=colors[i])
    ax.set_title(f'F-score @ {thr*100:.0f}cm', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.4)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=7)

axes[0].set_ylabel('F-score')
fig.suptitle('F-score at various thresholds – suzanne_point_colored_shadows', fontsize=12)
plt.tight_layout()
plt.savefig(str(OUT_ROOT / "fscore_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("F-score chart saved.")

# ─── 7. Precision-Recall curves ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cmap = plt.cm.tab10(np.linspace(0, 1, len(RUNS_ORDER)))

for idx, run in enumerate(runs_with_m):
    curves = load_curves(run)
    if curves is None:
        continue
    label = SHORT_LABELS.get(run, run)
    color = cmap[idx]
    thrs = curves.get("thresholds", THRESHOLDS)
    prec = curves.get("precision", [all_metrics[run]["precision"][ti] for ti in range(len(THRESHOLDS))])
    rec = curves.get("recall", [all_metrics[run]["recall"][ti] for ti in range(len(THRESHOLDS))])
    fs = curves.get("fscore", [all_metrics[run]["fscore"][ti] for ti in range(len(THRESHOLDS))])

    axes[0].plot(thrs, prec, 'o-', color=color, label=label, markersize=4)
    axes[0].plot(thrs, rec, 's--', color=color, markersize=4)
    axes[1].plot(thrs, fs, 'o-', color=color, label=label, markersize=4)

for ax in axes:
    ax.set_xlabel('Threshold (scene units)')
    ax.set_xscale('log')
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8)

axes[0].set_ylabel('Score')
axes[0].set_title('Precision (solid) & Recall (dashed) vs threshold')
axes[1].set_ylabel('F-score')
axes[1].set_title('F-score vs threshold')
axes[1].set_ylim(0, 1.05)

fig.suptitle('Precision / Recall / F-score curves – suzanne_point_colored_shadows', fontsize=12)
plt.tight_layout()
plt.savefig(str(OUT_ROOT / "pr_curves.png"), dpi=150, bbox_inches='tight')
plt.close()
print("PR curves saved.")

# ─── 8. Summary table as CSV + Markdown ──────────────────────────────────────
thr_labels = ["1cm", "2cm", "5cm", "10cm", "20cm", "50cm"]

csv_lines = ["run,n_points,chamfer_pred2gt,chamfer_gt2pred,chamfer_mean," +
             ",".join([f"precision@{t}" for t in thr_labels]) + "," +
             ",".join([f"recall@{t}" for t in thr_labels]) + "," +
             ",".join([f"fscore@{t}" for t in thr_labels])]

md_lines = [
    "# Evaluation Summary – suzanne_point_colored_shadows / suzanne\n",
    "## Chamfer Distances\n",
    "| Run | n_points | Chamfer pred→GT↓ | Chamfer GT→pred↓ | Chamfer mean↓ |",
    "|-----|----------|------------------|------------------|---------------|",
]

for run in runs_with_m:
    m = all_metrics[run]
    label = SHORT_LABELS.get(run, run)
    row = (f"{run},{m['n_data_points']},{m['chamfer_a2b']:.6f},"
           f"{m['chamfer_b2a']:.6f},{m['chamfer']:.6f}," +
           ",".join([f"{v:.4f}" for v in m['precision']]) + "," +
           ",".join([f"{v:.4f}" for v in m['recall']]) + "," +
           ",".join([f"{v:.4f}" for v in m['fscore']]))
    csv_lines.append(row)
    md_lines.append(f"| {label} | {m['n_data_points']} | {m['chamfer_a2b']:.5f} | {m['chamfer_b2a']:.5f} | **{m['chamfer']:.5f}** |")

md_lines += [
    "",
    "## F-score",
    "| Run | " + " | ".join([f"F@{t}" for t in thr_labels]) + " |",
    "|-----|" + "|".join(["-------"] * len(thr_labels)) + "|",
]
for run in runs_with_m:
    m = all_metrics[run]
    label = SHORT_LABELS.get(run, run)
    frow = " | ".join([f"{v:.4f}" for v in m['fscore']])
    md_lines.append(f"| {label} | {frow} |")

md_lines += [
    "",
    "## Precision",
    "| Run | " + " | ".join([f"P@{t}" for t in thr_labels]) + " |",
    "|-----|" + "|".join(["-------"] * len(thr_labels)) + "|",
]
for run in runs_with_m:
    m = all_metrics[run]
    label = SHORT_LABELS.get(run, run)
    prow = " | ".join([f"{v:.4f}" for v in m['precision']])
    md_lines.append(f"| {label} | {prow} |")

md_lines += [
    "",
    "## Recall",
    "| Run | " + " | ".join([f"R@{t}" for t in thr_labels]) + " |",
    "|-----|" + "|".join(["-------"] * len(thr_labels)) + "|",
]
for run in runs_with_m:
    m = all_metrics[run]
    label = SHORT_LABELS.get(run, run)
    rrow = " | ".join([f"{v:.4f}" for v in m['recall']])
    md_lines.append(f"| {label} | {rrow} |")

md_lines += [
    "",
    "## Figures",
    "- `chamfer_comparison.png` – Bar chart of Chamfer distances",
    "- `fscore_comparison.png` – Bar charts of F-score at 1/2/5/10 cm",
    "- `pr_curves.png` – Precision/Recall/F-score curves vs threshold",
    "- `visualizations/accuracy/<view>.png` – Accuracy heat-maps side-by-side",
    "- `visualizations/completeness/<view>.png` – Completeness heat-maps side-by-side",
    "- `visualizations/uniform/<view>.png` – Uniform color renders side-by-side",
    "- `meshes/*.ply` – Reconstructed meshes (+ Groundtruth.ply)",
]

with open(OUT_ROOT / "metrics.csv", "w") as f:
    f.write("\n".join(csv_lines) + "\n")

with open(OUT_ROOT / "README.md", "w") as f:
    f.write("\n".join(md_lines) + "\n")

print("CSV and README saved.")
print(f"\nDone! Output in: {OUT_ROOT}")
print("\n=== Quick summary ===")
print(f"{'Run':<45} {'Chamfer↓':>10} {'F@1cm':>7} {'F@2cm':>7} {'F@5cm':>7}")
print("-" * 75)
for run in runs_with_m:
    m = all_metrics[run]
    label = SHORT_LABELS.get(run, run)
    print(f"{label:<45} {m['chamfer']:>10.5f} {m['fscore'][0]:>7.4f} {m['fscore'][1]:>7.4f} {m['fscore'][2]:>7.4f}")
