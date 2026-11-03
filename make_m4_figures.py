"""Milestone 4 figures: python make_m4_figures.py m4_results/runs report/figures"""
import sys, glob, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
runs_dir, out = sys.argv[1], sys.argv[2]
C = ["akiec","bcc","bkl","df","mel","nv","vasc"]; MEL = 4
COL = {"ce": "#2a78d6", "focal": "#eb6834"}; NAME = {"ce": "Cross-entropy", "focal": "Focal loss (γ=2)"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
runs = {}
for f in sorted(glob.glob(f"{runs_dir}/irv2_*_paper_seed*/test_predictions.csv")):
    n = os.path.basename(os.path.dirname(f))
    if n.endswith("quick"): continue
    loss, seed = n.split("_")[1], int(n.split("seed")[1])
    d = pd.read_csv(f); y = d["true"].map(C.index).values; p = d[[f"p_{c}" for c in C]].values
    runs[(loss, seed)] = (y, p)

# Figure 1: melanoma operating curves
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
for (loss, seed), (y, p) in sorted(runs.items()):
    s = p[:, MEL]; pos, neg = s[y == MEL], s[y != MEL]
    ts = np.unique(np.concatenate([s, [1.1]]))[::-1]
    sens = [(pos >= t).mean() for t in ts]; spec = [(neg < t).mean() for t in ts]
    ax.plot(spec, sens, color=COL[loss], lw=1.6, alpha=0.85, label=NAME[loss] if seed == 42 else None, drawstyle="steps-post")
    am = p.argmax(1)
    ax.plot(((am != MEL) & (y != MEL)).sum() / (y != MEL).sum(), ((am == MEL) & (y == MEL)).sum() / (y == MEL).sum(),
            "o", ms=7, color=COL[loss], mec="white", mew=1.5, zorder=5)
for x in (0.95, 0.90):
    ax.axvline(x, color=GRID, lw=1, zorder=0); ax.text(x, 0.305, f"spec {x:.2f}", color=MUTED, fontsize=7.5, ha="right", rotation=90, va="bottom")
ax.plot(0.713, 0.866, "D", ms=7, color=INK, mec="white", mew=1.2, zorder=6)
ax.annotate("Dermatologists\n(Haenssle 2018)", (0.713, 0.866), xytext=(0.725, 0.70), fontsize=7.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_xlim(1.0, 0.70); ax.set_ylim(0.3, 1.02)
ax.set_xlabel("Melanoma specificity  (← stricter)"); ax.set_ylabel("Melanoma recall (sensitivity)")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
h, l = ax.get_legend_handles_labels()
h.append(plt.Line2D([], [], marker="o", ls="", color=MUTED, mec="white", ms=7)); l.append("Arg-max operating point")
ax.legend(h, l, loc="lower right", frameon=False, fontsize=8)
ax.set_title("Melanoma recall vs specificity, 3 seeds per loss (test set, n = 828, 34 mel)", fontsize=9.5, color=INK, loc="left")
fig.tight_layout(); fig.savefig(f"{out}/m4_mel_operating_curves.png", facecolor="white"); plt.close(fig)

# Figure 2: per-seed small multiples
def met(y, p):
    am = p.argmax(1); rec = [((am == i) & (y == i)).sum() / (y == i).sum() for i in range(7)]
    return {"Accuracy": (am == y).mean(), "Macro recall": np.mean(rec), "Melanomas found (of 34)": ((am == MEL) & (y == MEL)).sum()}
seeds = sorted({s for _, s in runs}); M = {k: met(*v) for k, v in runs.items()}
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), dpi=200)
for ax, key in zip(axs, ["Accuracy", "Macro recall", "Melanomas found (of 34)"]):
    for j, loss in enumerate(["ce", "focal"]):
        xs = np.arange(len(seeds)) + (j - 0.5) * 0.22
        vals = [M[(loss, s)][key] for s in seeds]
        ax.plot(xs, vals, "o", ms=7, color=COL[loss], mec="white", mew=1.2, label=NAME[loss])
        ax.axhline(np.mean(vals), color=COL[loss], lw=1, ls="--", alpha=0.7)
    ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=7.5)
    ax.set_title(key, fontsize=9, color=INK, loc="left"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlim(-0.6, len(seeds) - 0.4)
axs[2].set_ylim(10, 30)
axs[0].legend(loc="lower left", frameon=False, fontsize=7.5)
fig.text(0.99, 0.01, "dashed line = mean of 3 seeds", ha="right", fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{out}/m4_per_seed.png", facecolor="white"); plt.close(fig)
print("ok")
