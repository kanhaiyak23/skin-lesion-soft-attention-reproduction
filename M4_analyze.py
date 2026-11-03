"""Milestone 4 analysis: H1 (focal loss vs cross-entropy) and H2 (melanoma threshold control).

Reads the test_predictions.csv written by train.py for each run and prints Markdown tables:
  1. per-run metrics,
  2. mean over seeds per loss,
  3. paired McNemar tests, CE vs focal on the same test set (same seed), per seed and pooled,
  4. H2: cross-fitted melanoma threshold at fixed specificity, for every run,
  5. the H1 verdict under the decision rule in M3_hypothesis.md (§4 and addendum).

    python M4_analyze.py work/runs > M4_results_tables.md
"""
import glob, math, os, re, sys
import numpy as np
import pandas as pd

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL = CLASSES.index("mel")
SPEC_TARGETS = [0.95, 0.90, 0.713]          # 0.713 = mean dermatologist specificity, Haenssle et al. 2018
DERM_SENS = 0.866


def load_runs(root):
    runs = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "test_predictions.csv"))):
        name = os.path.basename(os.path.dirname(f))
        m = re.match(r"irv2_(ce|focal)[\d.]*_paper_seed(\d+)$", name)   # m4/train.py writes "focal2"
        if not m:
            continue
        df = pd.read_csv(f).sort_values("image_id").reset_index(drop=True)
        prob = df[[f"p_{c}" for c in CLASSES]].to_numpy()
        y = df["true"].map(CLASSES.index).to_numpy()
        runs[(m.group(1), int(m.group(2)))] = {"ids": df["image_id"].to_numpy(), "prob": prob, "y": y}
    return runs


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value. b = only model A right, c = only model B right."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def auc(pos_scores, neg_scores):
    s = np.concatenate([pos_scores, neg_scores])
    ranks = pd.Series(s).rank().to_numpy()[: len(pos_scores)]
    return (ranks.sum() - len(pos_scores) * (len(pos_scores) + 1) / 2) / (len(pos_scores) * len(neg_scores))


def metrics(y, pred, prob):
    k = len(CLASSES)
    support = np.bincount(y, minlength=k)
    tp = np.array([((pred == i) & (y == i)).sum() for i in range(k)])
    npred = np.bincount(pred, minlength=k)
    prec = np.divide(tp, npred, out=np.zeros(k), where=npred > 0)
    rec = tp / support
    aucs = np.array([auc(prob[y == i, i], prob[y != i, i]) for i in range(k)])
    mel_neg = y != MEL
    return {
        "accuracy": (pred == y).mean(),
        "w_precision": (prec * support).sum() / support.sum(),
        "macro_recall": rec.mean(),
        "mel_found": int(tp[MEL]), "mel_n": int(support[MEL]),
        "mel_recall": rec[MEL],
        "mel_precision": prec[MEL],
        "mel_specificity": ((pred != MEL) & mel_neg).sum() / mel_neg.sum(),
        "mel_auc": aucs[MEL],
        "w_auc": (aucs * support).sum() / support.sum(),
        "nv_recall": rec[CLASSES.index("nv")],
    }


def threshold_predict(prob, t):
    """Predict mel if p(mel) >= t, otherwise the arg-max over the other six classes."""
    others = prob.copy()
    others[:, MEL] = -1
    return np.where(prob[:, MEL] >= t, MEL, others.argmax(1))


def pick_threshold(p_mel, y, spec):
    """Lowest threshold whose melanoma specificity on (p_mel, y) is >= spec."""
    neg = np.sort(p_mel[y != MEL])
    idx = math.ceil(spec * len(neg))          # this many negatives must fall strictly below t
    return np.nextafter(neg[idx - 1], 1.0) if idx > 0 else 0.0


def stratified_folds(y, k, rng):
    fold = np.empty(len(y), int)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        fold[idx] = np.arange(len(idx)) % k
    return fold


def cross_fitted(prob, y, spec, k=5, repeats=20, seed=0):
    rng = np.random.default_rng(seed)
    found, specs, ts, wprec = [], [], [], []
    for _ in range(repeats):
        fold = stratified_folds(y, k, rng)
        pred = np.empty(len(y), int)
        for f in range(k):
            tr, te = fold != f, fold == f
            t = pick_threshold(prob[tr, MEL], y[tr], spec)
            ts.append(t)
            pred[te] = threshold_predict(prob[te], t)
        m = metrics(y, pred, prob)
        found.append(m["mel_found"]); specs.append(m["mel_specificity"]); wprec.append(m["w_precision"])
    return {"mel_found": np.mean(found), "mel_spec": np.mean(specs), "w_precision": np.mean(wprec),
            "t_mean": np.mean(ts), "t_std": np.std(ts)}


def main(root):
    runs = load_runs(root)
    if not runs:
        sys.exit(f"no irv2_*_paper_seed*/test_predictions.csv under {root}")
    rows = {key: metrics(r["y"], r["prob"].argmax(1), r["prob"]) for key, r in runs.items()}

    print("## 1. Per-run results (arg-max)\n")
    print("| Loss | Seed | Accuracy | W. precision | Macro recall | mel found | mel recall (95% Wilson) | mel spec. | mel AUC | W. AUC |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (loss, seed), m in sorted(rows.items()):
        lo, hi = wilson(m["mel_found"], m["mel_n"])
        print(f"| {loss} | {seed} | {m['accuracy']:.3f} | {m['w_precision']:.3f} | {m['macro_recall']:.3f} | "
              f"{m['mel_found']}/{m['mel_n']} | {m['mel_recall']:.3f} ({lo:.2f}–{hi:.2f}) | "
              f"{m['mel_specificity']:.3f} | {m['mel_auc']:.3f} | {m['w_auc']:.3f} |")

    print("\n## 2. Mean over seeds (± std where there are ≥ 2 seeds)\n")
    cols = ["accuracy", "w_precision", "macro_recall", "mel_recall", "mel_specificity", "w_auc"]
    print("| Loss | Seeds | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for loss in ["ce", "focal"]:
        sub = pd.DataFrame([m for (l, _), m in rows.items() if l == loss])
        if sub.empty:
            continue
        seeds = sorted(s for (l, s) in rows if l == loss)
        cells = [f"{sub[c].mean():.3f} ± {sub[c].std(ddof=1):.3f}" if len(sub) > 1 else f"{sub[c].mean():.3f}"
                 for c in cols]
        print(f"| {loss} | {', '.join(map(str, seeds))} | " + " | ".join(cells) + " |")

    print("\n## 3. Paired comparison, CE vs focal on the same test set (exact McNemar)\n")
    print("| Seed | Images | only CE right | only focal right | p (all) | mel: only CE found | mel: only focal found | p (mel) |")
    print("|---|---|---|---|---|---|---|---|")
    pooled = [0, 0]
    seeds_better = []
    for seed in sorted({s for (_, s) in runs}):
        if ("ce", seed) not in runs or ("focal", seed) not in runs:
            continue
        a, b = runs[("ce", seed)], runs[("focal", seed)]
        assert (a["ids"] == b["ids"]).all(), f"seed {seed}: CE and focal test sets differ"
        y = a["y"]
        ca, cb = a["prob"].argmax(1) == y, b["prob"].argmax(1) == y
        mel = y == MEL
        b_all, c_all = int((ca & ~cb).sum()), int((~ca & cb).sum())
        b_mel, c_mel = int((ca & ~cb & mel).sum()), int((~ca & cb & mel).sum())
        pooled[0] += b_mel; pooled[1] += c_mel
        seeds_better.append(c_mel > b_mel)
        print(f"| {seed} | {len(y)} | {b_all} | {c_all} | {mcnemar_exact(b_all, c_all):.3f} | "
              f"{b_mel} | {c_mel} | {mcnemar_exact(b_mel, c_mel):.3f} |")
    p_pooled = mcnemar_exact(*pooled)
    print(f"| **pooled** | | | | | {pooled[0]} | {pooled[1]} | **{p_pooled:.3f}** |")

    print("\n## 4. H2: cross-fitted melanoma threshold (5-fold × 20 repeats, threshold never chosen on the images it scores)\n")
    print("| Loss | Seed | Operating point | t (mean ± std) | mel found | mel spec. | W. precision |")
    print("|---|---|---|---|---|---|---|")
    for (loss, seed), r in sorted(runs.items()):
        m = rows[(loss, seed)]
        print(f"| {loss} | {seed} | arg-max | — | {m['mel_found']}/{m['mel_n']} | {m['mel_specificity']:.3f} | {m['w_precision']:.3f} |")
        for spec in SPEC_TARGETS:
            cf = cross_fitted(r["prob"], r["y"], spec, seed=seed)
            label = f"spec ≥ {spec:.2f}" if spec != 0.713 else "spec ≥ 0.713 (dermatologists)"
            extra = f" (derm. sens. {DERM_SENS:.3f})" if spec == 0.713 else ""
            print(f"| {loss} | {seed} | {label} | {cf['t_mean']:.3f} ± {cf['t_std']:.3f} | "
                  f"{cf['mel_found']:.1f}/{m['mel_n']} = {cf['mel_found'] / m['mel_n']:.3f}{extra} | "
                  f"{cf['mel_spec']:.3f} | {cf['w_precision']:.3f} |")

    print("\n## 5. H1 verdict (M3 §4 rule + paired test, see the M3 addendum)\n")
    if len(seeds_better) < 2:
        print("Needs CE and focal runs on at least two seeds.")
        return
    ce_mean = np.mean([m["mel_found"] for (l, _), m in rows.items() if l == "ce"])
    fl_mean = np.mean([m["mel_found"] for (l, _), m in rows.items() if l == "focal"])
    wp_drop = (np.mean([m["w_precision"] for (l, _), m in rows.items() if l == "ce"])
               - np.mean([m["w_precision"] for (l, _), m in rows.items() if l == "focal"]))
    ce_found = [m["mel_found"] for (l, _), m in rows.items() if l == "ce"]
    sd_rule = fl_mean - ce_mean > np.std(ce_found, ddof=1)          # M3 §4: more than one CE standard deviation
    print(f"M3 §4 rule: focal mean − CE mean = {fl_mean - ce_mean:+.1f} melanomas, "
          f"CE std = {np.std(ce_found, ddof=1):.1f} -> {'met' if sd_rule else 'not met'}.\n")
    consistent = all(seeds_better) and sd_rule
    if consistent and p_pooled < 0.05 and wp_drop > 0.02:
        verdict = (f"PARTIAL SUPPORT: more melanomas found, but weighted precision falls by {wp_drop:.3f} (> 0.02); "
                   "compare with the H2 threshold rows at matched specificity.")
    elif consistent and p_pooled < 0.05:
        verdict = "SUPPORTED: focal finds more melanomas on every seed, by more than one CE standard deviation on average, and the pooled McNemar test is significant."
    elif fl_mean <= ce_mean:
        verdict = "REJECTED: focal finds no more melanomas than cross-entropy on average."
    else:
        verdict = ("INCONCLUSIVE (paired rule): focal finds more melanomas on average, "
                   "but not on every seed or not significantly.")
    print(f"Melanomas found, mean per seed: CE {ce_mean:.1f}, focal {fl_mean:.1f}. "
          f"Pooled McNemar on melanomas: p = {p_pooled:.3f}.\n\n**{verdict}**")
    # M3 §4, second rejection criterion: macro recall must improve
    ce_mr = np.mean([m["macro_recall"] for (l, _), m in rows.items() if l == "ce"])
    fl_mr = np.mean([m["macro_recall"] for (l, _), m in rows.items() if l == "focal"])
    print(f"\nM3 §4 macro-recall criterion: CE {ce_mr:.3f}, focal {fl_mr:.3f} -> "
          f"{'improves' if fl_mr > ce_mr else '**does not improve: reject under §4**'}. "
          f"Weighted precision change: {-wp_drop:+.3f}.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "work/runs")
