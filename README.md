# Skin Lesion Classification on HAM10000 — Reproducing Datta et al. (2021)

**Team:** Navnit Naman (230085), Kanhaiya Kumar (230062) — Newton School of Technology, Rishihood University

Reproduction of *"Soft-Attention Improves Skin Cancer Classification Performance"* (Datta et al., iMIMIC @ MICCAI 2021)
on the HAM10000 dataset, followed by our own hypothesis.

## Milestones

| Milestone | Contents |
|-----------|----------|
| M1 — Literature review & base paper | `literature_review_milestone_1.pdf`, `base_paper_selection.md`, `Precision_Skin_Lesion_Classification.pdf` |
| M2 — Baseline reproduction | `M2_reproduce_datta2021_baseline.ipynb`, Kaggle runs `cv_project_irv2.ipynb` (IRv2) and `cv_project_irv2_sa.ipynb` (IRv2 + Soft-Attention), `reproduction_gap_analysis.md` |
| M3 — Hypothesis & introduction | `M3_hypothesis.md`, `M3_introduction.md`, `M2_M3_reproduction_hypothesis.pptx` |
| M3 — Hypothesis 2 (threshold control) | `M3_hypothesis2_threshold.md`, `M3_hypothesis2_threshold.pdf` |
| M4 — Hypothesis results | `M4_results.md` / `M4_results.pdf`, code `train.py`, `run_pod.sh`, `M4_analyze.py`, `make_m4_figures.py`; per-run outputs in `m4_results/`; full write-up in the final report `report/report.tex` (Sections VII–X) |
| Full report (IEEE format) | `report/report.pdf`, LaTeX source `report/report.tex`, Markdown `report/report.md`, figures in `report/figures/` |

## Reproduction results

| Model | Accuracy | Weighted precision | Melanoma recall | Paper (w. precision) |
|-------|----------|--------------------|-----------------|----------------------|
| IRv2 | 0.916 | 0.909 | 0.441 | 0.905 |
| IRv2 + Soft-Attention | 0.918 | 0.913 | 0.618 | 0.937 |

The reported +3.2 pt Soft-Attention gain was not reproduced (+0.4 pt). See `reproduction_gap_analysis.md`.

## M3 hypothesis

Focal loss (γ = 2) on the IRv2 baseline improves minority-class (melanoma) recall. See `M3_hypothesis.md`.

## M4 results (6 runs, 3 seeds per loss)

| | Cross-entropy | Focal loss (γ = 2) |
|---|---|---|
| Accuracy | 0.922 ± 0.003 | 0.813 ± 0.054 |
| Melanomas found (of 34) | 20.3 ± 5.1 | 25.7 ± 0.6 |
| Macro recall | 0.755 | 0.684 |
| Melanomas found at specificity ≥ 0.95 (threshold tuned) | **25.0** | 19.3 |

H1 (focal loss): not supported. Rejected by the pre-registered macro-recall criterion (inconclusive under the paired rule); it finds more melanomas only by predicting melanoma far more often, and loses at matched specificity. H2 (threshold): supported. See `M4_results.md`.

