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
| Full report (IEEE format) | `report/report.pdf`, LaTeX source `report/report.tex`, Markdown `report/report.md`, figures in `report/figures/` |

## Reproduction results

| Model | Accuracy | Weighted precision | Melanoma recall | Paper (w. precision) |
|-------|----------|--------------------|-----------------|----------------------|
| IRv2 | 0.916 | 0.909 | 0.441 | 0.905 |
| IRv2 + Soft-Attention | 0.918 | 0.913 | 0.618 | 0.937 |

The reported +3.2 pt Soft-Attention gain was not reproduced (+0.4 pt). See `reproduction_gap_analysis.md`.

## M3 hypothesis

Focal loss (γ = 2) on the IRv2 baseline improves minority-class (melanoma) recall. See `M3_hypothesis.md`.
