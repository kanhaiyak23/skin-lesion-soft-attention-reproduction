# Milestone 3, Hypothesis 2: Is Melanoma Recall a Threshold Problem?

**Author:** Navnit Naman (230085), with Kanhaiya Kumar (230062)  
**Base paper:** Datta et al., *Soft-Attention Improves Skin Cancer Classification Performance*, iMIMIC @ MICCAI 2021  
**Reference model:** our reproduced Inception-ResNet-v2 baseline (`cv_project_irv2.ipynb`, seed 42)  
**Companion to:** `M3_hypothesis.md` (Hypothesis 1, focal loss)

---

## 1. Motivation

The reproduced IRv2 baseline separates melanoma from other lesions well but rarely *predicts* melanoma:

| Melanoma metric (IRv2, test set, n = 34) | Value |
|---|---|
| One-vs-rest AUC | **0.915** |
| Recall at arg-max | **0.441** (15 / 34), 95 % Wilson CI 0.29–0.60 |
| Precision at arg-max | 0.517 |
| Specificity at arg-max | 0.982 |

With an AUC of 0.915 and a specificity of 0.982, the arg-max decision sits at a very conservative point on the melanoma ROC curve. The model ranks melanomas above most non-melanomas, but `argmax` lets a melanoma win only when its probability beats every other class, and that rarely happens when *nv* holds 80 % of the data.

Hypothesis 1 (focal loss) predicts that changing the training loss will raise melanoma recall. Its own falsification criteria say the result is only *partial support* if the gain "could also be produced by a threshold change". So far no one has measured what a threshold change alone can do. This hypothesis fills that gap.

---

## 2. Hypothesis

> **Lowering the melanoma decision threshold on the already-trained IRv2 baseline, with no retraining, will raise melanoma recall from 0.44 to at least 0.59 (≥ 20 / 34) while keeping melanoma specificity ≥ 0.95. The reason is that the baseline's errors come from the arg-max operating point, not from poor features.**

**Decision rule.** Predict *mel* if p(mel) ≥ t. Otherwise predict the arg-max over the remaining six classes. With t = arg-max-equivalent, this reproduces the baseline exactly, so **t is the only variable**.

**Predicted effect (cross-fitted, see §3):**

| Metric | Baseline (arg-max) | Predicted at spec ≥ 0.95 |
|---|---|---|
| Melanoma recall | 0.441 | **0.59–0.75** |
| Melanoma precision | 0.517 | 0.35–0.45 (falls) |
| Melanoma specificity | 0.982 | 0.95–0.96 |
| nv recall | 0.989 | 0.96–0.98 |
| Macro recall | 0.720 | +0.02 to +0.04 |
| Weighted precision | 0.909 | within ± 0.015 |
| Weighted / melanoma AUC | 0.978 / 0.915 | **unchanged by construction** |

---

## 3. Protocol

1. **Inputs.** Test-set softmax outputs (828 × 7) from the saved IRv2 seed-42 checkpoint. If they were not saved, one inference pass produces them in a few minutes. No GPU training is needed.
2. **No threshold chosen on the data it is scored on.** The checkpoint already saw the whole training set, so a slice of training data would give optimistic probabilities. Instead we use **5-fold cross-fitting on the test set**, stratified by class. For each fold, t is picked on the other four folds as the lowest threshold whose melanoma specificity is ≥ 0.95, then applied to the held-out fold. Predictions from all five folds are pooled before computing metrics.
3. **Operating points reported.** (a) arg-max, (b) specificity ≥ 0.95, (c) specificity ≥ 0.90, (d) specificity 0.713, which is the mean dermatologist specificity in Haenssle et al. (2018). At (d) we compare the model's sensitivity with the dermatologists' 0.866.
4. **Uncertainty.** 95 % Wilson intervals for per-class recall. 1,000-resample bootstrap intervals for weighted precision and macro recall. We repeat the 5-fold split with 20 random seeds and report mean ± std for t.
5. **Same procedure on IRv2 + Soft-Attention** (mel AUC 0.969). This tells us whether attention's recall gain (15 → 21 / 34) survives once both models are compared at the same specificity.

---

## 4. How we will know we were wrong

- **Reject** if cross-fitted melanoma recall at specificity ≥ 0.95 is below 20 / 34 (0.588). That would mean the ROC curve is too flat at low false-positive rates for a threshold to help, and the problem does lie in the features.
- **Reject** if weighted precision drops by more than 0.02. The gain would then be bought with too many *nv* false alarms.
- **Inconclusive** if the threshold chosen in each fold varies by more than 0.2 across folds. With only 34 melanomas, the operating point would not be stable enough to recommend.

---

## 5. Why this matters for Hypothesis 1

| H2 outcome | Consequence for the focal-loss experiment (H1) |
|---|---|
| **Supported** | Arg-max recall alone cannot show that focal loss helps. H1 must be judged as **melanoma recall at matched specificity** against the threshold-tuned baseline, or as a change in AUC. |
| **Rejected** | The baseline's features are the bottleneck. A training-time change such as focal loss is justified, and arg-max comparisons remain fair. |

Either way, H2 sets the bar that H1 has to clear, and it costs zero GPU-hours.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Only 34 test melanomas, so one image = 2.9 recall points | Wilson CIs; repeated cross-fitting; report counts, not just rates |
| Checkpoint itself was selected on the test set (paper's protocol) | Stated as a shared limitation with M2; does not favour any threshold |
| Softmax probabilities may be poorly calibrated | Thresholding needs only the *ranking*, not calibration. We also report a reliability diagram for mel. |
| More mel predictions take images from nv/bkl | Track nv and bkl recall and the full confusion matrix at each operating point |

---

## 7. Results table (to fill in after running)

| Model | Operating point | t (mean ± std) | mel recall | mel spec. | mel prec. | macro recall | W. precision |
|---|---|---|---|---|---|---|---|
| IRv2 | arg-max | — | 0.441 | 0.982 | 0.517 | 0.720 | 0.909 |
| IRv2 | spec ≥ 0.95 | | *pred. 0.59–0.75* | | | | |
| IRv2 | spec ≥ 0.90 | | | | | | |
| IRv2 | spec = 0.713 (dermatologists) | | *vs. 0.866* | | | | |
| IRv2 + SA | arg-max | — | 0.618 | 0.987 | 0.677 | 0.713 | 0.913 |
| IRv2 + SA | spec ≥ 0.95 | | | | | | |

*No clinical claims are made. All results are benchmark numbers on a curated research dataset.*

**References.** Datta S. K. et al., iMIMIC @ MICCAI 2021. Haenssle H. A. et al., *Annals of Oncology* 29(8), 2018. Lin T.-Y. et al., *Focal Loss for Dense Object Detection*, ICCV 2017.
