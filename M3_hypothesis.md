# Milestone 3: Hypothesis (written before any code for the change)

**Team:** Navnit Naman (230085), Kanhaiya Kumar (230062)
**Base paper:** Datta et al., *Soft-Attention Improves Skin Cancer Classification Performance*, iMIMIC @ MICCAI 2021
**Reference model:** *our reproduced* Inception-ResNet-v2 baseline (`cv_project_irv2.ipynb`, seed 42), not the paper's number.

---

## 1. What the reproduced baseline tells us

| Observation (our IRv2 baseline) | Evidence |
|---|---|
| Overall metrics look strong | Accuracy 0.916, weighted precision 0.909, weighted AUC 0.978 |
| **Melanoma, the most dangerous class, is the weakest** | **Recall 0.441: 15 of 34 melanomas found, 19 missed** |
| The errors are mostly lesions absorbed into "nevus" | bkl → nv **19**, mel → nv **9**, mel → bkl 7, akiec → nv 4 |
| The model has **memorised** the training set | Training accuracy 0.98 (loss 0.08), but test accuracy 0.90–0.92 |
| Ranking is good; the decision is the problem | mel AUC 0.915, but recall only 0.44 at the argmax |
| Soft-Attention (the paper's change) gave no measurable gain | Weighted precision +0.004, intervals overlap; see `reproduction_gap_analysis.md` |

**Important detail:** the training set is already *balanced by oversampling* (≈ 5,900–8,000 augmented images per class). The remaining problem is therefore **not class frequency during training**. It is that ~98 % of training examples are already classified correctly. Their cross-entropy gradients, though individually small, are so numerous that they drown out the few **hard, confusable examples**: melanoma or benign keratosis that look like a nevus.

---

## 2. Hypothesis (one variable)

> **Replacing categorical cross-entropy with focal loss (γ = 2) will increase melanoma recall and macro recall on HAM10000, because focal loss down-weights the ~98 % of training examples the network already classifies correctly and concentrates the gradient on hard, confusable lesions (mel / bkl vs nv).**
>
> **Predicted effect (vs our reproduced IRv2 baseline, mean over seeds):**
> - **Melanoma recall: 0.44 → 0.50–0.60** (+2 to +5 of 34 melanomas found)
> - **Macro recall: 0.72 → 0.74–0.77** (+0.02 to +0.05)
> - bkl recall: 0.61 → 0.65–0.72; fewer bkl → nv errors (19 → ≤ 14)
> - **Weighted precision: unchanged to slightly lower** (0.909 → 0.89–0.91)
> - Accuracy may dip by ≤ 0.01; nv recall 0.989 → ~0.97–0.98
> - **Weighted AUC: unchanged (± 0.005).** Focal loss shifts the decision point more than it changes the ranking.

### The change, and only the change

| | Baseline (reproduced) | Hypothesis run |
|---|---|---|
| **Loss** | Categorical cross-entropy | **Categorical focal loss, γ = 2, α = 1** |
| Class weight (mel × 5) | yes | yes (unchanged) |
| Backbone / head | IRv2 cut at block8_9, no SA | same |
| Data / split / augmentation | seed 42, 9,187 / 828, 51,699 aug. | same |
| Optimiser / schedule | Adam lr 0.01 ε 0.1, 150 ep, ES patience 30 | same |
| Hardware | 1 × T4 | 1 × T4 |

Focal loss: FL(p_t) = −(1 − p_t)^γ · log(p_t). With γ = 0 it is exactly cross-entropy, so γ is the single variable being changed. α = 1 is chosen so that **no extra class re-weighting** is added on top of the existing ×5 melanoma weight.

**Why the baseline without SA, and not IRv2+SA?** In our reproduction, Soft-Attention produced no measurable gain. Applying the change to the simpler model isolates the effect of the loss function.

---

## 3. Mechanism: why this should work

1. With CE, an example predicted correctly at p = 0.9 still contributes −log 0.9 ≈ 0.105. With γ = 2 it contributes 0.01 × 0.105 ≈ 0.001, which is **100× less**. A hard example at p = 0.3 keeps about half its CE loss.
2. In our run, nearly all of the 51,699 training images end up "easy". Focal loss therefore shifts most of the gradient onto the small set of mel / bkl / akiec images that the network still confuses with nv.
3. That moves the decision boundary away from nv, so fewer minority lesions get absorbed into the majority class. This is exactly the error pattern in our confusion matrix.
4. We do not expect it to make the features better at ranking images. Hence the prediction that AUC stays flat while recall rises.

---

## 4. How we will know we were wrong (falsification)

- **Reject** if mean melanoma recall over seeds does not exceed the baseline mean by more than one standard deviation.
- **Reject** if macro recall does not improve, or improves only through df / vasc (6 and 10 images, where one image is 10–17 points).
- **Partial support** if recall rises but weighted precision falls by more than 0.02. In that case focal loss is just trading precision for recall, which a threshold change could also do.

## 5. Risks we already see
| Risk | Why | What we will check |
|---|---|---|
| **Run-to-run noise may be as large as the predicted effect** | One run per config so far; no deterministic GPU ops; 1 melanoma ≈ 3 pts recall | ≥ 2–3 seeds per config; report mean ± std |
| **Effective learning rate changes** | Focal loss values are smaller than CE, and Adam's ε = 0.1 is large, so Adam is *not* scale-invariant here: smaller gradients mean smaller steps | Compare training curves; note it as a confound |
| **Checkpoint picked on the test set** | The paper's protocol | Also report one `PROTOCOL="clean"` run |
| **Recall gain comes at nv's expense** | More minority predictions bring more nv false positives | Track nv recall and specificity |

---

## 6. Ablation plan (Milestone 4)

Fill in **after** running. Predictions are written now.

| Configuration | Seeds | mel recall | macro recall | W.Avg precision | W.Avg AUC | Notes |
|---|---|---|---|---|---|---|
| IRv2 baseline, CE (reproduced) | 42 | **0.441** | **0.720** | **0.909** | **0.978** | Milestone 2 |
| **IRv2 + focal loss γ = 2 (hypothesis)** | 42, 7, 2024 | predicted 0.50–0.60 | predicted 0.74–0.77 | predicted 0.89–0.91 | predicted ≈ 0.978 | one variable changed |
| IRv2 + Soft-Attention (paper's change, reproduced) | 42 | 0.618 | 0.713 | 0.913 | 0.981 | reference only |
| IRv2 + focal, `PROTOCOL="clean"` (optional) | 42 | | | | | measures test-as-validation bias |

**Compute budget:** one run takes ~53 epochs × ~8 min ≈ **7 GPU-hours** on a Kaggle T4. Kaggle gives ~30 GPU-h/week, so plan for about 4 runs per week. Order: focal seed 42 first, so it is directly comparable to the baseline. Then the extra seeds for both baseline and focal.

### Alternatives we considered (and why not now)
| Alternative single change | Why not chosen |
|---|---|
| CLAHE contrast preprocessing | Our errors are semantic confusions (mel / bkl vs nv), not low contrast |
| Backbone swap → EfficientNet-B0 | Changes capacity and preprocessing at once; Ali et al. showed capacity is not the bottleneck on HAM10000 |
| Class-balanced sampling | The training set is already balanced by oversampling |
| Remove the test-as-validation leak | An evaluation fix, not a model change; we report it as a side experiment |

*No clinical claims are made. All results are benchmark numbers on a curated research dataset.*
