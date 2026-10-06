# Reproduction Results and Gap Analysis: Datta et al. (2021) on HAM10000

**Milestone 2 · HAM10000 · 7 classes · 85/15 split · 828 test images**
Base paper: S. K. Datta, S. M. A. Hashemi, S. N. Srihari, M. Gao, *"Soft-Attention Improves Skin Cancer Classification Performance,"* iMIMIC @ MICCAI 2021.

| Notebook (Kaggle outputs) | Model | Protocol | Seed | Hardware | Epochs run (best) | Time/epoch |
|---|---|---|---|---|---|---|
| `cv_project_irv2.ipynb` | IRv2 baseline | paper | 42 | 1× T4 | 53 (best 52) | ~473 s |
| `cv_project_irv2_sa.ipynb` | IRv2 + Soft-Attention | paper | 42 | 2× T4 (MirroredStrategy) | 44 (best 38) | ~467 s |

Both runs reproduce the paper's data pipeline exactly. Each has 10,015 images and 7,470 lesions, a 9,187 / 828 train/test split, the same per-class test counts as the paper, and 51,699 augmented training images. The parameter counts are identical to the authors' `model.summary()`: **47,550,631** (IRv2) and **47,535,287** (IRv2+SA).

---

## 1. Verdict

| | Result |
|---|---|
| **IRv2 baseline** | ✅ **Reproduced.** Accuracy, weighted precision and AUC are all within ±0.012 of the paper. |
| **IRv2 + Soft-Attention** | ⚠️ **Partly reproduced.** AUC matches (−0.003), but weighted precision is 0.025 below the paper. |
| **The paper's claim: SA adds +3.2 pts weighted precision** | ❌ **Not reproduced.** We measure +0.4 pts, well inside run-to-run noise. |

---

## 2. Headline metrics

| Metric | IRv2 paper | **IRv2 ours** | Gap | IRv2+SA paper | **IRv2+SA ours** | Gap |
|---|---|---|---|---|---|---|
| Accuracy | 0.912 | **0.916** | +0.003 | 0.935 | **0.918** | −0.017 |
| Precision (weighted), *paper's headline* | 0.906 | **0.909** | +0.003 | 0.938 | **0.913** | −0.025 |
| Precision (macro) | 0.833 | **0.846** | +0.013 | 0.892 | **0.835** | −0.057 |
| Recall (macro) | 0.689 | **0.720** | +0.031 | 0.719 | **0.713** | −0.006 |
| F1 (macro) | — | **0.773** | | — | **0.766** | |
| AUC (weighted, OvR) | 0.983 | **0.978** | −0.005 | 0.984 | **0.981** | −0.003 |
| AUC (macro, OvR) | 0.985 | **0.973** | −0.012 | 0.985 | **0.982** | −0.003 |

The paper's values combine Table 1 / Table 3 of the paper with the printed outputs of the authors' own notebooks (recall and macro scores are not tabulated in the paper).

**95 % bootstrap intervals, from our notebooks:**

| | IRv2 ours | IRv2+SA ours |
|---|---|---|
| Precision (weighted) | 0.887 – 0.932 | 0.891 – 0.935 |
| Recall (macro) | 0.636 – 0.790 | 0.627 – 0.793 |
| Melanoma recall | 0.267 – 0.618 | 0.441 – 0.781 |
| AUC (weighted) | 0.966 – 0.987 | 0.973 – 0.989 |

---

## 3. Per-class results as image counts

Most classes have very few test images, so converting recall back to images shows how small the differences really are.

| Class | Test n | 1 image = | Correct: paper IRv2 | **ours IRv2** | **ours IRv2+SA** | Recall ours IRv2 → SA | Precision ours IRv2 → SA |
|---|---|---|---|---|---|---|---|
| akiec | 23 | 4.3 pts | 15 | 12 | 12 | 0.522 → 0.522 | 0.857 → 0.706 |
| bcc | 26 | 3.8 pts | 22 | 22 | 21 | 0.846 → 0.808 | 0.846 → 0.875 |
| bkl | 66 | 1.5 pts | 29 | 40 | 39 | 0.606 → 0.591 | 0.755 → 0.848 |
| **df** | **6** | **16.7 pts** | 2 | 5 | 4 | 0.833 → 0.667 | 1.000 → 0.800 |
| **mel** | 34 | 2.9 pts | 19 | **15** | **21** | **0.441 → 0.618** | **0.517 → 0.677** |
| nv | 663 | 0.15 pts | 656 | 656 | 655 | 0.989 → 0.988 | 0.947 → 0.940 |
| **vasc** | **10** | **10.0 pts** | 10 | 8 | 8 | 0.800 → 0.800 | 1.000 → 1.000 |
| **Total correct** | 828 | | ≈ 755 | **758** | **760** | | |

**95 % Wilson intervals for recall.** For every class, our intervals overlap the paper's.

| Class | Paper IRv2 | Ours IRv2 | Ours IRv2+SA |
|---|---|---|---|
| akiec | 0.45 – 0.81 | 0.33 – 0.71 | 0.33 – 0.71 |
| bcc | 0.66 – 0.94 | 0.66 – 0.94 | 0.62 – 0.91 |
| bkl | 0.33 – 0.56 | 0.49 – 0.71 | 0.47 – 0.70 |
| df | 0.10 – 0.70 | 0.44 – 0.97 | 0.30 – 0.90 |
| mel | 0.39 – 0.71 | 0.29 – 0.61 | 0.45 – 0.76 |
| nv | 0.98 – 0.99 | 0.98 – 0.99 | 0.98 – 0.99 |
| vasc | 0.72 – 1.00 | 0.49 – 0.94 | 0.49 – 0.94 |

**Most frequent errors.** These are the same in both models:

| True → predicted | IRv2 | IRv2+SA |
|---|---|---|
| bkl → nv | 19 | 19 |
| mel → nv | 9 | 9 |
| akiec → nv | 4 | 8 |
| mel ↔ bkl | 7 + 6 | 3 + 4 |
| nv → mel | 6 | 5 |

The dominant failure is **minority lesions being absorbed into nv**. Benign keratosis and melanoma are the classes that look most like nevi under dermoscopy.

![Per-class recall, paper vs ours](reproduction_gap_recall.png)

---

## 4. Why our numbers differ from the paper

### Cause 1: a different random test set (largest effect)
The authors call `train_test_split(test_size=0.15, stratify=dx)` **without `random_state`**, so every execution draws a different 828 test images. We fixed `SEED=42`. Our test set has the **same class counts** but **different lesions**, and the training set changes with it.

### Cause 2: tiny minority classes amplify everything
One image is worth **16.7 points** of df recall and **10 points** of vasc recall. Macro metrics give each class a 1/7 weight, so df alone can swing macro recall by ±0.07. Our +0.031 macro recall on IRv2 is mostly df (5/6 correct vs 2/6) plus bkl (+11 images).

### Cause 3: noisy training, with the checkpoint chosen on the test set
- The optimiser settings are aggressive: Adam with lr = 0.01 on a fully trainable 47.5 M-parameter network. Validation accuracy jumps by several points between consecutive epochs, both in the authors' log and in ours.
- `ModelCheckpoint(monitor="val_accuracy")` keeps the best epoch **on the test set**, which is also the validation set in their protocol. That epoch is optimised for overall accuracy, which is about 80 % nv. How the minority classes behave at that epoch is close to random.
- The notebook fixes the seed but does not call `tf.config.experimental.enable_op_determinism()`, so repeated runs can also differ through cuDNN non-determinism and the epoch the checkpoint lands on. With one run per configuration, this run-to-run variance is **not yet measured**; M4 will run multiple seeds.

### Cause 4: a different operating point, not a worse model
AUC does not depend on the decision threshold, and it matches the paper closely (0.978–0.981 vs 0.983–0.984). The models rank images about equally well. What differs is where the argmax decision lands for each class: for example, melanoma recall vs melanoma precision, and how many lesions get absorbed into nv.

### Cause 5: software, porting and hardware (small)
| Difference | Expected effect |
|---|---|
| TF 2.4 / Keras 2 → TF 2.20 / Keras 3.13 | Different kernels and non-determinism; small |
| `ImageDataGenerator` → OpenCV re-implementation of the same transform | Same distribution and image count, different random samples; small |
| PIL nearest → `tf.image.resize(nearest)` | Negligible |
| **SA run on 2 GPUs (MirroredStrategy)** | BatchNorm statistics are computed over 8 images per GPU instead of 16. This is a real, small protocol deviation for the SA run only. It also gave **no speed-up** (467 vs 473 s/epoch), so future runs should use 1 GPU. |

### Why the SA improvement did not reproduce
- The paper's +3.2 pts compares **one run of each model on two different unseeded test sets**. Our two runs differ by +0.4 pts in weighted precision, and the bootstrap intervals overlap almost completely (0.887–0.932 vs 0.891–0.935).
- The run-to-run variance of a single configuration (Cause 3) is unmeasured. A single pair of runs cannot establish a 3-point gain.
- SA did change one thing in our runs: **melanoma recall went from 15 to 21 of 34**, with higher melanoma precision too. That is suggestive, but the bootstrap intervals overlap (0.27–0.62 vs 0.44–0.78), so one run of each cannot establish it.
- **Honest conclusion:** under the paper's own protocol we cannot distinguish IRv2+SA from IRv2. Showing an effect would need several seeds.

---

## 5. What the gap is *not*
- **Not** a split bug. The sizes, the per-class test counts and the "no lesion in both train and test" assertion all match.
- **Not** a different architecture. Both parameter counts are identical to the authors'.
- **Not** an improvement over the paper. Our small gains are within noise.

## 6. Caveat that applies to the paper and to us
Under `PROTOCOL="paper"` **the test set is also the validation set**: it drives checkpoint selection and early stopping. All numbers above, both the paper's and ours, are therefore **optimistically biased**. `PROTOCOL="clean"` (a lesion-grouped validation split) gives the honest estimate.

## 7. Consequences for Milestones 3–5
1. **Our reproduced IRv2 baseline, not the paper's number, is the reference** for our hypothesis.
2. **Melanoma recall is the weakest clinically important metric:** 0.44 in our baseline, with 9 melanomas called nv and 7 called bkl. This is the target of our hypothesis.
3. Because run-to-run variance is unmeasured and each melanoma is ~3 points of recall, **the hypothesis must be tested with ≥ 2–3 seeds** and judged by mean ± std, not one number.
4. Use 1 GPU for all further runs, so BatchNorm behaves as in the paper.

---

### One-paragraph version for the report / PPT

> We reproduced Datta et al.'s pipeline on HAM10000 using the authors' code, ported to TensorFlow 2.20 / Keras 3. We matched the split (9,187 / 828), the augmentation (51,699 images) and both parameter counts exactly. The Inception-ResNet-v2 baseline reproduces: accuracy 0.916 (paper 0.912), weighted precision 0.909 (0.906), weighted AUC 0.978 (0.983). With Soft-Attention we obtain weighted precision 0.913 and AUC 0.981, against the paper's 0.938 and 0.984. The paper's +3.2-point gain from attention therefore shrinks to +0.4 points, well inside the bootstrap interval of either model. We attribute the gaps to an unseeded random test split in the original code, very small minority classes (6 df, 10 vasc), and checkpoint selection on the test set under a noisy lr = 0.01 schedule. Melanoma recall (0.44) and the absorption of benign-keratosis and melanoma lesions into the nevus class are the main weaknesses of the baseline.

*No clinical claims are made.*
