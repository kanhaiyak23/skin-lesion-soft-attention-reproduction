# Base Paper Selection — Milestone 2 (Reproduction)

**Project:** Deep Learning for Multi-Class Skin Lesion Classification on HAM10000
**Team:** Navnit Naman (230085), Kanhaiya Kumar (230062) — Newton School of Technology, Rishihood University
**Topic / dataset (from the project brief):** Skin lesion / melanoma → HAM10000 → Classification

---

## 1. Selected base paper

> S. K. Datta, S. M. A. Hashemi, S. N. Srihari, M. Gao,
> **"Soft-Attention Improves Skin Cancer Classification Performance,"**
> *Interpretability of Machine Intelligence in Medical Image Computing (iMIMIC), MICCAI 2021 Workshops*, LNCS vol. 12929, Springer, 2021, pp. 13–23.
> arXiv: https://arxiv.org/abs/2105.03358
> Code: https://github.com/skrantidatta/Attention-based-Skin-Cancer-Classification

**In one line:** a transfer-learning CNN (ImageNet-pretrained Inception-ResNet-v2) fine-tuned on HAM10000 for 7-class classification, with and without an inserted Soft-Attention block. The public code includes both versions.

---

## 2. Reproducibility checklist (format from the project brief, Stage 2)

| Checklist item | Our answer |
|---|---|
| **Public code?** | **Yes.** The GitHub repo is linked in the paper. It has Jupyter notebooks for all 5 backbones *without* Soft-Attention (VGG16, ResNet34, ResNet50, IRv2, DenseNet201), the same 5 *with* Soft-Attention, a Grad-CAM notebook, and IRv2+SA at 15/20/30 % test splits. |
| **Dataset public & size?** | **Yes.** HAM10000 has 10,015 dermatoscopic JPEGs (600×450) and 7 classes. It is free for research from the Harvard Dataverse and mirrored on Kaggle (`kmader/skin-cancer-mnist-ham10000`). The metadata CSV includes `lesion_id`, which the split relies on. |
| **Fits free GPU?** | **Yes, with care.** IRv2 runs at 299×299 with batch size 16 (47.5 M params). The authors' log shows 919 steps/epoch and about 146 s/epoch on their GPU, with early stopping at about epoch 58, so roughly 2.5 h. A free T4 will be slower, so expect about 3–5 h for one run. **Kaggle (P100/T4, 12 h sessions) is safer than Colab**, which disconnects. If compute runs out, the cheaper fallback is the ResNet50 baseline at 224×224 from the same repo. |
| **Last commit / usable?** | **Usable with porting.** The last push was Nov 2023 (48★, 26 forks). The code targets TF 2.4 / standalone Keras (`from keras.layers import Layer, InputSpec`, `K.` backend). Current Colab/Kaggle images ship newer TF/Keras, so we expect small import and API fixes (and `irv2.layers[-28]` must be re-checked). We will document every change. |
| **Metric defined?** | **Yes.** The paper defines precision, accuracy, sensitivity (recall), specificity and AUC. The HAM10000 headline metrics are **weighted-average precision** and **weighted one-vs-rest AUC**. The notebooks also print per-class precision/recall/F1, macro averages, and per-class AUC. |

**Overall:** reproducible. The gaps are known and listed in §5.

---

## 3. Why this paper (and not the others in our survey)

We scored all 10 surveyed papers against 6 criteria (from our Milestone 1 report, §IV-A):

| Criterion | Datta 2021 (chosen) | Ali 2021 (EfficientNet) | Yao 2022 (TMI) | Chaturvedi 2020 (MobileNet) | Shetty 2022 |
|---|:-:|:-:|:-:|:-:|:-:|
| C1 Evaluated on HAM10000 | ✅ | ✅ | ~ (via ISIC-18) | ✅ | ~ (280-img subset) |
| C2 Official public code | ✅ | ❌ | ✅ | partial | ❌ |
| C3 Baseline code released too | ✅ **(only one)** | ❌ | ❌ | ❌ | ❌ |
| C4 Reports more than accuracy (precision/sens./AUC) | ✅ | ~ (F1) | ✅ | ~ | ❌ |
| C5 Trainable on free GPU | ✅ (with care) | ✅ | ❌ (heavy schedule) | ✅ | ✅ |
| C6 Clear axis for one controlled change | ✅ | ✅ | ~ | ✅ | ❌ |

**Main reasons:**

1. **It matches the project brief exactly.** It is a transfer-learning CNN on the HAM10000 dataset assigned to our topic, it uses a downloadable dataset, and it has public code.
2. **Only one surveyed paper releases the baseline *and* the method.** The brief says *"First task is replication of the selected base paper for baseline formation."* Because this repo includes the plain IRv2 notebook (no attention), we can reproduce the real baseline, not a guess at it, and then check the claimed +3.2 % gain.
3. **It reports metrics beyond accuracy.** The brief requires sensitivity/recall and AUC ("not just accuracy"). The paper reports AUC and precision, and the code prints per-class recall.
4. **The protocol avoids the worst leakage trap.** The test set is built only from lesions that have a single image (`lesion_id` grouped), so the same mole never appears in both train and test. Most HAM10000 papers miss this.
5. **It leaves a clear opening for our hypothesis (Milestone 3).** The baseline is weak on exactly the classes that matter clinically: melanoma recall 0.56 and dermatofibroma recall 0.33 (see §4). One controlled change, such as the loss function, class weighting or attention, can be tested against that.
6. **It supports the later milestones.** The Grad-CAM notebook covers the failure analysis in Milestone 5. The 5-backbone × {±SA} grid gives a ready-made ablation structure for Milestone 4.

**Why not the alternatives:**
- **Ali et al. 2021:** a clean study, but it has no official code, so we would be reimplementing rather than reproducing. We keep it as our fallback.
- **Yao et al. 2022:** the best code, but it uses ISIC configurations and a training schedule beyond our compute budget.
- **Gessert et al. 2020:** an ensemble of multi-resolution EfficientNets that is far out of budget.
- **Shetty et al. 2022:** its 95.18 % comes from a 280-image test set with augmentation applied before the split, so the number is not a valid target.
- **DenseNet-II (Soft Computing 2022):** retracted, so it is excluded.

---

## 4. Target numbers to reproduce (HAM10000, 85/15 split, 828 test images)

From the paper (Table 1 / Table 3) and the authors' own notebook output:

| Model | W.Avg Precision | W.Avg AUC | Accuracy | Macro Precision | Macro Recall | Macro AUC |
|---|---|---|---|---|---|---|
| **IRv2 baseline (no SA)** — *our Milestone 2 target* | **0.905** | **0.982–0.983** | 0.912 | 0.833 | 0.689 | 0.985 |
| IRv2 + Soft-Attention (paper's best) | **0.937** | **0.984** | 0.934 | 0.892 | — | 0.984 |

Per-class baseline (IRv2, authors' notebook), the classes our hypothesis will target:

| Class | Support | Precision | **Recall** | AUC |
|---|---|---|---|---|
| akiec | 23 | 0.83 | 0.65 | 0.993 |
| bcc | 26 | 0.85 | 0.85 | 0.997 |
| bkl | 66 | 0.85 | **0.44** | 0.971 |
| df | 6 | 0.67 | **0.33** | 0.983 |
| **mel** | 34 | 0.70 | **0.56** | 0.966 |
| nv | 663 | 0.93 | 0.99 | 0.984 |
| vasc | 10 | 1.00 | 1.00 | 1.000 |

> **Note on "4.7 %" vs "3.2 %":** the abstract's +4.7 % compares against an *external* baseline (0.890 precision, ref [16] in the paper). The +3.2 % compares against the authors' *own* IRv2 without attention (0.905 → 0.937). The 3.2 % figure is the one we reproduce.

**Success criterion (per the brief):** land *near* 0.905 weighted precision / 0.98 AUC for the baseline. We will record any gap and analyse it in Stage 6.

---

## 5. Exact protocol (verified from the repo code, not only the paper text)

| Step | What the code does |
|---|---|
| Split | Group by `lesion_id`, keep lesions with exactly one image, then take a stratified 15 % → **828 test** images. Everything else (**9,187**) is train. `train_test_split` has **no `random_state`**. |
| Offline augmentation | Per class, `ImageDataGenerator` (rotation 180°, shift 0.1, zoom 0.1, h/v flip) is written to disk up to about 8,000 images/class, giving **51,699** train images. Augmentation runs *after* the split (correct). |
| Input | 299×299, `inception_resnet_v2.preprocess_input` |
| Model (baseline) | ImageNet IRv2, cut at `layers[-28]` (8×8×192 feature map), then ReLU → Dropout 0.5 → Flatten → Dense(7, softmax). All layers trainable. |
| Optimiser | Adam, lr = 0.01, ε = 0.1; categorical cross-entropy |
| Class weights | **mel = 5.0**, others 1.0 (in the code only, not mentioned in the paper) |
| Schedule | 150 epochs max, `steps_per_epoch = len(train_df)/10` ≈ 919, batch 16, EarlyStopping(val_loss, patience 30) |
| Checkpoint | Best `val_accuracy` |

### Gaps and issues we found (inputs to Milestone 3 and Stage 6)

1. **The test set is also the validation set.** `validation_data=test_batches` drives both the checkpoint choice (best val_accuracy) and early stopping. The reported numbers are therefore picked using the test set, which is optimistic. We will reproduce the protocol faithfully first, then report a clean version with a separate lesion-grouped validation split.
2. **No random seed** for the split or training. Each run uses a different test set. We will fix seeds and report mean ± std over 3 runs if time allows.
3. **The paper does not mention the undocumented class weight (mel × 5).** We keep it for a faithful reproduction.
4. **Weighted averages hide the minority classes.** nv is 663 of 828 test images (80 %), so weighted precision mostly measures nv. We will report **macro recall and per-class recall** as primary metrics, as the brief requires.
5. **The test set is tiny for rare classes:** 6 df and 10 vasc images. A single image changes df recall by about 17 points, so per-class confidence intervals are wide.
6. **No trained weights or pinned environment.** We have to retrain from scratch and port to current TF/Keras.

### Correction to our Milestone 1 report

Our M1 report lists the second author as "M. A. Shaikh". The paper's second author is actually **Seyed Mohammad Abuzar Hashemi**. Shaikh et al. is the earlier handwriting-attention paper that Datta et al. build on. Please fix reference [8] in the report and the PPT.

---

## 6. Plan for Milestone 2 (next step)

1. Download HAM10000 on Kaggle/Colab and check the counts (10,015 images, 7,470 lesions, class distribution).
2. Port `IRV2.ipynb` (baseline) to the current TF/Keras with the same split logic, preprocessing, hyper-parameters and class weights. Fix seeds.
3. Train and report precision/recall/F1 per class, weighted and macro averages, AUC, confusion matrix, and a comparison with the paper's numbers (gap table).
4. If GPU time allows, run `IRV2+SA` to reproduce the 0.937 claim.
5. Use the per-class weaknesses (mel / bkl / df recall) to write the **one-variable hypothesis** for Milestone 3, e.g. focal loss or a class-balanced loss in place of CE + ad-hoc mel weight.

*No clinical claims are made. This is a benchmark reproduction on a research dataset.*
