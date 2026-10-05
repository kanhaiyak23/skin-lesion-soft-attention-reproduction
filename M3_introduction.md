# Introduction (for the report, IEEE format)

*Drop-in replacement for Section I of the literature-survey report. Reference numbers follow the existing report ([1]–[16]); one new reference, [17], is added at the end. A LaTeX version is at the bottom of this file.*

---

## I. INTRODUCTION

Skin cancer is among the most commonly diagnosed cancers worldwide. Melanoma accounts for only a small fraction of skin-cancer cases but for most skin-cancer deaths, and survival depends strongly on the stage at which it is detected [1]. Dermoscopy, a non-invasive magnified imaging technique, improves diagnostic accuracy over naked-eye examination. However, its interpretation is subjective and requires substantial training: in a reader study, 58 dermatologists reached a mean melanoma sensitivity of only 86.6 % [4]. This has motivated a decade of work on automated classification of dermatoscopic images, beginning with the demonstration that a single ImageNet-pretrained convolutional neural network (CNN) can match board-certified dermatologists on binary tasks [1].

For multi-class lesion classification, the HAM10000 dataset [2] has become the standard public benchmark. It contains 10,015 dermatoscopic images in seven diagnostic categories. Two properties of HAM10000 make published results harder to interpret than headline numbers suggest.

- **Severe class imbalance.** Melanocytic nevi make up 66.9 % of the images, while dermatofibroma and vascular lesions make up about 1 % each. A classifier that ignores melanoma entirely can still exceed 66 % accuracy, so accuracy and accuracy-weighted metrics are misleading.
- **Repeated lesions.** The 10,015 images show only 7,470 distinct lesions. Unless the data are split by lesion, near-duplicate views of the same lesion leak between training and test sets [12].

Several highly cited results do not survive these checks. Some report accuracy on small or pre-augmented test subsets [13], and at least one has been retracted [16].

Against this background, we chose to reproduce rather than to propose a new architecture. Our base paper is Datta *et al.* [8]. They insert a Soft-Attention module into ImageNet-pretrained backbones and report that Inception-ResNet-v2 with Soft-Attention reaches 93.7 % weighted precision and 98.4 % AUC on HAM10000, 3.2 points above the same backbone without attention. We selected it because:

- it is evaluated on HAM10000;
- it is the only surveyed work whose public code contains both the baseline and the proposed model;
- its split protocol draws the test set from single-image lesions, which avoids lesion-level leakage.

Our reproduction exposes three properties of the original protocol that the paper does not state:

- the train/test split is unseeded;
- the test set also serves as the validation set, for checkpoint selection and early stopping;
- the code applies an undocumented five-fold class weight to melanoma.

Using the authors' code ported to a current TensorFlow release, with an identical split procedure, augmentation pipeline and parameter count, we reproduce the baseline within 0.01 on accuracy, weighted precision and weighted AUC (0.916, 0.909 and 0.978, against 0.912, 0.906 and 0.983). The Soft-Attention model reaches 0.913 weighted precision. The reported 3.2-point gain therefore shrinks to 0.4 points, which lies inside the run-to-run variability of a single configuration.

More importantly, the reproduction shows where the baseline actually fails. Despite more than 90 % accuracy, it detects only 15 of the 34 test melanomas (recall 0.44). Its dominant errors are benign-keratosis and melanoma lesions misclassified as nevi. Because the training set is already balanced by oversampling, we attribute this not to class frequency but to the large number of already well-classified examples dominating the cross-entropy gradient.

We therefore hypothesise that **replacing cross-entropy with focal loss** [17] (γ = 2, all other settings unchanged) will raise melanoma recall from 0.44 to 0.50–0.60 and macro recall by 0.02–0.05, while leaving AUC essentially unchanged and weighted precision unchanged or slightly lower.

The contributions of this work are:

1. **A verified reproduction** of Datta *et al.* [8] on HAM10000, with exact agreement in split sizes, augmented-set size and model parameter counts, and an itemised account of every porting change;
2. **An analysis of the reproduction gap**, showing that per-class differences are a few images each, and that the claimed benefit of Soft-Attention is not distinguishable from noise under the original protocol;
3. **A single-variable hypothesis**, stated with its predicted direction, magnitude and mechanism *before* implementation, and tested with multiple seeds against our own reproduced baseline;
4. **Reporting that follows clinical relevance:** per-class sensitivity, specificity, AUC and bootstrap confidence intervals rather than accuracy alone.

No clinical claims are made. All results are benchmark measurements on a curated, two-centre research dataset.

The remainder of this paper is organised as follows. Section II reviews related work. Section III describes the dataset, the reproduced method and our proposed change. Section IV reports the reproduction and ablation results. Section V analyses failure cases and the reproduction gap, and Section VI discusses limitations before concluding.

---

**New reference to add:**
[17] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 2980–2988.

**Correction to existing reference [8]:** the second author is **S. M. A. Hashemi**, not M. A. Shaikh:
[8] S. K. Datta, S. M. A. Hashemi, S. N. Srihari, and M. Gao, "Soft-attention improves skin cancer classification performance," in *Interpretability of Machine Intelligence in Medical Image Computing (iMIMIC), MICCAI Workshops*, LNCS vol. 12929, Springer, 2021, pp. 13–23.

---

## LaTeX version (IEEEtran)

```latex
\section{Introduction}
Skin cancer is among the most commonly diagnosed cancers worldwide. Melanoma accounts for only a small fraction of skin-cancer cases but for most skin-cancer deaths, and survival depends strongly on the stage at which it is detected~\cite{esteva2017}. Dermoscopy improves diagnostic accuracy over naked-eye examination, but its interpretation is subjective and requires substantial training: in a reader study, 58 dermatologists reached a mean melanoma sensitivity of only 86.6\%~\cite{haenssle2018}. This has motivated a decade of work on automated classification of dermatoscopic images, beginning with the demonstration that a single ImageNet-pretrained convolutional neural network (CNN) can match board-certified dermatologists on binary tasks~\cite{esteva2017}.

For multi-class lesion classification, the HAM10000 dataset~\cite{tschandl2018} has become the standard public benchmark. It contains 10{,}015 dermatoscopic images in seven diagnostic categories. Two properties of HAM10000 make published results harder to interpret than headline numbers suggest. First, it is severely imbalanced: melanocytic nevi make up 66.9\% of the images, while dermatofibroma and vascular lesions make up about 1\% each. A classifier that ignores melanoma entirely can still exceed 66\% accuracy, so accuracy and accuracy-weighted metrics are misleading. Second, the 10{,}015 images show only 7{,}470 distinct lesions. Unless the data are split by lesion, near-duplicate views leak between training and test sets~\cite{abhishek2025}. Several highly cited results do not survive these checks: some report accuracy on small or pre-augmented test subsets~\cite{shetty2022}, and at least one has been retracted~\cite{densenet2_retraction}.

Against this background, we chose to reproduce rather than to propose a new architecture. Our base paper is Datta \emph{et al.}~\cite{datta2021}. They insert a Soft-Attention module into ImageNet-pretrained backbones and report that Inception-ResNet-v2 with Soft-Attention reaches 93.7\% weighted precision and 98.4\% AUC on HAM10000, 3.2 points above the same backbone without attention. We selected it because it is evaluated on HAM10000, because it is the only surveyed work whose public code contains both the baseline and the proposed model, and because its test set is drawn from single-image lesions, which avoids lesion-level leakage. Our reproduction exposes three properties of the original protocol that the paper does not state: the train/test split is unseeded, the test set also serves as the validation set for checkpoint selection and early stopping, and the code applies an undocumented five-fold class weight to melanoma.

Using the authors' code ported to a current TensorFlow release, with an identical split procedure, augmentation pipeline and parameter count, we reproduce the baseline within 0.01 on accuracy, weighted precision and weighted AUC (0.916, 0.909 and 0.978, against 0.912, 0.906 and 0.983). The Soft-Attention model reaches 0.913 weighted precision. The reported 3.2-point gain therefore shrinks to 0.4 points, which lies inside the run-to-run variability of a single configuration. More importantly, the reproduction shows where the baseline actually fails. Despite more than 90\% accuracy, it detects only 15 of the 34 test melanomas (recall 0.44), and its dominant errors are benign-keratosis and melanoma lesions misclassified as nevi. Because the training set is already balanced by oversampling, we attribute this not to class frequency but to the large number of already well-classified examples dominating the cross-entropy gradient. We therefore hypothesise that replacing cross-entropy with focal loss~\cite{lin2017focal} ($\gamma=2$, all other settings unchanged) will raise melanoma recall from 0.44 to 0.50--0.60 and macro recall by 0.02--0.05, while leaving AUC essentially unchanged and weighted precision unchanged or slightly lower.

The contributions of this work are:
\begin{enumerate}
  \item a verified reproduction of~\cite{datta2021} on HAM10000, with exact agreement in split sizes, augmented-set size and model parameter counts, and an itemised account of every porting change;
  \item an analysis of the reproduction gap, showing that per-class differences are a few images each and that the claimed benefit of Soft-Attention is not distinguishable from noise under the original protocol;
  \item a single-variable hypothesis, stated with its predicted direction, magnitude and mechanism before implementation, and tested with multiple seeds against our own reproduced baseline;
  \item reporting that follows clinical relevance: per-class sensitivity, specificity, AUC and bootstrap confidence intervals rather than accuracy alone.
\end{enumerate}
No clinical claims are made; all results are benchmark measurements on a curated, two-centre research dataset. Section~II reviews related work, Section~III describes the dataset, the reproduced method and our proposed change, Section~IV reports the reproduction and ablation results, Section~V analyses failure cases and the reproduction gap, and Section~VI discusses limitations before concluding.
```
