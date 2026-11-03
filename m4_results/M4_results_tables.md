## 1. Per-run results (arg-max)

| Loss | Seed | Accuracy | W. precision | Macro recall | mel found | mel recall (95% Wilson) | mel spec. | mel AUC | W. AUC |
|---|---|---|---|---|---|---|---|---|---|
| ce | 7 | 0.924 | 0.930 | 0.808 | 26/34 | 0.765 (0.60–0.88) | 0.976 | 0.951 | 0.979 |
| ce | 42 | 0.918 | 0.914 | 0.735 | 16/34 | 0.471 (0.31–0.63) | 0.977 | 0.957 | 0.984 |
| ce | 2024 | 0.924 | 0.921 | 0.722 | 19/34 | 0.559 (0.39–0.71) | 0.987 | 0.965 | 0.986 |
| focal | 7 | 0.860 | 0.906 | 0.786 | 26/34 | 0.765 (0.60–0.88) | 0.929 | 0.943 | 0.966 |
| focal | 42 | 0.825 | 0.887 | 0.651 | 25/34 | 0.735 (0.57–0.85) | 0.895 | 0.916 | 0.963 |
| focal | 2024 | 0.755 | 0.878 | 0.616 | 26/34 | 0.765 (0.60–0.88) | 0.841 | 0.923 | 0.952 |

## 2. Mean over seeds (± std where there are ≥ 2 seeds)

| Loss | Seeds | accuracy | w_precision | macro_recall | mel_recall | mel_specificity | w_auc |
|---|---|---|---|---|---|---|---|
| ce | 7, 42, 2024 | 0.922 ± 0.003 | 0.922 ± 0.008 | 0.755 ± 0.046 | 0.598 ± 0.151 | 0.980 ± 0.006 | 0.983 ± 0.004 |
| focal | 7, 42, 2024 | 0.813 ± 0.054 | 0.890 ± 0.014 | 0.684 ± 0.090 | 0.755 ± 0.017 | 0.889 ± 0.044 | 0.961 ± 0.008 |

## 3. Paired comparison, CE vs focal on the same test set (exact McNemar)

| Seed | Images | only CE right | only focal right | p (all) | mel: only CE found | mel: only focal found | p (mel) |
|---|---|---|---|---|---|---|---|
| 7 | 828 | 75 | 22 | 0.000 | 1 | 1 | 1.000 |
| 42 | 828 | 104 | 27 | 0.000 | 5 | 14 | 0.064 |
| 2024 | 828 | 161 | 21 | 0.000 | 4 | 11 | 0.118 |
| **pooled** | | | | | 10 | 26 | **0.011** |

## 4. H2: cross-fitted melanoma threshold (5-fold × 20 repeats, threshold never chosen on the images it scores)

| Loss | Seed | Operating point | t (mean ± std) | mel found | mel spec. | W. precision |
|---|---|---|---|---|---|---|
| ce | 7 | arg-max | — | 26/34 | 0.976 | 0.930 |
| ce | 7 | spec ≥ 0.95 | 0.130 ± 0.026 | 27.1/34 = 0.799 | 0.950 | 0.923 |
| ce | 7 | spec ≥ 0.90 | 0.036 ± 0.004 | 28.0/34 = 0.824 | 0.899 | 0.919 |
| ce | 7 | spec ≥ 0.713 (dermatologists) | 0.001 ± 0.000 | 32.0/34 = 0.943 (derm. sens. 0.866) | 0.713 | 0.922 |
| ce | 42 | arg-max | — | 16/34 | 0.977 | 0.914 |
| ce | 42 | spec ≥ 0.95 | 0.078 ± 0.022 | 24.4/34 = 0.716 | 0.950 | 0.922 |
| ce | 42 | spec ≥ 0.90 | 0.005 ± 0.002 | 28.6/34 = 0.840 | 0.899 | 0.921 |
| ce | 42 | spec ≥ 0.713 (dermatologists) | 0.000 ± 0.000 | 34.0/34 = 1.000 (derm. sens. 0.866) | 0.713 | 0.950 |
| ce | 2024 | arg-max | — | 19/34 | 0.987 | 0.921 |
| ce | 2024 | spec ≥ 0.95 | 0.048 ± 0.013 | 23.6/34 = 0.694 | 0.950 | 0.919 |
| ce | 2024 | spec ≥ 0.90 | 0.004 ± 0.001 | 31.4/34 = 0.922 | 0.899 | 0.928 |
| ce | 2024 | spec ≥ 0.713 (dermatologists) | 0.000 ± 0.000 | 34.0/34 = 1.000 (derm. sens. 0.866) | 0.713 | 0.945 |
| focal | 7 | arg-max | — | 26/34 | 0.929 | 0.906 |
| focal | 7 | spec ≥ 0.95 | 0.456 ± 0.011 | 22.4/34 = 0.657 | 0.949 | 0.899 |
| focal | 7 | spec ≥ 0.90 | 0.301 ± 0.017 | 27.0/34 = 0.794 | 0.900 | 0.902 |
| focal | 7 | spec ≥ 0.713 (dermatologists) | 0.067 ± 0.004 | 33.0/34 = 0.971 (derm. sens. 0.866) | 0.713 | 0.912 |
| focal | 42 | arg-max | — | 25/34 | 0.895 | 0.887 |
| focal | 42 | spec ≥ 0.95 | 0.520 ± 0.006 | 16.1/34 = 0.474 | 0.950 | 0.878 |
| focal | 42 | spec ≥ 0.90 | 0.426 ± 0.008 | 25.1/34 = 0.740 | 0.899 | 0.885 |
| focal | 42 | spec ≥ 0.713 (dermatologists) | 0.121 ± 0.006 | 32.0/34 = 0.943 (derm. sens. 0.866) | 0.712 | 0.885 |
| focal | 2024 | arg-max | — | 26/34 | 0.841 | 0.878 |
| focal | 2024 | spec ≥ 0.95 | 0.616 ± 0.010 | 19.3/34 = 0.568 | 0.949 | 0.866 |
| focal | 2024 | spec ≥ 0.90 | 0.511 ± 0.010 | 25.0/34 = 0.735 | 0.899 | 0.872 |
| focal | 2024 | spec ≥ 0.713 (dermatologists) | 0.136 ± 0.009 | 33.4/34 = 0.981 (derm. sens. 0.866) | 0.712 | 0.887 |

## 5. H1 verdict (M3 §4 rule + paired test, see the M3 addendum)

M3 §4 rule: focal mean − CE mean = +5.3 melanomas, CE std = 5.1 -> met.

Melanomas found, mean per seed: CE 20.3, focal 25.7. Pooled McNemar on melanomas: p = 0.011.

**INCONCLUSIVE (paired rule): focal finds more melanomas on average, but not on every seed or not significantly.**

M3 §4 macro-recall criterion: CE 0.755, focal 0.684 -> **does not improve: reject under §4**. Weighted precision change: -0.031.
