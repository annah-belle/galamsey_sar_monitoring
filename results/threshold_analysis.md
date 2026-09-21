# Probability Threshold Analysis

## Overview

The classification workflow produces continuous probability outputs for galamsey occurrence. The thesis evaluates probability cut-offs using accuracy, precision, recall and F1-score.

## Selected monitoring threshold

**P(galamsey) ≥ 0.70**

Pixels at or above 0.70 are classified as galamsey (`1`); pixels below 0.70 are classified as non-galamsey (`0`).

## Best threshold by model

| Model | Best threshold | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7 | 0.851 | 0.806 | 0.924 | 0.861 |
| Random Forest | 0.8 | 0.860 | 0.816 | 0.929 | 0.869 |
| SVC | 0.6 | 0.849 | 0.790 | 0.951 | 0.863 |
| ExtraTrees | 0.7 | 0.858 | 0.816 | 0.924 | 0.867 |

## Performance at 0.70

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.851 | 0.806 | 0.924 | 0.861 |
| Random Forest | 0.851 | 0.799 | 0.938 | 0.863 |
| SVC | 0.847 | 0.805 | 0.916 | 0.857 |
| ExtraTrees | 0.858 | 0.816 | 0.924 | 0.867 |

## Probability to binary workflow

```text
P(galamsey) ≥ 0.70  →  Galamsey (1)
P(galamsey) < 0.70  →  Non-galamsey (0)
```

The thesis retains the continuous probability maps while also generating binary outputs. Scene-level outputs are clipped to the area of interest and aggregated into monthly probability and binary composites before the spatially consistent CMF-based temporal analysis.

## Related figure

![Threshold analysis](../figures/threshold_analysis.png)
