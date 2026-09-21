# Feature Importance

## Predictors

The model-training workflow uses five Sentinel-1-derived predictors:

- VV
- VH
- VV–VH Difference
- VV/VH Ratio
- Normalized Polarisation Index (NPI)

## Reported values

| Feature | Logistic Regression | Random Forest | ExtraTrees | SVC |
|---|---:|---:|---:|---:|
| VH | 1.80 | 0.40 | 0.43 | 0.11 |
| VV | 1.74 | 0.37 | 0.39 | 0.08 |
| NPI | 0.43 | 0.08 | 0.06 | 0.00 |
| VV–VH Difference | 0.30 | 0.08 | 0.06 | 0.00 |
| VV/VH Ratio | 0.19 | 0.08 | 0.06 | 0.01 |

These values are reproduced from the thesis feature-importance comparison.

The table is presented as a cross-model comparison. Importance values depend on the model and importance measure and should not be interpreted as a universal ranking of Sentinel-1 variables.

## Feature engineering

The supplied workflow derives:

```text
VV–VH Difference = VV - VH
VV/VH Ratio = linear(VV) / linear(VH)
NPI = (linear(VV) - linear(VH)) /
      (linear(VV) + linear(VH))
```

The input VV and VH values are treated as dB and converted to linear scale where required for the ratio and NPI calculations.

## Related figure

![Feature importance](../figures/feature_importance.png)
