# Methodology and Code Notes

This document records what the supplied scripts actually implement and identifies items that should be confirmed against the final thesis before public release.

## 1. Sentinel-1 preprocessing

The supplied batch script implements the following chain:

1. Apply Orbit File
2. Remove GRD Border Noise
3. Thermal Noise Removal
4. Calibration to Sigma0 for VV/VH
5. Terrain Correction using SRTM 3Sec and 10 m pixel spacing
6. Conversion of Sigma0 VV/VH to dB using `10*log10(...)`
7. GeoTIFF-BigTIFF export

The supplied single-scene script implements the same broad chain **except that it does not call `Remove-GRD-Border-Noise`**. Because the source scripts are the requested basis, the cleaned repository does not silently add or remove this step from the single-scene implementation.

### Projection

Both supplied SNAP scripts use the SNAP parameter value `AUTO:42001`. The repository preserves that value. If the final thesis specifies a fixed CRS such as a particular UTM EPSG code, this should be reconciled before publication rather than inferred from the code.

### SNAP Python module

The supplied single-scene script imports `snappy`, while the supplied batch script imports `esa_snappy`. The cleaned repository preserves these imports in their respective scripts and documents the difference. A future release should standardise this after confirming the actual SNAP installation used for the thesis.

## 2. Feature engineering

The supplied model-training script uses five Sentinel-1 predictors:

- `VV`
- `VH`
- `VV_VH_diff = VV - VH` in dB
- `VV_VH_ratio` computed after converting VV and VH from dB to linear scale
- `NPI = (VV_linear - VH_linear) / (VV_linear + VH_linear + epsilon)`

The script assumes VV and VH inputs are already in dB.

## 3. Model training

The supplied script uses four classifiers:

- Logistic Regression
- Random Forest
- SVC with RBF kernel and probability estimates
- ExtraTrees

The data are separated by `plot_id`, with default held-out plots `1` and `5`. Only the training plots enter model fitting and GroupKFold hyperparameter tuning.

The tuning objective is recall for Class 1 (galamsey), implemented with `recall_score(pos_label=1)`.

### Important evaluation limitation

The source script creates `X_test` and `y_test`, prints the held-out sample count, and then does not calculate predictions or test metrics. Therefore, this repository does not claim held-out accuracy, precision, recall, F1, ROC-AUC, or confusion-matrix results from this script.

Any performance numbers shown in the portfolio should be attributed to the thesis results or another explicitly evaluated workflow, not to this training script unless the evaluation code is added and run.

## 4. Hotspot evolution

The supplied annual-stack script divides the 2015–2025 sequence as follows:

- Early period: 2015–2019
- Late period: 2020–2025
- Recent period: 2022–2025

Definitions are preserved exactly from the supplied script:

- **Persistent:** activity in at least 8 of 11 years.
- **Emerging:** no activity in 2015–2019 and activity in at least 2 years in 2020–2025.
- **Abandoned:** activity in at least 2 years in 2015–2019 and no activity in 2022–2025.

The combined class uses priority ordering inherited from the source code: persistent overwrites emerging, and emerging overwrites abandoned where masks overlap.

The cleaned implementation adds checks that the input set contains exactly one raster for every year 2015–2025 and that all rasters have matching dimensions, CRS, and affine transform. These checks do not alter the classification definitions.

## 5. Data handling

Raw Sentinel-1 products, credentials, generated rasters, and trained model binaries should not be committed to the public repository by default. The code expects these resources to be supplied locally.

## 6. Items to confirm before GitHub publication

- Confirm whether border-noise removal should be present in both single-scene and batch workflows.
- Confirm the intended SNAP Python bridge: `snappy` or `esa_snappy`.
- Confirm the intended map projection/CRS represented by `AUTO:42001` in the actual thesis workflow.
- Confirm whether a separate thesis evaluation script/notebook exists for the held-out plots; if so, add it as an explicit evaluation component rather than modifying the training script's historical behavior.
- Confirm exact data/licensing/attribution requirements for the satellite data and external DEM.
