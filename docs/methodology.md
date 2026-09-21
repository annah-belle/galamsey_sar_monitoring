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

## 6. Items to verify in future revisions

- Confirm whether border-noise removal should be present in both single-scene and batch workflows.
- Confirm the intended SNAP Python bridge: `snappy` or `esa_snappy`.
- Confirm the intended map projection/CRS represented by `AUTO:42001` in the actual thesis workflow.
- Confirm whether a separate thesis evaluation script/notebook exists for the held-out plots; if so, add it as an explicit evaluation component rather than modifying the training script's historical behavior.
- Confirm exact data/licensing/attribution requirements for the satellite data and external DEM.

## 7. Reproducibility and methodological limitations

This repository is intended to document and support reproduction of the project workflow while keeping large source datasets and external software dependencies outside GitHub.

### Data availability

The full Sentinel-1 GRD archive used in the thesis is not distributed with this repository because of its size. Sentinel-1 products should be obtained from the Copernicus Data Space Ecosystem or another appropriate source and supplied locally.

Sentinel-2 data were used as reference information for sample labelling and interpretation. They are not used as predictor variables in the Sentinel-1 machine-learning workflow documented here.

### Spatial validation

Reference samples are associated with spatially defined plots. Plots 1 and 5 are designated as held-out plots in the supplied training workflow, while the remaining plots are used for model fitting and GroupKFold hyperparameter tuning.

The current training script creates the held-out test subset but does not calculate predictions or held-out performance metrics. Consequently, the repository does not claim held-out accuracy, precision, recall, F1, ROC-AUC, or confusion-matrix results from that script.

The thesis analysis documents a major spatial-generalisation limitation on the held-out Plot 5. This result should therefore be considered when interpreting model transferability beyond the training plots.

### Temporal comparability

The temporal analysis uses a Common Monitoring Footprint (CMF) derived from the June Sentinel-1 acquisitions for 2015–2025. Restricting analysis to a common spatial footprint supports comparison across years by reducing differences caused by changing spatial coverage.

The hotspot-evolution workflow expects exactly one compatible annual raster for each year from 2015 through 2025. The cleaned implementation additionally checks raster dimensions, CRS, and affine transform before combining the annual layers.

### Workflow differences and implementation notes

The single-scene and batch Sentinel-1 SNAP scripts are intentionally preserved as separate workflows. The batch workflow includes GRD border-noise removal, whereas the supplied single-scene workflow does not.

The scripts also preserve the different SNAP Python module imports present in the supplied source material (`snappy` and `esa_snappy`). These differences should be standardised only after confirming the SNAP installation and workflow used for the thesis.

The SNAP terrain-correction scripts preserve the `AUTO:42001` parameter from the supplied implementation. The intended projection should be confirmed against the final thesis workflow before changing this value.

### Interpretation of portfolio results

Performance values presented in the portfolio are documented as thesis results or results from explicitly evaluated analysis components. They should not be interpreted as newly reproduced model metrics unless the corresponding evaluation workflow has been executed from the supplied data and code.

The repository therefore distinguishes between:
- documented thesis results,
- workflow code preserved from the project,
- portfolio visualisations based on documented results, and
- analyses that require the original source data and external software environment for full reproduction.
