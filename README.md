# Sentinel-1 SAR Monitoring of Galamsey Activity in Southern Ghana

A thesis-derived geospatial portfolio project demonstrating the use of **Sentinel-1 C-band SAR**, feature engineering, machine learning, and spatial-temporal analysis to investigate galamsey (illegal small-scale gold mining) activity in Southern Ghana.

> **Portfolio scope:** This repository presents research and a reproducible processing framework derived from the MSc Forest Information Technology thesis. It should not be interpreted as a deployed real-time operational monitoring service.

## Project overview

**Research question:** Can Sentinel-1 SAR time-series information support timely detection and monitoring of galamsey activity where optical imagery may be constrained by cloud cover?

The project combines:

- Sentinel-1 GRD C-band SAR data, primarily using VV and VH polarisation.
- SNAP-based preprocessing to calibrated, terrain-corrected backscatter.
- Sentinel-1 feature engineering: VV, VH, VV–VH difference, VV/VH ratio, and Normalized Polarisation Index (NPI).
- Four supervised classifiers: Logistic Regression, Random Forest, Support Vector Classifier, and ExtraTrees.
- Plot-based separation and GroupKFold cross-validation to reduce spatial leakage during model tuning.
- Annual binary maps and a 2015–2025 hotspot-evolution classification.

## Study area

The thesis focused on galamsey hotspots across **Ashanti, Ahafo, Central, Western, and Western North** regions of Ghana.

## Data and workflow

```text
Sentinel-1 GRD
     │
     ▼
SNAP preprocessing
Orbit → Border Noise* → Thermal Noise → Calibration →
Terrain Correction → dB conversion → GeoTIFF
     │
     ▼
VV / VH samples
     │
     ▼
Feature engineering
VV, VH, VV–VH, VV/VH, NPI
     │
     ▼
Plot-based split + GroupKFold CV
     │
     ├── Logistic Regression
     ├── Random Forest
     ├── SVC
     └── ExtraTrees
     │
     ▼
Probability-based classification
     │
     ▼
Annual CMF-restricted binary maps
     │
     ▼
Persistent / Emerging / Abandoned hotspots

* Border-noise removal is present in the supplied batch script but not in the
  supplied single-scene script. See docs/methodology.md.
```

## Repository structure

```text
galamsey_sar_monitoring_repo/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── docs/
│   └── methodology.md
├── figures/
│   └── README.md
├── notebooks/
│   └── README.md
├── results/
│   └── README.md
└── scripts/
    ├── download_s1_cdse_s3.py
    ├── s1_snap_preprocess.py
    ├── s1_snap_batch_preprocess.py
    ├── train_s1_models.py
    └── hotspot_evolution.py
```

## Reproducibility notes

### Sentinel-1 download

`download_s1_cdse_s3.py` authenticates with the Copernicus Data Space Ecosystem using environment variables. Set credentials locally; never commit them.

```bash
export CDSE_USERNAME="your_username"
export CDSE_PASSWORD="your_password"
python scripts/download_s1_cdse_s3.py \
  --product-name "S1A_IW_GRDH_1SDV_..." \
  --output-dir ./sentinel1_raw
```

### SNAP preprocessing

The preprocessing scripts require an installed ESA SNAP environment and a compatible Python bridge (`snappy`/`esa_snappy`). These are not ordinary pip-only dependencies.

Single scene:

```bash
python scripts/s1_snap_preprocess.py \
  --input ./sentinel1_raw/S1_scene.zip \
  --output ./processed/S1_scene_ARD.tif
```

Batch:

```bash
python scripts/s1_snap_batch_preprocess.py \
  --input-dir ./sentinel1_raw \
  --output-dir ./processed
```

### Model training

```bash
python scripts/train_s1_models.py \
  --csv-path ./data/s1_raster_values.csv \
  --out-dir ./model_outputs_s1 \
  --test-plots 1 5 \
  --seed 42
```

The training script tunes for recall of Class 1 (galamsey), matching the supplied thesis training code. It does **not** calculate held-out test metrics; it only excludes the specified test plots from model training.

### Hotspot evolution

Input rasters should contain one annual binary GeoTIFF for each year from 2015 through 2025. The cleaned script extracts the year from filenames and verifies that all rasters share shape, CRS, and transform.

```bash
python scripts/hotspot_evolution.py \
  --input-dir ./data/cmf_june_binary \
  --output-dir ./results/hotspot_evolution
```

## Thesis-derived hotspot definitions

| Class | Definition in supplied thesis script |
|---|---|
| Persistent | Active in at least 8 of 11 years (2015–2025) |
| Emerging | No activity in 2015–2019 and active in at least 2 years in 2020–2025 |
| Abandoned | Active in at least 2 years in 2015–2019 and no activity in 2022–2025 |

Combined raster codes are **0 = no hotspot, 1 = persistent, 2 = emerging, 3 = abandoned**.

## Important methodological notes

This repository intentionally preserves the supplied code where it differs across scripts. In particular:

1. The batch preprocessing script includes `Remove-GRD-Border-Noise`; the supplied single-scene script did not.
2. The supplied scripts use both `snappy` and `esa_snappy`, which may reflect different SNAP/Python environments.
3. The supplied SNAP scripts use `AUTO:42001` as the map projection parameter. This has not been silently replaced with another CRS.
4. The model-training script creates a held-out plot split but does not evaluate that held-out set.
5. Trained `.joblib` models and satellite rasters are intentionally excluded from the repository by default because they may be large and/or environment-sensitive.

See [`docs/methodology.md`](docs/methodology.md) for the detailed source-to-code reconciliation notes.

## Portfolio assets

A visual portfolio PDF and figure assets were prepared separately from this code repository. The repository is the technical/code component; the PDF is the recruiter-facing project case study.

## Author

**Anabel Bonsu**  
MSc Forest Information Technology  
Eberswalde University for Sustainable Development (HNEE)
