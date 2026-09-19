# Sentinel-1 SAR Monitoring of Galamsey Activity in Southern Ghana

**Geospatial portfolio project | Synthetic Aperture Radar (SAR) | Remote Sensing | GIS | Machine Learning | Python**

A thesis-derived geospatial research project demonstrating how **Copernicus Sentinel-1 C-band SAR** data can be processed and analysed to support the detection and temporal monitoring of artisanal and small-scale gold-mining activity (*galamsey*) in Southern Ghana.

> **Portfolio scope**  
> This repository is a technical and reproducibility component of my MSc Forest Information Technology research. It demonstrates a SAR-based monitoring framework; it is **not presented as a deployed real-time operational monitoring service**.

---

## Project at a glance

| | |
|---|---|
| **Study area** | Ashanti, Ahafo, Central, Western and Western North, Ghana |
| **Primary data** | Copernicus Sentinel-1 GRD C-band SAR |
| **Reference data** | Sentinel-2 imagery / labelled reference samples |
| **Processing** | ESA SNAP + Python |
| **GIS / spatial analysis** | Raster processing, spatially grouped validation, temporal hotspot analysis |
| **Machine learning** | Logistic Regression, Random Forest, SVC, ExtraTrees |
| **SAR features** | VV, VH, VV–VH difference, VV/VH ratio, NPI |
| **Analysis period** | 2015–2025 |
| **Programming** | Python, NumPy, pandas, scikit-learn, rasterio |

---

## 1. The problem

Artisanal and small-scale gold mining can produce rapid changes in land cover and surface conditions. Optical satellite imagery can be affected by cloud cover, creating challenges for frequent monitoring in tropical environments.

This project investigates whether **Sentinel-1 SAR time-series information** can provide a complementary source of evidence for detecting and monitoring mining-related activity.

### Research question

> **Can Sentinel-1 SAR time-series information support timely detection and monitoring of galamsey activity in Southern Ghana?**

---

## 2. Objectives

The project was designed to:

1. Process Sentinel-1 SAR imagery into analysis-ready backscatter data.
2. Engineer polarimetric/backscatter-derived features for classification.
3. Train and compare supervised machine-learning models for galamsey detection.
4. Reduce the risk of spatial leakage through plot-based data separation and grouped cross-validation.
5. Produce annual binary detection outputs within a common monitoring footprint.
6. Analyse the spatial persistence and evolution of detected hotspots from **2015 to 2025**.

---

## 3. Study area

The study focused on mining-affected areas across five regions of Southern Ghana:

- **Ashanti**
- **Ahafo**
- **Central**
- **Western**
- **Western North**

The geographic focus was selected to represent areas where artisanal and small-scale mining activity has been documented and where satellite-based monitoring can provide useful spatial and temporal information.

> **Map:** A study-area figure can be added to `figures/` when the final portfolio visual set is committed to the repository.

---

## 4. Data

### Sentinel-1

The primary analytical dataset consists of **Sentinel-1 Ground Range Detected (GRD) C-band SAR** observations, using VV and VH polarisation information.

The research workflow used imagery spanning **2015–2025** and restricted annual analysis to a common monitoring footprint so that temporal comparisons were made over comparable spatial coverage.

### Reference information

Sentinel-2 imagery was used as reference information for identifying and labelling samples. The machine-learning predictor set is based on Sentinel-1-derived features rather than Sentinel-2 spectral bands.

The supplied thesis workflow contains **1,425 labelled samples**, comprising **720 galamsey** and **705 non-galamsey** observations across six reference plots.

---

## 5. Processing workflow

```text
                    COPERNICUS SENTINEL-1 GRD
                              │
                              ▼
                     ESA SNAP PREPROCESSING
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
      Orbit file        Noise handling       Calibration
                              │
                              ▼
                     Terrain correction
                              │
                              ▼
                        dB conversion
                              │
                              ▼
                       GeoTIFF outputs
                              │
                              ▼
                   VV / VH sample extraction
                              │
                              ▼
                     FEATURE ENGINEERING
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
       VV / VH           VV–VH difference     VV/VH + NPI
                              │
                              ▼
                     MACHINE LEARNING
                              │
       ┌──────────────┬──────────────┬──────────────┐
       ▼              ▼              ▼              ▼
       LR             RF            SVC         ExtraTrees
                              │
                              ▼
                  PROBABILITY-BASED OUTPUT
                              │
                              ▼
                    ANNUAL BINARY MAPS
                              │
                              ▼
                  HOTSPOT EVOLUTION 2015–2025
```

---

## 6. SAR preprocessing

The preprocessing workflow converts Sentinel-1 GRD scenes into analysis-ready backscatter products.

The supplied batch-processing implementation includes:

1. Apply Orbit File
2. Remove GRD Border Noise
3. Thermal Noise Removal
4. Radiometric Calibration to Sigma0 for VV/VH
5. Terrain Correction using SRTM
6. Conversion to decibel (dB) representation
7. GeoTIFF/BigTIFF export

### Important implementation note

The supplied single-scene and batch scripts are not completely identical. The **batch script includes GRD border-noise removal**, whereas the supplied single-scene script does not. The repository documents this difference rather than silently changing the original research implementation.

The scripts also use different Python SNAP bridge import names (`snappy` and `esa_snappy`), reflecting the supplied code/environment setup.

---

## 7. Feature engineering

Five Sentinel-1-derived predictors are used by the supplied model-training workflow:

| Feature | Description |
|---|---|
| **VV** | VV backscatter in dB |
| **VH** | VH backscatter in dB |
| **VV–VH difference** | Difference between VV and VH backscatter |
| **VV/VH ratio** | Ratio calculated after conversion from dB to linear scale |
| **NPI** | Normalized Polarisation Index derived from linear VV and VH |

The feature engineering implementation handles invalid/infinite derived values before model fitting, with median imputation included in the model pipelines.

---

## 8. Machine-learning workflow

Four supervised classifiers are trained:

- **Logistic Regression**
- **Random Forest**
- **Support Vector Classifier (SVC)**
- **ExtraTrees**

The training workflow uses scikit-learn pipelines and includes preprocessing appropriate to each model family. Hyperparameter optimisation is performed using `RandomizedSearchCV` with grouped cross-validation.

### Why grouped validation?

Nearby pixels can be spatially correlated. Randomly splitting individual pixels can therefore produce overly optimistic estimates if neighbouring observations appear in both training and validation data.

The supplied workflow instead separates observations by **reference plot** and uses `GroupKFold` during model tuning. This provides a more spatially independent validation strategy.

The supplied thesis workflow uses plots **2, 3, 4 and 6 for model development**, with plots **1 and 5 held out** from model training.

> **Important:** The current training script creates the held-out test split but does not calculate held-out test metrics. Therefore, this repository does not claim test-set performance from that script.

---

## 9. Probability-based detection

The classification workflow produces class probabilities that can be converted into binary detection outputs using a probability threshold.

The thesis workflow uses an operational threshold of:

**P(galamsey) ≥ 0.70**

Pixels meeting or exceeding this threshold are classified as detected galamsey activity for the relevant analysis.

---

## 10. Temporal hotspot analysis

Annual binary outputs are analysed across the **2015–2025** period to characterise different patterns of hotspot persistence and change.

The supplied hotspot-analysis script defines:

| Hotspot class | Definition |
|---|---|
| **Persistent** | Active in at least 8 of 11 years |
| **Emerging** | No activity in 2015–2019 and active in at least 2 years during 2020–2025 |
| **Abandoned** | Active in at least 2 years during 2015–2019 and no activity during 2022–2025 |

The combined raster uses:

```text
0 = No hotspot
1 = Persistent
2 = Emerging
3 = Abandoned
```

This provides a simple spatial representation of how detected activity changes over time rather than treating each annual map as an isolated observation.

---

## 11. Results

The thesis reports model-performance comparisons, probability-based detection, annual detected-area/probability trends, and hotspot-evolution analysis.

The portfolio version of this repository intentionally separates **verified research results** from the processing code. Detailed result figures and tables will be added under `figures/` and `results/` once the final thesis visual set is committed.

> **Why this matters:** reproducible geospatial work should make a clear distinction between the code used to generate an analysis and the numerical results actually produced by a particular run.

---

## 12. Repository structure

```text
galamsey_sar_monitoring/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── docs/
│   └── methodology.md
│
├── figures/
│   └── README.md
│
├── notebooks/
│   └── README.md
│
├── results/
│   └── README.md
│
└── scripts/
    ├── download_s1_cdse_s3.py
    ├── s1_snap_preprocess.py
    ├── s1_snap_batch_preprocess.py
    ├── train_s1_models.py
    └── hotspot_evolution.py
```

---

## 13. Reproducibility

### Sentinel-1 data access

The download script authenticates through the **Copernicus Data Space Ecosystem** using environment variables.

Credentials should be configured locally and must **never** be committed to GitHub.

Example:

```bash
export CDSE_USERNAME="your_username"
export CDSE_PASSWORD="your_password"
```

### SNAP preprocessing

The preprocessing scripts require an installed ESA SNAP environment and a compatible Python bridge. SNAP is an external dependency and is therefore not installed through the standard Python requirements file alone.

### Machine learning

The model-training script expects a CSV containing the required class, plot and Sentinel-1 feature fields. It performs feature engineering, grouped hyperparameter tuning and model serialisation.

### Large satellite datasets

Raw Sentinel-1 products, large GeoTIFF outputs and trained model artefacts are excluded from the Git repository by design. See `data/README.md` and `results/README.md` for the intended structure.

---

## 14. Technologies & skills demonstrated

### GIS & geospatial analysis

- Raster data processing
- Spatial data management
- Temporal raster analysis
- Spatially grouped validation
- Hotspot mapping and classification
- QGIS

### Remote sensing

- Sentinel-1 SAR
- Sentinel-2 reference imagery
- VV/VH polarisation analysis
- SAR preprocessing
- Terrain correction
- Backscatter analysis
- Change and hotspot monitoring

### Programming & data science

- Python
- NumPy
- pandas
- scikit-learn
- rasterio
- Machine-learning pipelines
- Feature engineering
- Hyperparameter optimisation

### Remote-sensing software & platforms

- ESA SNAP
- Google Earth Engine
- Copernicus Data Space Ecosystem

---

## 15. Research limitations

Several limitations are important when interpreting the project:

- The repository represents a **research monitoring framework**, not a production monitoring service.
- The supplied scripts do not implement identical preprocessing chains across single-scene and batch workflows.
- The current model-training script does not calculate held-out test metrics after creating the test split.
- Large satellite datasets and trained model artefacts are not included in the repository.
- The hotspot categories depend on the temporal thresholds defined in the supplied analysis script.

These limitations are documented deliberately so that the portfolio remains transparent about what the code does and does not demonstrate.

---

## 16. Author

**Anabel Bonsu**  
MSc Forest Information Technology  
Eberswalde University for Sustainable Development (HNEE)

This project is based on MSc research in geospatial information technology, remote sensing and environmental monitoring.

---

## 17. Citation

If you use this repository or build on the workflow, please cite the associated research project using the information provided in [`CITATION.cff`](CITATION.cff).

---

### Portfolio takeaway

This project demonstrates an end-to-end geospatial workflow: **satellite data acquisition → SAR preprocessing → feature engineering → spatially aware machine learning → probability-based detection → temporal hotspot analysis**.

It reflects practical experience working across **GIS, remote sensing, Python, raster data, machine learning and environmental monitoring** rather than treating these as separate technical areas.
