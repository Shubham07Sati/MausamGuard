# AI Forecast Bust Detection — SIH 2026
> **Team:** Person A (ML) + Person B (Data/Geo)  
> **Problem:** Detect when the GFS numerical weather prediction model makes catastrophically wrong forecasts over India  
> **Approach:** LightGBM binary classifier trained on (GFS forecast, ERA5 reanalysis) error statistics

---

## 🗂️ Project Structure
```
.
├── configs/
│   └── india_domain.yaml       ← All domain, variable, path config
├── data/
│   ├── raw/
│   │   ├── era5/               ← .nc files from Copernicus CDS
│   │   └── gfs/                ← .grib2 files from NOAA AWS
│   ├── interim/                ← Intermediate regridded files
│   └── processed/              ← Final .parquet training data
├── logs/                       ← Download and training logs
├── models/                     ← Saved .pkl model files
├── notebooks/                  ← Jupyter exploration notebooks
├── scripts/
│   ├── download_era5.py        ← ERA5 downloader (CDS API)
│   ├── download_gfs.py         ← GFS downloader (AWS S3)
│   ├── download_gfs_nomads.py  ← GFS downloader fallback (NOMADS HTTP)
│   ├── verify_downloads.py     ← Sanity check downloaded files
│   ├── preprocess.py           ← Regrid + merge + Parquet (Day 2)
│   ├── features.py             ← Feature engineering (Day 2)
│   ├── train.py                ← LightGBM training (Day 3)
│   ├── infer.py                ← Inference pipeline (Day 4)
│   └── evaluate.py             ← Metrics + SHAP (Day 5)
├── api/
│   └── main.py                 ← FastAPI backend (Day 3)
├── dashboard/
│   └── app.py                  ← Streamlit dashboard (Day 4)
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up CDS API key (for ERA5)
Create `~/.cdsapirc` (in your home directory):
```
url: https://cds.climate.copernicus.eu/api/v2
key: <YOUR_UID>:<YOUR_API_KEY>
```
Register free at https://cds.climate.copernicus.eu

### 3. Download ERA5 data (Day 1 – Person B)
```bash
# Monsoon season (June–Sept) for 2019–2022
python scripts/download_era5.py --start 2019-01-01 --end 2022-12-31 --season monsoon

# Single year test
python scripts/download_era5.py --years 2023 --season monsoon
```

### 4. Download GFS data (Day 1 – Person B)
```bash
# Method A: AWS S3 (full GRIB2, large files)
python scripts/download_gfs.py --start 2019-06-01 --end 2019-09-30 --season monsoon

# Method B: NOMADS HTTP filter (small, India-only subset) ← RECOMMENDED
python scripts/download_gfs_nomads.py --start 2019-06-01 --end 2019-09-30 --season monsoon

# Dry run (print URLs only, no download)
python scripts/download_gfs_nomads.py --start 2023-06-01 --end 2023-06-03 --dry-run
```

### 5. Verify downloads
```bash
python scripts/verify_downloads.py --era5 --gfs
```

---

## 📐 Domain

| Parameter | Value |
|-----------|-------|
| Region | India |
| Lat range | 6°N – 37°N |
| Lon range | 66°E – 98°E |
| Resolution | 0.25° × 0.25° |
| Training | 2019–2022 |
| Test | 2023 |
| Season focus | Monsoon (JJAS) |
| Variables | T2m, TP, Z500, MSLP, U10, V10 |

---

## 🎯 Bust Definition

$$B_{i,t} = 1 \text{ (Bust) if } |Forecast_{i,t} - Observation_{i,t}| > \tau_i$$

Where $\tau_i$ = 90th percentile of errors for location $i$ over the training set.

---

## 📊 Metrics

| Metric | Why |
|--------|-----|
| **PR-AUC** | Primary — handles class imbalance |
| **Brier Score** | Probability calibration |
| **F1 (Macro)** | Hard binary threshold |
| **Baseline** | Naive climatology (1σ deviation) |

---

## 🗓️ 7-Day Plan

| Day | Person A (ML) | Person B (Data/Geo) |
|-----|---------------|----------------------|
| 1 | Project scaffold, dummy data generator | **ERA5 + GFS download scripts** ← You are here |
| 2 | Feature engineering pipeline | Regrid + tabularize to Parquet |
| 3 | LightGBM training, bust labels | FastAPI backend |
| 4 | Inference pipeline | Streamlit dashboard |
| 5 | SHAP explainability, evaluation | Connect frontend to API |
| 6 | End-to-end testing | Docker + integration |
| 7 | Pre-compute demo scenarios | Hardcode fallback scenarios |
