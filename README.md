---
title: PQR Anomaly Detection
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: mit
---

# 🔬 PQR Anomaly Detection Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-4.36.1-orange?logo=gradio&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![HuggingFace](https://img.shields.io/badge/🤗%20Spaces-Live-yellow)

> **Ensemble anomaly detection for pharmaceutical batch quality review.**  
> Combines Z-Score, Isolation Forest, Autoencoder, and Hotelling T² into a single weighted ensemble — compliant with ICH Q10 methodology.

---

## ✨ Features

### 🔍 Anomaly Detection
- **Z-Score** — univariate outlier detection per parameter
- **Isolation Forest** — unsupervised tree-based anomaly detection
- **Vanilla Autoencoder** — reconstruction-error-based deep learning detector
- **Hotelling T²** — multivariate statistical process control
- **Weighted Ensemble** — combines all four models into a final risk score

### ⚠️ Risk Assessment
- OOS (Out-of-Spec) / OOT (Out-of-Trend) classification per parameter
- % distance to pharmacopoeial specification limits
- **Fraud / copy-paste detection** — exact duplicates, near-duplicates, suspicious rounding
- **Mann-Kendall trend analysis** — monotonic trend significance testing
- **CUSUM shift detection** — cumulative sum control charts (Montgomery, 2009)

### 🏭 Process Capability
- Cp, Cpk, CPL, CPU per parameter per product
- Color-coded interpretation (Excellent / Capable / Marginal / Not Capable)

### 🎯 Model Performance
- Precision, Recall, F1 per model and ensemble
- Ground truth: synthetic OOS, Fraud, and Outlier batches embedded in dataset

### 📄 Reporting & Export
- Auto-generated executive summary (PQR / CAPA ready)
- Downloadable CSV and multi-sheet Excel
- Interactive Plotly charts throughout
- Audit-trail-ready output format

---

## 🚀 Quick Start

### Run on Hugging Face Spaces
Just open the Space — no installation needed.  
Select **"Use synthetic demo dataset"** and click **🚀 Run Analysis**.

### Run Locally

```bash
# 1. Clone the repo
git clone https://huggingface.co/spaces/<your-username>/pqr-anomaly-detection
cd pqr-anomaly-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
python app.py
# Open http://localhost:7860
```

> **Python 3.10+** is required. TensorFlow 2.16 requires pip ≥ 23.

---

## 📂 Project Structure

```
├── app.py                  # Main Gradio application
├── config.py               # All tunable parameters (edit here only)
├── requirements.txt
├── data/
│   ├── dataset_pqr_v5.xlsx # Synthetic demo dataset
│   └── spec_limits.xlsx    # Pharmacopoeial specification limits
└── modules/
    ├── data_loader.py      # Data ingestion & validation
    ├── fraud.py            # Duplicate / copy-paste detection
    ├── preprocessing.py    # Per-product scaling
    ├── models.py           # Z-Score, Isolation Forest, Autoencoder, Hotelling T²
    ├── ensemble.py         # Weighted ensemble scoring
    ├── risk_context.py     # OOS/OOT contextualization
    ├── trend.py            # Mann-Kendall + CUSUM
    ├── capability.py       # Cp/Cpk analysis
    ├── evaluation.py       # Model performance metrics
    ├── executive.py        # Auto-generated report text
    └── charts.py           # Plotly chart builders
```

---

## 📋 Input Data Format

Upload an `.xlsx` file with a sheet named **`Data`** containing these columns:

| Column | Type | Description |
|---|---|---|
| `year` | int | Batch production year |
| `month` | int | Batch production month (1–12) |
| `product-name` | str | Product name |
| `batch-number` | str | Unique batch identifier |
| `yield` | float | % yield |
| `hardness` | float | Tablet hardness (N) |
| `friability` | float | % friability |
| `thickness` | float | Tablet thickness (mm) |
| `disintegration-time` | float | Disintegration time (min) |
| `dissolution-rate` | float | % dissolution |
| `% assay` | float | API assay (%) |

Specification limits are loaded from `data/spec_limits.xlsx` (columns: `parameter`, `LSL`, `USL`, `product-name`).

---

## ⚙️ Configuration

All model parameters are in **`config.py`** — no need to touch any other file:

```python
# Ensemble weights
VOTE_WEIGHTS = {
    'zscore':      0.20,
    'iforest':     0.20,
    'autoencoder': 0.30,
    'hotelling':   0.30,
}
WEIGHTED_THRESHOLD = 0.40   # score above this → anomaly

# Z-Score
ZSCORE_THRESHOLD = 3.0

# Isolation Forest
IF_CONTAMINATION = 0.05

# Autoencoder
AE_EPOCHS          = 100
AE_THRESHOLD_SIGMA = 2.0    # anomaly if recon. error > N × σ

# CUSUM
CUSUM_K = 0.5   # reference value (k·σ)
CUSUM_H = 5.0   # decision interval (h·σ)
```

---

## 🗂️ Dashboard Tabs

| Tab | Description |
|---|---|
| ⚙️ Settings & Run | Upload data, review config, launch pipeline |
| 📊 Overview | Batch status summary, AE vs T² scatter, agreement heatmap, fraud results |
| 🔍 Anomaly Detail | Per-product control charts and flagged batch table |
| ⚠️ Risk Context | OOS/OOT distance to spec limits, per-parameter risk labels |
| 📈 Trends & CUSUM | Mann-Kendall trend lines and CUSUM shift charts |
| 🏭 Capability | Cp/Cpk table with color-coded interpretation |
| 🎯 Model Performance | Precision / Recall / F1 comparison across models |
| 📄 Executive Summary | Auto-generated PQR narrative, downloadable as `.txt` |
| 💾 Export | Download full results as CSV or multi-sheet Excel |

---

## 📖 Methodology & References

- **ICH Q10** — Pharmaceutical Quality System
- **Montgomery, D.C. (2009)** — *Introduction to Statistical Quality Control*, 6th ed.
- **AIAG SPC Manual** — Statistical Process Control reference
- **Mann-Kendall test** — non-parametric monotonic trend detection (α = 0.05)
- **CUSUM** — cumulative sum control chart for mean shift detection

---

## 📄 License

MIT © 2024 — free to use, modify, and distribute with attribution.
