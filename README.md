# 🔬 PQR Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

An ensemble anomaly detection system for pharmaceutical batch quality review (PQR) — built for QA/QC teams who need more than a single model and a spreadsheet.

**Live demo:** [pqr-anomaly-detection.streamlit.app]([https://pqr-anomaly-detection.streamlit.app](https://pqr-anomaly-detection.streamlit.app/))

---

## 📸 Screenshots

> *(Add screenshots after deploying — Overview tab, Anomaly Detail tab, Trend & CUSUM tab)*

---

## 🎯 What it does

Upload a batch dataset → the system runs 4 anomaly detection models in parallel, combines their votes using a weighted ensemble, and produces:

- **Risk-tiered batch classification** — Confirmed Anomaly / Suspected / Watch List / Normal
- **Fraud detection** — exact duplicates, near-duplicates (copy-paste with minor edits), suspicious rounding
- **Risk contextualization** — how close each flagged parameter is to LSL/USL
- **Trend analysis** — Mann-Kendall test per parameter per product (ICH Q10 Section 3.2)
- **CUSUM chart** — detects small sustained process shifts
- **Auto-generated executive summary** — ready for PQR reports, CAPAs, or management presentations
- **Export** — CSV + Excel (one sheet per product + Risk Context + Trend Analysis)

---

## 🧠 Models & Methodology

| Model | Type | Why it's included |
|-------|------|-------------------|
| Z-Score | Univariate | Fast, interpretable, catches single-parameter spikes |
| Isolation Forest | Multivariate | Robust cluster outlier detection |
| Vanilla Autoencoder | Deep learning | Captures non-linear parameter correlations |
| Hotelling T² | Multivariate statistical | Optimal for correlated parameters; F-distribution UCL (Montgomery 2009) |

### Weighted Ensemble

Each model vote is weighted by its theoretical strength for multivariate pharmaceutical data:

```
weighted_score = 0.20 × Z-Score
              + 0.20 × Isolation Forest
              + 0.30 × Autoencoder
              + 0.30 × Hotelling T²
```

| Score | Risk Tier |
|-------|-----------|
| Fraud flag | 🔴 Data Fraud |
| ≥ 0.40 | 🔴 Confirmed Anomaly — Investigate & CAPA |
| ≥ 0.25 | 🟠 Suspected Anomaly — Enhanced Monitoring |
| ≥ 0.15 | 🟡 Watch List — Re-check Next Batch |
| < 0.15 | ✅ Normal |

**Why 0.40 and not 0.50?** In pharmaceutical manufacturing, the cost of a false negative (missing a real anomaly) vastly outweighs the cost of a false positive. At 0.40, any two models agreeing is sufficient to flag a batch. See [Design Rationale](#design-rationale) for full reasoning.

---

## 🚀 Quick Start

### Run locally

```bash
# Clone the repo
git clone https://github.com/yourusername/pqr-anomaly-detection.git
cd pqr-anomaly-detection

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

The app will open at `http://localhost:8501`. A synthetic demo dataset is included — click **Run Analysis** to see it in action immediately.

### Deploy to Streamlit Community Cloud

1. Fork or push this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo → set **Main file path** to `app.py`
4. Click **Deploy**

No environment variables required.

---

## 📁 Project Structure

```
pqr-anomaly-detection/
├── app.py                  # Streamlit entry point — UI + pipeline orchestration
├── config.py               # All global constants — edit here, nowhere else
├── requirements.txt
├── data/
│   └── dataset_pqr_v5.xlsx # Synthetic demo dataset (192 batches, 4 products)
└── modules/
    ├── data_loader.py       # Load, validate, fill missing values, load spec limits
    ├── fraud.py             # Fraud detection (exact dup, near-dup, rounding)
    ├── preprocessing.py     # StandardScaler per product
    ├── models.py            # Z-Score, Isolation Forest, Autoencoder, Hotelling T²
    ├── ensemble.py          # Merge results + weighted ensemble scoring
    ├── risk_context.py      # OOS / OOT classification per parameter
    ├── trend.py             # Mann-Kendall trend test + risk assessment
    ├── executive.py         # Auto-generate executive summary text
    └── charts.py            # All Plotly visualisations
```

Each module is independent — swap out any model in `models.py` without touching `app.py`, as long as the function signature stays the same.

---

## 📊 Dataset Format

The system accepts `.xlsx` files with a `Data` sheet containing these columns:

| Column | Type | Description |
|--------|------|-------------|
| `year` | int | Batch year |
| `month` | int | Batch month (1–12) |
| `product-name` | str | Product name |
| `batch-number` | str | Unique batch identifier |
| `yield` | float | % yield |
| `hardness` | float | kP |
| `friability` | float | % |
| `thickness` | float | mm |
| `disintegration-time` | float | minutes |
| `dissolution-rate` | float | % |
| `% assay` | float | % |

### Specification Limits (LSL/USL)

Two options — the system auto-detects which one you're using:

**Option 1:** Include LSL/USL directly in the dataset as additional columns:
```
LSL_yield, USL_yield, LSL_hardness, USL_hardness, ...
```

**Option 2:** Provide a separate `data/spec_limits.xlsx` with columns:
```
product-name | parameter | LSL | USL
```

---

## 🔬 Design Rationale

### Why 4 models?
No single model is sufficient. Z-Score misses multivariate anomalies. Isolation Forest is sensitive to dataset size. The Autoencoder needs sufficient training data. Hotelling T² assumes multivariate normality. The ensemble compensates for each model's blind spot.

### Why train the Autoencoder on non-fraud batches only?
The fundamental principle of autoencoder-based anomaly detection: *train on normal data, measure reconstruction error on new data.* Including fraudulent (copy-pasted) batches in training teaches the model to reconstruct them well — the opposite of what we want.

### Why Mann-Kendall instead of linear regression for trends?
Mann-Kendall requires no normality assumption, is robust to outliers, and is the ICH Q10-recommended method for pharmaceutical QC trending. A single OOS batch can heavily distort a regression slope.

### Why CUSUM in addition to Mann-Kendall?
Mann-Kendall answers: *"Is there a trend over the entire history?"* CUSUM answers: *"Has the process shifted recently, and from which batch?"* Together they cover both long-term trend and recent shift detection.

---

## 📚 References

- ICH Q10 Pharmaceutical Quality System, Section 3.2
- Montgomery, D.C. (2009). *Introduction to Statistical Quality Control*, 6th Ed. — Hotelling T², CUSUM
- AIAG SPC Manual, 2nd Ed. — Process Capability (Cp/Cpk)
- Western Electric Statistical Quality Control Handbook

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built for pharmaceutical QA/QC teams who need audit-ready, ICH-compliant batch anomaly detection.*
