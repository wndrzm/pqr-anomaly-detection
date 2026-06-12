---
title: Pqr Anomaly Detection
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.18.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🔬 PQR Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-6.18.0-orange?logo=gradio&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow--CPU-2.16-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

An ensemble anomaly detection system for pharmaceutical batch quality review (PQR) — built for QA/QC teams who need more than a single model and a spreadsheet.

**Live demo:** [pqr-anomaly-detection on Hugging Face Spaces](https://huggingface.co/spaces/harujionhitori/pqr-anomaly-detection)

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
