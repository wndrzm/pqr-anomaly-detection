# =============================================================================
# config.py — Global configuration
# Edit here, nowhere else.
# =============================================================================

PARAMETERS = [
    'yield', 'hardness', 'friability', 'thickness',
    'disintegration-time', 'dissolution-rate', '% assay'
]

IDENTITY_COLS = ['year', 'month', 'product-name', 'batch-number']

DEFAULT_DATASET = 'data/dataset_pqr_v5.xlsx'
SHEET_NAME      = 'Data'
SPEC_PATH       = 'data/spec_limits.xlsx'

# Z-Score
ZSCORE_THRESHOLD = 3.0

# Isolation Forest
IF_CONTAMINATION = 0.05
IF_RANDOM_STATE  = 42

# Autoencoder
AE_EPOCHS          = 100
AE_BATCH_SIZE      = 16
AE_PATIENCE        = 10
AE_VAL_SPLIT       = 0.1
AE_MIN_BATCHES     = 20
AE_THRESHOLD_SIGMA = 2.0   # anomaly if reconstruction error > N * sigma

# Hotelling T²
T2_ALPHA         = 0.05
T2_ALPHA_WARNING = 0.01

# Fraud detection
FRAUD_NEAR_THRESHOLD = 0.50
FRAUD_ROUND_RATIO    = 0.70

# Ensemble
VOTE_WEIGHTS = {
    'zscore':      0.20,
    'iforest':     0.20,
    'autoencoder': 0.30,
    'hotelling':   0.30,
}
WEIGHTED_THRESHOLD = 0.40

STATUS_ORDER = [
    '🔴 Data Fraud — Duplicate/Copy-Paste',
    '🔴 Confirmed Anomaly — Investigate & CAPA',
    '🟠 Suspected Anomaly — Enhanced Monitoring',
    '🟡 Watch List — Re-check Next Batch',
    '✅ Normal',
]

COLOR_MAP = {
    '🔴 Data Fraud — Duplicate/Copy-Paste':       '#9b2226',
    '🔴 Confirmed Anomaly — Investigate & CAPA':  '#e63946',
    '🟠 Suspected Anomaly — Enhanced Monitoring': '#f4a261',
    '🟡 Watch List — Re-check Next Batch':        '#e9c46a',
    '✅ Normal':                                   '#457b9d',
}

# Trend analysis
TREND_CONFIG = {
    'alpha':         0.05,
    'min_batches':   8,
    'window_recent': None,
}

# CUSUM
CUSUM_K = 0.5
CUSUM_H = 5.0

# Parameters where an UPWARD trend is bad
PARAMS_HIGH_IS_BAD = ['friability', 'disintegration-time']

# Parameters where a DOWNWARD trend is bad
PARAMS_LOW_IS_BAD  = ['yield', 'hardness', 'dissolution-rate', '% assay', 'thickness']
