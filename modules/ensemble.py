# =============================================================================
# modules/ensemble.py — Sections 8 & 9: Merge + Ensemble Scoring
# =============================================================================

import pandas as pd
from config import (
    PARAMETERS, IDENTITY_COLS,
    VOTE_WEIGHTS, WEIGHTED_THRESHOLD, STATUS_ORDER, COLOR_MAP,
)


def merge_results(
    df:      pd.DataFrame,
    df_zs_if: pd.DataFrame,
    df_ae:   pd.DataFrame,
    df_t2:   pd.DataFrame,
) -> pd.DataFrame:
    """Merge all model results into one DataFrame."""
    KEY = ['batch-number', 'product-name']

    merged = df[IDENTITY_COLS + PARAMETERS + [
        'fraud_exact_dup', 'fraud_near_dup', 'fraud_round', 'fraud_any'
    ]].copy()

    zs_cols = KEY + ['zscore_is_anomaly', 'zscore_max_z', 'iforest_is_anomaly', 'iforest_score']
    merged  = merged.merge(df_zs_if[zs_cols], on=KEY, how='left')

    ae_cols = KEY + ['ae_reconstruction_error', 'ae_is_anomaly', 'ae_severity', 'ae_sigma2']
    merged  = merged.merge(df_ae[ae_cols], on=KEY, how='left')

    t2_cols = KEY + ['t2_distance', 't2_is_outlier', 't2_severity', 't2_ucl', 't2_warning']
    merged  = merged.merge(df_t2[t2_cols], on=KEY, how='left')

    return merged


def run_ensemble(merged: pd.DataFrame) -> pd.DataFrame:
    """Add vote_count, weighted_score, ensemble_is_anomaly, final_status."""

    merged['vote_count'] = (
        merged['zscore_is_anomaly'].fillna(0).astype(int) +
        merged['iforest_is_anomaly'].fillna(0).astype(int) +
        merged['ae_is_anomaly'].fillna(0).astype(int) +
        merged['t2_is_outlier'].fillna(0).astype(int)
    )

    merged['weighted_score'] = (
        merged['zscore_is_anomaly']  * VOTE_WEIGHTS['zscore']      +
        merged['iforest_is_anomaly'] * VOTE_WEIGHTS['iforest']     +
        merged['ae_is_anomaly']      * VOTE_WEIGHTS['autoencoder'] +
        merged['t2_is_outlier']      * VOTE_WEIGHTS['hotelling']
    )

    merged['ensemble_is_anomaly'] = (
        merged['weighted_score'] >= WEIGHTED_THRESHOLD
    ).astype(int)

    def classify(row):
        if row['fraud_any']:
            return '🔴 Data Fraud — Duplicate/Copy-Paste'
        v = row['weighted_score']
        if v >= 0.40: return '🔴 Confirmed Anomaly — Investigate & CAPA'
        if v >= 0.25: return '🟠 Suspected Anomaly — Enhanced Monitoring'
        if v >= 0.15: return '🟡 Watch List — Re-check Next Batch'
        return '✅ Normal'

    merged['final_status'] = merged.apply(classify, axis=1)
    return merged


def get_ensemble_summary(merged: pd.DataFrame) -> dict:
    """Return summary counts and flagged DataFrame for display."""
    counts  = merged['final_status'].value_counts()
    flagged = merged[merged['final_status'] != '✅ Normal'].copy()

    return {
        'counts':         {s: int(counts.get(s, 0)) for s in STATUS_ORDER},
        'total':          len(merged),
        'n_flagged':      len(flagged),
        'flagged':        flagged,
        'normal_pct':     round(counts.get('✅ Normal', 0) / len(merged) * 100, 1),
    }
