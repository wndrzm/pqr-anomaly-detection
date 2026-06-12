# =============================================================================
# modules/fraud.py — Section 3: Fraud Detection
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from config import PARAMETERS, FRAUD_NEAR_THRESHOLD, FRAUD_ROUND_RATIO


def detect_fraud(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run fraud detection per product.

    Adds four columns:
        fraud_exact_dup : bool — exact duplicate of another batch
        fraud_near_dup  : bool — near-duplicate within FRAUD_NEAR_THRESHOLD
        fraud_round     : bool — suspicious rounding ratio > FRAUD_ROUND_RATIO
        fraud_any       : bool — flagged by at least one check
    """
    result_frames = []

    for product in df['product-name'].unique():
        subset = df[df['product-name'] == product].copy()

        # Check 1: Exact duplicate
        exact_dup = subset.duplicated(subset=PARAMETERS, keep=False)

        # Check 2: Near-duplicate
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(subset[PARAMETERS])
        dist_mat = euclidean_distances(X_scaled)
        np.fill_diagonal(dist_mat, np.inf)

        min_dist = np.min(dist_mat, axis=1)
        near_dup = pd.Series(min_dist < FRAUD_NEAR_THRESHOLD, index=subset.index)
        near_dup = near_dup & ~exact_dup

        # Check 3: Suspicious rounding
        def _is_rounded(row):
            rounded_count = sum((v * 10) % 5 == 0 for v in row)
            return (rounded_count / len(row)) > FRAUD_ROUND_RATIO

        round_flag = subset[PARAMETERS].apply(_is_rounded, axis=1)

        subset['fraud_exact_dup'] = exact_dup.values
        subset['fraud_near_dup']  = near_dup.values
        subset['fraud_round']     = round_flag.values
        subset['fraud_any']       = (
            subset['fraud_exact_dup'] |
            subset['fraud_near_dup']  |
            subset['fraud_round']
        )

        result_frames.append(subset)

    return pd.concat(result_frames).sort_index()


def get_fraud_summary(df: pd.DataFrame) -> dict:
    """Return summary counts for display."""
    return {
        'exact_duplicates':    int(df['fraud_exact_dup'].sum()),
        'near_duplicates':     int(df['fraud_near_dup'].sum()),
        'suspicious_rounding': int(df['fraud_round'].sum()),
        'total_flagged':       int(df['fraud_any'].sum()),
        'flagged_batches':     df[df['fraud_any']][
            ['batch-number', 'product-name',
             'fraud_exact_dup', 'fraud_near_dup', 'fraud_round']
        ].reset_index(drop=True),
    }
