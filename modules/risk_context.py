# =============================================================================
# modules/risk_context.py — Section 10: Risk Contextualization
# =============================================================================

import numpy as np
import pandas as pd
from config import PARAMETERS


def compute_risk_context(df: pd.DataFrame, flagged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute risk context for flagged batches.
    Returns summary DataFrame per batch per parameter.
    """
    results = []

    for _, flagged_row in flagged_df.iterrows():
        batch = flagged_row['batch-number']
        prod  = flagged_row['product-name']

        row = df[
            (df['batch-number'] == batch) &
            (df['product-name'] == prod)
        ]
        if row.empty:
            continue
        row = row.iloc[0]

        for param in PARAMETERS:
            val = row.get(param, np.nan)
            if pd.isna(val):
                continue

            lsl = row.get(f'LSL_{param}', np.nan)
            usl = row.get(f'USL_{param}', np.nan)

            oos_low  = (not pd.isna(lsl)) and (val < lsl)
            oos_high = (not pd.isna(usl)) and (val > usl)
            is_oos   = oos_low or oos_high

            distances = []
            if not pd.isna(lsl) and lsl != 0:
                distances.append(('LSL', lsl, ((val - lsl) / lsl) * 100))
            if not pd.isna(usl) and usl != 0:
                distances.append(('USL', usl, ((usl - val) / usl) * 100))

            if not distances:
                continue

            nearest_limit, nearest_val, nearest_dist = min(distances, key=lambda x: abs(x[2]))

            if is_oos:
                risk_label  = '🔴 OOS'
                risk_color  = '#e63946'
            elif abs(nearest_dist) <= 5:
                risk_label  = '🟠 OOT Near Limit'
                risk_color  = '#f4a261'
            elif abs(nearest_dist) <= 15:
                risk_label  = '🟡 OOT Monitor'
                risk_color  = '#e9c46a'
            else:
                risk_label  = '🟢 Within Spec'
                risk_color  = '#2a9d8f'

            results.append({
                'batch-number':    batch,
                'product-name':    prod,
                'parameter':       param,
                'actual_value':    round(val, 3),
                'LSL':             round(lsl, 3) if not pd.isna(lsl) else '—',
                'USL':             round(usl, 3) if not pd.isna(usl) else '—',
                'nearest_limit':   nearest_limit,
                'dist_to_limit_%': round(nearest_dist, 2),
                'is_OOS':          is_oos,
                'risk_label':      risk_label,
                'risk_color':      risk_color,
            })

    return pd.DataFrame(results)


def get_risk_summary(risk_df: pd.DataFrame) -> dict:
    """Return summary counts for display."""
    if risk_df.empty:
        return {'n_oos': 0, 'n_near': 0, 'n_monitor': 0, 'n_ok': 0}

    return {
        'n_oos':     int(risk_df['is_OOS'].sum()),
        'n_near':    int(risk_df['risk_label'].str.contains('Near Limit').sum()),
        'n_monitor': int(risk_df['risk_label'].str.contains('Monitor').sum()),
        'n_ok':      int(risk_df['risk_label'].str.contains('Within Spec').sum()),
    }
