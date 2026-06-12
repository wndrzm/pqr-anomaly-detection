# =============================================================================
# modules/trend.py — Section 11: Trend Analysis (Mann-Kendall)
# =============================================================================

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from config import PARAMETERS, TREND_CONFIG, PARAMS_HIGH_IS_BAD, PARAMS_LOW_IS_BAD


def mann_kendall_test(series: pd.Series) -> dict:
    """Mann-Kendall trend test using scipy.stats.kendalltau."""
    series = series.dropna()
    n      = len(series)

    if n < TREND_CONFIG['min_batches']:
        return {
            'tau': None, 'p_value': None,
            'trend': 'insufficient_data',
            'trend_label': f'Insufficient data (min. {TREND_CONFIG["min_batches"]})',
            'significant': False,
        }

    tau, p_value = kendalltau(np.arange(n), series.values)
    significant  = p_value < TREND_CONFIG['alpha']

    if not significant:
        trend, trend_label = 'no_trend', '✅ No significant trend'
    elif tau > 0:
        trend, trend_label = 'increasing', '⬆️ Upward trend (significant)'
    else:
        trend, trend_label = 'decreasing', '⬇️ Downward trend (significant)'

    return {
        'tau':         round(tau, 4),
        'p_value':     round(p_value, 4),
        'trend':       trend,
        'trend_label': trend_label,
        'significant': significant,
        'n_batches':   n,
    }


def assess_trend_risk(param: str, trend: str) -> tuple[str, str]:
    """
    Returns (risk_label, risk_color).
    """
    if trend in ('no_trend', 'insufficient_data'):
        return '🟢 Safe', '#2a9d8f'
    if trend == 'increasing' and param in PARAMS_HIGH_IS_BAD:
        return '🔴 ATTENTION — approaching upper limit', '#e63946'
    if trend == 'decreasing' and param in PARAMS_LOW_IS_BAD:
        return '🔴 ATTENTION — approaching lower limit', '#e63946'
    if trend == 'increasing' and param in PARAMS_LOW_IS_BAD:
        return '🟡 Monitor — moving away from lower limit', '#e9c46a'
    if trend == 'decreasing' and param in PARAMS_HIGH_IS_BAD:
        return '🟡 Monitor — moving away from upper limit', '#e9c46a'
    return '🟢 Safe', '#2a9d8f'


def run_trend_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Run Mann-Kendall for all products and parameters."""
    results = []

    for prod in df['product-name'].unique():
        sub = df[df['product-name'] == prod].sort_values('month').reset_index(drop=True)

        if TREND_CONFIG['window_recent']:
            sub = sub.tail(TREND_CONFIG['window_recent'])

        for param in PARAMETERS:
            if param not in sub.columns:
                continue

            mk             = mann_kendall_test(sub[param])
            risk, color    = assess_trend_risk(param, mk['trend'])

            series_clean = sub[param].dropna()
            slope = round(
                np.polyfit(np.arange(len(series_clean)), series_clean.values, 1)[0], 4
            ) if len(series_clean) >= 2 else None

            results.append({
                'product-name':    prod,
                'parameter':       param,
                'n_batches':       mk.get('n_batches', len(sub)),
                'tau':             mk['tau'],
                'p_value':         mk['p_value'],
                'slope_per_batch': slope,
                'trend':           mk['trend'],
                'trend_label':     mk['trend_label'],
                'significant':     mk['significant'],
                'risk_assessment': risk,
                'risk_color':      color,
            })

    return pd.DataFrame(results)


def get_trend_summary(trend_df: pd.DataFrame) -> dict:
    """Return summary counts for display."""
    return {
        'n_attention': int(trend_df['risk_assessment'].str.contains('ATTENTION').sum()),
        'n_monitor':   int(trend_df['risk_assessment'].str.contains('Monitor').sum()),
        'n_safe':      int(trend_df['risk_assessment'].str.contains('Safe').sum()),
        'alerts':      trend_df[trend_df['risk_assessment'].str.contains('ATTENTION')].reset_index(drop=True),
    }
