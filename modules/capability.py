# =============================================================================
# modules/capability.py — Section 15: Process Capability Analysis
# =============================================================================

import numpy as np
import pandas as pd
from config import PARAMETERS


def compute_capability(values: np.ndarray, lsl: float, usl: float) -> dict:
    """
    Compute Cp, Cpk, and interpretation.

    For batch release data where each batch = one observation,
    within-subgroup and overall std are equivalent — Cp = Pp, Cpk = Ppk.
    This is expected and consistent with industry practice for annual PQR data.

    One-sided specs (LSL only or USL only) return only the relevant index.
    """
    values = values[~np.isnan(values)]
    n      = len(values)

    if n < 2:
        return _empty_result()

    mean = np.mean(values)
    std  = np.std(values, ddof=1)

    if std == 0:
        return _empty_result()

    has_lsl = not (pd.isna(lsl) or lsl == 0)
    has_usl = not (pd.isna(usl) or usl == 0)

    # Cp — requires both limits
    cp = round((usl - lsl) / (6 * std), 3) if (has_lsl and has_usl) else None

    # Cpk — one or two-sided
    cpu = round((usl - mean) / (3 * std), 3) if has_usl else None
    cpl = round((mean - lsl) / (3 * std), 3) if has_lsl else None

    if cpu is not None and cpl is not None:
        cpk = round(min(cpu, cpl), 3)
    elif cpu is not None:
        cpk = cpu
    elif cpl is not None:
        cpk = cpl
    else:
        cpk = None

    return {
        'n':    n,
        'mean': round(mean, 4),
        'std':  round(std, 4),
        'cp':   cp,
        'cpk':  cpk,
        'cpu':  cpu,
        'cpl':  cpl,
        'interpretation': _interpret_cpk(cpk),
    }


def _empty_result() -> dict:
    return {
        'n': 0, 'mean': None, 'std': None,
        'cp': None, 'cpk': None, 'cpu': None, 'cpl': None,
        'interpretation': 'Insufficient data',
    }


def _interpret_cpk(cpk) -> str:
    if cpk is None:
        return '—'
    if cpk >= 1.67:
        return '✅ Excellent (Six Sigma)'
    if cpk >= 1.33:
        return '✅ Capable'
    if cpk >= 1.00:
        return '🟡 Marginally Capable — Monitor'
    return '🔴 Not Capable — Investigate'


def run_capability_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run capability analysis for all products and parameters.
    Returns summary DataFrame.
    """
    results = []

    for product in df['product-name'].unique():
        subset = df[df['product-name'] == product]

        for param in PARAMETERS:
            if param not in subset.columns:
                continue

            values  = subset[param].values.astype(float)
            lsl_col = f'LSL_{param}'
            usl_col = f'USL_{param}'

            lsl = subset[lsl_col].iloc[0] if lsl_col in subset.columns else np.nan
            usl = subset[usl_col].iloc[0] if usl_col in subset.columns else np.nan

            cap = compute_capability(values, lsl, usl)

            results.append({
                'product-name':     product,
                'parameter':        param,
                'n':                cap['n'],
                'mean':             cap['mean'],
                'std':              cap['std'],
                'LSL':              round(lsl, 3) if not pd.isna(lsl) else '—',
                'USL':              round(usl, 3) if not pd.isna(usl) else '—',
                'Cp':               cap['cp'],
                'Cpk':              cap['cpk'],
                'CPU':              cap['cpu'],
                'CPL':              cap['cpl'],
                'interpretation':   cap['interpretation'],
            })

    return pd.DataFrame(results)


def get_capability_summary(cap_df: pd.DataFrame) -> dict:
    """Return summary counts for display."""
    if cap_df.empty:
        return {'n_excellent': 0, 'n_capable': 0, 'n_marginal': 0, 'n_not_capable': 0}

    return {
        'n_excellent':   int(cap_df['interpretation'].str.contains('Excellent').sum()),
        'n_capable':     int(cap_df['interpretation'].str.contains('Capable$').sum()),
        'n_marginal':    int(cap_df['interpretation'].str.contains('Marginally').sum()),
        'n_not_capable': int(cap_df['interpretation'].str.contains('Not Capable').sum()),
    }
