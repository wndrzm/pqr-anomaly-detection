# =============================================================================
# modules/data_loader.py — Section 2: Load & Validate Data
# =============================================================================

import os
import pandas as pd
import numpy as np
import streamlit as st
from config import PARAMETERS, IDENTITY_COLS, SPEC_PATH


def load_data(uploaded_file=None, default_path: str = None) -> pd.DataFrame:
    """
    Load dataset from uploaded file or default path.
    Returns cleaned DataFrame or raises on validation failure.
    """
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name='Data')
    elif default_path and os.path.exists(default_path):
        df = pd.read_excel(default_path, sheet_name='Data')
    else:
        raise FileNotFoundError('No dataset provided and default not found.')

    df.columns = df.columns.str.strip()
    return df


def validate_data(df: pd.DataFrame) -> tuple[bool, list]:
    """
    Validate required columns exist.
    Returns (is_valid, missing_cols).
    """
    required = IDENTITY_COLS + PARAMETERS
    missing  = [c for c in required if c not in df.columns]
    return len(missing) == 0, missing


def fill_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Fill missing values with per-product median.
    Returns (df, report) where report is {param: n_filled}.
    """
    report      = {}
    null_counts = df[PARAMETERS].isnull().sum()

    for param in PARAMETERS:
        n = null_counts[param]
        if n > 0:
            df[param] = df.groupby('product-name')[param].transform('median')
            report[param] = int(n)

    return df, report


def load_spec_limits(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Load LSL/USL — flexible source:
      1. Already in df (LSL_*/USL_* columns) → use as-is
      2. SPEC_PATH file exists → merge
      3. Neither → return df unchanged with warning message

    Returns (df, status_message).
    """
    lsl_exist = all(f'LSL_{p}' in df.columns for p in PARAMETERS)
    usl_exist = all(f'USL_{p}' in df.columns for p in PARAMETERS)

    if lsl_exist and usl_exist:
        return df, 'Spec limits found in dataset — used as-is.'

    if os.path.exists(SPEC_PATH):
        spec = pd.read_excel(SPEC_PATH)
        spec.columns = spec.columns.str.strip()

        for param in PARAMETERS:
            spec_sub = spec[spec['parameter'] == param][['product-name', 'LSL', 'USL']]
            spec_sub = spec_sub.rename(columns={
                'LSL': f'LSL_{param}',
                'USL': f'USL_{param}'
            })
            df = df.merge(spec_sub, on='product-name', how='left')

        missing = [p for p in PARAMETERS if df[f'LSL_{p}'].isna().sum() > 0]
        if missing:
            return df, f'Warning: missing spec limits for {missing}'
        return df, f'Spec limits loaded from {SPEC_PATH}.'

    return df, 'No spec limits found — Risk Contextualization will be skipped.'


def run_data_pipeline(uploaded_file=None, default_path=None):
    """
    Full data loading pipeline. Returns dict with all results.
    Raises on critical errors (missing columns).
    """
    df = load_data(uploaded_file, default_path)

    is_valid, missing = validate_data(df)
    if not is_valid:
        raise ValueError(f'Missing required columns: {missing}')

    df, missing_report = fill_missing_values(df)
    df, spec_status    = load_spec_limits(df)

    products = df['product-name'].unique()
    batch_counts = {p: len(df[df['product-name'] == p]) for p in products}

    return {
        'df':            df,
        'products':      products,
        'batch_counts':  batch_counts,
        'missing_report': missing_report,
        'spec_status':   spec_status,
    }
