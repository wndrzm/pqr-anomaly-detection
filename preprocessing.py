# =============================================================================
# modules/preprocessing.py — Section 4: Preprocessing Per Product
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import PARAMETERS


def run_preprocessing(df: pd.DataFrame, products) -> tuple[dict, dict]:
    """
    Fit StandardScaler per product.

    Returns:
        scalers     : { product_name: fitted StandardScaler }
        scaled_dict : { product_name: np.ndarray of scaled features }
    """
    scalers     = {}
    scaled_dict = {}

    for product in products:
        mask   = df['product-name'] == product
        raw    = df.loc[mask, PARAMETERS].values

        scaler = StandardScaler()
        scaled = scaler.fit_transform(raw)

        scalers[product]     = scaler
        scaled_dict[product] = scaled

    return scalers, scaled_dict
