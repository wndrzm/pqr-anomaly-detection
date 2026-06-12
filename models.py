# =============================================================================
# modules/models.py — Sections 5, 6, 7: All Detection Models
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f as f_dist
from sklearn.ensemble import IsolationForest
from config import (
    PARAMETERS, IDENTITY_COLS,
    ZSCORE_THRESHOLD, IF_CONTAMINATION, IF_RANDOM_STATE,
    AE_EPOCHS, AE_BATCH_SIZE, AE_PATIENCE, AE_VAL_SPLIT, AE_MIN_BATCHES,
    AE_THRESHOLD_SIGMA,    # ← ini
    T2_ALPHA, T2_ALPHA_WARNING,
)


# =============================================================================
# SECTION 5 — Z-Score & Isolation Forest
# =============================================================================

def run_zscore(data: pd.DataFrame) -> tuple:
    """Z-Score per parameter. Returns (z_scores, flags, is_anomaly, max_z)."""
    z_scores = pd.DataFrame(index=data.index)
    flags    = pd.DataFrame(index=data.index)

    for param in PARAMETERS:
        z = np.abs(stats.zscore(data[param], ddof=1))
        z_scores[f'z_{param}']        = z
        flags[f'zscore_flag_{param}'] = z > ZSCORE_THRESHOLD

    is_anomaly = flags.any(axis=1)
    max_z      = z_scores.max(axis=1)
    return z_scores, flags, is_anomaly, max_z


def run_isolation_forest(data: pd.DataFrame, X_scaled: np.ndarray, X_train: np.ndarray = None):
    """
    Isolation Forest using pre-scaled features (from Section 4 scaler).
    Trained on non-fraud batches only (X_train), scored on all batches (X_scaled).
    Returns is_anomaly (bool Series) and anomaly_score (float Series).
    """
    contamination = max(IF_CONTAMINATION, 1 / len(data))

    clf = IsolationForest(
        n_estimators  = 200,
        contamination = contamination,
        random_state  = IF_RANDOM_STATE,
        n_jobs        = -1,
    )
    clf.fit(X_train if X_train is not None else X_scaled)

    preds      = clf.predict(X_scaled)
    raw_scores = clf.score_samples(X_scaled)

    is_anomaly    = pd.Series(preds == -1, index=data.index)
    anomaly_score = pd.Series(raw_scores,  index=data.index)
    return is_anomaly, anomaly_score


def run_zscore_iforest(df: pd.DataFrame, products, scaled_dict: dict) -> pd.DataFrame:
    """Run Z-Score + Isolation Forest for all products. Returns df_zs_if."""
    results_zs_if = []

    for product in products:
        subset   = df[df['product-name'] == product].copy()
        X_scaled = scaled_dict[product]

        # Z-Score
        z_scores, z_flags, zs_anomaly, max_z = run_zscore(subset)

        # Isolation Forest — train on non-fraud batches only
        non_fraud_mask = subset['fraud_any'].values == False
        X_train        = X_scaled[non_fraud_mask]
        if_anomaly, if_score = run_isolation_forest(subset, X_scaled, X_train=X_train)

        # Build result DataFrame
        result = subset[IDENTITY_COLS + PARAMETERS].copy()
        result = pd.concat([result, z_scores, z_flags], axis=1)
        result['zscore_max_z']       = max_z
        result['zscore_is_anomaly']  = zs_anomaly.astype(int)
        result['iforest_score']      = if_score
        result['iforest_is_anomaly'] = if_anomaly.astype(int)
        result['zs_if_both']         = (
            (result['zscore_is_anomaly'] == 1) &
            (result['iforest_is_anomaly'] == 1)
        ).astype(int)

        results_zs_if.append(result)  

    return pd.concat(results_zs_if, ignore_index=True)


# =============================================================================
# SECTION 6 — Vanilla Autoencoder
# =============================================================================

def build_autoencoder(input_dim: int):
    """Build and compile a fresh Vanilla Autoencoder."""
    try:
        import tensorflow as tf
        from tensorflow.keras import Model
        import tensorflow.keras.layers as Layers

        inp        = Layers.Input(shape=(input_dim,))
        encoded    = Layers.Dense(4, activation='relu')(inp)
        bottleneck = Layers.Dense(2, activation='relu')(encoded)
        decoded    = Layers.Dense(4, activation='relu')(bottleneck)
        out        = Layers.Dense(input_dim, activation='linear')(decoded)

        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='mse')
        return model
    except ImportError:
        return None


def _severity_labels(mae, s1, s2, s3):
    return np.where(
        mae > s3, '🔴 High Risk — CAPA Required',
        np.where(
            mae > s2, '🟠 Medium Risk — Investigate',
            np.where(
                mae > s1, '🟡 Low Risk — Monitor',
                '✅ Normal'
            )
        )
    )


def _pick_threshold(s1, s2, s3):
    """Pick AE anomaly threshold based on AE_THRESHOLD_SIGMA config."""
    if AE_THRESHOLD_SIGMA == 1.0: return s1
    if AE_THRESHOLD_SIGMA == 3.0: return s3
    return s2   # default: 2.0


def run_autoencoder_product(
    X_scaled: np.ndarray,
    product_name: str,
    normal_mask: np.ndarray = None,
) -> dict:
    """
    Train or fallback autoencoder for one product.
    Falls back to RMS proxy if n < AE_MIN_BATCHES or TF unavailable.
    Anomaly threshold is controlled by AE_THRESHOLD_SIGMA global config.
    """
    n = len(X_scaled)

    try:
        import tensorflow as tf
        tf_available = True
    except ImportError:
        tf_available = False

    if n < AE_MIN_BATCHES or not tf_available:
        mean_vec = np.mean(X_scaled, axis=0)
        mae      = np.sqrt(np.mean((X_scaled - mean_vec) ** 2, axis=1))
        mean_err = np.mean(mae)
        std_err  = np.std(mae, ddof=1)
        s1, s2, s3 = mean_err + std_err, mean_err + 2*std_err, mean_err + 3*std_err

        ae_threshold = _pick_threshold(s1, s2, s3)

        return {
            'mae'          : mae,
            'sigma1'       : s1, 'sigma2': s2, 'sigma3': s3,
            'ae_is_anomaly': (mae > ae_threshold).astype(int),
            'ae_severity'  : _severity_labels(mae, s1, s2, s3),
            'reconstructed': np.tile(mean_vec, (n, 1)),
            'model'        : None,
            'skipped'      : True,
        }

    import tensorflow as tf
    tf.random.set_seed(42)

    model      = build_autoencoder(X_scaled.shape[1])
    X_train    = X_scaled[normal_mask] if normal_mask is not None else X_scaled
    n_train    = len(X_train)
    val_split  = AE_VAL_SPLIT if n_train >= 10 else 0.0

    callbacks = []
    if val_split > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=AE_PATIENCE, restore_best_weights=True
        ))

    model.fit(
        X_train, X_train,
        epochs           = AE_EPOCHS,
        batch_size       = min(AE_BATCH_SIZE, n_train),
        validation_split = val_split,
        shuffle          = True,
        verbose          = 0,
        callbacks        = callbacks,
    )

    reconstructed = model.predict(X_scaled, verbose=0)
    mae           = np.mean(np.abs(reconstructed - X_scaled), axis=1)
    mean_err      = np.mean(mae)
    std_err       = np.std(mae, ddof=1)
    s1, s2, s3    = mean_err + std_err, mean_err + 2*std_err, mean_err + 3*std_err

    ae_threshold = _pick_threshold(s1, s2, s3)

    return {
        'mae'          : mae,
        'sigma1'       : s1, 'sigma2': s2, 'sigma3': s3,
        'ae_is_anomaly': (mae > ae_threshold).astype(int),
        'ae_severity'  : _severity_labels(mae, s1, s2, s3),
        'reconstructed': reconstructed,
        'model'        : model,
        'skipped'      : False,
    }


def run_autoencoder_all(df: pd.DataFrame, products, scaled_dict: dict) -> tuple:
    """Run autoencoder for all products. Returns (df_ae, ae_results)."""
    ae_results = {}
    results    = []

    for product in products:
        subset      = df[df['product-name'] == product].copy()
        X_scaled    = scaled_dict[product]
        normal_mask = ~subset['fraud_any'].values

        res = run_autoencoder_product(X_scaled, product, normal_mask)
        ae_results[product] = res

        subset['ae_reconstruction_error'] = res['mae']
        subset['ae_severity']             = res['ae_severity']
        subset['ae_is_anomaly']           = res['ae_is_anomaly']
        subset['ae_sigma1']               = res['sigma1']
        subset['ae_sigma2']               = res['sigma2']
        subset['ae_sigma3']               = res['sigma3']

        results.append(subset[IDENTITY_COLS + [
            'ae_reconstruction_error', 'ae_severity',
            'ae_is_anomaly', 'ae_sigma1', 'ae_sigma2', 'ae_sigma3'
        ]])

    return pd.concat(results, ignore_index=True), ae_results


# =============================================================================
# SECTION 7 — Hotelling T²
# =============================================================================

def run_hotelling_t2_product(X_scaled: np.ndarray) -> tuple:
    """Compute Hotelling T² distances and UCL for one product."""
    n, p     = X_scaled.shape
    mean_vec = np.mean(X_scaled, axis=0)
    cov_mat  = np.cov(X_scaled, rowvar=False) + np.eye(p) * 1e-6
    inv_cov  = np.linalg.inv(cov_mat)

    t2 = np.array([
        (row - mean_vec) @ inv_cov @ (row - mean_vec)
        for row in X_scaled
    ])

    ucl     = (p * (n-1) * f_dist.ppf(1 - T2_ALPHA,         p, n-p)) / (n-p)
    warning = (p * (n-1) * f_dist.ppf(1 - T2_ALPHA_WARNING, p, n-p)) / (n-p)

    return t2, ucl, warning


def run_hotelling_all(df: pd.DataFrame, products, scaled_dict: dict) -> tuple:
    """Run Hotelling T² for all products. Returns (df_t2, t2_params)."""
    t2_params = {}
    results   = []

    for product in products:
        subset   = df[df['product-name'] == product].copy()
        X_scaled = scaled_dict[product]

        t2, ucl, warning = run_hotelling_t2_product(X_scaled)
        t2_params[product] = {'ucl': ucl, 'warning': warning}

        severity = np.where(
            t2 > ucl * 1.5, '🔴 High Risk — CAPA Required',
            np.where(
                t2 > ucl,     '🟠 Medium Risk — Investigate',
                np.where(
                    t2 > warning, '🟡 Low Risk — Monitor',
                    '✅ Normal'
                )
            )
        )

        subset['t2_distance']   = t2
        subset['t2_ucl']        = ucl
        subset['t2_warning']    = warning
        subset['t2_severity']   = severity
        subset['t2_is_outlier'] = (t2 > ucl).astype(int)

        results.append(subset[IDENTITY_COLS + [
            't2_distance', 't2_ucl', 't2_warning',
            't2_severity', 't2_is_outlier'
        ]])

    return pd.concat(results, ignore_index=True), t2_params
