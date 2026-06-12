# =============================================================================
# modules/evaluation.py — Section 18: Model Evaluation (Precision/Recall/F1)
# =============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Ground truth — anomalies embedded in dataset_pqr_v5
# Sources: OOS, Fraud, Outlier (Trend excluded — not batch-level label)
GROUND_TRUTH = {
    # OOS
    'PAR-2023-030',                                        # Paracetamol — yield OOS
    'AMO-2023-015',                                        # Amoxicillin — dissolution OOS
    'MET-2023-021', 'MET-2023-022',                        # Metformin   — hardness OOS
    'IBU-2023-041',                                        # Ibuprofen   — % assay OOS
    # Fraud
    'PAR-2023-012', 'PAR-2023-013',                        # Paracetamol — copy-paste
    'AMO-2023-026', 'AMO-2023-027', 'AMO-2023-028',        # Amoxicillin — near-identical
    'MET-2023-034', 'MET-2023-035',                        # Metformin   — near-identical
    'IBU-2023-007', 'IBU-2023-008',                        # Ibuprofen   — near-identical
    # Outlier
    'PAR-2023-006', 'PAR-2023-023', 'PAR-2023-039',        # Paracetamol
    'AMO-2023-009', 'AMO-2023-020', 'AMO-2023-042',        # Amoxicillin
    'MET-2023-004', 'MET-2023-028', 'MET-2023-045',        # Metformin
    'IBU-2023-011', 'IBU-2023-032', 'IBU-2023-044',        # Ibuprofen
}


def evaluate_model(
    df_eval:    pd.DataFrame,
    pred_col:   str,
    model_name: str,
    gt_set:     set = None,
) -> dict:
    """
    Compute precision, recall, F1 for one model vs ground truth.

    pred_col : binary column (1 = anomaly, 0 = normal)
               OR string column where non-'✅ Normal' = anomaly (for ensemble)
    """
    if gt_set is None:
        gt_set = GROUND_TRUTH

    if df_eval[pred_col].dtype == object:
        y_pred = (df_eval[pred_col] != '✅ Normal').astype(int)
    else:
        y_pred = df_eval[pred_col].fillna(0).astype(int)

    y_true = df_eval['batch-number'].isin(gt_set).astype(int)

    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'model':     model_name,
        'TP':        TP,
        'FP':        FP,
        'FN':        FN,
        'TN':        TN,
        'precision': round(precision, 3),
        'recall':    round(recall,    3),
        'f1':        round(f1,        3),
    }


def run_evaluation(merged: pd.DataFrame) -> pd.DataFrame:
    """Run evaluation for all models. Returns eval_df."""
    models_to_eval = [
        ('zscore_is_anomaly',   'Z-Score'),
        ('iforest_is_anomaly',  'Isolation Forest'),
        ('ae_is_anomaly',       'Autoencoder'),
        ('t2_is_outlier',       'Hotelling T²'),
        ('fraud_any',           'Fraud Detection'),
        ('ensemble_is_anomaly', 'Ensemble (weighted)'),
        ('final_status',        'Ensemble (final)'),
    ]

    results = []
    for col, name in models_to_eval:
        if col not in merged.columns:
            continue
        results.append(evaluate_model(merged, col, name))

    return pd.DataFrame(results)


def plot_evaluation(eval_df: pd.DataFrame) -> go.Figure:
    """Bar chart + confusion matrix heatmap side by side."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Precision / Recall / F1 per Model',
            'Confusion Matrix Components'
        )
    )

    models = eval_df['model'].tolist()
    x      = list(range(len(models)))
    colors = {'precision': '#457b9d', 'recall': '#e63946', 'f1': '#2a9d8f'}

    # Panel 1: Bar chart
    for metric, color in colors.items():
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=models,
            y=eval_df[metric].tolist(),
            marker_color=color,
            opacity=0.85,
            text=[f'{v:.2f}' for v in eval_df[metric]],
            textposition='outside',
            textfont=dict(size=9),
        ), row=1, col=1)

    # Panel 2: Heatmap
    conf_data = eval_df[['TP', 'FP', 'FN', 'TN']].values.T.tolist()
    fig.add_trace(go.Heatmap(
        z=conf_data,
        x=models,
        y=['TP', 'FP', 'FN', 'TN'],
        colorscale='Blues',
        showscale=False,
        text=[[str(v) for v in row] for row in conf_data],
        texttemplate='%{text}',
        hovertemplate='Model: %{x}<br>%{y}: %{z}<extra></extra>',
    ), row=1, col=2)

    fig.update_layout(
        barmode='group',
        template='plotly_white',
        height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.05),
        xaxis=dict(tickangle=20),
        xaxis2=dict(tickangle=20),
    )

    fig.update_yaxes(range=[0, 1.15], row=1, col=1)

    return fig


def get_evaluation_summary(eval_df: pd.DataFrame) -> dict:
    """Return best model and ensemble performance for display."""
    ensemble_row = eval_df[eval_df['model'] == 'Ensemble (final)']

    if ensemble_row.empty:
        return {}

    best_recall = eval_df.loc[eval_df['recall'].idxmax()]

    return {
        'ensemble_precision': float(ensemble_row['precision'].iloc[0]),
        'ensemble_recall':    float(ensemble_row['recall'].iloc[0]),
        'ensemble_f1':        float(ensemble_row['f1'].iloc[0]),
        'best_recall_model':  best_recall['model'],
        'best_recall_value':  float(best_recall['recall']),
        'n_ground_truth':     len(GROUND_TRUTH),
    }