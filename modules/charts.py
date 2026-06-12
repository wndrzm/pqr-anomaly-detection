# =============================================================================
# modules/charts.py — Section 14: All Visualisations (Plotly)
# =============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from config import PARAMETERS, COLOR_MAP, STATUS_ORDER, CUSUM_K, CUSUM_H


# =============================================================================
# CHART A — Reconstruction Error Control Chart
# =============================================================================

def plot_ae_control_chart(merged: pd.DataFrame, product: str) -> go.Figure:
    sub  = merged[merged['product-name'] == product].reset_index(drop=True)
    s1   = sub['ae_sigma1'].iloc[0] if 'ae_sigma1' in sub.columns else None
    s2   = sub['ae_sigma2'].iloc[0]
    s3   = sub['ae_sigma3'].iloc[0] if 'ae_sigma3' in sub.columns else None
    mean = sub['ae_reconstruction_error'].mean()

    norm = sub[sub['ae_is_anomaly'] == 0]
    anom = sub[sub['ae_is_anomaly'] == 1]

    fig = go.Figure()

    # Normal points
    fig.add_trace(go.Scatter(
        x=norm.index, y=norm['ae_reconstruction_error'],
        mode='lines+markers',
        name='Normal',
        line=dict(color='#457b9d', width=1.5),
        marker=dict(size=6),
        hovertemplate='Batch: %{customdata}<br>MAE: %{y:.4f}<extra></extra>',
        customdata=norm['batch-number'],
    ))

    # Anomaly points
    if len(anom) > 0:
        fig.add_trace(go.Scatter(
            x=anom.index, y=anom['ae_reconstruction_error'],
            mode='markers+text',
            name='Anomaly',
            marker=dict(color='#e63946', size=10, symbol='circle'),
            text=anom['batch-number'],
            textposition='top center',
            textfont=dict(size=9, color='#e63946'),
            hovertemplate='Batch: %{customdata}<br>MAE: %{y:.4f}<extra></extra>',
            customdata=anom['batch-number'],
        ))

    # Control lines
    fig.add_hline(y=mean, line_color='green',  line_dash='solid',  annotation_text=f'CL ({mean:.4f})')
    if s1: fig.add_hline(y=s1, line_color='gold',   line_dash='dash',   annotation_text=f'1σ ({s1:.4f})')
    fig.add_hline(y=s2,        line_color='orange', line_dash='dash',   annotation_text=f'2σ UCL ({s2:.4f})')
    if s3: fig.add_hline(y=s3, line_color='red',    line_dash='dash',   annotation_text=f'3σ ({s3:.4f})')

    fig.update_layout(
        title=f'Autoencoder Reconstruction Error — {product}',
        xaxis_title='Batch Index',
        yaxis_title='MAE',
        template='plotly_white',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    fig.update_xaxes(
        tickvals=sub.index[::4],
        ticktext=sub['batch-number'].iloc[::4],
        tickangle=45,
    )
    return fig


# =============================================================================
# CHART B — Hotelling T² Control Chart
# =============================================================================

def plot_t2_control_chart(merged: pd.DataFrame, product: str) -> go.Figure:
    sub     = merged[merged['product-name'] == product].reset_index(drop=True)
    ucl     = sub['t2_ucl'].iloc[0]
    warning = sub['t2_warning'].iloc[0] if 't2_warning' in sub.columns else None

    norm = sub[sub['t2_is_outlier'] == 0]
    out  = sub[sub['t2_is_outlier'] == 1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=norm.index, y=norm['t2_distance'],
        mode='lines+markers', name='Normal',
        line=dict(color='#457b9d', width=1.5),
        marker=dict(size=6),
        hovertemplate='Batch: %{customdata}<br>T²: %{y:.2f}<extra></extra>',
        customdata=norm['batch-number'],
    ))

    if len(out) > 0:
        fig.add_trace(go.Scatter(
            x=out.index, y=out['t2_distance'],
            mode='markers+text', name='Outlier',
            marker=dict(color='#e63946', size=10),
            text=out['batch-number'],
            textposition='top center',
            textfont=dict(size=9, color='#e63946'),
            hovertemplate='Batch: %{customdata}<br>T²: %{y:.2f}<extra></extra>',
            customdata=out['batch-number'],
        ))

    if warning:
        fig.add_hline(y=warning, line_color='gold', line_dash='dash',
                      annotation_text=f'Warning α=0.01 ({warning:.2f})')
    fig.add_hline(y=ucl, line_color='red', line_dash='dash',
                  annotation_text=f'UCL α=0.05 ({ucl:.2f})')

    fig.update_layout(
        title=f'Hotelling T² Control Chart — {product}',
        xaxis_title='Batch Index',
        yaxis_title='T² Distance',
        template='plotly_white',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    fig.update_xaxes(
        tickvals=sub.index[::4],
        ticktext=sub['batch-number'].iloc[::4],
        tickangle=45,
    )
    return fig


# =============================================================================
# CHART C — Detection Agreement Heatmap
# =============================================================================

def plot_agreement_heatmap(flagged: pd.DataFrame) -> go.Figure:
    if flagged.empty:
        fig = go.Figure()
        fig.add_annotation(text='No flagged batches', showarrow=False,
                           font=dict(size=16), xref='paper', yref='paper', x=0.5, y=0.5)
        return fig

    method_cols   = ['zscore_is_anomaly', 'iforest_is_anomaly',
                     'ae_is_anomaly', 't2_is_outlier', 'fraud_any']
    method_labels = ['Z-Score', 'Isolation Forest', 'Autoencoder', 'Hotelling T²', 'Fraud']

    flagged_sorted = flagged.sort_values(['product-name', 'batch-number']).reset_index(drop=True)
    hm_data = flagged_sorted[method_cols].fillna(0).astype(int).values.T

    text_matrix = [['✓' if v else '–' for v in row] for row in hm_data]

    fig = go.Figure(go.Heatmap(
        z=hm_data,
        x=flagged_sorted['batch-number'].tolist(),
        y=method_labels,
        text=text_matrix,
        texttemplate='%{text}',
        colorscale=[[0, '#f0f0f0'], [1, '#e63946']],
        showscale=False,
        hoverongaps=False,
        hovertemplate='Batch: %{x}<br>Method: %{y}<br>Flagged: %{z}<extra></extra>',
    ))

    fig.update_layout(
        title='Detection Agreement Heatmap (Flagged Batches)',
        xaxis_title='Batch Number',
        template='plotly_white',
        height=max(250, len(flagged_sorted) * 20 + 150),
        xaxis=dict(tickangle=45),
    )
    return fig


# =============================================================================
# CHART D — Final Status Summary (bar chart)
# =============================================================================

def plot_status_summary(merged: pd.DataFrame) -> go.Figure:
    summary = (
        merged.groupby(['product-name', 'final_status'])
        .size().reset_index(name='count')
    )

    fig = go.Figure()
    for status in STATUS_ORDER:
        sub = summary[summary['final_status'] == status]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            name=status.split('—')[0].strip(),
            x=sub['product-name'],
            y=sub['count'],
            marker_color=COLOR_MAP.get(status, '#888'),
            hovertemplate=f'{status}<br>Count: %{{y}}<extra></extra>',
        ))

    fig.update_layout(
        barmode='stack',
        title='Final Status by Product',
        xaxis_title='',
        yaxis_title='Number of Batches',
        template='plotly_white',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=10)),
    )
    return fig


# =============================================================================
# CHART E — Scatter: AE Error vs T² Distance
# =============================================================================

def plot_ae_vs_t2(merged: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    for status, color in COLOR_MAP.items():
        grp = merged[merged['final_status'] == status]
        if grp.empty:
            continue
        fig.add_trace(go.Scatter(
            x=grp['ae_reconstruction_error'],
            y=grp['t2_distance'],
            mode='markers',
            name=status.split('—')[0].strip(),
            marker=dict(color=color, size=8, opacity=0.85,
                        line=dict(color='white', width=0.5)),
            hovertemplate=(
                'Batch: %{customdata}<br>'
                'AE Error: %{x:.4f}<br>'
                'T²: %{y:.2f}<extra></extra>'
            ),
            customdata=grp['batch-number'],
        ))

    fig.update_layout(
        title='Autoencoder Error vs Hotelling T²',
        xaxis_title='Reconstruction Error (AE)',
        yaxis_title='T² Distance',
        template='plotly_white',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=10)),
    )
    return fig


# =============================================================================
# CHART F — Trend line per parameter per product
# =============================================================================

def plot_trend_line(df: pd.DataFrame, product: str, param: str) -> go.Figure:
    sub = df[df['product-name'] == product].sort_values('month').reset_index(drop=True)

    lsl_col = f'LSL_{param}'
    usl_col = f'USL_{param}'
    lsl = sub[lsl_col].iloc[0] if lsl_col in sub.columns else None
    usl = sub[usl_col].iloc[0] if usl_col in sub.columns else None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sub.index,
        y=sub[param],
        mode='lines+markers',
        name=param,
        line=dict(color='#457b9d', width=2),
        marker=dict(size=6),
        hovertemplate='Batch: %{customdata}<br>Value: %{y:.3f}<extra></extra>',
        customdata=sub['batch-number'],
    ))

    # Trend line
    if len(sub[param].dropna()) >= 2:
        x_arr = np.arange(len(sub))
        slope, intercept = np.polyfit(x_arr, sub[param].fillna(sub[param].mean()), 1)
        trend_y = slope * x_arr + intercept
        fig.add_trace(go.Scatter(
            x=x_arr, y=trend_y,
            mode='lines', name='Trend',
            line=dict(color='orange', dash='dash', width=1.5),
        ))

    if lsl: fig.add_hline(y=lsl, line_color='red',   line_dash='dot', annotation_text=f'LSL ({lsl})')
    if usl: fig.add_hline(y=usl, line_color='red',   line_dash='dot', annotation_text=f'USL ({usl})')

    fig.update_layout(
        title=f'{param} — {product}',
        xaxis_title='Batch Index',
        yaxis_title=param,
        template='plotly_white',
        height=350,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    fig.update_xaxes(
        tickvals=sub.index[::4],
        ticktext=sub['batch-number'].iloc[::4],
        tickangle=45,
    )
    return fig


# =============================================================================
# CHART G — CUSUM Chart
# =============================================================================

def plot_cusum(df: pd.DataFrame, product: str, param: str) -> go.Figure:
    sub    = df[df['product-name'] == product].sort_values('month').reset_index(drop=True)
    values = sub[param].values.astype(float)
    n      = len(values)
    mean   = np.mean(values)
    std    = np.std(values, ddof=1)

    if std == 0:
        fig = go.Figure()
        fig.add_annotation(text='std=0 — CUSUM not applicable',
                           showarrow=False, xref='paper', yref='paper', x=0.5, y=0.5)
        return fig

    z       = (values - mean) / std
    c_plus  = np.zeros(n)
    c_minus = np.zeros(n)
    for i in range(1, n):
        c_plus[i]  = max(0.0, c_plus[i-1]  + z[i] - CUSUM_K)
        c_minus[i] = max(0.0, c_minus[i-1] - z[i] - CUSUM_K)

    sig_up  = c_plus  > CUSUM_H
    sig_dn  = c_minus > CUSUM_H
    x_vals  = sub['batch-number'].tolist()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals, y=c_plus,
        mode='lines+markers', name='C⁺ (upward)',
        line=dict(color='#e63946', width=1.5), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=c_minus,
        mode='lines+markers', name='C⁻ (downward)',
        line=dict(color='#457b9d', width=1.5), marker=dict(size=5),
    ))

    if sig_up.any():
        fig.add_trace(go.Scatter(
            x=[x_vals[i] for i in range(n) if sig_up[i]],
            y=c_plus[sig_up],
            mode='markers', name='⬆ Signal',
            marker=dict(color='#e63946', size=12, symbol='star'),
        ))
    if sig_dn.any():
        fig.add_trace(go.Scatter(
            x=[x_vals[i] for i in range(n) if sig_dn[i]],
            y=c_minus[sig_dn],
            mode='markers', name='⬇ Signal',
            marker=dict(color='#457b9d', size=12, symbol='star'),
        ))

    fig.add_hline(y=CUSUM_H, line_color='black', line_dash='dash',
                  annotation_text=f'h = {CUSUM_H} (UCL)')

    n_signals = int(sig_up.sum() + sig_dn.sum())
    fig.update_layout(
        title=f'CUSUM — {param} | {product} | signals={n_signals}',
        xaxis_title='Batch Number',
        yaxis_title='CUSUM Statistic',
        template='plotly_white',
        height=350,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        xaxis=dict(tickangle=45),
    )
    return fig
