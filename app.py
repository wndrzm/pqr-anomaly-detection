# =============================================================================
# app.py — PQR Anomaly Detection Dashboard (Gradio 6.x)
# =============================================================================

import gradio as gr
import pandas as pd
import numpy as np
import io
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    PARAMETERS, DEFAULT_DATASET, STATUS_ORDER, COLOR_MAP, VOTE_WEIGHTS,
    WEIGHTED_THRESHOLD, ZSCORE_THRESHOLD, CUSUM_K, CUSUM_H,
)
from modules.data_loader   import run_data_pipeline
from modules.fraud         import detect_fraud, get_fraud_summary
from modules.preprocessing import run_preprocessing
from modules.models        import run_zscore_iforest, run_autoencoder_all, run_hotelling_all
from modules.ensemble      import merge_results, run_ensemble, get_ensemble_summary
from modules.risk_context  import compute_risk_context, get_risk_summary
from modules.trend         import run_trend_analysis, get_trend_summary
from modules.capability    import run_capability_analysis, get_capability_summary
from modules.evaluation    import run_evaluation, plot_evaluation, get_evaluation_summary
from modules.executive     import generate_executive_summary
from modules.charts        import (
    plot_ae_control_chart, plot_t2_control_chart,
    plot_agreement_heatmap, plot_status_summary,
    plot_ae_vs_t2, plot_trend_line, plot_cusum,
)

# =============================================================================
# GLOBAL STATE
# =============================================================================

EMPTY_STATE = {
    'data_loaded':      False,
    'model_run':        False,
    'df':               None,
    'products':         [],
    'merged':           None,
    'risk_df':          None,
    'trend_df':         None,
    'cap_df':           None,
    'eval_df':          None,
    'ae_results':       None,
    'scaled_dict':      None,
    'fraud_summary':    None,
    'ensemble_summary': None,
}

# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(uploaded_file, use_demo, state):
    log  = []
    state = dict(state)

    try:
        log.append('📂 Loading and validating data...')
        result = run_data_pipeline(
            uploaded_file = None if use_demo == 'Use synthetic demo dataset' else uploaded_file,
            default_path  = DEFAULT_DATASET,
        )
        df       = result['df']
        products = result['products']
        if result['missing_report']:
            log.append(f"⚠️  Missing values filled: {result['missing_report']}")
        log.append(f"✅ {result['spec_status']}")
    except Exception as e:
        return state, f'❌ Data loading failed: {e}', *_empty_outputs()

    log.append('🔍 Running fraud detection...')
    df = detect_fraud(df)
    fraud_summary = get_fraud_summary(df)
    log.append(f"{'⚠️  ' + str(fraud_summary['total_flagged']) + ' batch(es) flagged' if fraud_summary['total_flagged'] > 0 else '✅ No fraud detected'}")

    log.append('⚙️  Preprocessing...')
    scalers, scaled_dict = run_preprocessing(df, products)

    log.append('📊 Running Z-Score & Isolation Forest...')
    df_zs_if = run_zscore_iforest(df, products, scaled_dict)

    log.append('🧠 Training Autoencoder...')
    df_ae, ae_results = run_autoencoder_all(df, products, scaled_dict)

    log.append('📐 Running Hotelling T²...')
    df_t2, t2_params = run_hotelling_all(df, products, scaled_dict)

    log.append('🗳️  Computing ensemble scores...')
    merged = merge_results(df, df_zs_if, df_ae, df_t2)
    merged = run_ensemble(merged)
    ensemble_summary = get_ensemble_summary(merged)

    log.append('⚠️  Risk Contextualization & Trend Analysis...')
    flagged  = merged[merged['final_status'] != '✅ Normal'].copy()
    risk_df  = compute_risk_context(df, flagged)
    trend_df = run_trend_analysis(df)

    log.append('🏭 Process Capability Analysis...')
    cap_df = run_capability_analysis(df)

    log.append('🎯 Model Evaluation...')
    eval_df = run_evaluation(merged)

    log.append('✅ Analysis complete!')

    state.update({
        'data_loaded': True, 'model_run': True,
        'df': df, 'products': products, 'merged': merged,
        'risk_df': risk_df, 'trend_df': trend_df, 'cap_df': cap_df,
        'eval_df': eval_df, 'ae_results': ae_results, 'scaled_dict': scaled_dict,
        'fraud_summary': fraud_summary, 'ensemble_summary': ensemble_summary,
    })

    return (state, '\n'.join(log)) + tuple(_build_outputs(state))


def _empty_outputs():
    empty_dropdown = gr.Dropdown(choices=[], value=None)
    return (
        '_Run analysis to see results._',   # overview_metrics_md
        None, None, None,                   # 3 plots
        '',                                 # fraud_md
        pd.DataFrame(),                     # fraud_table
        empty_dropdown,                     # tab2_product
        None, None,                         # ae/t2 charts
        pd.DataFrame(),                     # flagged_table
        empty_dropdown,                     # tab3_product
        pd.DataFrame(),                     # risk_table
        empty_dropdown,                     # tab4_product
        empty_dropdown,                     # tab4_param
        '',                                 # trend_metrics_md
        pd.DataFrame(),                     # trend_alerts
        None, None,                         # trend/cusum plots
        '',                                 # mk_info
        empty_dropdown,                     # tab5_product
        pd.DataFrame(),                     # cap_table
        '',                                 # eval_metrics_md
        None,                               # eval plot
        pd.DataFrame(),                     # eval_table
        '',                                 # exec_text
        pd.DataFrame(),                     # preview_table
    )


def _build_outputs(state):
    df               = state['df']
    products         = state['products']
    merged           = state['merged']
    risk_df          = state['risk_df']
    trend_df         = state['trend_df']
    cap_df           = state['cap_df']
    eval_df          = state['eval_df']
    fraud_summary    = state['fraud_summary']
    ensemble_summary = state['ensemble_summary']

    # TAB 1
    counts = ensemble_summary['counts']
    overview_md = (
        f"**Total Batches:** {ensemble_summary['total']} &nbsp;|&nbsp; "
        f"🔴 Data Fraud: **{counts.get('🔴 Data Fraud — Duplicate/Copy-Paste', 0)}** &nbsp;|&nbsp; "
        f"🔴 Confirmed: **{counts.get('🔴 Confirmed Anomaly — Investigate & CAPA', 0)}** &nbsp;|&nbsp; "
        f"🟠 Suspected: **{counts.get('🟠 Suspected Anomaly — Enhanced Monitoring', 0)}** &nbsp;|&nbsp; "
        f"🟡 Watch List: **{counts.get('🟡 Watch List — Re-check Next Batch', 0)}**"
    )
    fig_status    = plot_status_summary(merged)
    fig_ae_t2     = plot_ae_vs_t2(merged)
    fig_agreement = plot_agreement_heatmap(merged[merged['final_status'] != '✅ Normal'].copy())

    fraud_md  = ''
    fraud_tbl = pd.DataFrame()
    if fraud_summary['total_flagged'] > 0:
        fraud_md = (
            f"🚨 **Exact Duplicates:** {fraud_summary['exact_duplicates']} &nbsp;|&nbsp; "
            f"**Near-Duplicates:** {fraud_summary['near_duplicates']} &nbsp;|&nbsp; "
            f"**Suspicious Rounding:** {fraud_summary['suspicious_rounding']}"
        )
        fraud_tbl = fraud_summary['flagged_batches']

    # TAB 2
    first_product = products[0] if products else None
    display_cols = [
        'batch-number', 'product-name', 'month',
        'zscore_is_anomaly', 'iforest_is_anomaly', 'ae_is_anomaly', 't2_is_outlier',
        'fraud_any', 'vote_count', 'weighted_score', 'final_status'
    ]
    rename_map = {
        'zscore_is_anomaly': 'Z-Score', 'iforest_is_anomaly': 'IF',
        'ae_is_anomaly': 'AE', 't2_is_outlier': 'T²',
        'fraud_any': 'Fraud', 'vote_count': 'Votes',
        'weighted_score': 'Score', 'final_status': 'Status',
    }
    fig_ae_cc = fig_t2_cc = None
    flagged_prod = pd.DataFrame()
    if first_product:
        fig_ae_cc    = plot_ae_control_chart(merged, first_product)
        fig_t2_cc    = plot_t2_control_chart(merged, first_product)
        flagged_prod = merged[
            (merged['product-name'] == first_product) &
            (merged['final_status'] != '✅ Normal')
        ][[c for c in display_cols if c in merged.columns]].rename(columns=rename_map)

    tab2_dd = gr.Dropdown(choices=products, value=first_product, label='Select product')

    # TAB 3
    risk_cols = ['batch-number', 'product-name', 'parameter',
                 'actual_value', 'LSL', 'USL', 'dist_to_limit_%', 'risk_label']
    risk_tbl = risk_df[[c for c in risk_cols if c in risk_df.columns]] if not risk_df.empty else pd.DataFrame()
    tab3_dd  = gr.Dropdown(choices=['All'] + products, value='All', label='Filter by product')

    # TAB 4
    trend_summary    = get_trend_summary(trend_df)
    trend_metrics_md = (
        f"🔴 Attention: **{trend_summary['n_attention']}** &nbsp;|&nbsp; "
        f"🟡 Monitor: **{trend_summary['n_monitor']}** &nbsp;|&nbsp; "
        f"🟢 Safe: **{trend_summary['n_safe']}**"
    )
    trend_alerts = trend_summary['alerts'][[
        'product-name', 'parameter', 'tau', 'p_value', 'slope_per_batch', 'risk_assessment'
    ]] if not trend_summary['alerts'].empty else pd.DataFrame()

    first_param = PARAMETERS[0]
    fig_trend = plot_trend_line(df, first_product, first_param) if first_product else None
    fig_cusum = plot_cusum(df, first_product, first_param) if first_product else None

    mk_info = ''
    if first_product:
        mk_row = trend_df[
            (trend_df['product-name'] == first_product) &
            (trend_df['parameter']    == first_param)
        ]
        if not mk_row.empty:
            r = mk_row.iloc[0]
            mk_info = (
                f"**Mann-Kendall:** {r['trend_label']} | τ = {r['tau']} | "
                f"p = {r['p_value']} | slope = {r['slope_per_batch']:+.4f}/batch | "
                f"**{r['risk_assessment']}**"
            )

    tab4_prod_dd  = gr.Dropdown(choices=products, value=first_product, label='Product')
    tab4_param_dd = gr.Dropdown(choices=PARAMETERS, value=first_param, label='Parameter')

    # TAB 5
    cap_cols = ['product-name', 'parameter', 'n', 'mean', 'std',
                'LSL', 'USL', 'Cp', 'Cpk', 'CPL', 'CPU', 'interpretation']
    cap_tbl = cap_df[[c for c in cap_cols if c in cap_df.columns]] if not cap_df.empty else pd.DataFrame()
    tab5_dd = gr.Dropdown(choices=['All'] + products, value='All', label='Filter by product')

    # TAB 6
    eval_summary = get_evaluation_summary(eval_df)
    eval_md = (
        f"**Precision:** {eval_summary['ensemble_precision']:.3f} &nbsp;|&nbsp; "
        f"**Recall:** {eval_summary['ensemble_recall']:.3f} &nbsp;|&nbsp; "
        f"**F1:** {eval_summary['ensemble_f1']:.3f}  \n"
        f"Best recall: **{eval_summary['best_recall_model']}** ({eval_summary['best_recall_value']:.3f})"
    ) if eval_summary else 'No evaluation data.'
    fig_eval = plot_evaluation(eval_df)

    # TAB 7
    report_year = int(df['year'].max()) if 'year' in df.columns else 2023
    exec_text = generate_executive_summary(df, risk_df, trend_df, year=report_year)

    # TAB 8 preview
    export_cols = [
        'batch-number', 'product-name', 'year', 'month', *PARAMETERS,
        'fraud_any', 'zscore_is_anomaly', 'iforest_is_anomaly',
        'ae_is_anomaly', 't2_is_outlier',
        'vote_count', 'weighted_score', 'ensemble_is_anomaly', 'final_status',
    ]
    preview = merged[[c for c in export_cols if c in merged.columns]].head(20)

    return [
        overview_md, fig_status, fig_ae_t2, fig_agreement,
        fraud_md, fraud_tbl,
        tab2_dd, fig_ae_cc, fig_t2_cc, flagged_prod,
        tab3_dd, risk_tbl,
        tab4_prod_dd, tab4_param_dd,
        trend_metrics_md, trend_alerts,
        fig_trend, fig_cusum, mk_info,
        tab5_dd, cap_tbl,
        eval_md, fig_eval, eval_df,
        exec_text,
        preview,
    ]


# =============================================================================
# CALLBACKS
# =============================================================================

def update_tab2(product, state):
    if not state.get('model_run') or not product:
        return None, None, pd.DataFrame()
    merged = state['merged']
    display_cols = [
        'batch-number', 'product-name', 'month',
        'zscore_is_anomaly', 'iforest_is_anomaly', 'ae_is_anomaly', 't2_is_outlier',
        'fraud_any', 'vote_count', 'weighted_score', 'final_status'
    ]
    rename_map = {
        'zscore_is_anomaly': 'Z-Score', 'iforest_is_anomaly': 'IF',
        'ae_is_anomaly': 'AE', 't2_is_outlier': 'T²',
        'fraud_any': 'Fraud', 'vote_count': 'Votes',
        'weighted_score': 'Score', 'final_status': 'Status',
    }
    return (
        plot_ae_control_chart(merged, product),
        plot_t2_control_chart(merged, product),
        merged[
            (merged['product-name'] == product) &
            (merged['final_status'] != '✅ Normal')
        ][[c for c in display_cols if c in merged.columns]].rename(columns=rename_map),
    )


def update_tab3(product_filter, state):
    if not state.get('model_run'):
        return pd.DataFrame()
    risk_df   = state['risk_df']
    risk_cols = ['batch-number', 'product-name', 'parameter',
                 'actual_value', 'LSL', 'USL', 'dist_to_limit_%', 'risk_label']
    if risk_df.empty:
        return pd.DataFrame()
    df = risk_df if product_filter == 'All' else risk_df[risk_df['product-name'] == product_filter]
    return df[[c for c in risk_cols if c in df.columns]]


def update_tab4(product, param, state):
    if not state.get('model_run') or not product or not param:
        return None, None, ''
    df       = state['df']
    trend_df = state['trend_df']
    fig_trend = plot_trend_line(df, product, param)
    fig_cusum = plot_cusum(df, product, param)
    mk_row = trend_df[
        (trend_df['product-name'] == product) &
        (trend_df['parameter']    == param)
    ]
    mk_info = ''
    if not mk_row.empty:
        r = mk_row.iloc[0]
        mk_info = (
            f"**Mann-Kendall:** {r['trend_label']} | τ = {r['tau']} | "
            f"p = {r['p_value']} | slope = {r['slope_per_batch']:+.4f}/batch | "
            f"**{r['risk_assessment']}**"
        )
    return fig_trend, fig_cusum, mk_info


def update_tab5(product_filter, state):
    if not state.get('model_run'):
        return pd.DataFrame()
    cap_df   = state['cap_df']
    cap_cols = ['product-name', 'parameter', 'n', 'mean', 'std',
                'LSL', 'USL', 'Cp', 'Cpk', 'CPL', 'CPU', 'interpretation']
    if cap_df.empty:
        return pd.DataFrame()
    df = cap_df if product_filter == 'All' else cap_df[cap_df['product-name'] == product_filter]
    return df[[c for c in cap_cols if c in df.columns]]


def save_and_download_csv(state):
    if not state.get('model_run'):
        return None
    merged = state['merged']
    export_cols = [
        'batch-number', 'product-name', 'year', 'month', *PARAMETERS,
        'fraud_exact_dup', 'fraud_near_dup', 'fraud_round', 'fraud_any',
        'zscore_max_z', 'zscore_is_anomaly', 'iforest_score', 'iforest_is_anomaly',
        'ae_reconstruction_error', 'ae_is_anomaly', 't2_distance', 't2_is_outlier',
        'vote_count', 'weighted_score', 'ensemble_is_anomaly', 'final_status',
    ]
    out  = merged[[c for c in export_cols if c in merged.columns]].sort_values(['product-name', 'batch-number'])
    path = '/tmp/pqr_anomaly_results.csv'
    out.to_csv(path, index=False)
    return path


def save_and_download_excel(state):
    if not state.get('model_run'):
        return None
    merged   = state['merged']
    products = state['products']
    export_cols = [
        'batch-number', 'product-name', 'year', 'month', *PARAMETERS,
        'fraud_exact_dup', 'fraud_near_dup', 'fraud_round', 'fraud_any',
        'zscore_max_z', 'zscore_is_anomaly', 'iforest_score', 'iforest_is_anomaly',
        'ae_reconstruction_error', 'ae_is_anomaly', 't2_distance', 't2_is_outlier',
        'vote_count', 'weighted_score', 'ensemble_is_anomaly', 'final_status',
    ]
    out  = merged[[c for c in export_cols if c in merged.columns]].sort_values(['product-name', 'batch-number'])
    path = '/tmp/pqr_anomaly_results.xlsx'
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        out.to_excel(writer, sheet_name='All Products', index=False)
        for product in products:
            safe = product.replace(' ', '_').replace('/', '-')[:31]
            out[out['product-name'] == product].to_excel(writer, sheet_name=safe, index=False)
        for sheet_name, df_obj in [
            ('Risk Context', state['risk_df']),
            ('Trend Analysis', state['trend_df']),
            ('Capability', state['cap_df']),
            ('Model Evaluation', state['eval_df']),
        ]:
            if df_obj is not None and not df_obj.empty:
                df_obj.to_excel(writer, sheet_name=sheet_name, index=False)
    return path


def save_exec_txt(text):
    path = '/tmp/executive_summary_PQR.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


# =============================================================================
# UI
# =============================================================================

MODEL_WEIGHTS_MD = f"""
| Model | Weight |
|---|---|
| Z-Score | {VOTE_WEIGHTS['zscore']} |
| Isolation Forest | {VOTE_WEIGHTS['iforest']} |
| Autoencoder | {VOTE_WEIGHTS['autoencoder']} |
| Hotelling T² | {VOTE_WEIGHTS['hotelling']} |
| **Threshold** | **{WEIGHTED_THRESHOLD}** |
"""

CSS = """
#run-btn { font-weight: 700; font-size: 1rem; }
"""

with gr.Blocks(title='PQR Anomaly Detection', theme=gr.themes.Soft(), css=CSS) as demo:

    state = gr.State(dict(EMPTY_STATE))

    gr.Markdown("""
## 🔬 PQR Anomaly Detection Dashboard
**Ensemble anomaly detection for pharmaceutical batch quality review** — Z-Score · Isolation Forest · Autoencoder · Hotelling T²
""")

    with gr.Tabs():

        # ── Settings ─────────────────────────────────────────────────────────
        with gr.Tab('⚙️ Settings & Run'):
            with gr.Row():
                with gr.Column():
                    gr.Markdown('### 📂 Dataset')
                    use_demo = gr.Radio(
                        ['Use synthetic demo dataset', 'Upload my own dataset'],
                        value='Use synthetic demo dataset',
                        label='Data source',
                    )
                    uploaded_file = gr.File(
                        label='Upload Excel (.xlsx)',
                        file_types=['.xlsx'],
                        visible=False,
                    )
                    gr.Markdown('_Must have a "Data" sheet: year, month, product-name, batch-number + parameters_')
                with gr.Column():
                    gr.Markdown('### ⚙️ Model Weights')
                    gr.Markdown(MODEL_WEIGHTS_MD)

            run_btn = gr.Button('🚀 Run Analysis', variant='primary', elem_id='run-btn')
            run_log = gr.Textbox(label='Pipeline log', lines=14, interactive=False)

            use_demo.change(
                fn=lambda v: gr.update(visible=(v == 'Upload my own dataset')),
                inputs=use_demo, outputs=uploaded_file,
            )

        # ── Tab 1: Overview ───────────────────────────────────────────────────
        with gr.Tab('📊 Overview'):
            overview_metrics_md = gr.Markdown('_Run analysis to see results._')
            with gr.Row():
                plot_status   = gr.Plot(label='Batch Status Distribution')
                plot_ae_vs_t2 = gr.Plot(label='AE vs Hotelling T²')
            plot_agreement = gr.Plot(label='Detection Agreement Heatmap')
            fraud_md       = gr.Markdown()
            fraud_table    = gr.Dataframe(label='Flagged Fraud Batches')

        # ── Tab 2: Anomaly Detail ─────────────────────────────────────────────
        with gr.Tab('🔍 Anomaly Detail'):
            tab2_product = gr.Dropdown(choices=[], label='Select product')
            with gr.Row():
                plot_ae_cc = gr.Plot(label='Autoencoder Control Chart')
                plot_t2_cc = gr.Plot(label='Hotelling T² Control Chart')
            flagged_table = gr.Dataframe(label='Flagged Batches')
            tab2_product.change(fn=update_tab2, inputs=[tab2_product, state],
                                outputs=[plot_ae_cc, plot_t2_cc, flagged_table])

        # ── Tab 3: Risk Context ───────────────────────────────────────────────
        with gr.Tab('⚠️ Risk Context'):
            gr.Markdown('_How close are flagged batch parameters to specification limits?_')
            tab3_product = gr.Dropdown(choices=[], label='Filter by product')
            risk_table   = gr.Dataframe(label='Risk Context Table')
            tab3_product.change(fn=update_tab3, inputs=[tab3_product, state], outputs=risk_table)

        # ── Tab 4: Trends & CUSUM ─────────────────────────────────────────────
        with gr.Tab('📈 Trends & CUSUM'):
            trend_metrics_md   = gr.Markdown()
            trend_alerts_table = gr.Dataframe(label='Parameters requiring attention')
            with gr.Row():
                tab4_product = gr.Dropdown(choices=[], label='Product')
                tab4_param   = gr.Dropdown(choices=[], label='Parameter')
            with gr.Tabs():
                with gr.Tab('Mann-Kendall Trend'):
                    plot_trend = gr.Plot()
                    mk_info_md = gr.Markdown()
                with gr.Tab('CUSUM Chart'):
                    plot_cusum_fig = gr.Plot()
                    gr.Markdown(f'k = {CUSUM_K}σ | h = {CUSUM_H}σ | Reference: Montgomery (2009)')
            for inp in [tab4_product, tab4_param]:
                inp.change(fn=update_tab4, inputs=[tab4_product, tab4_param, state],
                           outputs=[plot_trend, plot_cusum_fig, mk_info_md])

        # ── Tab 5: Capability ─────────────────────────────────────────────────
        with gr.Tab('🏭 Capability'):
            gr.Markdown('_Cp/Cpk per parameter per product._')
            tab5_product = gr.Dropdown(choices=[], label='Filter by product')
            cap_table    = gr.Dataframe(label='Capability Table')
            tab5_product.change(fn=update_tab5, inputs=[tab5_product, state], outputs=cap_table)

        # ── Tab 6: Model Performance ──────────────────────────────────────────
        with gr.Tab('🎯 Model Performance'):
            eval_metrics_md = gr.Markdown()
            plot_eval_fig   = gr.Plot(label='Precision / Recall / F1')
            eval_table      = gr.Dataframe(label='Detailed Metrics')

        # ── Tab 7: Executive Summary ──────────────────────────────────────────
        with gr.Tab('📄 Executive Summary'):
            exec_summary_text = gr.Textbox(label='Executive Summary', lines=20, interactive=False)
            dl_txt_btn  = gr.Button('⬇️ Download as .txt')
            dl_txt_file = gr.File(label='Download .txt')
            dl_txt_btn.click(fn=save_exec_txt, inputs=exec_summary_text, outputs=dl_txt_file)

        # ── Tab 8: Export ─────────────────────────────────────────────────────
        with gr.Tab('💾 Export'):
            with gr.Row():
                dl_csv_btn   = gr.Button('⬇️ Download CSV')
                dl_excel_btn = gr.Button('⬇️ Download Excel (all sheets)')
            dl_csv_file   = gr.File(label='CSV')
            dl_excel_file = gr.File(label='Excel')
            preview_table = gr.Dataframe(label='Preview (first 20 rows)')
            dl_csv_btn.click(fn=save_and_download_csv, inputs=state, outputs=dl_csv_file)
            dl_excel_btn.click(fn=save_and_download_excel, inputs=state, outputs=dl_excel_file)

    # ── Wire run button ───────────────────────────────────────────────────────
    run_btn.click(
        fn=run_pipeline,
        inputs=[uploaded_file, use_demo, state],
        outputs=[
            state, run_log,
            overview_metrics_md, plot_status, plot_ae_vs_t2, plot_agreement,
            fraud_md, fraud_table,
            tab2_product, plot_ae_cc, plot_t2_cc, flagged_table,
            tab3_product, risk_table,
            tab4_product, tab4_param,
            trend_metrics_md, trend_alerts_table,
            plot_trend, plot_cusum_fig, mk_info_md,
            tab5_product, cap_table,
            eval_metrics_md, plot_eval_fig, eval_table,
            exec_summary_text,
            preview_table,
        ],
    )

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860)
