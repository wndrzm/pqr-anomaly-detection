# =============================================================================
# app.py — PQR Anomaly Detection Dashboard (Gradio Version)
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
# WRAPPER PIPELINE FOR GRADIO
# =============================================================================

def execute_pipeline(data_source, uploaded_file):
    # Determine file path
    file_to_load = None
    if data_source == 'Upload my own dataset':
        if uploaded_file is None:
            return "Please upload an Excel file first when selecting the upload option.", None, None, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        file_to_load = uploaded_file.name

    try:
        # Step 1: Load data
        result = run_data_pipeline(uploaded_file=file_to_load, default_path=DEFAULT_DATASET)
        df = result['df']
        products = list(result['products'])

        # Step 2: Fraud detection
        df = detect_fraud(df)
        fraud_summary = get_fraud_summary(df)

        # Step 3: Preprocessing
        scalers, scaled_dict = run_preprocessing(df, products)

        # Step 4: Z-Score + Isolation Forest
        df_zs_if = run_zscore_iforest(df, products, scaled_dict)

        # Step 5: Autoencoder
        df_ae, ae_results = run_autoencoder_all(df, products, scaled_dict)

        # Step 6: Hotelling T²
        df_t2, t2_params = run_hotelling_all(df, products, scaled_dict)

        # Step 7: Merge + Ensemble
        merged = merge_results(df, df_zs_if, df_ae, df_t2)
        merged = run_ensemble(merged)
        ensemble_summary = get_ensemble_summary(merged)

        # Step 8: Risk Context + Trend
        flagged = merged[merged['final_status'] != '✅ Normal'].copy()
        risk_df = compute_risk_context(df, flagged)
        trend_df = run_trend_analysis(df)

        # Step 9: Capability Analysis
        cap_df = run_capability_analysis(df)

        # Step 10: Model Evaluation
        eval_df = run_evaluation(merged)
        eval_summary = get_evaluation_summary(eval_df)

        # Step 11: Executive Summary Text
        report_year = int(df['year'].max()) if 'year' in df.columns else 2026
        summary_txt = generate_executive_summary(df, risk_df, trend_df, year=report_year)

        # --- PREPARE EXPORT FILES ---
        export_cols = [
            'batch-number', 'product-name', 'year', 'month', *PARAMETERS,
            'fraud_exact_dup', 'fraud_near_dup', 'fraud_round', 'fraud_any',
            'zscore_max_z', 'zscore_is_anomaly', 'iforest_score', 'iforest_is_anomaly',
            'ae_reconstruction_error', 'ae_is_anomaly', 't2_distance', 't2_is_outlier',
            'vote_count', 'weighted_score', 'ensemble_is_anomaly', 'final_status',
        ]
        out_df = merged[[c for c in export_cols if c in merged.columns]].sort_values(['product-name', 'batch-number'])
        
        # Save CSV Temp
        csv_path = "pqr_anomaly_results.csv"
        out_df.to_csv(csv_path, index=False)

        # Save Excel Temp
        excel_path = "pqr_anomaly_results.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            out_df.to_excel(writer, sheet_name='All Products', index=False)
            for prod in products:
                safe_name = prod.replace(' ', '_').replace('/', '-')[:31]
                out_df[out_df['product-name'] == prod].to_excel(writer, sheet_name=safe_name, index=False)
            if not risk_df.empty: risk_df.to_excel(writer, sheet_name='Risk Context', index=False)
            if not trend_df.empty: trend_df.to_excel(writer, sheet_name='Trend Analysis', index=False)
            if not cap_df.empty: cap_df.to_excel(writer, sheet_name='Capability', index=False)
            if not eval_df.empty: eval_df.to_excel(writer, sheet_name='Model Evaluation', index=False)

        # --- PREPARE METRICS & PLOTS ---
        counts = ensemble_summary['counts']
        overview_metrics = f"""
        ### 📊 Overview Metrics
        * **Total Batches:** {ensemble_summary['total']}
        * **🔴 Data Fraud:** {counts.get('🔴 Data Fraud — Duplicate/Copy-Paste', 0)}
        * **🔴 Confirmed Anomaly:** {counts.get('🔴 Confirmed Anomaly — Investigate & CAPA', 0)}
        * **🟠 Suspected Anomaly:** {counts.get('🟠 Suspected Anomaly — Enhanced Monitoring', 0)}
        * **🟡 Watch List:** {counts.get('🟡 Watch List — Re-check Next Batch', 0)}
        """

        risk_sum = get_risk_summary(risk_df)
        risk_metrics = f"""
        ### ⚠️ Risk Context Summary
        * **🔴 OOS:** {risk_sum['n_oos']} | **🟠 Near Limit:** {risk_sum['n_near']} | **🟡 OOT Monitor:** {risk_sum['n_monitor']} | **🟢 Within Spec:** {risk_sum['n_ok']}
        """

        trend_sum = get_trend_summary(trend_df)
        trend_metrics = f"""
        ### 📈 Trend & CUSUM Summary
        * **🔴 Attention:** {trend_sum['n_attention']} | **🟡 Monitor:** {trend_sum['n_monitor']} | **🟢 Safe:** {trend_sum['n_safe']}
        """

        cap_sum = get_capability_summary(cap_df)
        cap_metrics = f"""
        ### 🏭 Process Capability Summary
        * **✅ Excellent:** {cap_sum['n_excellent']} | **✅ Capable:** {cap_sum['n_capable']} | **🟡 Marginal:** {cap_sum['n_marginal']} | **🔴 Not Capable:** {cap_sum['n_not_capable']}
        """

        eval_metrics = f"""
        ### 🎯 Ensemble Performance
        * **Precision:** {eval_summary['ensemble_precision']:.3f} | **Recall:** {eval_summary['ensemble_recall']:.3f} | **F1-Score:** {eval_summary['ensemble_f1']:.3f}
        * Best Recall Model: **{eval_summary['best_recall_model']}** ({eval_summary['best_recall_value']:.3f})
        """

        # Charts
        fig_status = plot_status_summary(merged)
        fig_ae_v_t2 = plot_ae_vs_t2(merged)
        fig_heatmap = plot_agreement_heatmap(flagged)
        fig_eval = plot_evaluation(eval_df)

        # Detail Table Clean Up
        display_cols = ['batch-number', 'product-name', 'month', 'zscore_is_anomaly', 'iforest_is_anomaly', 'ae_is_anomaly', 't2_is_outlier', 'fraud_any', 'vote_count', 'weighted_score', 'final_status']
        rename_map = {'zscore_is_anomaly': 'Z-Score', 'iforest_is_anomaly': 'IF', 'ae_is_anomaly': 'AE', 't2_is_outlier': 'T²', 'fraud_any': 'Fraud', 'vote_count': 'Votes', 'weighted_score': 'Score', 'final_status': 'Status'}
        detail_table = merged[(merged['final_status'] != '✅ Normal')][display_cols].rename(columns=rename_map).sort_values('Score', ascending=False)

        # Risk Table Clean up
        risk_table = risk_df[['batch-number', 'product-name', 'parameter', 'actual_value', 'LSL', 'USL', 'dist_to_limit_%', 'risk_label']] if not risk_df.empty else pd.DataFrame()

        return (
            "✅ Analysis Complete!", 
            overview_metrics, fig_status, fig_ae_v_t2, fig_heatmap,
            gr.update(choices=products, value=products[0]), detail_table,
            risk_metrics, risk_table,
            trend_metrics, trend_df[['product-name', 'parameter', 'tau', 'p_value', 'slope_per_batch', 'risk_assessment']],
            cap_metrics, cap_df[['product-name', 'parameter', 'n', 'mean', 'std', 'LSL', 'USL', 'Cp', 'Cpk', 'interpretation']],
            eval_metrics, fig_eval, eval_df,
            summary_txt, csv_path, excel_path,
            df, merged # Return raw data for dynamic dropdown chart triggers
        )

    except Exception as e:
        return f"❌ Error: {str(e)}", None, None, None, None, gr.update(), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

def update_tab2_charts(product, merged_data):
    if merged_data is None: return None, None
    fig_ae = plot_ae_control_chart(merged_data, product)
    fig_t2 = plot_t2_control_chart(merged_data, product)
    return fig_ae, fig_t2

def update_tab4_charts(product, parameter, df_data):
    if df_data is None: return None, None
    fig_trend = plot_trend_line(df_data, product, parameter)
    fig_cusum = plot_cusum(df_data, product, parameter)
    return fig_trend, fig_cusum

# =============================================================================
# GRADIO INTERFACE DESIGN
# =============================================================================

custom_css = """
footer {visibility: hidden}
.output-markdown {font-family: 'Source Sans Pro', sans-serif;}
"""

with gr.Blocks(css=custom_css, title="PQR Anomaly Detection") as demo:
    
    # Internal state memory tags
    raw_df_state = gr.State()
    merged_df_state = gr.State()

    gr.Markdown("# 🔬 PQR Anomaly Detection Dashboard")
    gr.Markdown("### Ensemble anomaly detection for pharmaceutical batch quality review — Z-Score · Isolation Forest · Autoencoder · Hotelling T²")
    
    with gr.Row():
        # --- SIDEBAR COMPONENT ---
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 📂 Dataset Settings")
            data_source = gr.Radio(
                label="Data source:",
                choices=['Use synthetic demo dataset', 'Upload my own dataset'],
                value='Use synthetic demo dataset'
            )
            
            uploaded_file = gr.File(
                label="Upload Excel (.xlsx)",
                file_types=['.xlsx'],
                visible=False
            )
            
            # Show/Hide Upload File input based on choice
            data_source.change(
                fn=lambda x: gr.update(visible=(x == 'Upload my own dataset')),
                inputs=data_source,
                outputs=uploaded_file
            )

            gr.Markdown("### ⚙️ Configuration")
            with gr.Accordion("Model Weights Info", open=False):
                gr.Markdown("""
                | Model | Weight |
                |-------|--------|
                | Z-Score | 1.0 |
                | Isolation Forest | 1.5 |
                | Autoencoder | 2.5 |
                | Hotelling T² | 2.0 |
                | **Threshold** | **3.5** |
                """)
                
            with gr.Accordion("About this system", open=False):
                gr.Markdown("**Models:** Z-Score · Isolation Forest · Autoencoder · Hotelling T²\n\n**Extra:** Fraud / copy-paste detection\n\n**Reference:** ICH Q10 · Montgomery (2009) · AIAG SPC Manual")

            run_btn = gr.Button("🚀 Run Analysis", variant="primary")
            status_output = gr.Textbox(label="System Status", value="Ready to Analyze.")

        # --- MAIN PANELS (TABS) ---
        with gr.Column(scale=4):
            with gr.Tabs():
                
                # TAB 1: OVERVIEW
                with gr.Tab("📊 Overview"):
                    overview_md = gr.Markdown("Please run the analysis first.")
                    with gr.Row():
                        plot_status = gr.Plot(label="Status Summary")
                        plot_ae_t2 = gr.Plot(label="AE vs Hotelling T²")
                    gr.Markdown("### 🗳️ Detection Agreement Heatmap")
                    plot_heatmap = gr.Plot()

                # TAB 2: ANOMALY DETAIL
                with gr.Tab("🔍 Anomaly Detail"):
                    tab2_product_drop = gr.Dropdown(label="Select Product", choices=["Run analysis first"])
                    with gr.Row():
                        plot_ae_control = gr.Plot(label="Autoencoder Control Chart")
                        plot_t2_control = gr.Plot(label="Hotelling T² Control Chart")
                    gr.Markdown("### 📋 Flagged Batches — Detail Table")
                    table_flagged_detail = gr.Dataframe()

                # TAB 3: RISK CONTEXT
                with gr.Tab("⚠️ Risk Context"):
                    risk_md = gr.Markdown()
                    table_risk = gr.Dataframe()

                # TAB 4: TRENDS & CUSUM
                with gr.Tab("📈 Trends & CUSUM"):
                    trend_md = gr.Markdown()
                    with gr.Row():
                        tab4_product_drop = gr.Dropdown(label="Select Product", choices=["Run analysis first"])
                        tab4_param_drop = gr.Dropdown(label="Select Parameter", choices=PARAMETERS, value=PARAMETERS[0])
                    with gr.Row():
                        plot_trend = gr.Plot(label="Mann-Kendall Trend Chart")
                        plot_cusum_chart = gr.Plot(label="CUSUM Chart")
                    table_trend = gr.Dataframe(label="Parameters Requiring Attention")

                # TAB 5: CAPABILITY
                with gr.Tab("🏭 Capability"):
                    cap_md = gr.Markdown()
                    table_cap = gr.Dataframe()

                # TAB 6: PERFORMANCE
                with gr.Tab("🎯 Model Performance"):
                    eval_md = gr.Markdown()
                    plot_eval_chart = gr.Plot()
                    table_eval = gr.Dataframe()

                # TAB 7: EXECUTIVE SUMMARY
                with gr.Tab("📄 Executive Summary"):
                    exec_summary_text = gr.Textbox(label="Generated Executive Summary Report", lines=20)
                    download_txt_btn = gr.File(label="📥 Download Summary Text File")

                # TAB 8: EXPORT
                with gr.Tab("💾 Export"):
                    gr.Markdown("### 💾 Download Final Reports & Dataset Sheets")
                    with gr.Row():
                        download_csv_btn = gr.File(label="📥 Download CSV Results")
                        download_xlsx_btn = gr.File(label="📥 Download Full Excel Workbook")

    # =============================================================================
    # EVENT BACKEND TRIGGERS
    # =============================================================================
    
    # Main RUN button trigger
    run_btn.click(
        fn=execute_pipeline,
        inputs=[data_source, uploaded_file],
        outputs=[
            status_output, overview_md, plot_status, plot_ae_t2, plot_heatmap,
            tab2_product_drop, table_flagged_detail,
            risk_md, table_risk,
            trend_md, table_trend,
            cap_md, table_cap,
            gr.Markdown(visible=True), plot_eval_chart, table_eval,
            exec_summary_text, download_csv_btn, download_xlsx_btn,
            raw_df_state, merged_df_state # save raw data into state
        ]
    )

    # Trigger dropdown changes in Tab 2 (Detail Anomaly Chart)
    tab2_product_drop.change(
        fn=update_tab2_charts,
        inputs=[tab2_product_drop, merged_df_state],
        outputs=[plot_ae_control, plot_t2_control]
    )

    # Sync tab 2 dropdown choices to tab 4 for better UX flow
    tab2_product_drop.change(fn=lambda x: x, inputs=tab2_product_drop, outputs=tab4_product_drop)

    # Trigger dropdown changes for product/parameter in Tab 4 (Trend & CUSUM)
    gr.on(
        triggers=[tab4_product_drop.change, tab4_param_drop.change],
        fn=update_tab4_charts,
        inputs=[tab4_product_drop, tab4_param_drop, raw_df_state],
        outputs=[plot_trend, plot_cusum_chart]
    )

if __name__ == "__main__":
    demo.launch()
