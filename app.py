# =============================================================================
# app.py — PQR Anomaly Detection Dashboard
# =============================================================================

import streamlit as st
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
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title = 'PQR Anomaly Detection',
    page_icon  = '🔬',
    layout     = 'wide',
    initial_sidebar_state = 'expanded',
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1d3557;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #457b9d;
        margin-bottom: 0.5rem;
    }
    .alert-red    { border-left-color: #e63946 !important; }
    .alert-orange { border-left-color: #f4a261 !important; }
    .alert-green  { border-left-color: #2a9d8f !important; }
    .section-divider {
        border-top: 2px solid #e0e0e0;
        margin: 1.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================

def init_state():
    defaults = {
        'data_loaded':  False,
        'model_run':    False,
        'df':           None,
        'products':     None,
        'merged':       None,
        'risk_df':      None,
        'trend_df':     None,
        'cap_df':       None,
        'eval_df':      None,
        'ae_results':   None,
        'scaled_dict':  None,
        'fraud_summary':    None,
        'ensemble_summary': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.image('https://img.icons8.com/fluency/96/microscope.png', width=60)
    st.markdown('## PQR Anomaly Detection')
    st.markdown('*Pharmaceutical Batch Quality Review*')
    st.divider()

    st.markdown('### 📂 Dataset')
    use_default = st.radio(
        'Data source:',
        ['Use synthetic demo dataset', 'Upload my own dataset'],
        index=0,
    )

    uploaded_file = None
    if use_default == 'Upload my own dataset':
        uploaded_file = st.file_uploader(
            'Upload Excel (.xlsx)',
            type=['xlsx'],
            help='Must have a "Data" sheet with columns: year, month, product-name, batch-number + parameters'
        )
        if uploaded_file is None:
            st.info('Waiting for upload...')

    st.divider()
    st.markdown('### ⚙️ Configuration')
    with st.expander('Model weights'):
        st.markdown(f"""
        | Model | Weight |
        |-------|--------|
        | Z-Score | {VOTE_WEIGHTS['zscore']} |
        | Isolation Forest | {VOTE_WEIGHTS['iforest']} |
        | Autoencoder | {VOTE_WEIGHTS['autoencoder']} |
        | Hotelling T² | {VOTE_WEIGHTS['hotelling']} |
        | **Threshold** | **{WEIGHTED_THRESHOLD}** |
        """)

    with st.expander('About this system'):
        st.markdown("""
        **Models:** Z-Score · Isolation Forest · Autoencoder · Hotelling T²

        **Extra:** Fraud / copy-paste detection

        **Reference:** ICH Q10 · Montgomery (2009) · AIAG SPC Manual

        Built with ❤️ for pharmaceutical QA teams.
        """)

    st.divider()

    run_btn = st.button(
        '🚀 Run Analysis',
        type='primary',
        use_container_width=True,
        disabled=(use_default == 'Upload my own dataset' and uploaded_file is None),
    )

# =============================================================================
# HEADER
# =============================================================================

st.markdown('<div class="main-header">🔬 PQR Anomaly Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ensemble anomaly detection for pharmaceutical batch quality review — Z-Score · Isolation Forest · Autoencoder · Hotelling T²</div>', unsafe_allow_html=True)

# =============================================================================
# RUN PIPELINE
# =============================================================================

if run_btn:
    st.session_state['data_loaded'] = False
    st.session_state['model_run']   = False

    with st.status('Running analysis...', expanded=True) as status:

        # Step 1: Load data
        st.write('📂 Loading and validating data...')
        try:
            result = run_data_pipeline(
                uploaded_file = uploaded_file,
                default_path  = DEFAULT_DATASET,
            )
            df       = result['df']
            products = result['products']

            if result['missing_report']:
                st.warning(f"Missing values filled: {result['missing_report']}")
            if 'Warning' in result['spec_status']:
                st.warning(result['spec_status'])
            else:
                st.write(f"✅ {result['spec_status']}")

        except Exception as e:
            st.error(f'Data loading failed: {e}')
            st.stop()

        # Step 2: Fraud detection
        st.write('🔍 Running fraud detection...')
        df = detect_fraud(df)
        fraud_summary = get_fraud_summary(df)
        if fraud_summary['total_flagged'] > 0:
            st.warning(f"⚠️ {fraud_summary['total_flagged']} batch(es) flagged for fraud")
        else:
            st.write('✅ No fraud detected')

        # Step 3: Preprocessing
        st.write('⚙️ Preprocessing (scaling per product)...')
        scalers, scaled_dict = run_preprocessing(df, products)

        # Step 4: Z-Score + Isolation Forest
        st.write('📊 Running Z-Score & Isolation Forest...')
        df_zs_if = run_zscore_iforest(df, products, scaled_dict)

        # Step 5: Autoencoder
        st.write('🧠 Training Autoencoder...')
        df_ae, ae_results = run_autoencoder_all(df, products, scaled_dict)

        # Step 6: Hotelling T²
        st.write('📐 Running Hotelling T²...')
        df_t2, t2_params = run_hotelling_all(df, products, scaled_dict)

        # Step 7: Merge + Ensemble
        st.write('🗳️ Computing ensemble scores...')
        merged = merge_results(df, df_zs_if, df_ae, df_t2)
        merged = run_ensemble(merged)
        ensemble_summary = get_ensemble_summary(merged)

        # Step 8: Risk Context + Trend
        st.write('⚠️ Running Risk Contextualization & Trend Analysis...')
        flagged  = merged[merged['final_status'] != '✅ Normal'].copy()
        risk_df  = compute_risk_context(df, flagged)
        trend_df = run_trend_analysis(df)

        # Step 9: Capability Analysis
        st.write('🏭 Running Process Capability Analysis...')
        cap_df = run_capability_analysis(df)

        # Step 10: Model Evaluation
        st.write('🎯 Running Model Evaluation...')
        eval_df = run_evaluation(merged)

        # Store in session state
        st.session_state.update({
            'data_loaded':      True,
            'model_run':        True,
            'df':               df,
            'products':         products,
            'merged':           merged,
            'risk_df':          risk_df,
            'trend_df':         trend_df,
            'cap_df':           cap_df,
            'eval_df':          eval_df,
            'ae_results':       ae_results,
            'scaled_dict':      scaled_dict,
            'fraud_summary':    fraud_summary,
            'ensemble_summary': ensemble_summary,
        })

        status.update(label='✅ Analysis complete!', state='complete')

# =============================================================================
# MAIN CONTENT — only shown after run
# =============================================================================

if not st.session_state['model_run']:
    st.info('👈 Configure your dataset in the sidebar and click **Run Analysis** to start.')

    st.markdown('---')
    st.markdown('### 📋 What this dashboard does')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🔍 Anomaly Detection**
        - Z-Score (univariate)
        - Isolation Forest
        - Vanilla Autoencoder
        - Hotelling T² (multivariate)
        - Weighted ensemble scoring
        """)
    with col2:
        st.markdown("""
        **⚠️ Risk Assessment**
        - OOS / OOT classification per parameter
        - % distance to spec limit
        - Fraud / copy-paste detection
        - Trend analysis (Mann-Kendall)
        - CUSUM shift detection
        """)
    with col3:
        st.markdown("""
        **📄 Reporting**
        - Auto-generated executive summary
        - Downloadable CSV / Excel results
        - Interactive Plotly charts
        - Audit trail ready
        - ICH Q10 compliant methodology
        """)
    st.stop()

# ── Pull from session state ───────────────────────────────────────────────────
df               = st.session_state['df']
products         = st.session_state['products']
merged           = st.session_state['merged']
risk_df          = st.session_state['risk_df']
trend_df         = st.session_state['trend_df']
cap_df           = st.session_state['cap_df']
eval_df          = st.session_state['eval_df']
ae_results       = st.session_state['ae_results']
scaled_dict      = st.session_state['scaled_dict']
fraud_summary    = st.session_state['fraud_summary']
ensemble_summary = st.session_state['ensemble_summary']

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    '📊 Overview',
    '🔍 Anomaly Detail',
    '⚠️ Risk Context',
    '📈 Trends & CUSUM',
    '🏭 Capability',
    '🎯 Model Performance',
    '📄 Executive Summary',
    '💾 Export',
])

# =============================================================================
# TAB 1 — OVERVIEW
# =============================================================================

with tab1:
    st.markdown('### 📊 Analysis Overview')

    c1, c2, c3, c4, c5 = st.columns(5)
    counts = ensemble_summary['counts']

    c1.metric('Total Batches',  ensemble_summary['total'])
    c2.metric('🔴 Data Fraud',  counts.get('🔴 Data Fraud — Duplicate/Copy-Paste', 0))
    c3.metric('🔴 Confirmed',   counts.get('🔴 Confirmed Anomaly — Investigate & CAPA', 0))
    c4.metric('🟠 Suspected',   counts.get('🟠 Suspected Anomaly — Enhanced Monitoring', 0))
    c5.metric('🟡 Watch List',  counts.get('🟡 Watch List — Re-check Next Batch', 0))

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(plot_status_summary(merged), use_container_width=True)
    with col_right:
        st.plotly_chart(plot_ae_vs_t2(merged), use_container_width=True)

    st.divider()
    st.markdown('### 🗳️ Detection Agreement Heatmap')
    flagged = merged[merged['final_status'] != '✅ Normal'].copy()
    st.plotly_chart(plot_agreement_heatmap(flagged), use_container_width=True)

    if fraud_summary['total_flagged'] > 0:
        st.divider()
        st.markdown('### 🚨 Fraud Detection Results')
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric('Exact Duplicates',    fraud_summary['exact_duplicates'])
        fc2.metric('Near-Duplicates',     fraud_summary['near_duplicates'])
        fc3.metric('Suspicious Rounding', fraud_summary['suspicious_rounding'])
        st.dataframe(fraud_summary['flagged_batches'], use_container_width=True)

# =============================================================================
# TAB 2 — ANOMALY DETAIL
# =============================================================================

with tab2:
    st.markdown('### 🔍 Anomaly Detection Detail')

    selected_product = st.selectbox('Select product:', products, key='tab2_product')

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            plot_ae_control_chart(merged, selected_product),
            use_container_width=True
        )
    with col_b:
        st.plotly_chart(
            plot_t2_control_chart(merged, selected_product),
            use_container_width=True
        )

    st.divider()
    st.markdown('### 📋 Flagged Batches — Detail Table')

    display_cols = [
        'batch-number', 'product-name', 'month',
        'zscore_is_anomaly', 'iforest_is_anomaly',
        'ae_is_anomaly', 't2_is_outlier',
        'fraud_any', 'vote_count', 'weighted_score', 'final_status'
    ]
    rename_map = {
        'zscore_is_anomaly':  'Z-Score',
        'iforest_is_anomaly': 'IF',
        'ae_is_anomaly':      'AE',
        't2_is_outlier':      'T²',
        'fraud_any':          'Fraud',
        'vote_count':         'Votes',
        'weighted_score':     'Score',
        'final_status':       'Status',
    }

    flagged_prod = merged[
        (merged['product-name'] == selected_product) &
        (merged['final_status'] != '✅ Normal')
    ][display_cols].rename(columns=rename_map).sort_values('Score', ascending=False)

    if flagged_prod.empty:
        st.success(f'✅ No anomalies detected for {selected_product}.')
    else:
        st.dataframe(flagged_prod, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 3 — RISK CONTEXT
# =============================================================================

with tab3:
    st.markdown('### ⚠️ Risk Contextualization')
    st.caption('How close are flagged batch parameters to specification limits?')

    risk_summary = get_risk_summary(risk_df)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric('🔴 OOS',         risk_summary['n_oos'])
    r2.metric('🟠 Near Limit',  risk_summary['n_near'])
    r3.metric('🟡 OOT Monitor', risk_summary['n_monitor'])
    r4.metric('🟢 Within Spec', risk_summary['n_ok'])

    st.divider()

    if risk_df.empty:
        st.info('No batches were flagged — nothing to contextualize.')
    else:
        product_filter = st.selectbox(
            'Filter by product:',
            ['All'] + list(products),
            key='tab3_product'
        )

        display_df = risk_df if product_filter == 'All' else \
                     risk_df[risk_df['product-name'] == product_filter]

        def color_risk(val):
            colors = {
                '🔴 OOS':            'background-color: #ffe0e0',
                '🟠 OOT Near Limit': 'background-color: #fff3e0',
                '🟡 OOT Monitor':    'background-color: #fffde0',
                '🟢 Within Spec':    'background-color: #e8f5e9',
            }
            return colors.get(val, '')

        styled = display_df[[
            'batch-number', 'product-name', 'parameter',
            'actual_value', 'LSL', 'USL', 'dist_to_limit_%', 'risk_label'
        ]].style.applymap(color_risk, subset=['risk_label'])

        st.dataframe(styled, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 4 — TRENDS & CUSUM
# =============================================================================

with tab4:
    st.markdown('### 📈 Trend Analysis & CUSUM')

    trend_summary = get_trend_summary(trend_df)
    t1, t2, t3 = st.columns(3)
    t1.metric('🔴 Attention', trend_summary['n_attention'])
    t2.metric('🟡 Monitor',   trend_summary['n_monitor'])
    t3.metric('🟢 Safe',      trend_summary['n_safe'])

    if not trend_summary['alerts'].empty:
        st.warning('⚠️ Parameters requiring attention:')
        st.dataframe(
            trend_summary['alerts'][['product-name', 'parameter', 'tau',
                                     'p_value', 'slope_per_batch', 'risk_assessment']],
            use_container_width=True, hide_index=True
        )

    st.divider()

    col_prod, col_param = st.columns(2)
    with col_prod:
        sel_product = st.selectbox('Product:', products, key='tab4_product')
    with col_param:
        sel_param   = st.selectbox('Parameter:', PARAMETERS, key='tab4_param')

    tab4a, tab4b = st.tabs(['Mann-Kendall Trend', 'CUSUM Chart'])

    with tab4a:
        st.plotly_chart(
            plot_trend_line(df, sel_product, sel_param),
            use_container_width=True
        )

        mk_row = trend_df[
            (trend_df['product-name'] == sel_product) &
            (trend_df['parameter']    == sel_param)
        ]
        if not mk_row.empty:
            r = mk_row.iloc[0]
            st.info(
                f"**Mann-Kendall result:** {r['trend_label']}  |  "
                f"τ = {r['tau']}  |  p = {r['p_value']}  |  "
                f"slope = {r['slope_per_batch']:+.4f}/batch  |  "
                f"**{r['risk_assessment']}**"
            )

    with tab4b:
        st.plotly_chart(
            plot_cusum(df, sel_product, sel_param),
            use_container_width=True
        )
        st.caption(f'k = {CUSUM_K}σ (reference value) | h = {CUSUM_H}σ (decision interval) | Reference: Montgomery (2009)')

# =============================================================================
# TAB 5 — CAPABILITY
# =============================================================================

with tab5:
    st.markdown('### 🏭 Process Capability Analysis')
    st.caption('Cp/Cpk per parameter per product. For batch release data, Cp = Pp and Cpk = Ppk (no subgroup structure).')

    cap_summary = get_capability_summary(cap_df)
    ca1, ca2, ca3, ca4 = st.columns(4)
    ca1.metric('✅ Excellent',  cap_summary['n_excellent'])
    ca2.metric('✅ Capable',    cap_summary['n_capable'])
    ca3.metric('🟡 Marginal',   cap_summary['n_marginal'])
    ca4.metric('🔴 Not Capable', cap_summary['n_not_capable'])

    st.divider()

    cap_filter = st.selectbox(
        'Filter by product:',
        ['All'] + list(products),
        key='tab5_product'
    )

    display_cap = cap_df if cap_filter == 'All' else \
                  cap_df[cap_df['product-name'] == cap_filter]

    def color_cap(val):
        if isinstance(val, str):
            if 'Excellent' in val or 'Capable' in val and 'Not' not in val and 'Marginally' not in val:
                return 'background-color: #e8f5e9'
            if 'Marginally' in val:
                return 'background-color: #fffde0'
            if 'Not Capable' in val:
                return 'background-color: #ffe0e0'
        return ''

    styled_cap = display_cap[[
        'product-name', 'parameter', 'n', 'mean', 'std',
        'LSL', 'USL', 'Cp', 'Cpk', 'CPL', 'CPU', 'interpretation'
    ]].style.applymap(color_cap, subset=['interpretation'])

    st.dataframe(styled_cap, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 6 — MODEL PERFORMANCE
# =============================================================================

with tab6:
    st.markdown('### 🎯 Model Performance Evaluation')
    st.caption(f'Ground truth: {len(__import__("modules.evaluation", fromlist=["GROUND_TRUTH"]).GROUND_TRUTH)} known anomalous batches embedded in dataset (OOS + Fraud + Outlier).')

    eval_summary = get_evaluation_summary(eval_df)

    if eval_summary:
        e1, e2, e3 = st.columns(3)
        e1.metric('Ensemble Precision', f"{eval_summary['ensemble_precision']:.3f}")
        e2.metric('Ensemble Recall',    f"{eval_summary['ensemble_recall']:.3f}")
        e3.metric('Ensemble F1',        f"{eval_summary['ensemble_f1']:.3f}")

        st.caption(f"Best recall: **{eval_summary['best_recall_model']}** ({eval_summary['best_recall_value']:.3f})")

    st.divider()
    st.plotly_chart(plot_evaluation(eval_df), use_container_width=True)

    st.divider()
    st.markdown('### 📋 Detailed Metrics Table')
    st.dataframe(
        eval_df.style.highlight_max(subset=['precision', 'recall', 'f1'], color='#d4edda'),
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.markdown('### ℹ️ Methodology Note')
    st.info("""
    **Ground truth** consists of synthetic anomalies embedded during dataset generation:
    - **OOS**: batches with values outside pharmacopoeial specification limits
    - **Fraud**: copy-paste and near-identical batches
    - **Outlier**: statistically extreme batches (multivariate)

    Trend anomalies are excluded from ground truth as they are multi-batch phenomena,
    not single-batch labels.

    *Recall is prioritized over precision in pharmaceutical anomaly detection —
    the cost of a missed anomaly (false negative) exceeds the cost of a false alarm (false positive).*
    """)

# =============================================================================
# TAB 7 — EXECUTIVE SUMMARY
# =============================================================================

with tab7:
    st.markdown('### 📄 Executive Summary')
    st.caption('Auto-generated summary ready for PQR reports, CAPAs, or management presentations.')

    report_year = int(df['year'].max()) if 'year' in df.columns else 2023
    summary_txt = generate_executive_summary(df, risk_df, trend_df, year=report_year)

    st.text_area(
        'Executive Summary',
        value=summary_txt,
        height=500,
        label_visibility='collapsed',
    )

    st.download_button(
        label='⬇️ Download as .txt',
        data=summary_txt.encode('utf-8'),
        file_name=f'executive_summary_PQR_{report_year}.txt',
        mime='text/plain',
        use_container_width=True,
    )

# =============================================================================
# TAB 8 — EXPORT
# =============================================================================

with tab8:
    st.markdown('### 💾 Export Results')

    export_cols = [
        'batch-number', 'product-name', 'year', 'month',
        *PARAMETERS,
        'fraud_exact_dup', 'fraud_near_dup', 'fraud_round', 'fraud_any',
        'zscore_max_z', 'zscore_is_anomaly',
        'iforest_score', 'iforest_is_anomaly',
        'ae_reconstruction_error', 'ae_is_anomaly',
        't2_distance', 't2_is_outlier',
        'vote_count', 'weighted_score', 'ensemble_is_anomaly', 'final_status',
    ]

    out = merged[[c for c in export_cols if c in merged.columns]] \
            .sort_values(['product-name', 'batch-number'])

    # CSV
    csv_bytes = out.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='⬇️ Download CSV',
        data=csv_bytes,
        file_name='pqr_anomaly_results.csv',
        mime='text/csv',
        use_container_width=True,
    )

    st.divider()

    # Excel — all sheets
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        out.to_excel(writer, sheet_name='All Products', index=False)
        for product in products:
            safe = product.replace(' ', '_').replace('/', '-')[:31]
            out[out['product-name'] == product].to_excel(
                writer, sheet_name=safe, index=False
            )
        if not risk_df.empty:
            risk_df.to_excel(writer, sheet_name='Risk Context', index=False)
        if not trend_df.empty:
            trend_df.to_excel(writer, sheet_name='Trend Analysis', index=False)
        if not cap_df.empty:
            cap_df.to_excel(writer, sheet_name='Capability', index=False)
        if not eval_df.empty:
            eval_df.to_excel(writer, sheet_name='Model Evaluation', index=False)

    st.download_button(
        label='⬇️ Download Excel (all sheets)',
        data=excel_buffer.getvalue(),
        file_name='pqr_anomaly_results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        use_container_width=True,
    )

    st.divider()
    st.markdown('### 👁️ Preview')
    st.dataframe(out.head(20), use_container_width=True, hide_index=True)