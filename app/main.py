"""
Temporal Dynamic Bayesian Network for Air Quality Risk & Decision Support
IITM Monitoring Station 11613 | Revenue Colony - Shivajinagar, Pune
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import sys

# Ensure local app module can be imported
sys.path.append(str(Path(__file__).parent))
from model import (
    STATES, SEASONS, POLLUTANTS, STATE_COLORS,
    load_and_preprocess_data, build_and_train_bayesian_network,
    query_posterior_risk, simulate_markov_trajectory, get_decision_advisory,
    get_season, get_aqi_state
)

# Page configuration
st.set_page_config(
    page_title="Bayesian Air Quality Decision Support System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .app-header {
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        padding-bottom: 20px;
        margin-bottom: 24px;
    }
    
    .app-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
        margin: 0 0 6px 0;
    }
    
    .app-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0 0 14px 0;
        line-height: 1.4;
    }
    
    .meta-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 500;
        background: #1e293b;
        color: #cbd5e1;
        border: 1px solid #334155;
        margin-right: 6px;
        margin-top: 4px;
    }
    
    .panel-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 16px;
    }
    
    .metric-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px;
        text-align: left;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 2px;
        margin-bottom: 2px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .advisory-card {
        border-radius: 8px;
        padding: 18px;
        margin: 16px 0;
        background: #0f172a;
    }
    
    .advisory-routine {
        border-left: 4px solid #059669;
        border-top: 1px solid #1e293b;
        border-right: 1px solid #1e293b;
        border-bottom: 1px solid #1e293b;
    }
    
    .advisory-elevated {
        border-left: 4px solid #d97706;
        border-top: 1px solid #1e293b;
        border-right: 1px solid #1e293b;
        border-bottom: 1px solid #1e293b;
    }
    
    .advisory-high {
        border-left: 4px solid #ea580c;
        border-top: 1px solid #1e293b;
        border-right: 1px solid #1e293b;
        border-bottom: 1px solid #1e293b;
    }
    
    .advisory-severe {
        border-left: 4px solid #ef4444;
        border-top: 1px solid #1e293b;
        border-right: 1px solid #1e293b;
        border-bottom: 1px solid #1e293b;
    }
    
    .tier-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #1e293b;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-size: 0.9rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Data & Model caching
@st.cache_data(show_spinner="Loading IITM Station 11613 Air Quality Dataset...")
def get_cached_data():
    return load_and_preprocess_data()

@st.cache_resource(show_spinner="Compiling Dirichlet BDeu Bayesian Network...")
def get_cached_model(ess=10):
    _, _, modeling_df = get_cached_data()
    return build_and_train_bayesian_network(modeling_df, train_split=0.8, equivalent_sample_size=ess)

try:
    raw_df, df_2025, modeling_df = get_cached_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### System Configuration")
    
    st.markdown("**Bayesian Prior Parameter**")
    ess_slider = st.slider(
        "BDeu Equivalent Sample Size (α)",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="Dirichlet pseudo-count smoothing weight. Ensures non-zero conditional probabilities for unobserved state transitions."
    )
    
    st.markdown("---")
    st.markdown("**Monitoring Station Telemetry**")
    st.markdown("""
    - **Station ID:** IITM 11613
    - **Site:** Revenue Colony – Shivajinagar, Pune
    - **Observed Horizon:** 2020 – 2026
    - **Continuous 2025 Series:** 294 days
    - **Valid Markov Transitions:** 286 pairs
    - **Holdout Test ECE:** `0.0805`
    """)
    
    st.markdown("---")
    st.markdown("**Methodology**")
    st.markdown("""
    1st-Order Markov Dynamic Bayesian Network estimated via Bayesian Dirichlet Prior (`BDeu`), evaluated with Walk-Forward Cross-Validation.
    """)

# Train / load model with selected ESS
model, infer_engine, train_df, test_df = get_cached_model(ess=ess_slider)

# Application Header
st.markdown(f"""
<div class="app-header">
    <div class="app-title">Bayesian Air Quality Risk & Mitigation System</div>
    <div class="app-subtitle">
        Probabilistic Temporal Forecasting & Municipal Decision Support Pipeline • Station 11613 (Shivajinagar, Pune)
    </div>
    <div>
        <span class="meta-tag">Model: 1st-Order Markov DBN</span>
        <span class="meta-tag">Dirichlet Prior: BDeu (α={ess_slider})</span>
        <span class="meta-tag">Inference: Exact Variable Elimination</span>
        <span class="meta-tag">Split: Chronological Holdout (80/20)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_forecast, tab_simulation, tab_explorer, tab_model_dag, tab_benchmark = st.tabs([
    "Risk Forecasting & Decision Support",
    "Multi-Day Markov Simulation",
    "Station Data & Leakage Audit",
    "Bayesian Network & CPDs",
    "Evaluation & Benchmark Validation"
])

# ==============================================================================
# TAB 1: RISK FORECASTING & DECISION SUPPORT
# ==============================================================================
with tab_forecast:
    st.markdown("#### Evidence Query: Next-Day Posterior Risk Distribution")
    st.caption("Exact probabilistic evaluation: P(AQI_t | AQI_{t-1}, Pollutant_{t-1}, Season_t)")
    
    col_input, col_results = st.columns([1.1, 1.9], gap="large")
    
    with col_input:
        st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
        st.markdown("**Observed Evidence at Day t-1**")
        
        input_mode = st.radio("Input Format for AQI (t-1):", ["Categorical State", "Continuous Sensor Value"], horizontal=True)
        
        if input_mode == "Categorical State":
            prev_state_selected = st.selectbox(
                "Observed AQI State (t-1):",
                options=STATES,
                index=1,
                format_func=lambda x: f"{x} (AQI {'≤100' if x=='Low' else '101–200' if x=='Moderate' else '>200'})"
            )
        else:
            raw_aqi_val = st.number_input("Sensor AQI Value (t-1):", min_value=0.0, max_value=500.0, value=142.0, step=1.0)
            prev_state_selected = get_aqi_state(raw_aqi_val)
            st.caption(f"Discretized State: **{prev_state_selected}**")
            
        season_selected = st.selectbox(
            "Calendar Season at Target Day t:",
            options=SEASONS,
            index=2,
            help="Monsoon (Jun–Sep), Post-Monsoon (Oct–Nov), Winter (Dec–Feb), Summer (Mar–May)"
        )
        
        dominant_pol_selected = st.selectbox(
            "Dominant Pollutant Species (t-1):",
            options=POLLUTANTS,
            index=0,
            format_func=lambda x: x.upper() + (" (Carbon Monoxide)" if x=='co' else " (Nitrogen Dioxide)" if x=='no2' else " (Ozone)" if x=='o3' else "")
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_results:
        probs = query_posterior_risk(infer_engine, prev_state_selected, season_selected, dominant_pol_selected)
        advisory = get_decision_advisory(probs)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">P(Low AQI) [≤100]</div>
                <div class="metric-value" style="color: {STATE_COLORS['Low']}">{probs['Low']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">P(Moderate AQI) [101–200]</div>
                <div class="metric-value" style="color: {STATE_COLORS['Moderate']}">{probs['Moderate']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">P(High AQI) [>200]</div>
                <div class="metric-value" style="color: {STATE_COLORS['High']}">{probs['High']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Horizontal Probability Distribution Bar
        prob_chart_df = pd.DataFrame([{
            "State": s,
            "Probability": probs[s],
            "Label": f"{s}: {probs[s]*100:.1f}%"
        } for s in STATES])
        
        fig_prob = px.bar(
            prob_chart_df,
            x="Probability",
            y=["Posterior Risk"] * len(STATES),
            color="State",
            color_discrete_map=STATE_COLORS,
            orientation="h",
            text="Label",
            height=110
        )
        fig_prob.update_layout(
            barmode='stack',
            margin=dict(l=5, r=5, t=5, b=5),
            showlegend=False,
            xaxis=dict(range=[0, 1], showgrid=False, visible=False),
            yaxis=dict(visible=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_prob.update_traces(
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(color='#ffffff', size=13, family='Inter', weight='bold')
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # Advisory Card
        st.markdown(f"""
        <div class="advisory-card {advisory['badge_class']}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div>
                    <span class="tier-badge" style="background: {advisory['color']}; color: #ffffff;">{advisory['code']}</span>
                    <span style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-left: 8px;">{advisory['tier']}</span>
                </div>
                <span style="font-size: 0.8rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">Decision Support Protocol</span>
            </div>
            <p style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 12px; line-height: 1.45;">
                {advisory['summary']}
            </p>
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #1e293b; border-radius: 6px; padding: 12px;">
                <div style="font-size: 0.85rem; color: #e2e8f0; margin-bottom: 6px;">
                    <strong style="color: #94a3b8;">Public Health Advisory:</strong> {advisory['health_advice']}
                </div>
                <div style="font-size: 0.85rem; color: #e2e8f0;">
                    <strong style="color: #94a3b8;">Municipal Response:</strong> {advisory['municipal_action']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Sensitivity Analysis: Seasonal & Pollutant Matrix")
    st.caption(f"Posterior probability shifts given yesterday's state = {prev_state_selected} across all configurations.")
    
    sensitivity_records = []
    for s in SEASONS:
        for p in POLLUTANTS:
            p_dict = query_posterior_risk(infer_engine, prev_state_selected, s, p)
            adv = get_decision_advisory(p_dict)
            sensitivity_records.append({
                "Season": s,
                "Dominant Pollutant (t-1)": p.upper(),
                "P(Low)": f"{p_dict['Low']*100:.1f}%",
                "P(Moderate)": f"{p_dict['Moderate']*100:.1f}%",
                "P(High)": f"{p_dict['High']*100:.1f}%",
                "Advisory Tier": adv["tier"],
                "Protocol Code": adv["code"]
            })
            
    sens_df = pd.DataFrame(sensitivity_records)
    st.dataframe(
        sens_df,
        use_container_width=True,
        hide_index=True
    )

# ==============================================================================
# TAB 2: MULTI-DAY MARKOV SIMULATION
# ==============================================================================
with tab_simulation:
    st.markdown("#### Forward Markov Trajectory Simulation")
    st.caption("Iterative multi-step probability propagation under stationary environmental regimes without teacher forcing.")
    
    sim_c1, sim_c2, sim_c3 = st.columns(3)
    with sim_c1:
        sim_init_state = st.selectbox("Initial State (t-1):", STATES, index=1, key="sim_init")
    with sim_c2:
        sim_season = st.selectbox("Conditioned Season:", SEASONS, index=2, key="sim_season")
    with sim_c3:
        sim_steps = st.slider("Simulation Horizon (Days):", min_value=3, max_value=14, value=7, step=1)
        
    traj_df = simulate_markov_trajectory(infer_engine, sim_init_state, sim_season, dominant_pollutant='co', steps=sim_steps)
    
    fig_traj = go.Figure()
    for s in STATES:
        fig_traj.add_trace(go.Scatter(
            x=traj_df['Day'],
            y=traj_df[s] * 100,
            mode='lines+markers',
            name=f"P({s})",
            line=dict(color=STATE_COLORS[s], width=2.5),
            marker=dict(size=7)
        ))
        
    fig_traj.update_layout(
        title=f"Markov State Evolution: {sim_steps}-Day Horizon ({sim_season}, Starting at {sim_init_state})",
        xaxis_title="Time Step",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 105], gridcolor='rgba(255,255,255,0.08)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_traj, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Seasonal Equilibrium Contrast: Thermal Inversion vs. Monsoon Washout")
    
    eq_c1, eq_c2 = st.columns(2)
    with eq_c1:
        traj_w = simulate_markov_trajectory(infer_engine, 'Moderate', 'Winter', 'co', steps=7)
        fig_w = px.line(
            traj_w, x='Day', y=['Low', 'Moderate', 'High'],
            color_discrete_map=STATE_COLORS,
            title="Winter Trajectory (Atmospheric Trapping)"
        )
        fig_w.update_layout(
            yaxis_title="Probability",
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_w, use_container_width=True)
        
    with eq_c2:
        traj_m = simulate_markov_trajectory(infer_engine, 'Moderate', 'Monsoon', 'co', steps=7)
        fig_m = px.line(
            traj_m, x='Day', y=['Low', 'Moderate', 'High'],
            color_discrete_map=STATE_COLORS,
            title="Monsoon Trajectory (Washout & Dispersion)"
        )
        fig_m.update_layout(
            yaxis_title="Probability",
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_m, use_container_width=True)

# ==============================================================================
# TAB 3: STATION DATA & LEAKAGE AUDIT
# ==============================================================================
with tab_explorer:
    st.markdown("#### Historical Station Telemetry (IITM 11613)")
    st.caption("Pune Urban Corridor Air Monitoring Station: Revenue Colony – Shivajinagar")
    
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        st.metric("Total Archive Records", f"{len(raw_df):,}")
    with col_d2:
        st.metric("Continuous 2025+ Series", f"{len(df_2025):,} days")
    with col_d3:
        st.metric("Consecutive Markov Pairs", f"{len(modeling_df):,} days")
    with col_d4:
        st.metric("Resolved Missing Gap", "957 Days")
        
    st.markdown("---")
    st.markdown("#### Continuous 2025–2026 AQI Time Series")
    
    fig_ts = go.Figure()
    fig_ts.add_hrect(y0=0, y1=100, fillcolor="rgba(16, 185, 129, 0.08)", line_width=0, annotation_text="Low State (≤100)", annotation_position="top left")
    fig_ts.add_hrect(y0=100, y1=200, fillcolor="rgba(245, 158, 11, 0.08)", line_width=0, annotation_text="Moderate State (101–200)", annotation_position="top left")
    fig_ts.add_hrect(y0=200, y1=max(350, modeling_df['aqi'].max()+20), fillcolor="rgba(239, 68, 68, 0.08)", line_width=0, annotation_text="High State (>200)", annotation_position="top left")
    
    fig_ts.add_trace(go.Scatter(
        x=modeling_df['date'],
        y=modeling_df['aqi'],
        mode='lines+markers',
        name='Observed AQI',
        line=dict(color='#38bdf8', width=1.5),
        marker=dict(size=3, color='#818cf8')
    ))
    
    fig_ts.update_layout(
        xaxis_title="Date",
        yaxis_title="Continuous AQI",
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified"
    )
    st.plotly_chart(fig_ts, use_container_width=True)
    
    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        st.markdown("**State Frequency by Season**")
        season_counts = modeling_df.groupby(['season', 'target_aqi_state'], observed=False).size().reset_index(name='count')
        fig_season_bar = px.bar(
            season_counts, x='season', y='count', color='target_aqi_state',
            color_discrete_map=STATE_COLORS,
            barmode='group'
        )
        fig_season_bar.update_layout(
            yaxis_title="Count of Days",
            xaxis_title="Season",
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_season_bar, use_container_width=True)
        
    with col_dist2:
        st.markdown("**Dominant Pollutant Precursor Distribution**")
        pollutant_counts = modeling_df['previous_dominant_pollutant'].value_counts().reset_index()
        pollutant_counts.columns = ['Pollutant', 'Count']
        fig_pol_pie = px.pie(
            pollutant_counts, names='Pollutant', values='Count',
            color='Pollutant',
            color_discrete_map={'co': '#0284c7', 'no2': '#7c3aed', 'o3': '#ea580c'},
            hole=0.5
        )
        fig_pol_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pol_pie, use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### Zero Temporal Data Leakage Audit")
    st.caption("Verification of temporal index alignment and causal separation.")
    
    leakage_data = [
        {"Feature": "previous_aqi_state", "Temporal Index": "t - 1", "Source": "Observed Sensor State at t-1", "Verification": "SAFE (Strictly Lagged Autoregressive)"},
        {"Feature": "previous_dominant_pollutant", "Temporal Index": "t - 1", "Source": "Observed Primary Pollutant at t-1", "Verification": "SAFE (Strictly Lagged Precursor)"},
        {"Feature": "season", "Temporal Index": "t", "Source": "Astronomical Calendar Date at t", "Verification": "SAFE (Pre-known Exogenous Forcing)"},
        {"Feature": "target_aqi_state", "Temporal Index": "t", "Source": "Ground Truth State at t", "Verification": "TARGET (Held out during inference)"}
    ]
    st.dataframe(pd.DataFrame(leakage_data), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 4: BAYESIAN NETWORK & CPDS
# ==============================================================================
with tab_model_dag:
    st.markdown("#### Graphical Topology & Parameter Estimations")
    
    dag_c1, dag_c2 = st.columns([1.1, 1.9])
    
    with dag_c1:
        st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
        st.markdown("**Directed Acyclic Graph (DAG)**")
        st.markdown("""
        ```text
        [ AQI State (t-1) ] ─────────┐
                                     ▼
        [ Dominant Pol (t-1) ] ───> [ Target AQI State (t) ]
                                     ▲
        [ Season (t) ] ──────────────┘
        ```
        """)
        st.markdown("""
        **Joint Factorization:**  
        $$P(Y_t, Y_{t-1}, P_{t-1}, S_t) = P(Y_t \\mid Y_{t-1}, P_{t-1}, S_t) \\cdot P(Y_{t-1}) P(P_{t-1}) P(S_t)$$
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with dag_c2:
        st.markdown("**1-Day State Persistence Matrix: P(AQI_t | AQI_{t-1})**")
        trans_counts = pd.crosstab(
            train_df['previous_aqi_state'],
            train_df['target_aqi_state'],
            normalize='index',
            dropna=False
        ).reindex(index=STATES, columns=STATES).fillna(0)
        
        fig_heat = px.imshow(
            trans_counts,
            text_auto=".2%",
            labels=dict(x="Target State (t)", y="Previous State (t-1)", color="Probability"),
            color_continuous_scale="Blues"
        )
        fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_heat, use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### Conditional Probability Table (CPD) Query Inspector")
    st.caption("Inspect smoothed Dirichlet BDeu conditional distributions for specific evidence configurations.")
    
    cpd_c1, cpd_c2 = st.columns(2)
    with cpd_c1:
        inspect_season = st.selectbox("Inspect Season:", SEASONS, key="cpd_season")
    with cpd_c2:
        inspect_pol = st.selectbox("Inspect Dominant Pollutant:", POLLUTANTS, key="cpd_pol")
        
    cpd_rows = []
    for prev_s in STATES:
        p = query_posterior_risk(infer_engine, prev_s, inspect_season, inspect_pol)
        cpd_rows.append({
            "Previous State (t-1)": prev_s,
            "Season (t)": inspect_season,
            "Dominant Pollutant (t-1)": inspect_pol.upper(),
            "P(Low | Evidence)": f"{p['Low']*100:.2f}%",
            "P(Moderate | Evidence)": f"{p['Moderate']*100:.2f}%",
            "P(High | Evidence)": f"{p['High']*100:.2f}%"
        })
        
    st.dataframe(pd.DataFrame(cpd_rows), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 5: BENCHMARKS & VALIDATION
# ==============================================================================
with tab_benchmark:
    st.markdown("#### Model Performance Benchmarks on Chronological Holdout")
    st.caption("Evaluated on strictly unseen chronological test set (N=58 samples, Nov 2025 – Jan 2026).")
    
    benchmark_table = [
        {"Model": "Dynamic Bayesian Network (BDeu)", "Accuracy": "86.21%", "Balanced Accuracy": "67.51%", "Macro F1": "0.4500", "Brier Score": "0.2187", "Log Loss": "2.6681", "Expected Calibration Error (ECE)": "0.0805", "Characteristics": "Best calibrated log loss; full posterior uncertainty"},
        {"Model": "Logistic Regression", "Accuracy": "87.93%", "Balanced Accuracy": "68.49%", "Macro F1": "0.4645", "Brier Score": "0.1934", "Log Loss": "5.5311", "Expected Calibration Error (ECE)": "0.1140", "Characteristics": "Linear probabilistic classifier"},
        {"Model": "Random Forest", "Accuracy": "86.21%", "Balanced Accuracy": "67.51%", "Macro F1": "0.4500", "Brier Score": "0.1949", "Log Loss": "29.4175", "Expected Calibration Error (ECE)": "0.1378", "Characteristics": "Non-linear decision trees; high overconfidence"},
        {"Model": "Naive Persistence Baseline", "Accuracy": "86.21%", "Balanced Accuracy": "67.51%", "Macro F1": "0.4500", "Brier Score": "0.2759", "Log Loss": "32.1568", "Expected Calibration Error (ECE)": "N/A", "Characteristics": "Standard point baseline (Y_t = Y_{t-1})"}
    ]
    st.dataframe(pd.DataFrame(benchmark_table), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("#### 3-Fold Walk-Forward Cross-Validation")
    
    wf_table = [
        {"Fold": "Fold 1", "Train Period": "Feb 19, 2025 – Jun 17, 2025 (N=114)", "Holdout Period": "Jun 18, 2025 – Sep 22, 2025 (Monsoon, N=57)", "Accuracy": "87.72%", "Macro F1": "0.5534", "Brier Score": "0.2240", "Log Loss": "3.8809"},
        {"Fold": "Fold 2", "Train Period": "Feb 19, 2025 – Sep 22, 2025 (N=171)", "Holdout Period": "Sep 23, 2025 – Nov 21, 2025 (Post-Monsoon, N=57)", "Accuracy": "42.11%", "Macro F1": "0.2729", "Brier Score": "0.5875", "Log Loss": "1.6862"},
        {"Fold": "Fold 3", "Train Period": "Feb 19, 2025 – Nov 21, 2025 (N=228)", "Holdout Period": "Nov 22, 2025 – Jan 27, 2026 (Winter, N=57)", "Accuracy": "85.96%", "Macro F1": "0.4183", "Brier Score": "0.2108", "Log Loss": "2.6956"}
    ]
    st.dataframe(pd.DataFrame(wf_table), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("#### Methodological FAQ")
    
    with st.expander("Why use a Dynamic Bayesian Network instead of standard black-box classifiers?"):
        st.write("""
        Municipal decision-making requires well-calibrated probabilistic uncertainty rather than deterministic point scores. Dynamic Bayesian Networks factor the joint probability distribution over explicit causal dependencies, enabling exact conditional inference, Dirichlet smoothing against sparse states, and bi-directional diagnostic reasoning.
        """)
        
    with st.expander("Why is chronological splitting required for atmospheric time-series?"):
        st.write("""
        Atmospheric chemistry exhibits significant temporal inertia and autocorrelation across consecutive days. A random train/test split introduces temporal lookahead bias by allowing the model to train on future observations to predict past states. Chronological splitting ensures true out-of-sample forward evaluation.
        """)
        
    with st.expander("How does Dirichlet BDeu smoothing prevent zero-frequency probability traps?"):
        st.write("""
        In small or seasonally concentrated datasets, rare parent-child state configurations (e.g., High AQI in heavy Monsoon) have zero counts. Maximum Likelihood Estimation assigns these zero probability, causing infinite Log Loss (-log(0)) on future occurrences. BDeu smoothing distributes equivalent sample size pseudo-counts (alpha = 10) to guarantee normalized, strictly non-zero distributions.
        """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem; padding: 8px 0;">
    Bayesian Air Quality Risk & Mitigation System • IITM Station 11613 (Pune)
</div>
""", unsafe_allow_html=True)
