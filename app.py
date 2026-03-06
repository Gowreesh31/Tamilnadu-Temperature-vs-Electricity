"""
🌡️ Tamil Nadu Temperature vs Electricity — Elite Analytics Dashboard

A comprehensive data analytics dashboard demonstrating:
  - API-based data ingestion (Open-Meteo, World Bank, data.gov.in)
  - Advanced statistical analysis (Granger Causality, Mann-Kendall, ADF)
  - Multi-model ML comparison (Linear, Polynomial, Random Forest, SARIMAX)
  - Interactive EDA with distribution analysis and anomaly detection
  - Professional Plotly + Streamlit visualization

Run: streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Project imports ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import COLOR_PALETTE, MONTHS, START_YEAR, END_YEAR
from src.data_loader import DataLoader
from src.analysis import (
    compute_correlation,
    standardize_series,
    compute_seasonal_pattern,
)
from src.advanced_analysis import (
    granger_causality_test,
    adf_stationarity_test,
    mann_kendall_trend_test,
    rolling_correlation,
    detect_anomalies,
    profile_dataframe,
    generate_insights,
)
from src.models import (
    forecast_sarimax,
    regression_temp_demand,
    polynomial_regression,
    random_forest_regression,
    compare_models,
)
from src.visualizations import (
    plot_temperature_heatmap,
    plot_temperature_trend,
    plot_temperature_comparison,
    plot_demand_heatmap,
    plot_demand_trend,
    plot_demand_comparison,
    plot_production_pie,
    plot_consumption_pie,
    plot_standardized_overlay,
    plot_scatter_regression,
    plot_gdp_trend,
    plot_profit_loss,
    plot_gdp_state_comparison,
    plot_forecast,
    plot_population_pie,
    plot_urban_rural,
    # ── Advanced charts ──
    plot_seasonal_decomposition,
    plot_rolling_correlation,
    plot_anomaly_detection,
    plot_distribution,
    plot_model_comparison,
    plot_feature_importance,
    plot_residual_analysis,
    plot_correlation_matrix,
    plot_granger_results,
)


# ═══════════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TN Temperature × Electricity",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
        border: 1px solid rgba(78, 205, 196, 0.2);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }

    h1 {
        background: linear-gradient(90deg, #FF6B6B, #FFE66D, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    h2 { color: #4ECDC4; border-bottom: 2px solid rgba(78, 205, 196, 0.3); padding-bottom: 8px; }
    h3 { color: #FFE66D; }
    hr { border-color: rgba(78, 205, 196, 0.2); }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 10px 20px; }

    .api-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; margin: 2px;
    }
    .api-live { background: rgba(78,205,196,0.2); color: #4ECDC4; border: 1px solid #4ECDC4; }
    .api-csv { background: rgba(255,107,107,0.2); color: #FF6B6B; border: 1px solid #FF6B6B; }

    .insight-card {
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
        border: 1px solid rgba(78, 205, 196, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .insight-card h4 { color: #FFE66D; margin: 0 0 6px 0; font-size: 1rem; }
    .insight-card p { color: #e0e0e0; margin: 0; font-size: 0.9rem; }

    .stat-test-box {
        background: rgba(78,205,196,0.08);
        border-left: 3px solid #4ECDC4;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Data Loading (cached)
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="🔄 Fetching data from APIs…")
def load_all_data():
    """Load all datasets with API-first strategy. Cached by Streamlit."""
    loader = DataLoader(prefer_api=True)
    return {
        "temperature": loader.get_temperature(),
        "ap_temperature": loader.get_ap_temperature(),
        "electricity_demand": loader.get_electricity_demand(),
        "generation": loader.get_generation_capacity(),
        "consumption": loader.get_consumption_share(),
        "national_gdp": loader.get_national_gdp(),
        "state_gdp": loader.get_state_gdp(),
        "gdp_comparison": loader.get_state_gdp_comparison(),
        "profit_loss": loader.get_profit_loss(),
        "revenue": loader.get_revenue(),
        "tariff": loader.get_tariff_rates(),
        "population": loader.get_population(),
        "gsdp_gva": loader.get_gsdp_gva(),
    }


data = load_all_data()

# Helper function for column detection
def _col(df, preferred, fallback_idx=-1):
    """Get column name safely."""
    return preferred if preferred in df.columns else df.columns[fallback_idx]


# ═══════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# ⚡ TN Dashboard")
    st.markdown("---")

    page = st.radio(
        "📑 Navigate",
        [
            "🏠 Overview",
            "🔍 EDA",
            "🌡️ Temperature",
            "⚡ Electricity",
            "📊 Correlation",
            "💰 Economics",
            "🤖 ML Models",
            "🔮 Predictions",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🔌 Data Sources")
    st.markdown("""
    <span class="api-badge api-live">🌐 Open-Meteo</span>
    <span class="api-badge api-live">🌐 World Bank</span>
    <span class="api-badge api-live">🌐 data.gov.in</span>
    <span class="api-badge api-csv">📁 CSV Fallback</span>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧪 Methodology")
    st.caption("Pearson · Spearman · Granger Causality · Mann-Kendall · ADF · SARIMAX · Random Forest · Polynomial Regression")
    st.markdown("---")
    st.caption(f"Data: {START_YEAR}–{END_YEAR} · Built with Streamlit + Plotly")


# ═══════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.markdown("# 🌡️ Tamil Nadu: Temperature × Electricity")
    st.markdown(
        "> **An end-to-end data analytics project investigating the relationship between "
        "temperature patterns, electricity demand, and economic factors in Tamil Nadu (2015–2024).**"
    )
    st.markdown("---")

    # ── KPI Cards ──
    col1, col2, col3, col4 = st.columns(4)
    temp_df = data["temperature"]
    demand_df = data["electricity_demand"]
    pl_df = data["profit_loss"]
    pop_df = data["population"]

    if not temp_df.empty:
        avg_temp = temp_df[_col(temp_df, "Temperature")].mean()
        col1.metric("🌡️ Avg Temperature", f"{avg_temp:.1f}°C", "Tamil Nadu")
    if not demand_df.empty:
        dc = _col(demand_df, "Peak Demand (in MW)")
        col2.metric("⚡ Peak Demand", f"{demand_df[dc].max():,.0f} MW", "Maximum")
    if not pl_df.empty:
        col3.metric("💰 TANGEDCO P&L", f"₹{pl_df['Profit and Loss (in Rs Crores)'].iloc[-1]:,.0f} Cr", "Latest")
    if not pop_df.empty and "State (in Cr)" in pop_df.columns:
        col4.metric("👥 Population", f"{pop_df['State (in Cr)'].iloc[0]} Cr", "Census")

    st.markdown("---")

    # ── Auto-Generated Insights ──
    st.markdown("### 🧠 Auto-Generated Insights")
    insights = generate_insights(temp_df, demand_df, pl_df)
    for ins in insights[:6]:
        st.markdown(
            f'<div class="insight-card">'
            f'<h4>{ins["icon"]} {ins["title"]}</h4>'
            f'<p>{ins["body"]}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Dataset Summary + Population ──
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 📊 Dataset Summary")
        summary = pd.DataFrame({
            "Dataset": ["Temperature", "Electricity Demand", "Generation", "GDP (State)", "GDP (National)", "P&L"],
            "Records": [len(temp_df), len(demand_df), len(data["generation"]),
                       len(data["state_gdp"]), len(data["national_gdp"]), len(pl_df)],
            "Source": ["Open-Meteo / CSV", "data.gov.in / CSV", "CSV", "data.gov.in / CSV", "World Bank / CSV", "CSV"],
        })
        st.dataframe(summary, width="stretch", hide_index=True)
    with col_r:
        if not pop_df.empty:
            st.plotly_chart(plot_population_pie(), width="stretch")

    # ── Tech & Methodology ──
    st.markdown("---")
    st.markdown("### 🛠️ Analytics Pipeline")
    st.markdown("""
    ```
    Data Ingestion → EDA & Profiling → Statistical Testing → Feature Engineering → Modelling → Visualization
    (APIs + CSV)     (Distribution,      (ADF, Mann-Kendall,   (Seasonal,          (SARIMAX,     (Plotly
                      Anomalies)          Granger Causality)    Lag Features)       RF, Poly)     Dashboard)
    ```
    """)


# ═══════════════════════════════════════════════════════════════════
# PAGE: EDA  (NEW)
# ═══════════════════════════════════════════════════════════════════

elif page == "🔍 EDA":
    st.markdown("# 🔍 Exploratory Data Analysis")
    st.markdown("> *Systematic exploration of data quality, distributions, and patterns before modelling.*")

    tab1, tab2, tab3 = st.tabs(["📋 Data Profile", "📊 Distributions", "🔎 Anomalies"])

    with tab1:
        st.markdown("### 📋 Data Quality Report")
        selected_ds = st.selectbox("Select Dataset", ["Temperature", "Electricity Demand", "Profit & Loss"])

        ds_map = {"Temperature": data["temperature"], "Electricity Demand": data["electricity_demand"], "Profit & Loss": data["profit_loss"]}
        ds = ds_map[selected_ds]

        if not ds.empty:
            profile = profile_dataframe(ds, selected_ds)

            # Summary metrics
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Rows", f"{profile['n_rows']:,}")
            mcol2.metric("Columns", profile['n_cols'])
            mcol3.metric("Completeness", f"{profile['completeness']}%")
            mcol4.metric("Memory", f"{profile['memory_mb']} MB")

            # Column profiles table
            st.markdown("### Column Details")
            prof_df = pd.DataFrame(profile["column_profiles"])
            display_cols = ["column", "dtype", "non_null", "null_pct", "unique"]
            extra = [c for c in ["mean", "std", "min", "max", "skewness", "kurtosis"] if c in prof_df.columns]
            st.dataframe(prof_df[display_cols + extra], width="stretch", hide_index=True)

            # Skewness interpretation
            for cp in profile["column_profiles"]:
                if cp.get("skewness") is not None and abs(cp["skewness"]) > 1:
                    direction = "right-skewed" if cp["skewness"] > 0 else "left-skewed"
                    st.markdown(
                        f'<div class="stat-test-box">⚠️ <b>{cp["column"]}</b> is significantly '
                        f'<b>{direction}</b> (skewness={cp["skewness"]:.3f}). Consider log transform.</div>',
                        unsafe_allow_html=True)

    with tab2:
        st.markdown("### 📊 Distribution Analysis")
        temp_df = data["temperature"]
        demand_df = data["electricity_demand"]

        col1, col2 = st.columns(2)
        with col1:
            if not temp_df.empty:
                tc = _col(temp_df, "Temperature")
                st.plotly_chart(
                    plot_distribution(temp_df[tc], name="Temperature (°C)", title="Temperature Distribution"),
                    width="stretch",
                )
                # Descriptive stats
                desc = temp_df[tc].describe()
                st.markdown(f"**Mean**: {desc['mean']:.2f}°C  |  **Std**: {desc['std']:.2f}  |  "
                           f"**Skew**: {temp_df[tc].skew():.3f}  |  **Kurt**: {temp_df[tc].kurtosis():.3f}")

        with col2:
            if not demand_df.empty:
                dc = _col(demand_df, "Peak Demand (in MW)")
                st.plotly_chart(
                    plot_distribution(demand_df[dc], name="Peak Demand (MW)", title="Electricity Demand Distribution"),
                    width="stretch",
                )
                desc = demand_df[dc].describe()
                st.markdown(f"**Mean**: {desc['mean']:,.0f} MW  |  **Std**: {desc['std']:,.0f}  |  "
                           f"**Skew**: {demand_df[dc].skew():.3f}  |  **Kurt**: {demand_df[dc].kurtosis():.3f}")

    with tab3:
        st.markdown("### 🔎 Anomaly Detection (Modified Z-Score)")
        anomaly_target = st.selectbox("Analyze", ["Temperature", "Electricity Demand"], key="anomaly_ds")
        threshold = st.slider("Z-Score Threshold", 1.5, 4.0, 2.5, 0.1, key="anomaly_thresh")

        if anomaly_target == "Temperature" and not data["temperature"].empty:
            series = data["temperature"][_col(data["temperature"], "Temperature")]
            ylabel = "Temperature (°C)"
        else:
            series = data["electricity_demand"][_col(data["electricity_demand"], "Peak Demand (in MW)")]
            ylabel = "Peak Demand (MW)"

        result = detect_anomalies(series.dropna(), threshold=threshold)

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Total Anomalies", result["total_anomalies"])
        mcol2.metric("High Anomalies", result["high_anomalies"])
        mcol3.metric("% Anomalous", f"{result['pct_anomalies']}%")

        st.plotly_chart(
            plot_anomaly_detection(series.dropna(), result["z_scores"], result["anomaly_indices"],
                                  title=f"{anomaly_target} — Anomaly Detection", y_label=ylabel),
            width="stretch",
        )


# ═══════════════════════════════════════════════════════════════════
# PAGE: Temperature
# ═══════════════════════════════════════════════════════════════════

elif page == "🌡️ Temperature":
    st.markdown("# 🌡️ Temperature Analysis")
    temp_df = data["temperature"]

    if temp_df.empty:
        st.warning("Temperature data not available.")
    else:
        tc = _col(temp_df, "Temperature")
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🗺️ Heatmap", "🔬 Decomposition", "🔄 Comparison"])

        with tab1:
            st.plotly_chart(plot_temperature_trend(temp_df, title="Tamil Nadu Temperature Trend (2015–2024)"), width="stretch")

            # Mann-Kendall trend test
            mk = mann_kendall_trend_test(temp_df[tc], name="Temperature")
            st.markdown(f'<div class="stat-test-box">{mk["interpretation"]}</div>', unsafe_allow_html=True)

            # Monthly stats
            st.markdown("### 📊 Monthly Statistics")
            mcol = "Month_Name" if "Month_Name" in temp_df.columns else "Month"
            stats = temp_df.groupby(mcol)[tc].agg(["mean", "min", "max", "std"]).round(2)
            st.dataframe(stats, width="stretch")

        with tab2:
            st.plotly_chart(plot_temperature_heatmap(temp_df), width="stretch")
            st.info("💡 April–June consistently show the highest temperatures, coinciding with peak demand.")

        with tab3:
            st.markdown("### 🔬 Seasonal Decomposition")
            decomp = compute_seasonal_pattern(temp_df[tc])
            if decomp["trend"] is not None:
                st.plotly_chart(
                    plot_seasonal_decomposition(
                        decomp["observed"], decomp["trend"],
                        decomp["seasonal"], decomp["residual"],
                        title="Temperature — Seasonal Decomposition",
                    ),
                    width="stretch",
                )
                st.info("💡 The **seasonal component** reveals a clear annual cycle. "
                       "The **trend** shows long-term direction, while **residuals** capture noise.")

            # ADF stationarity test
            adf = adf_stationarity_test(temp_df[tc], name="Temperature")
            st.markdown(f'<div class="stat-test-box">{adf.get("interpretation", "Test unavailable")}</div>', unsafe_allow_html=True)

        with tab4:
            ap_df = data["ap_temperature"]
            if not ap_df.empty:
                tn_vals = temp_df[tc]
                ap_vals = ap_df[_col(ap_df, "Temperature")]
                st.plotly_chart(plot_temperature_comparison(tn_vals, ap_vals), width="stretch")
            else:
                st.info("AP comparison data not available.")


# ═══════════════════════════════════════════════════════════════════
# PAGE: Electricity
# ═══════════════════════════════════════════════════════════════════

elif page == "⚡ Electricity":
    st.markdown("# ⚡ Electricity Demand & Production")
    demand_df = data["electricity_demand"]

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🗺️ Heatmap", "🏭 Production Mix", "🍰 Consumption"])

    with tab1:
        if not demand_df.empty:
            dc = _col(demand_df, "Peak Demand (in MW)")
            st.plotly_chart(plot_demand_trend(demand_df, title="Peak Demand Trend (2015–2024)"), width="stretch")

            # Mann-Kendall
            mk = mann_kendall_trend_test(demand_df[dc], name="Electricity Demand")
            st.markdown(f'<div class="stat-test-box">{mk["interpretation"]}</div>', unsafe_allow_html=True)

            # Decomposition
            decomp = compute_seasonal_pattern(demand_df[dc])
            if decomp["trend"] is not None:
                st.markdown("### 🔬 Seasonal Decomposition")
                st.plotly_chart(
                    plot_seasonal_decomposition(
                        decomp["observed"], decomp["trend"],
                        decomp["seasonal"], decomp["residual"],
                        title="Electricity Demand — Seasonal Pattern",
                    ),
                    width="stretch",
                )

    with tab2:
        if not demand_df.empty:
            st.plotly_chart(plot_demand_heatmap(demand_df), width="stretch")

    with tab3:
        gen_df = data["generation"]
        if not gen_df.empty and "Year" in gen_df.columns:
            years = sorted(gen_df["Year"].unique(), reverse=True)
            yr = st.selectbox("Year", years, key="gen_yr")
            st.plotly_chart(plot_production_pie(gen_df, yr), width="stretch")

    with tab4:
        cons_df = data["consumption"]
        if not cons_df.empty and "Year" in cons_df.columns:
            years = sorted(cons_df["Year"].unique(), reverse=True)
            yr = st.selectbox("Year", years, key="cons_yr")
            st.plotly_chart(plot_consumption_pie(cons_df, yr), width="stretch")


# ═══════════════════════════════════════════════════════════════════
# PAGE: Correlation  (Enhanced)
# ═══════════════════════════════════════════════════════════════════

elif page == "📊 Correlation":
    st.markdown("# 📊 Temperature × Electricity Correlation")
    temp_df = data["temperature"]
    demand_df = data["electricity_demand"]

    if temp_df.empty or demand_df.empty:
        st.warning("Insufficient data for correlation analysis.")
    else:
        tc = _col(temp_df, "Temperature")
        dc = _col(demand_df, "Peak Demand (in MW)")
        t_vals = temp_df[tc]
        d_vals = demand_df[dc]
        min_len = min(len(t_vals), len(d_vals))
        t_vals = t_vals.iloc[:min_len]
        d_vals = pd.Series(d_vals.values[:min_len])

        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Overlay", "📍 Scatter", "🧪 Granger Causality", "🔄 Rolling Correlation"
        ])

        with tab1:
            std_t, std_d = standardize_series(t_vals, d_vals)
            st.plotly_chart(
                plot_standardized_overlay(std_t, std_d, "Temperature (Z)", "Demand (Z)",
                                         title="Temperature vs Demand (Standardized)"),
                width="stretch",
            )

        with tab2:
            reg = regression_temp_demand(t_vals, d_vals)
            if reg["model"]:
                st.plotly_chart(
                    plot_scatter_regression(t_vals, d_vals, y_pred=reg["predictions"]),
                    width="stretch",
                )
                st.markdown(f"**Equation**: `{reg['equation']}`")
                mcols = st.columns(4)
                m = reg["metrics"]
                mcols[0].metric("R²", m.get("R²", "N/A"))
                mcols[1].metric("MAE", m.get("MAE", "N/A"))
                mcols[2].metric("RMSE", m.get("RMSE", "N/A"))
                mcols[3].metric("MAPE", f"{m.get('MAPE (%)', 'N/A')}%")

            # Correlation coefficients
            st.markdown("---")
            pcol, scol = st.columns(2)
            pearson = compute_correlation(t_vals, d_vals, method="pearson")
            spearman = compute_correlation(t_vals, d_vals, method="spearman")
            with pcol:
                st.markdown("#### Pearson")
                st.metric("r", pearson["coefficient"])
                st.markdown(f"p={pearson['p_value']} · **{pearson['strength']}** · {'✅ Sig' if pearson['significant'] else '❌ Not Sig'}")
            with scol:
                st.markdown("#### Spearman")
                st.metric("ρ", spearman["coefficient"])
                st.markdown(f"p={spearman['p_value']} · **{spearman['strength']}** · {'✅ Sig' if spearman['significant'] else '❌ Not Sig'}")

        with tab3:
            st.markdown("### 🧪 Granger Causality Test")
            st.markdown("*Does temperature **cause** changes in electricity demand?*")
            max_lag = st.slider("Max Lag (months)", 1, 12, 6, key="granger_lag")
            granger = granger_causality_test(t_vals, d_vals, max_lag=max_lag)

            if "error" not in granger:
                st.plotly_chart(plot_granger_results(granger), width="stretch")
                st.markdown(f'<div class="stat-test-box">📋 **Conclusion**: {granger["conclusion"]}</div>',
                           unsafe_allow_html=True)
                st.markdown(f"**Best lag**: {granger['best_lag']} months · **Best p-value**: {granger['best_p_value']:.6f}")
            else:
                st.error(granger.get("error", "Test failed"))

        with tab4:
            st.markdown("### 🔄 Rolling Correlation")
            window = st.slider("Window Size (months)", 6, 24, 12, key="roll_window")
            roll_corr = rolling_correlation(t_vals, d_vals, window=window)
            st.plotly_chart(
                plot_rolling_correlation(roll_corr, window=window,
                                        title=f"Rolling Correlation ({window}-month)"),
                width="stretch",
            )
            st.info("💡 This shows how the temperature-demand relationship **changes over time**. "
                   "Peaks indicate periods where the two variables move together most strongly.")


# ═══════════════════════════════════════════════════════════════════
# PAGE: Economics
# ═══════════════════════════════════════════════════════════════════

elif page == "💰 Economics":
    st.markdown("# 💰 Economic Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 TN GDP", "🏛️ National GDP", "🗺️ State Comparison", "📊 TANGEDCO"])

    with tab1:
        sgdp = data["state_gdp"]
        if not sgdp.empty:
            st.plotly_chart(plot_gdp_trend(sgdp, title="Tamil Nadu GDP Growth"), width="stretch")

    with tab2:
        ngdp = data["national_gdp"]
        if not ngdp.empty:
            vc = "GDP_Lakh_Crore" if "GDP_Lakh_Crore" in ngdp.columns else \
                 "Price (in Rs.Lakh Crore)" if "Price (in Rs.Lakh Crore)" in ngdp.columns else \
                 ngdp.select_dtypes(include=[np.number]).columns[-1]
            st.plotly_chart(plot_gdp_trend(ngdp, value_col=vc, title="India GDP (World Bank)"), width="stretch")

    with tab3:
        comp_df = data["gdp_comparison"]
        if not comp_df.empty:
            try:
                temp = comp_df["Price (in Rs.Lakh Crore)"]
                arr1 = np.zeros(39)
                for i, idx in enumerate([140, 179, 293]):
                    for j in range(13):
                        arr1[i * 13 + j] = temp.iloc[idx + j]
                st.plotly_chart(
                    plot_gdp_state_comparison(arr1[:13].tolist(), arr1[13:26].tolist(),
                                             arr1[26:39].tolist(), list(range(2011, 2024))),
                    width="stretch",
                )
            except (IndexError, KeyError):
                st.info("State comparison data format mismatch.")

    with tab4:
        pldf = data["profit_loss"]
        if not pldf.empty:
            st.plotly_chart(plot_profit_loss(pldf), width="stretch")
            st.warning("⚠️ TANGEDCO has been operating at a loss for most years.")

        rev = data["revenue"]
        if not rev.empty:
            st.markdown("### Average Cost vs Revenue")
            st.dataframe(rev, width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE: ML Models  (NEW)
# ═══════════════════════════════════════════════════════════════════

elif page == "🤖 ML Models":
    st.markdown("# 🤖 Machine Learning Model Comparison")
    st.markdown("> *Systematic evaluation of Linear, Polynomial, and Random Forest models for predicting electricity demand from temperature.*")

    temp_df = data["temperature"]
    demand_df = data["electricity_demand"]

    if temp_df.empty or demand_df.empty:
        st.warning("Insufficient data for ML modelling.")
    else:
        tc = _col(temp_df, "Temperature")
        dc = _col(demand_df, "Peak Demand (in MW)")
        t_vals = temp_df[tc]
        d_vals = demand_df[dc]
        min_len = min(len(t_vals), len(d_vals))
        t_vals = pd.Series(t_vals.values[:min_len])
        d_vals = pd.Series(d_vals.values[:min_len])
        m_vals = temp_df["Month"].iloc[:min_len] if "Month" in temp_df.columns else None

        tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard", "🌲 Random Forest", "📐 Residual Analysis"])

        with tab1:
            st.markdown("### 🏆 Model Performance Leaderboard")

            with st.spinner("Training 4 models…"):
                leaderboard = compare_models(t_vals, d_vals, m_vals)

            if leaderboard:
                # Leaderboard table
                lb_data = []
                for r in leaderboard:
                    lb_data.append({
                        "🏅 Rank": r["rank"],
                        "Model": r["name"],
                        "R²": r["metrics"].get("R²", "N/A"),
                        "MAE": r["metrics"].get("MAE", "N/A"),
                        "RMSE": r["metrics"].get("RMSE", "N/A"),
                        "MAPE (%)": r["metrics"].get("MAPE (%)", "N/A"),
                    })
                st.dataframe(pd.DataFrame(lb_data), width="stretch", hide_index=True)

                # Visual comparison
                st.plotly_chart(plot_model_comparison(leaderboard), width="stretch")

                # Winner announcement
                winner = leaderboard[0]
                st.success(f"🏆 **Best Model**: {winner['name']} with R² = {winner['metrics']['R²']}")

                if winner.get("equation"):
                    st.markdown(f"**Equation**: `{winner['equation']}`")

        with tab2:
            st.markdown("### 🌲 Random Forest Deep Dive")

            rf = random_forest_regression(t_vals, d_vals, m_vals)
            if rf["model"]:
                # Feature importance
                st.plotly_chart(
                    plot_feature_importance(rf["feature_importance"]),
                    width="stretch",
                )

                # Cross-validation
                if rf["cv_r2_mean"] is not None:
                    st.markdown("### 📊 Cross-Validation (5-Fold)")
                    mcol1, mcol2 = st.columns(2)
                    mcol1.metric("CV R² (Mean)", rf["cv_r2_mean"])
                    mcol2.metric("CV R² (Std)", f"±{rf['cv_r2_std']}")
                    st.info("💡 Cross-validation ensures the model generalizes and isn't just memorizing training data.")

                # Scatter: actual vs predicted
                st.markdown("### Predicted vs Actual")
                st.plotly_chart(
                    plot_scatter_regression(
                        d_vals.iloc[:len(rf["predictions"])], pd.Series(rf["predictions"]),
                        x_label="Actual (MW)", y_label="Predicted (MW)",
                        title="Random Forest: Actual vs Predicted",
                    ),
                    width="stretch",
                )

        with tab3:
            st.markdown("### 📐 Residual Analysis")
            st.markdown("*Good residuals should be normally distributed around zero with no patterns.*")

            model_choice = st.selectbox("Model", ["Polynomial (Degree 2)", "Random Forest"], key="resid_model")

            if model_choice == "Polynomial (Degree 2)":
                poly = polynomial_regression(t_vals, d_vals, degree=2)
                if poly["model"]:
                    st.plotly_chart(
                        plot_residual_analysis(poly["residuals"], poly["predictions"]),
                        width="stretch",
                    )
            else:
                rf = random_forest_regression(t_vals, d_vals, m_vals)
                if rf["model"]:
                    st.plotly_chart(
                        plot_residual_analysis(rf["residuals"], rf["predictions"]),
                        width="stretch",
                    )

            st.info("💡 **Residuals vs Fitted**: Look for random scatter. Patterns indicate model misspecification.\n\n"
                   "**Distribution**: Should be approximately normal (bell curve).\n\n"
                   "**Q-Q Plot**: Points on the diagonal = normally distributed residuals.")


# ═══════════════════════════════════════════════════════════════════
# PAGE: Predictions
# ═══════════════════════════════════════════════════════════════════

elif page == "🔮 Predictions":
    st.markdown("# 🔮 Time Series Forecasting")

    demand_df = data["electricity_demand"]
    temp_df = data["temperature"]

    tab1, tab2 = st.tabs(["⚡ Demand Forecast", "🌡️ Temperature Forecast"])

    with tab1:
        if not demand_df.empty:
            dc = _col(demand_df, "Peak Demand (in MW)")
            series = demand_df[dc].reset_index(drop=True)

            # Stationarity check
            adf = adf_stationarity_test(series, name="Demand Series")
            st.markdown(f'<div class="stat-test-box">{adf.get("interpretation", "")}</div>', unsafe_allow_html=True)

            mcol1, mcol2, mcol3 = st.columns(3)
            forecast_periods = mcol1.slider("Forecast Months", 6, 36, 24, key="d_fp")
            test_size = mcol2.slider("Test Size", 6, 24, 12, key="d_ts")
            mcol3.markdown("**Model**: SARIMAX(1,1,1)(1,1,0,12)")

            if st.button("🚀 Run Demand Forecast", type="primary", key="btn_d"):
                with st.spinner("Training SARIMAX…"):
                    result = forecast_sarimax(series, forecast_steps=forecast_periods, test_size=test_size)

                if result["train_metrics"]:
                    mcols = st.columns(4)
                    m = result["train_metrics"]
                    mcols[0].metric("MAE", m.get("MAE", "N/A"))
                    mcols[1].metric("RMSE", m.get("RMSE", "N/A"))
                    mcols[2].metric("R²", m.get("R²", "N/A"))
                    mcols[3].metric("MAPE", f"{m.get('MAPE (%)', 'N/A')}%")

                    st.plotly_chart(
                        plot_forecast(series, result["forecast"], result["forecast_ci"],
                                     result["test_actual"], result["test_predicted"],
                                     title=f"Demand Forecast ({forecast_periods} months)", y_label="MW"),
                        width="stretch",
                    )

                    if result.get("aic"):
                        st.markdown(f"**AIC**: {result['aic']} | **BIC**: {result['bic']}")
                    with st.expander("📋 Model Summary"):
                        st.code(result["model_summary"])

                    # Download forecast
                    fc_df = pd.DataFrame({"Period": range(1, forecast_periods + 1), "Forecast_MW": result["forecast"].values.round(1)})
                    st.download_button("📥 Download Forecast CSV", fc_df.to_csv(index=False), "demand_forecast.csv", "text/csv")

    with tab2:
        if not temp_df.empty:
            tc = _col(temp_df, "Temperature")
            series = temp_df[tc].reset_index(drop=True)

            mcol1, mcol2 = st.columns(2)
            fp = mcol1.slider("Forecast Months", 6, 36, 12, key="t_fp")
            ts = mcol2.slider("Test Size", 6, 24, 12, key="t_ts")

            if st.button("🚀 Run Temperature Forecast", type="primary", key="btn_t"):
                with st.spinner("Training SARIMAX…"):
                    result = forecast_sarimax(series, order=(1,1,1), seasonal_order=(1,1,1,12), forecast_steps=fp, test_size=ts)

                if result["train_metrics"]:
                    mcols = st.columns(4)
                    m = result["train_metrics"]
                    mcols[0].metric("MAE", f"{m.get('MAE', 'N/A')}°C")
                    mcols[1].metric("RMSE", f"{m.get('RMSE', 'N/A')}°C")
                    mcols[2].metric("R²", m.get("R²", "N/A"))
                    mcols[3].metric("MAPE", f"{m.get('MAPE (%)', 'N/A')}%")

                    st.plotly_chart(
                        plot_forecast(series, result["forecast"], result["forecast_ci"],
                                     result["test_actual"], result["test_predicted"],
                                     title=f"Temperature Forecast ({fp} months)", y_label="°C"),
                        width="stretch",
                    )


# ═══════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "🌡️ Tamil Nadu Temperature × Electricity — Advanced Analytics Dashboard<br>"
    "APIs: Open-Meteo · World Bank · data.gov.in | "
    "ML: SARIMAX · Random Forest · Polynomial Regression<br>"
    "Stats: Granger Causality · Mann-Kendall · ADF · Pearson · Spearman"
    "</div>",
    unsafe_allow_html=True,
)
