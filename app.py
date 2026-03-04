"""
🌡️ Tamil Nadu Temperature vs Electricity — Interactive Dashboard

A comprehensive Streamlit dashboard analyzing the relationship between
temperature patterns and electricity consumption in Tamil Nadu, India.

Data sourced via APIs (Open-Meteo, World Bank, data.gov.in) with CSV fallback.

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
from src.models import forecast_sarimax, regression_temp_demand
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

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
        border: 1px solid rgba(78, 205, 196, 0.2);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }

    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Headers */
    h1 {
        background: linear-gradient(90deg, #FF6B6B, #FFE66D, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    h2 {
        color: #4ECDC4;
        border-bottom: 2px solid rgba(78, 205, 196, 0.3);
        padding-bottom: 8px;
    }

    h3 {
        color: #FFE66D;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
    }

    /* Divider */
    hr {
        border-color: rgba(78, 205, 196, 0.2);
    }

    /* Info boxes */
    .api-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .api-live { background: rgba(78,205,196,0.2); color: #4ECDC4; border: 1px solid #4ECDC4; }
    .api-csv { background: rgba(255,107,107,0.2); color: #FF6B6B; border: 1px solid #FF6B6B; }
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
            "🌡️ Temperature",
            "⚡ Electricity",
            "📊 Correlation",
            "💰 Economics",
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
    st.caption("Built with Streamlit + Plotly")
    st.caption(f"Data: {START_YEAR}–{END_YEAR}")


# ═══════════════════════════════════════════════════════════════════
# Page: Overview
# ═══════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.markdown("# 🌡️ Tamil Nadu: Temperature × Electricity")
    st.markdown(
        "> **Analyzing the intricate relationship between rising temperatures "
        "and electricity demand in Tamil Nadu using data science & machine learning.**"
    )

    st.markdown("---")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    temp_df = data["temperature"]
    demand_df = data["electricity_demand"]

    if not temp_df.empty:
        avg_temp = temp_df["Temperature"].mean() if "Temperature" in temp_df.columns else 0
        col1.metric("🌡️ Avg Temperature", f"{avg_temp:.1f}°C", "Tamil Nadu")

    if not demand_df.empty:
        demand_col = "Peak Demand (in MW)" if "Peak Demand (in MW)" in demand_df.columns else demand_df.columns[-1]
        max_demand = demand_df[demand_col].max()
        col2.metric("⚡ Peak Demand", f"{max_demand:,.0f} MW", "Maximum Recorded")

    pl_df = data["profit_loss"]
    if not pl_df.empty:
        latest_pl = pl_df["Profit and Loss (in Rs Crores)"].iloc[-1]
        col3.metric("💰 TANGEDCO P&L", f"₹{latest_pl:,.0f} Cr",
                    "Latest Year",
                    delta_color="inverse")

    pop_df = data["population"]
    if not pop_df.empty and "State (in Cr)" in pop_df.columns:
        pop = pop_df["State (in Cr)"].iloc[0]
        col4.metric("👥 Population", f"{pop} Cr", "Census 2011")

    st.markdown("---")

    # Overview charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📊 Dataset Summary")
        summary_data = {
            "Dataset": ["Temperature", "Electricity Demand", "Generation Capacity",
                        "GDP (State)", "GDP (National)", "Profit/Loss"],
            "Records": [
                len(temp_df), len(demand_df), len(data["generation"]),
                len(data["state_gdp"]), len(data["national_gdp"]), len(pl_df)
            ],
            "Source": ["Open-Meteo API / CSV", "data.gov.in / CSV", "CSV",
                      "data.gov.in / CSV", "World Bank API / CSV", "CSV"],
        }
        st.dataframe(pd.DataFrame(summary_data), width='stretch', hide_index=True)

    with col_right:
        if not pop_df.empty:
            st.plotly_chart(plot_population_pie(), width='stretch')

    # Project description
    st.markdown("---")
    st.markdown("### 🎯 Project Objectives")
    st.markdown("""
    1. **Analyze** historical temperature patterns across Tamil Nadu (2015–2024)
    2. **Investigate** the correlation between temperature and electricity demand
    3. **Compare** Tamil Nadu with neighboring states (Andhra Pradesh)
    4. **Forecast** future electricity demand using SARIMAX time series models
    5. **Evaluate** economic impact through GDP and TANGEDCO financial analysis
    6. **Fetch live data** from Open-Meteo, World Bank, and data.gov.in APIs
    """)

    st.markdown("### 🛠️ Tech Stack")
    tech_cols = st.columns(4)
    tech_cols[0].markdown("**Data**\n\nPandas · NumPy · Requests")
    tech_cols[1].markdown("**ML/Stats**\n\nScikit-learn · Statsmodels · SciPy")
    tech_cols[2].markdown("**Visualization**\n\nPlotly · Seaborn · Matplotlib")
    tech_cols[3].markdown("**Dashboard**\n\nStreamlit · Python-dotenv")


# ═══════════════════════════════════════════════════════════════════
# Page: Temperature
# ═══════════════════════════════════════════════════════════════════

elif page == "🌡️ Temperature":
    st.markdown("# 🌡️ Temperature Analysis")

    temp_df = data["temperature"]

    if temp_df.empty:
        st.warning("Temperature data not available. Check API connection or CSV files.")
    else:
        tab1, tab2, tab3 = st.tabs(["📈 Trends", "🗺️ Heatmap", "🔄 State Comparison"])

        with tab1:
            st.plotly_chart(
                plot_temperature_trend(temp_df, title="Tamil Nadu Temperature Trend (2015–2024)"),
                width='stretch',
            )

            # Monthly statistics
            st.markdown("### 📊 Monthly Temperature Statistics")
            if "Month_Name" in temp_df.columns or "Month" in temp_df.columns:
                month_col = "Month_Name" if "Month_Name" in temp_df.columns else "Month"
                temp_col = "Temperature" if "Temperature" in temp_df.columns else temp_df.columns[-1]
                stats = temp_df.groupby(month_col)[temp_col].agg(["mean", "min", "max", "std"]).round(2)
                st.dataframe(stats, width='stretch')

        with tab2:
            st.plotly_chart(
                plot_temperature_heatmap(temp_df),
                width='stretch',
            )
            st.info("💡 **Insight**: April–June consistently show the highest temperatures, "
                   "coinciding with peak electricity demand season.")

        with tab3:
            ap_df = data["ap_temperature"]
            if not ap_df.empty and not temp_df.empty:
                tn_vals = temp_df["Temperature"] if "Temperature" in temp_df.columns else temp_df.iloc[:, -1]
                ap_vals = ap_df["Temperature"] if "Temperature" in ap_df.columns else ap_df.iloc[:, -1]
                st.plotly_chart(
                    plot_temperature_comparison(tn_vals, ap_vals),
                    width='stretch',
                )
            else:
                st.info("AP temperature comparison data not available.")


# ═══════════════════════════════════════════════════════════════════
# Page: Electricity
# ═══════════════════════════════════════════════════════════════════

elif page == "⚡ Electricity":
    st.markdown("# ⚡ Electricity Demand & Production")

    demand_df = data["electricity_demand"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Demand Trends", "🗺️ Demand Heatmap",
        "🏭 Production Mix", "🍰 Consumption Sectors"
    ])

    with tab1:
        if not demand_df.empty:
            st.plotly_chart(
                plot_demand_trend(demand_df, title="Electricity Peak Demand Trend (2015–2024)"),
                width='stretch',
            )

            # Year comparison
            st.markdown("### 📊 2015 vs 2024 Comparison")
            demand_col = "Peak Demand (in MW)" if "Peak Demand (in MW)" in demand_df.columns else demand_df.columns[-1]
            values = demand_df[demand_col].values

            if len(values) >= 120:  # 10 years × 12 months
                cons_2024 = list(values[:12])
                cons_2015 = list(values[-12:])
                st.plotly_chart(
                    plot_demand_comparison(cons_2015, cons_2024, "2015", "2024"),
                    width='stretch',
                )

                growth = ((sum(cons_2024) - sum(cons_2015)) / sum(cons_2015)) * 100
                st.success(f"📈 Electricity demand grew **{growth:.1f}%** from 2015 to 2024!")

    with tab2:
        if not demand_df.empty:
            st.plotly_chart(
                plot_demand_heatmap(demand_df),
                width='stretch',
            )

    with tab3:
        gen_df = data["generation"]
        if not gen_df.empty and "Year" in gen_df.columns:
            years = sorted(gen_df["Year"].unique(), reverse=True)
            selected_year = st.selectbox("Select Year", years, key="gen_year")
            st.plotly_chart(
                plot_production_pie(gen_df, selected_year),
                width='stretch',
            )

    with tab4:
        cons_df = data["consumption"]
        if not cons_df.empty and "Year" in cons_df.columns:
            years = sorted(cons_df["Year"].unique(), reverse=True)
            selected_year = st.selectbox("Select Year", years, key="cons_year")
            st.plotly_chart(
                plot_consumption_pie(cons_df, selected_year),
                width='stretch',
            )


# ═══════════════════════════════════════════════════════════════════
# Page: Correlation
# ═══════════════════════════════════════════════════════════════════

elif page == "📊 Correlation":
    st.markdown("# 📊 Temperature × Electricity Correlation")

    temp_df = data["temperature"]
    demand_df = data["electricity_demand"]

    if temp_df.empty or demand_df.empty:
        st.warning("Insufficient data for correlation analysis.")
    else:
        temp_col = "Temperature" if "Temperature" in temp_df.columns else temp_df.columns[-1]
        demand_col = "Peak Demand (in MW)" if "Peak Demand (in MW)" in demand_df.columns else demand_df.columns[-1]

        temp_vals = temp_df[temp_col]
        demand_vals = demand_df[demand_col]

        # Align lengths
        min_len = min(len(temp_vals), len(demand_vals))

        # Reverse demand if it's in reverse chronological order
        d_vals = demand_vals.iloc[:min_len]
        if len(d_vals) > 1:
            # Check if it's reverse by looking at year column
            first_year = str(demand_df["Year"].iloc[0])
            last_year = str(demand_df["Year"].iloc[-1])
            if first_year > last_year:
                d_vals = pd.Series(d_vals.values[::-1])

        t_vals = temp_vals.iloc[:min_len]

        tab1, tab2, tab3 = st.tabs(["📈 Overlay", "📍 Scatter Plot", "📊 Statistics"])

        with tab1:
            st.markdown("### Standardized Comparison")
            std_temp, std_demand = standardize_series(t_vals, d_vals)
            st.plotly_chart(
                plot_standardized_overlay(
                    std_temp, std_demand,
                    "Temperature (Z-score)", "Electricity Demand (Z-score)",
                    title="Temperature vs Electricity Demand (Standardized)",
                ),
                width='stretch',
            )

        with tab2:
            st.markdown("### Scatter Plot with Regression Line")
            reg_result = regression_temp_demand(t_vals, d_vals)

            if reg_result["model"] is not None:
                st.plotly_chart(
                    plot_scatter_regression(
                        t_vals.iloc[:min_len], d_vals.iloc[:min_len],
                        y_pred=reg_result["predictions"],
                    ),
                    width='stretch',
                )

                st.markdown(f"**Regression Equation**: `{reg_result['equation']}`")

                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                metrics = reg_result["metrics"]
                mcol1.metric("R² Score", f"{metrics.get('R²', 'N/A')}")
                mcol2.metric("MAE", f"{metrics.get('MAE', 'N/A')}")
                mcol3.metric("RMSE", f"{metrics.get('RMSE', 'N/A')}")
                mcol4.metric("MAPE", f"{metrics.get('MAPE (%)', 'N/A')}%")

        with tab3:
            st.markdown("### Correlation Coefficients")

            col1, col2 = st.columns(2)

            pearson = compute_correlation(t_vals, d_vals, method="pearson")
            spearman = compute_correlation(t_vals, d_vals, method="spearman")

            with col1:
                st.markdown("#### Pearson Correlation")
                st.metric("Coefficient", pearson["coefficient"])
                st.metric("P-value", pearson["p_value"])
                st.markdown(f"**Strength**: {pearson['strength']}")
                st.markdown(f"**Significant**: {'✅ Yes' if pearson['significant'] else '❌ No'}")

            with col2:
                st.markdown("#### Spearman Correlation")
                st.metric("Coefficient", spearman["coefficient"])
                st.metric("P-value", spearman["p_value"])
                st.markdown(f"**Strength**: {spearman['strength']}")
                st.markdown(f"**Significant**: {'✅ Yes' if spearman['significant'] else '❌ No'}")

            if pearson["significant"]:
                st.success(
                    f"🔬 **Statistically significant** {pearson['strength'].lower()} correlation "
                    f"detected between temperature and electricity demand (r = {pearson['coefficient']}, "
                    f"p = {pearson['p_value']})."
                )


# ═══════════════════════════════════════════════════════════════════
# Page: Economics
# ═══════════════════════════════════════════════════════════════════

elif page == "💰 Economics":
    st.markdown("# 💰 Economic Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 TN GDP", "🏛️ National GDP", "🗺️ State Comparison", "📊 TANGEDCO P&L"
    ])

    with tab1:
        state_gdp = data["state_gdp"]
        if not state_gdp.empty:
            st.plotly_chart(
                plot_gdp_trend(state_gdp, title="Tamil Nadu GDP Growth"),
                width='stretch',
            )
        else:
            st.info("State GDP data not available.")

    with tab2:
        nat_gdp = data["national_gdp"]
        if not nat_gdp.empty:
            value_col = "GDP_Lakh_Crore" if "GDP_Lakh_Crore" in nat_gdp.columns else \
                        "Price (in Rs.Lakh Crore)" if "Price (in Rs.Lakh Crore)" in nat_gdp.columns else \
                        nat_gdp.select_dtypes(include=[np.number]).columns[-1]
            st.plotly_chart(
                plot_gdp_trend(nat_gdp, value_col=value_col, title="India National GDP (World Bank API)"),
                width='stretch',
            )
            st.info("💡 Data sourced from **World Bank Indicators API** (`NY.GDP.MKTP.CN`)")

    with tab3:
        comp_df = data["gdp_comparison"]
        if not comp_df.empty:
            # Extract TN, Karnataka, Maharashtra
            temp = comp_df["Price (in Rs.Lakh Crore)"]
            try:
                arr1 = np.zeros(39)
                lst1 = [140, 179, 293]
                for i in range(3):
                    for j in range(13):
                        arr1[i * 13 + j] = temp.iloc[(lst1[i]) + j]
                TN = arr1[:13].tolist()
                KA = arr1[13:26].tolist()
                MH = arr1[26:39].tolist()
                years = list(range(2011, 2024))
                st.plotly_chart(
                    plot_gdp_state_comparison(TN, KA, MH, years),
                    width='stretch',
                )
            except (IndexError, KeyError):
                st.info("State comparison data format mismatch.")

    with tab4:
        pl_df = data["profit_loss"]
        if not pl_df.empty:
            st.plotly_chart(
                plot_profit_loss(pl_df),
                width='stretch',
            )
            st.warning("⚠️ TANGEDCO has been operating at a loss for most years, "
                      "highlighting challenges in the power distribution sector.")

        # Revenue analysis
        rev_df = data["revenue"]
        if not rev_df.empty:
            st.markdown("### 📊 Average Cost vs Revenue")
            st.dataframe(rev_df, width='stretch', hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# Page: Predictions
# ═══════════════════════════════════════════════════════════════════

elif page == "🔮 Predictions":
    st.markdown("# 🔮 Time Series Forecasting")

    demand_df = data["electricity_demand"]
    temp_df = data["temperature"]

    tab1, tab2 = st.tabs(["⚡ Demand Forecast", "🌡️ Temperature Forecast"])

    with tab1:
        if not demand_df.empty:
            demand_col = "Peak Demand (in MW)" if "Peak Demand (in MW)" in demand_df.columns else demand_df.columns[-1]
            series = demand_df[demand_col].reset_index(drop=True)

            st.markdown("### ⚙️ Model Configuration")
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                forecast_periods = st.slider("Forecast Periods (months)", 6, 36, 24, key="demand_periods")
            with mcol2:
                test_size = st.slider("Test Size (months)", 6, 24, 12, key="demand_test")
            with mcol3:
                st.markdown("**Model**: SARIMAX(1,1,1)(1,1,0,12)")

            if st.button("🚀 Run Demand Forecast", type="primary", key="btn_demand"):
                with st.spinner("Training SARIMAX model…"):
                    result = forecast_sarimax(
                        series,
                        forecast_steps=forecast_periods,
                        test_size=test_size,
                    )

                if result["train_metrics"]:
                    st.markdown("### 📊 Model Performance (Test Set)")
                    mcols = st.columns(4)
                    metrics = result["train_metrics"]
                    mcols[0].metric("MAE", metrics.get("MAE", "N/A"))
                    mcols[1].metric("RMSE", metrics.get("RMSE", "N/A"))
                    mcols[2].metric("R²", metrics.get("R²", "N/A"))
                    mcols[3].metric("MAPE", f"{metrics.get('MAPE (%)', 'N/A')}%")

                    st.markdown("### 📈 Forecast Visualization")
                    st.plotly_chart(
                        plot_forecast(
                            series,
                            result["forecast"],
                            result["forecast_ci"],
                            result["test_actual"],
                            result["test_predicted"],
                            title=f"Electricity Demand Forecast ({forecast_periods} months)",
                            y_label="Peak Demand (MW)",
                        ),
                        width='stretch',
                    )

                    # Model info
                    if result.get("aic"):
                        st.markdown(f"**AIC**: {result['aic']} | **BIC**: {result['bic']}")

                    with st.expander("📋 View Full Model Summary"):
                        st.code(result["model_summary"])
                else:
                    st.error("Model training failed. Check the data and try different parameters.")

    with tab2:
        if not temp_df.empty:
            temp_col = "Temperature" if "Temperature" in temp_df.columns else temp_df.columns[-1]
            series = temp_df[temp_col].reset_index(drop=True)

            st.markdown("### ⚙️ Model Configuration")
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                forecast_periods = st.slider("Forecast Periods (months)", 6, 36, 12, key="temp_periods")
            with mcol2:
                test_size = st.slider("Test Size (months)", 6, 24, 12, key="temp_test")

            if st.button("🚀 Run Temperature Forecast", type="primary", key="btn_temp"):
                with st.spinner("Training SARIMAX model…"):
                    result = forecast_sarimax(
                        series,
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 12),
                        forecast_steps=forecast_periods,
                        test_size=test_size,
                    )

                if result["train_metrics"]:
                    st.markdown("### 📊 Model Performance")
                    mcols = st.columns(4)
                    metrics = result["train_metrics"]
                    mcols[0].metric("MAE", f"{metrics.get('MAE', 'N/A')}°C")
                    mcols[1].metric("RMSE", f"{metrics.get('RMSE', 'N/A')}°C")
                    mcols[2].metric("R²", metrics.get("R²", "N/A"))
                    mcols[3].metric("MAPE", f"{metrics.get('MAPE (%)', 'N/A')}%")

                    st.plotly_chart(
                        plot_forecast(
                            series,
                            result["forecast"],
                            result["forecast_ci"],
                            result["test_actual"],
                            result["test_predicted"],
                            title=f"Temperature Forecast ({forecast_periods} months)",
                            y_label="Temperature (°C)",
                        ),
                        width='stretch',
                    )


# ═══════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "🌡️ Tamil Nadu Temperature × Electricity Dashboard | "
    "Data: Open-Meteo · World Bank · data.gov.in | "
    "Built with Streamlit + Plotly"
    "</div>",
    unsafe_allow_html=True,
)
