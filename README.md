<div align="center">

# 🌡️ Tamil Nadu: Temperature × Electricity ⚡

**Analyzing the relationship between rising temperatures and electricity demand in Tamil Nadu using Data Science, Machine Learning, and Live API data.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 Overview

This project investigates how **temperature patterns** correlate with **electricity consumption** in Tamil Nadu, India (2015–2024). Using a combination of live API data, statistical analysis, and machine learning, we uncover actionable insights about energy planning for one of India's most industrialized states.

### 🔑 Key Findings

| Insight | Detail |
|---------|--------|
| 🌡️ **Peak summer** temperatures (Apr–Jun) directly correlate with electricity demand spikes |  |
| ⚡ **Electricity demand grew significantly** from 2015 to 2024 across all months |  |
| 📉 **TANGEDCO** has been operating at financial losses consistently |  |
| 🔮 **SARIMAX forecasting** captures seasonal demand patterns with quantified accuracy (MAE, RMSE, R²) |  |
| 🏭 **Thermal power** dominates Tamil Nadu's energy mix, but renewable share is growing |  |

---

## ✨ Features

- 🌐 **Live API Data Fetching** — Temperature (Open-Meteo), GDP (World Bank), Electricity (data.gov.in)  
- 📁 **Smart Fallback** — Gracefully falls back to bundled CSV data if APIs are unavailable  
- 📊 **Interactive Dashboard** — Streamlit + Plotly web dashboard with **8 analysis pages**  
- 🔍 **Exploratory Data Analysis** — Data profiling, distribution analysis, box plots, skewness/kurtosis  
- 🧪 **Advanced Statistical Tests** — Granger Causality, Mann-Kendall trend, ADF stationarity  
- 🤖 **Multi-Model ML Comparison** — Linear, Polynomial (deg 2,3), Random Forest with ranked leaderboard  
- 🔮 **Time Series Forecasting** — SARIMAX with train/test, confidence intervals, downloadable CSVs  
- 📈 **Correlation Analysis** — Pearson/Spearman, rolling correlation, causal relationship testing  
- 🔎 **Anomaly Detection** — Modified Z-score based outlier identification  
- 🧠 **Auto-Insight Engine** — Automatically discovers and articulates key data findings  
- 🌡️ **Seasonal Decomposition** — Trend/seasonal/residual breakdown for time series  
- 💰 **Economic Analysis** — GDP trends, state comparisons, TANGEDCO financials  

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9+ |
| **Data** | Pandas, NumPy, Requests |
| **ML / Statistics** | Scikit-learn, Statsmodels, SciPy |
| **Visualization** | Plotly, Seaborn, Matplotlib |
| **Dashboard** | Streamlit |
| **APIs** | Open-Meteo, World Bank, data.gov.in |
| **Config** | python-dotenv |

---

## 🏗️ Project Architecture

```
├── app.py                    # 8-page Streamlit dashboard
├── config.py                 # API endpoints, paths, constants
├── requirements.txt          # Dependencies
├── .env.example              # API key template
│
├── src/
│   ├── data_fetcher.py       # API clients (Open-Meteo, World Bank, data.gov.in)
│   ├── data_loader.py        # API-first data loading with CSV fallback
│   ├── analysis.py           # Correlation, standardization, seasonal analysis
│   ├── advanced_analysis.py  # Granger causality, ADF, Mann-Kendall, anomalies
│   ├── models.py             # SARIMAX, Linear, Polynomial, Random Forest
│   └── visualizations.py     # 25+ reusable Plotly chart functions
│
├── data/
│   ├── raw/                  # Bundled CSV/XLSX datasets (fallback)
│   │   ├── temperature/      # TN & AP temperature data
│   │   ├── electricity/      # Demand, generation, consumption
│   │   ├── economic/         # GDP, P&L, revenue, tariffs
│   │   └── demographic/      # Population statistics
│   └── processed/            # API-fetched cached data
│
├── notebooks/                # Jupyter analysis notebooks
└── outputs/                  # Generated charts
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Gowreesh31/Tamilnadu-Temperature-vs-Electricity.git
cd Tamilnadu-Temperature-vs-Electricity

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys (optional — CSV fallback works without this)
cp .env.example .env
# Edit .env and add your free data.gov.in API key
```

### Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at **http://localhost:8501** 🎉

---

## 🌐 API Integration

| API | Data | Auth | Docs |
|-----|------|------|------|
| **Open-Meteo** | Historical daily temperature for TN cities | ❌ No key needed | [open-meteo.com](https://open-meteo.com/en/docs/historical-weather-api) |
| **World Bank** | India national GDP indicators | ❌ No key needed | [api.worldbank.org](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392) |
| **data.gov.in** | State electricity demand, state GDP | ✅ Free API key | [data.gov.in](https://data.gov.in) |

> **Note**: The project works without any API keys — it gracefully falls back to bundled CSV datasets with a warning message.

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **🏠 Overview** | KPI cards, auto-generated insights, dataset summary, analytics pipeline |
| **🔍 EDA** | Data profiling, distribution analysis (histogram+KDE+box), anomaly detection |
| **🌡️ Temperature** | Heatmaps, trends, seasonal decomposition, Mann-Kendall test, state comparison |
| **⚡ Electricity** | Demand trends, heatmaps, production mix, seasonal decomposition |
| **📊 Correlation** | Standardized overlay, scatter regression, **Granger causality**, rolling correlation |
| **💰 Economics** | State/national GDP, multi-state comparison, TANGEDCO P&L |
| **🤖 ML Models** | Multi-model leaderboard, Random Forest deep-dive, residual analysis |
| **🔮 Predictions** | SARIMAX forecasting with configurable params, downloadable forecast CSVs |

---

## 🧪 Statistical Methodology

| Test | Purpose | Implementation |
|------|---------|---------------|
| **Granger Causality** | Does temperature *cause* demand? | `statsmodels.grangercausalitytests` |
| **Mann-Kendall** | Non-parametric monotonic trend detection | Custom implementation |
| **ADF (Augmented Dickey-Fuller)** | Time series stationarity test | `statsmodels.adfuller` |
| **Pearson / Spearman** | Linear / rank correlation | `scipy.stats` |
| **Modified Z-Score** | Robust anomaly detection | MAD-based outlier identification |
| **Rolling Correlation** | Time-varying relationship strength | Windowed Pearson r |

---

## 🤖 Machine Learning

### Model Comparison Leaderboard

All models are trained and ranked automatically:

| Model | Type | Features |
|-------|------|----------|
| **Linear Regression** | Parametric | Temperature |
| **Polynomial (Degree 2)** | Non-linear | Temperature (quadratic) |
| **Polynomial (Degree 3)** | Non-linear | Temperature (cubic) |
| **Random Forest** | Ensemble | Temperature + Month (5-fold CV) |
| **SARIMAX** | Time Series | Lag features + seasonality |

### Evaluation Metrics
- **R²**, **MAE**, **RMSE**, **MAPE** for all models
- **5-fold cross-validation** for Random Forest
- **AIC/BIC** for SARIMAX model comparison
- **Residual analysis**: Residuals vs Fitted, Q-Q plot, distribution check

---

## 📂 Data Sources

| Dataset | Source | Period |
|---------|--------|--------|
| Temperature (TN) | Open-Meteo API / IMD | 2015–2024 |
| Temperature (AP) | Open-Meteo API / IMD | 2018–2023 |
| Electricity Demand | data.gov.in / CEA | 2015–2024 |
| Power Generation | State Generation Reports | 2015–2024 |
| GDP (National) | World Bank API | 2011–2024 |
| GDP (State) | data.gov.in / RBI | 2011–2024 |
| TANGEDCO Financials | Annual Reports | 2015–2024 |

---

## 🔮 Future Improvements

- [ ] Add deep learning models (LSTM / Prophet) for comparison
- [ ] Real-time temperature monitoring with streaming data
- [ ] District-level granular analysis
- [ ] Renewable energy transition impact study
- [ ] Deploy on Streamlit Cloud for public access
- [ ] Automated PDF report generation

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <b>Made with ❤️ for Tamil Nadu</b><br>
  <sub>If you found this useful, please ⭐ the repository!</sub>
</div>