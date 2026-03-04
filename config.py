"""
Centralized configuration for the Tamil Nadu Temperature vs Electricity project.

Loads environment variables from .env and defines API endpoints,
data paths, and project-wide constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment variables ──────────────────────────────────────
load_dotenv()

# ── Project Paths ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Configuration ──────────────────────────────────────────────

# Open-Meteo — Historical Weather API (no API key required)
OPEN_METEO_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Chennai coordinates (representative of Tamil Nadu)
TAMIL_NADU_CITIES = {
    "Chennai": {"latitude": 13.0827, "longitude": 80.2707},
    "Coimbatore": {"latitude": 11.0168, "longitude": 76.9558},
    "Madurai": {"latitude": 9.9252, "longitude": 78.1198},
    "Tiruchirappalli": {"latitude": 10.7905, "longitude": 78.7047},
    "Salem": {"latitude": 11.6643, "longitude": 78.1460},
}
# Andhra Pradesh comparison city
AP_CITIES = {
    "Vijayawada": {"latitude": 16.5062, "longitude": 80.6480},
}

# World Bank — Indicators API (no API key required)
WORLD_BANK_BASE_URL = "https://api.worldbank.org/v2"
WORLD_BANK_GDP_INDICATOR = "NY.GDP.MKTP.CN"  # GDP in current LCU (INR)
WORLD_BANK_COUNTRY = "IN"

# data.gov.in — Indian Government Open Data API
DATA_GOV_IN_API_KEY = os.getenv("DATA_GOV_IN_API_KEY", "")
DATA_GOV_IN_BASE_URL = "https://api.data.gov.in/resource"

# ── Analysis Constants ──────────────────────────────────────────────
START_YEAR = 2015
END_YEAR = 2024
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# ── Streamlit Theme ────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_PALETTE = {
    "primary": "#FF6B6B",
    "secondary": "#4ECDC4",
    "accent": "#FFE66D",
    "bg_dark": "#1a1a2e",
    "bg_card": "#16213e",
    "text": "#e0e0e0",
    "gradient_warm": ["#FF6B6B", "#FF8E53", "#FFE66D"],
    "gradient_cool": ["#4ECDC4", "#44B3C2", "#2980B9"],
    "heatmap": "YlOrRd",
}
