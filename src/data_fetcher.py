"""
API-based data fetching for Tamil Nadu Temperature vs Electricity project.

Fetches data from three free APIs:
  - Open-Meteo: Historical daily temperature (no API key)
  - World Bank: India national GDP indicators (no API key)
  - data.gov.in: State electricity demand & GDP (free API key)

Each function returns a pandas DataFrame and caches results locally.
Falls back to CSV files in data/raw/ if API calls fail.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OPEN_METEO_BASE_URL,
    TAMIL_NADU_CITIES,
    AP_CITIES,
    WORLD_BANK_BASE_URL,
    WORLD_BANK_GDP_INDICATOR,
    WORLD_BANK_COUNTRY,
    DATA_GOV_IN_API_KEY,
    DATA_GOV_IN_BASE_URL,
    PROCESSED_DATA_DIR,
    START_YEAR,
    END_YEAR,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════

def _cache_path(name: str) -> Path:
    """Return path for a cached processed file."""
    return PROCESSED_DATA_DIR / f"{name}.csv"


def _load_cache(name: str) -> Optional[pd.DataFrame]:
    """Load cached DataFrame if it exists."""
    path = _cache_path(name)
    if path.exists():
        logger.info("Loading cached data: %s", path)
        return pd.read_csv(path)
    return None


def _save_cache(df: pd.DataFrame, name: str) -> None:
    """Save DataFrame to local cache."""
    path = _cache_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Cached data saved: %s", path)


# ═══════════════════════════════════════════════════════════════════
# 1. Open-Meteo — Historical Temperature
# ═══════════════════════════════════════════════════════════════════

def fetch_temperature_data(
    cities: Optional[dict] = None,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical daily maximum temperature from Open-Meteo API.

    Parameters
    ----------
    cities : dict, optional
        Dict of {city_name: {latitude, longitude}}. Defaults to TN cities.
    start_year, end_year : int
        Date range for historical data.
    use_cache : bool
        If True, return cached data when available.

    Returns
    -------
    pd.DataFrame
        Columns: Date, City, Temperature_Max_C, Temperature_Mean_C, Month, Year
    """
    cache_name = "temperature_api"
    if use_cache:
        cached = _load_cache(cache_name)
        if cached is not None:
            return cached

    if cities is None:
        cities = TAMIL_NADU_CITIES

    all_frames = []

    for city_name, coords in cities.items():
        logger.info("Fetching temperature for %s (%d–%d)…", city_name, start_year, end_year)
        params = {
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
            "daily": "temperature_2m_max,temperature_2m_mean",
            "timezone": "Asia/Kolkata",
        }

        try:
            resp = requests.get(OPEN_METEO_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            daily = data.get("daily", {})
            df = pd.DataFrame({
                "Date": pd.to_datetime(daily["time"]),
                "Temperature_Max_C": daily["temperature_2m_max"],
                "Temperature_Mean_C": daily["temperature_2m_mean"],
            })
            df["City"] = city_name
            all_frames.append(df)

        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            logger.warning("Open-Meteo API failed for %s: %s", city_name, e)

    if not all_frames:
        logger.error("No temperature data fetched from API.")
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    result["Month"] = result["Date"].dt.month
    result["Year"] = result["Date"].dt.year
    result["Month_Name"] = result["Date"].dt.strftime("%b")

    _save_cache(result, cache_name)
    return result


def fetch_ap_temperature_data(
    start_year: int = 2018,
    end_year: int = END_YEAR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch temperature for Andhra Pradesh (comparison state)."""
    cache_name = "ap_temperature_api"
    if use_cache:
        cached = _load_cache(cache_name)
        if cached is not None:
            return cached

    result = fetch_temperature_data(
        cities=AP_CITIES,
        start_year=start_year,
        end_year=end_year,
        use_cache=False,
    )

    if not result.empty:
        _save_cache(result, cache_name)
    return result


# ═══════════════════════════════════════════════════════════════════
# 2. World Bank — India GDP
# ═══════════════════════════════════════════════════════════════════

def fetch_national_gdp(
    start_year: int = 2011,
    end_year: int = END_YEAR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch India's national GDP from the World Bank Indicators API.

    Returns
    -------
    pd.DataFrame
        Columns: Year, GDP_Current_LCU (in Rs.)
    """
    cache_name = "national_gdp_api"
    if use_cache:
        cached = _load_cache(cache_name)
        if cached is not None:
            return cached

    url = (
        f"{WORLD_BANK_BASE_URL}/country/{WORLD_BANK_COUNTRY}"
        f"/indicator/{WORLD_BANK_GDP_INDICATOR}"
    )
    params = {
        "date": f"{start_year}:{end_year}",
        "format": "json",
        "per_page": 50,
    }

    try:
        logger.info("Fetching India GDP from World Bank API…")
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        if len(payload) < 2 or not payload[1]:
            logger.warning("World Bank API returned empty data.")
            return pd.DataFrame()

        records = [
            {"Year": int(item["date"]), "GDP_Current_LCU": float(item["value"])}
            for item in payload[1]
            if item["value"] is not None
        ]
        result = pd.DataFrame(records).sort_values("Year").reset_index(drop=True)

        # Convert to Lakh Crore for consistency with local datasets
        result["GDP_Lakh_Crore"] = result["GDP_Current_LCU"] / 1e12

        _save_cache(result, cache_name)
        return result

    except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
        logger.warning("World Bank API failed: %s", e)
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════
# 3. data.gov.in — State Electricity & GDP
# ═══════════════════════════════════════════════════════════════════

def fetch_electricity_demand(use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch monthly peak electricity demand for Tamil Nadu from data.gov.in.

    Requires DATA_GOV_IN_API_KEY in .env.
    Resource: Monthly Peak Demand for Power (state-wise).
    """
    cache_name = "electricity_demand_api"
    if use_cache:
        cached = _load_cache(cache_name)
        if cached is not None:
            return cached

    if not DATA_GOV_IN_API_KEY or DATA_GOV_IN_API_KEY == "your_api_key_here":
        logger.warning(
            "data.gov.in API key not set. Set DATA_GOV_IN_API_KEY in .env. "
            "Falling back to local CSV."
        )
        return pd.DataFrame()

    # Resource ID for Monthly Peak Demand — Tamil Nadu
    resource_id = "2a7aaed5-0e88-4b85-bdf2-c6c7d2809ef5"
    url = f"{DATA_GOV_IN_BASE_URL}/{resource_id}"

    params = {
        "api-key": DATA_GOV_IN_API_KEY,
        "format": "json",
        "limit": 500,
        "filters[state]": "Tamil Nadu",
    }

    try:
        logger.info("Fetching electricity demand from data.gov.in…")
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        records = data.get("records", [])
        if not records:
            logger.warning("data.gov.in returned no records for electricity demand.")
            return pd.DataFrame()

        result = pd.DataFrame(records)
        _save_cache(result, cache_name)
        return result

    except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
        logger.warning("data.gov.in API failed for electricity demand: %s", e)
        return pd.DataFrame()


def fetch_state_gdp(use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch state-wise GDP data from data.gov.in.

    Returns DataFrame with columns for state, year, and GDP value.
    """
    cache_name = "state_gdp_api"
    if use_cache:
        cached = _load_cache(cache_name)
        if cached is not None:
            return cached

    if not DATA_GOV_IN_API_KEY or DATA_GOV_IN_API_KEY == "your_api_key_here":
        logger.warning(
            "data.gov.in API key not set. Falling back to local CSV."
        )
        return pd.DataFrame()

    # Resource ID for State-wise GDP
    resource_id = "f23b5c02-4248-4ed8-87fe-2c3b7f1e7acf"
    url = f"{DATA_GOV_IN_BASE_URL}/{resource_id}"

    params = {
        "api-key": DATA_GOV_IN_API_KEY,
        "format": "json",
        "limit": 1000,
    }

    try:
        logger.info("Fetching state GDP from data.gov.in…")
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        records = data.get("records", [])
        if not records:
            logger.warning("data.gov.in returned no records for state GDP.")
            return pd.DataFrame()

        result = pd.DataFrame(records)
        _save_cache(result, cache_name)
        return result

    except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
        logger.warning("data.gov.in API failed for state GDP: %s", e)
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════
# Convenience: Fetch All
# ═══════════════════════════════════════════════════════════════════

def fetch_all_data() -> dict:
    """
    Fetch all datasets from APIs. Returns a dict of DataFrames.
    Empty DataFrames indicate API failure (caller should fall back to CSV).
    """
    return {
        "temperature_tn": fetch_temperature_data(),
        "temperature_ap": fetch_ap_temperature_data(),
        "national_gdp": fetch_national_gdp(),
        "electricity_demand": fetch_electricity_demand(),
        "state_gdp": fetch_state_gdp(),
    }
