"""
Data loading module with API-first, CSV-fallback strategy.

Provides the DataLoader class that:
  1. Attempts to fetch fresh data from APIs (via data_fetcher)
  2. Falls back to local CSV/XLSX files in data/raw/ if APIs fail
  3. Cleans and normalizes all datasets into consistent DataFrames
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR, MONTHS, START_YEAR, END_YEAR
from src.data_fetcher import (
    fetch_temperature_data,
    fetch_ap_temperature_data,
    fetch_national_gdp,
    fetch_electricity_demand,
    fetch_state_gdp,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified data loader for all project datasets.

    Usage
    -----
    >>> loader = DataLoader()
    >>> temp_df = loader.get_temperature()
    >>> demand_df = loader.get_electricity_demand()
    """

    def __init__(self, prefer_api: bool = True):
        """
        Parameters
        ----------
        prefer_api : bool
            If True, try API first; if False, load directly from CSV.
        """
        self.prefer_api = prefer_api
        self._raw = RAW_DATA_DIR
        self._cache: dict[str, pd.DataFrame] = {}

    # ── Temperature ──────────────────────────────────────────────

    def get_temperature(self) -> pd.DataFrame:
        """
        Get monthly average max temperature for Tamil Nadu cities.

        Returns DataFrame with: Year, Month, Month_Name, Temperature
        (monthly average of daily max across TN cities).
        """
        if "temperature" in self._cache:
            return self._cache["temperature"]

        df = pd.DataFrame()

        # Try API
        if self.prefer_api:
            api_df = fetch_temperature_data()
            if not api_df.empty:
                # Aggregate: monthly avg of daily max across all TN cities
                monthly = (
                    api_df.groupby(["Year", "Month", "Month_Name"])
                    ["Temperature_Max_C"]
                    .mean()
                    .reset_index()
                    .rename(columns={"Temperature_Max_C": "Temperature"})
                )
                monthly = monthly.sort_values(["Year", "Month"]).reset_index(drop=True)
                df = monthly

        # Fallback to CSV
        if df.empty:
            logger.info("Falling back to CSV for temperature data.")
            csv_path = self._raw / "temperature" / "temperature.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                if "Month" in df.columns and "Year" in df.columns:
                    # Add Month_Name if missing
                    if "Month_Name" not in df.columns:
                        month_map = dict(enumerate(MONTHS, 1))
                        df["Month_Name"] = df["Month"].map(month_map) if df["Month"].dtype != object else df["Month"]

        self._cache["temperature"] = df
        return df

    def get_ap_temperature(self) -> pd.DataFrame:
        """Get monthly temperature for Andhra Pradesh (comparison)."""
        if "ap_temperature" in self._cache:
            return self._cache["ap_temperature"]

        df = pd.DataFrame()

        if self.prefer_api:
            api_df = fetch_ap_temperature_data()
            if not api_df.empty:
                monthly = (
                    api_df.groupby(["Year", "Month", "Month_Name"])
                    ["Temperature_Max_C"]
                    .mean()
                    .reset_index()
                    .rename(columns={"Temperature_Max_C": "Temperature"})
                )
                df = monthly.sort_values(["Year", "Month"]).reset_index(drop=True)

        # Fallback: load individual AP CSVs
        if df.empty:
            logger.info("Falling back to CSV for AP temperature data.")
            frames = []
            for year in range(2018, 2024):
                csv_path = self._raw / "temperature" / f"AP{year}.csv"
                if csv_path.exists():
                    frames.append(pd.read_csv(csv_path))
            if frames:
                raw = pd.concat(frames, ignore_index=True)
                if "Maximum Temprature (in C)" in raw.columns:
                    raw = raw.rename(columns={"Maximum Temprature (in C)": "Temperature"})
                df = raw

        self._cache["ap_temperature"] = df
        return df

    # ── Electricity ──────────────────────────────────────────────

    def get_electricity_demand(self) -> pd.DataFrame:
        """
        Get monthly peak electricity demand (MW) for Tamil Nadu.

        Returns DataFrame with: Year, Month, Peak Demand (in MW)
        """
        if "electricity_demand" in self._cache:
            return self._cache["electricity_demand"]

        df = pd.DataFrame()

        if self.prefer_api:
            api_df = fetch_electricity_demand()
            if not api_df.empty:
                df = api_df

        if df.empty:
            logger.info("Falling back to CSV for electricity demand.")
            csv_path = self._raw / "electricity" / "Monthly_Peak_Demand_Sheet.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)

        self._cache["electricity_demand"] = df
        return df

    def get_generation_capacity(self) -> pd.DataFrame:
        """Get source-wise electricity generation data."""
        if "generation" in self._cache:
            return self._cache["generation"]

        csv_path = self._raw / "electricity" / "Geo_Cap_Gen_Sheet.csv"
        df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        self._cache["generation"] = df
        return df

    def get_consumption_share(self) -> pd.DataFrame:
        """Get sector-wise electricity consumption share."""
        if "consumption" in self._cache:
            return self._cache["consumption"]

        csv_path = self._raw / "electricity" / "Cat_Elect_Cons_Share_Sheet.csv"
        df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        self._cache["consumption"] = df
        return df

    # ── Economic ─────────────────────────────────────────────────

    def get_national_gdp(self) -> pd.DataFrame:
        """
        Get India national GDP. Tries World Bank API first.
        """
        if "national_gdp" in self._cache:
            return self._cache["national_gdp"]

        df = pd.DataFrame()

        if self.prefer_api:
            api_df = fetch_national_gdp()
            if not api_df.empty:
                df = api_df

        if df.empty:
            logger.info("Falling back to CSV for national GDP.")
            csv_path = self._raw / "economic" / "National_GDP.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)

        self._cache["national_gdp"] = df
        return df

    def get_state_gdp(self) -> pd.DataFrame:
        """Get Tamil Nadu state GDP."""
        if "state_gdp" in self._cache:
            return self._cache["state_gdp"]

        df = pd.DataFrame()

        if self.prefer_api:
            api_df = fetch_state_gdp()
            if not api_df.empty:
                df = api_df

        if df.empty:
            logger.info("Falling back to CSV for state GDP.")
            csv_path = self._raw / "economic" / "State_GDP.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)

        self._cache["state_gdp"] = df
        return df

    def get_state_gdp_comparison(self) -> pd.DataFrame:
        """Get multi-state GDP comparison data."""
        if "gdp_comparison" in self._cache:
            return self._cache["gdp_comparison"]

        csv_path = self._raw / "economic" / "StateGDPMapIndia.csv"
        df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        self._cache["gdp_comparison"] = df
        return df

    def get_profit_loss(self) -> pd.DataFrame:
        """Get TANGEDCO profit and loss data."""
        if "profit_loss" in self._cache:
            return self._cache["profit_loss"]

        csv_path = self._raw / "economic" / "Profit_and_Loss_Sheet.csv"
        df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        self._cache["profit_loss"] = df
        return df

    def get_revenue(self) -> pd.DataFrame:
        """Get average cost and revenue data."""
        if "revenue" in self._cache:
            return self._cache["revenue"]

        csv_path = self._raw / "economic" / "Average_Cost_&_Revenue_Sheet.csv"
        df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        self._cache["revenue"] = df
        return df

    def get_tariff_rates(self) -> pd.DataFrame:
        """Get electricity tariff rates."""
        if "tariff" in self._cache:
            return self._cache["tariff"]

        csv_path = self._raw / "economic" / "Tariff_Rates_Sheet.csv"
        df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        self._cache["tariff"] = df
        return df

    # ── Demographic ──────────────────────────────────────────────

    def get_population(self) -> pd.DataFrame:
        """Get population data for Tamil Nadu."""
        if "population" in self._cache:
            return self._cache["population"]

        csv_path = self._raw / "demographic" / "Population_sheet.csv"
        df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        self._cache["population"] = df
        return df

    # ── GSDP & Sectoral ──────────────────────────────────────────

    def get_gsdp_gva(self) -> pd.DataFrame:
        """Get GSDP and GVA data."""
        if "gsdp_gva" in self._cache:
            return self._cache["gsdp_gva"]

        csv_path = self._raw / "economic" / "GSDP_GVA_Sheet.csv"
        df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        self._cache["gsdp_gva"] = df
        return df

    def get_sectoral_gva(self) -> pd.DataFrame:
        """Get sectoral GVA data."""
        if "sectoral_gva" in self._cache:
            return self._cache["sectoral_gva"]

        csv_path = self._raw / "economic" / "Sectoral_GVA_Sheet.csv"
        df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        self._cache["sectoral_gva"] = df
        return df
