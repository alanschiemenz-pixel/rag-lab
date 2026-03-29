"""
Weather data layer — historical and 14-day forecast.

Uses Open-Meteo (free, no API key required).
Responses are cached to .weather_cache.sqlite (1-hour TTL) to stay within
the free tier rate limits.

Falls back to synthetic mock data on any network/API error.
"""
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# ── Open-Meteo client (cached + retry) ───────────────────────────────────────

def _make_client():
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    session = requests_cache.CachedSession(
        ".weather_cache", expire_after=3600
    )
    session = retry(session, retries=3, backoff_factor=0.5)
    return openmeteo_requests.Client(session=session)


# ── Mock data ─────────────────────────────────────────────────────────────────

def _mock_weather(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """
    Generate synthetic hourly weather that has a plausible daily/seasonal pattern.
    Seeded from coordinates + start date for reproducibility.
    """
    seed = hash(f"{lat:.2f}{lon:.2f}{start}") % (2**31)
    rng = np.random.default_rng(seed)

    hours = pd.date_range(start, end, freq="h", inclusive="left")
    n = len(hours)
    doy = hours.day_of_year.values
    h = hours.hour.values

    # Temperature: seasonal base + daily swing + noise (Fahrenheit)
    seasonal_base = 60 + 25 * np.sin((doy - 80) * 2 * np.pi / 365)
    daily_swing = 10 * np.sin((h - 6) * np.pi / 12)
    temp_f = seasonal_base + daily_swing + rng.normal(0, 3, n)

    # Wind speed (mph)
    wind = np.abs(rng.normal(10, 5, n))

    # Precipitation (mm/hr) — mostly 0, occasional events
    precip = np.where(rng.random(n) < 0.05, rng.exponential(2, n), 0.0)

    # Cloud cover (%)
    cloud = np.clip(rng.normal(40, 30, n), 0, 100)

    return pd.DataFrame({
        "time":            hours,
        "temperature_2m":  temp_f.round(1),
        "wind_speed_10m":  wind.round(1),
        "precipitation":   precip.round(2),
        "cloud_cover":     cloud.round(0),
        "source":          "mock",
    })


# ── Real data ─────────────────────────────────────────────────────────────────

def _parse_openmeteo(response, start: date, end: date) -> pd.DataFrame:
    """Convert an Open-Meteo response object to a tidy DataFrame."""
    hourly = response.Hourly()
    # Derive timestamps from the response itself — avoids length mismatches
    # caused by DST transitions or API rounding of the requested date range.
    hours = pd.date_range(
        start=pd.to_datetime(hourly.Time(),    unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(),   unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    ).tz_localize(None)  # drop UTC tz so downstream merges stay tz-naive
    # Variable order must match the request's hourly list
    return pd.DataFrame({
        "time":           hours,
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy().round(1),
        "wind_speed_10m": hourly.Variables(1).ValuesAsNumpy().round(1),
        "precipitation":  hourly.Variables(2).ValuesAsNumpy().round(2),
        "cloud_cover":    hourly.Variables(3).ValuesAsNumpy().round(0),
        "source":         "real",
    })


# ── Public API ────────────────────────────────────────────────────────────────

def get_historical_weather(lat: float, lon: float,
                           start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly historical weather from Open-Meteo archive API.

    Returns a DataFrame with columns:
      time, temperature_2m (°F), wind_speed_10m (mph), precipitation (mm),
      cloud_cover (%), source
    """
    start = date.fromisoformat(start_date)
    end   = date.fromisoformat(end_date)

    if config.USE_MOCK_DATA:
        return _mock_weather(lat, lon, start, end)

    try:
        client = _make_client()
        params = {
            "latitude":        lat,
            "longitude":       lon,
            "start_date":      start_date,
            "end_date":        end_date,
            "hourly":          ["temperature_2m", "wind_speed_10m",
                                "precipitation", "cloud_cover"],
            "temperature_unit": "fahrenheit",
            "wind_speed_unit":  "mph",
            "timezone":         "America/Chicago",
        }
        responses = client.weather_api(
            "https://archive-api.open-meteo.com/v1/archive", params=params
        )
        df = _parse_openmeteo(responses[0], start, end)
        logger.info(f"[Weather] Historical data fetched: {len(df)} rows")
        return df
    except Exception as exc:
        logger.warning(f"[Weather] Historical fetch failed ({exc}). Using mock.")
        return _mock_weather(lat, lon, start, end)


def get_forecast_weather(lat: float, lon: float, forecast_days: int = 14) -> pd.DataFrame:
    """
    Fetch hourly weather forecast from Open-Meteo (up to 14 days ahead).

    Returns a DataFrame with columns:
      time, temperature_2m (°F), wind_speed_10m (mph), precipitation (mm),
      cloud_cover (%), source
    """
    forecast_days = min(forecast_days, 14)
    start = date.today()
    end   = start + timedelta(days=forecast_days)

    if config.USE_MOCK_DATA:
        return _mock_weather(lat, lon, start, end)

    try:
        client = _make_client()
        params = {
            "latitude":         lat,
            "longitude":        lon,
            "forecast_days":    forecast_days,
            "hourly":           ["temperature_2m", "wind_speed_10m",
                                 "precipitation", "cloud_cover"],
            "temperature_unit": "fahrenheit",
            "wind_speed_unit":  "mph",
            "timezone":         "America/Chicago",
        }
        responses = client.weather_api(
            "https://api.open-meteo.com/v1/forecast", params=params
        )
        df = _parse_openmeteo(responses[0], start, end)
        logger.info(f"[Weather] Forecast data fetched: {len(df)} rows")
        return df
    except Exception as exc:
        logger.warning(f"[Weather] Forecast fetch failed ({exc}). Using mock.")
        return _mock_weather(lat, lon, start, end)


def weather_for_iso(iso: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Convenience wrapper: look up coordinates by ISO name."""
    region = config.ISO_REGIONS[iso.upper()]
    return get_historical_weather(region["lat"], region["lon"], start_date, end_date)


def forecast_for_iso(iso: str, forecast_days: int = 14) -> pd.DataFrame:
    """Convenience wrapper: look up coordinates by ISO name."""
    region = config.ISO_REGIONS[iso.upper()]
    return get_forecast_weather(region["lat"], region["lon"], forecast_days)


def weather_for_hub(iso: str, hub: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical weather for a specific pricing hub's location."""
    hubs = config.ISO_HUB_LOCATIONS.get(iso.upper(), {})
    if hub not in hubs:
        # Fall back to ISO centroid if hub coordinates aren't defined
        return weather_for_iso(iso, start_date, end_date)
    loc = hubs[hub]
    return get_historical_weather(loc["lat"], loc["lon"], start_date, end_date)


def forecast_for_hub(iso: str, hub: str, forecast_days: int = 14) -> pd.DataFrame:
    """Fetch weather forecast for a specific pricing hub's location."""
    hubs = config.ISO_HUB_LOCATIONS.get(iso.upper(), {})
    if hub not in hubs:
        return forecast_for_iso(iso, forecast_days)
    loc = hubs[hub]
    return get_forecast_weather(loc["lat"], loc["lon"], forecast_days)
