"""
LangChain tools exposed to the LangGraph agent.

Each tool returns a formatted string so the LLM can interpret and relay results
to the user. A [DATA_SOURCE: mock|real] tag is embedded so the agent can
transparently disclose data provenance.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
from langchain_core.tools import tool

import config
from data import lmp as lmp_data
from data import weather as wx_data


# ── Helpers ───────────────────────────────────────────────────────────────────

def _source_tag(df: pd.DataFrame) -> str:
    src = df["source"].iloc[0] if "source" in df.columns else "unknown"
    return f"[DATA_SOURCE: {src}]"


def _lmp_summary(df: pd.DataFrame, iso: str) -> str:
    stats = df.groupby("Location")["LMP"].agg(["mean", "min", "max", "std"]).round(2)
    lines = [f"  {row.name}: mean=${row['mean']}/MWh  min=${row['min']}  "
             f"max=${row['max']}  std=${row['std']}"
             for _, row in stats.iterrows()]
    return "\n".join(lines)


def _date_range_30d() -> tuple[str, str]:
    end   = date.today() - timedelta(days=1)
    start = end - timedelta(days=30)
    return start.isoformat(), end.isoformat()


def _fit_lmp_model(merged: pd.DataFrame):
    """
    Fit a linear regression: LMP ~ temp + temp^2 + hour_sin + hour_cos + month_sin + month_cos
    Returns (coefficients, feature_builder_fn).
    """
    t    = merged["temperature_2m"].values
    h    = merged["Time"].dt.hour.values
    mon  = merged["Time"].dt.month.values
    y    = merged["LMP"].values

    X = np.column_stack([
        np.ones(len(t)),
        t,
        t ** 2,
        np.sin(2 * np.pi * h / 24),
        np.cos(2 * np.pi * h / 24),
        np.sin(2 * np.pi * mon / 12),
        np.cos(2 * np.pi * mon / 12),
    ])

    # Least-squares fit via numpy
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _predict_lmp(coef, temp: np.ndarray, hours) -> np.ndarray:
    hours = pd.DatetimeIndex(hours)
    h   = hours.hour
    mon = hours.month
    X = np.column_stack([
        np.ones(len(temp)),
        temp,
        temp ** 2,
        np.sin(2 * np.pi * h / 24),
        np.cos(2 * np.pi * h / 24),
        np.sin(2 * np.pi * mon / 12),
        np.cos(2 * np.pi * mon / 12),
    ])
    return X @ coef


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def get_lmp_data(iso: str, start_date: str, end_date: str) -> str:
    """
    Fetch hourly LMP (Locational Marginal Price) data for the given ISO market.

    Args:
        iso: Energy market — ERCOT, PJM, or SPP
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns a summary of LMP statistics by pricing node.
    """
    try:
        df = lmp_data.get_lmp(iso, start_date, end_date)
        tag = _source_tag(df)
        summary = _lmp_summary(df, iso)
        overall_mean = df["LMP"].mean().round(2)
        overall_max  = df["LMP"].max().round(2)
        spike_pct    = (df["LMP"] > 100).mean() * 100

        return (
            f"{tag}\n"
            f"ISO: {iso} | Period: {start_date} to {end_date}\n"
            f"Overall mean LMP: ${overall_mean}/MWh\n"
            f"Overall peak LMP: ${overall_max}/MWh\n"
            f"Hours with LMP > $100/MWh: {spike_pct:.1f}%\n\n"
            f"By node:\n{summary}"
        )
    except Exception as e:
        return f"Error fetching LMP data for {iso}: {e}"


@tool
def get_historical_weather(iso: str, start_date: str, end_date: str) -> str:
    """
    Fetch historical hourly weather for the region corresponding to an ISO market.

    Args:
        iso: Energy market — ERCOT, PJM, or SPP
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns daily weather averages (temperature, wind, precipitation).
    """
    try:
        df = wx_data.weather_for_iso(iso, start_date, end_date)
        tag = _source_tag(df)
        region = config.ISO_REGIONS[iso.upper()]

        daily = df.groupby(df["time"].dt.date).agg(
            temp_avg=("temperature_2m", "mean"),
            temp_max=("temperature_2m", "max"),
            temp_min=("temperature_2m", "min"),
            wind_max=("wind_speed_10m", "max"),
            precip_total=("precipitation", "sum"),
        ).round(1)

        lines = [
            f"  {d}: avg {r.temp_avg}°F (min {r.temp_min} / max {r.temp_max}) | "
            f"wind peak {r.wind_max} mph | precip {r.precip_total} mm"
            for d, r in daily.iterrows()
        ][:14]  # cap to 14 lines for readability

        return (
            f"{tag}\n"
            f"ISO: {iso} | Location: {region['label']}\n"
            f"Period: {start_date} to {end_date}\n\n"
            + "\n".join(lines)
        )
    except Exception as e:
        return f"Error fetching weather for {iso}: {e}"


@tool
def correlate_lmp_weather(iso: str, lookback_days: int = 30) -> str:
    """
    Compute the correlation between LMP prices and weather variables for the given ISO.

    Args:
        iso: Energy market — ERCOT, PJM, or SPP
        lookback_days: Number of past days to analyze (max 90)

    Returns Pearson correlation coefficients and an interpretation.
    """
    lookback_days = min(lookback_days, 90)
    start, end = _date_range_30d()
    # Override with caller-specified window
    end_dt   = date.today() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=lookback_days)
    start, end = start_dt.isoformat(), end_dt.isoformat()

    try:
        lmp_df = lmp_data.get_lmp(iso, start, end)
        wx_df  = wx_data.weather_for_iso(iso, start, end)

        # Use first node only for correlation
        node_df = lmp_df[lmp_df["Location"] == lmp_df["Location"].iloc[0]].copy()
        node_df["time"] = node_df["Time"].dt.floor("h")
        wx_df["time"]   = wx_df["time"].dt.floor("h")

        merged = node_df.merge(wx_df, on="time", how="inner")
        if len(merged) < 48:
            return f"Insufficient overlapping data for correlation ({len(merged)} hours)."

        tag = _source_tag(lmp_df)
        corr_temp  = merged["LMP"].corr(merged["temperature_2m"])
        corr_wind  = merged["LMP"].corr(merged["wind_speed_10m"])
        corr_precip = merged["LMP"].corr(merged["precipitation"])

        def interp(r):
            if abs(r) > 0.7:  return "strong"
            if abs(r) > 0.4:  return "moderate"
            if abs(r) > 0.2:  return "weak"
            return "negligible"

        return (
            f"{tag}\n"
            f"ISO: {iso} | Lookback: {lookback_days} days | Node: {lmp_df['Location'].iloc[0]}\n\n"
            f"Correlations with LMP (Pearson r):\n"
            f"  Temperature:   r = {corr_temp:+.3f}  → {interp(corr_temp)} {'positive' if corr_temp > 0 else 'negative'} relationship\n"
            f"  Wind speed:    r = {corr_wind:+.3f}  → {interp(corr_wind)} {'positive' if corr_wind > 0 else 'negative'} relationship\n"
            f"  Precipitation: r = {corr_precip:+.3f}  → {interp(corr_precip)} {'positive' if corr_precip > 0 else 'negative'} relationship\n\n"
            f"Interpretation: In {iso}, temperature {'drives' if abs(corr_temp) > 0.4 else 'has limited influence on'} "
            f"LMP pricing. {'High temps likely indicate cooling demand spikes.' if corr_temp > 0.4 else ''}"
            f"{'Wind generation appears to suppress prices.' if corr_wind < -0.3 else ''}"
        )
    except Exception as e:
        return f"Error computing correlation for {iso}: {e}"


@tool
def get_weather_forecast(iso: str, forecast_days: int = 7) -> str:
    """
    Fetch the weather forecast for the region corresponding to an ISO market.

    Args:
        iso: Energy market — ERCOT, PJM, or SPP
        forecast_days: Number of days to forecast (max 14)

    Returns a daily forecast summary.
    """
    forecast_days = min(forecast_days, 14)
    try:
        df = wx_data.forecast_for_iso(iso, forecast_days)
        tag = _source_tag(df)
        region = config.ISO_REGIONS[iso.upper()]

        daily = df.groupby(df["time"].dt.date).agg(
            temp_high=("temperature_2m", "max"),
            temp_low=("temperature_2m", "min"),
            wind_peak=("wind_speed_10m", "max"),
            precip_mm=("precipitation", "sum"),
        ).round(1)

        lines = [
            f"  {d}: high {r.temp_high}°F / low {r.temp_low}°F | "
            f"wind peak {r.wind_peak} mph | precip {r.precip_mm} mm"
            for d, r in daily.iterrows()
        ]

        return (
            f"{tag}\n"
            f"ISO: {iso} | Location: {region['label']}\n"
            f"{forecast_days}-Day Forecast:\n\n"
            + "\n".join(lines)
        )
    except Exception as e:
        return f"Error fetching forecast for {iso}: {e}"


@tool
def forecast_lmp(iso: str, horizon_days: int = 7) -> str:
    """
    Forecast LMP prices for the given ISO based on weather forecasts and
    historical LMP-weather correlations.

    Args:
        iso: Energy market — ERCOT, PJM, or SPP
        horizon_days: Number of days to forecast (max 14)

    Returns predicted daily average and peak LMP with uncertainty note.
    """
    horizon_days = min(horizon_days, 14)
    lookback = 45

    end_hist   = (date.today() - timedelta(days=1)).isoformat()
    start_hist = (date.today() - timedelta(days=lookback + 1)).isoformat()

    try:
        # 1. Gather historical LMP + weather for model fitting
        lmp_df = lmp_data.get_lmp(iso, start_hist, end_hist)
        wx_hist = wx_data.weather_for_iso(iso, start_hist, end_hist)

        node_df = lmp_df[lmp_df["Location"] == lmp_df["Location"].iloc[0]].copy()
        node_df["time"] = node_df["Time"].dt.floor("h")
        wx_hist["time"] = wx_hist["time"].dt.floor("h")
        merged = node_df.merge(wx_hist, on="time", how="inner")

        if len(merged) < 48:
            return "Insufficient historical data to build forecast model."

        # 2. Fit model
        coef = _fit_lmp_model(merged)

        # 3. Get weather forecast
        wx_fcast = wx_data.forecast_for_iso(iso, horizon_days)

        # 4. Predict
        pred = _predict_lmp(coef, wx_fcast["temperature_2m"].values, wx_fcast["time"])
        wx_fcast = wx_fcast.copy()
        wx_fcast["predicted_lmp"] = pred.round(2)

        daily = wx_fcast.groupby(wx_fcast["time"].dt.date).agg(
            lmp_avg=("predicted_lmp", "mean"),
            lmp_peak=("predicted_lmp", "max"),
            lmp_low=("predicted_lmp", "min"),
            temp_high=("temperature_2m", "max"),
            temp_low=("temperature_2m", "min"),
        ).round(2)

        tag = _source_tag(lmp_df)
        lines = [
            f"  {d}: avg ${r.lmp_avg}/MWh  peak ${r.lmp_peak}/MWh  "
            f"(temp {r.temp_low}–{r.temp_high}°F)"
            for d, r in daily.iterrows()
        ]

        return (
            f"{tag}\n"
            f"ISO: {iso} | {horizon_days}-Day LMP Forecast\n"
            f"Model: linear regression on temperature, hour-of-day, month (trained on {lookback} days)\n\n"
            + "\n".join(lines)
            + "\n\n⚠ Forecast uncertainty is significant beyond 3–5 days. "
            "Treat as indicative, not as a trading signal."
        )
    except Exception as e:
        return f"Error generating LMP forecast for {iso}: {e}"


@tool
def compare_markets(reference_date: str) -> str:
    """
    Compare LMP prices across all three ISO markets (ERCOT, PJM, SPP)
    for the same date.

    Args:
        reference_date: Date to compare in YYYY-MM-DD format

    Returns a side-by-side market comparison.
    """
    results = []
    for iso in ["ERCOT", "PJM", "SPP"]:
        try:
            df = lmp_data.get_lmp(iso, reference_date, reference_date)
            tag = df["source"].iloc[0]
            mean_lmp = df["LMP"].mean().round(2)
            peak_lmp = df["LMP"].max().round(2)
            spike_pct = (df["LMP"] > 100).mean() * 100
            results.append(
                f"  {iso:6s} [{tag}]:  mean ${mean_lmp:7.2f}/MWh | "
                f"peak ${peak_lmp:7.2f}/MWh | spikes {spike_pct:.1f}%"
            )
        except Exception as e:
            results.append(f"  {iso}: Error — {e}")

    return (
        f"Market Comparison — {reference_date}\n"
        + "\n".join(results)
        + "\n\n(Lowest mean price = most attractive for power buyers on this date)"
    )
