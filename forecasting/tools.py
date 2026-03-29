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
    Returns (coef, sigma_base) where sigma_base is the std of training residuals —
    the day-1 uncertainty used for the sqrt(t) confidence interval.
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

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    sigma_base = float(np.std(y - X @ coef))
    return coef, sigma_base


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


_CDD_BASE = 65.0  # °F — industry-standard thermal comfort threshold

def _add_cdd_hdd(df: pd.DataFrame) -> pd.DataFrame:
    """Add CDD/HDD columns to a DataFrame that already has temp_avg and temp_max."""
    df["CDD_avg"] = (df["temp_avg"] - _CDD_BASE).clip(lower=0)
    df["HDD_avg"] = (_CDD_BASE - df["temp_avg"]).clip(lower=0)
    df["CDD_max"] = (df["temp_max"] - _CDD_BASE).clip(lower=0)
    return df


def _aggregate_daily(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly merged LMP+weather DataFrame to daily averages,
    then derive CDD/HDD features.

    Exogenous columns returned: CDD_avg, HDD_avg, CDD_max
      CDD_avg = max(0, daily_avg_temp - 65)  — cooling demand proxy
      HDD_avg = max(0, 65 - daily_avg_temp)  — heating demand proxy
      CDD_max = max(0, daily_max_temp - 65)  — peak cooling intensity
    """
    daily = (
        merged.groupby(merged["time"].dt.date)
        .agg(
            lmp_avg=("LMP", "mean"),
            temp_avg=("temperature_2m", "mean"),
            temp_max=("temperature_2m", "max"),
        )
        .reset_index()
    )
    daily["time"] = pd.to_datetime(daily["time"])
    daily = daily.set_index("time").asfreq("D")
    return _add_cdd_hdd(daily)


def _fit_arima_model(merged: pd.DataFrame):
    """
    Fit a SARIMAX model on daily LMP with temperature as exogenous variables.
    Model: SARIMAX(1,1,1)(1,0,1,7) — AR(1), differenced, MA(1),
           seasonal AR/MA at 7-day lag (weekly pattern).

    The target is shifted-log transformed: log(LMP - floor) where
    floor = min(daily LMP) - 1, guaranteeing the argument is always >= 1
    before taking the log. This constrains back-transformed forecasts to
    always be greater than floor (i.e. never cross zero into nonsensical
    territory), while handling ERCOT negative prices correctly.

    Exogenous features use CDD/HDD rather than raw temperature:
      CDD_avg = max(0, daily_avg_temp - 65)  — cooling demand proxy
      HDD_avg = max(0, 65 - daily_avg_temp)  — heating demand proxy
      CDD_max = max(0, daily_max_temp - 65)  — peak cooling intensity
    This is asymmetric (cooling and heating have independent slopes),
    avoids the multicollinearity of T + T², and matches industry convention.

    Returns (result, sigma_base, daily_df, floor) where:
      result     — fitted statsmodels SARIMAXResultsWrapper (log scale)
      sigma_base — std of in-sample residuals on the log scale
      daily_df   — the daily training DataFrame (original $/MWh scale)
      floor      — shift constant; back-transform: exp(pred) + floor
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    daily = _aggregate_daily(merged)

    # Shifted-log transform so all values are strictly positive before log.
    # Use floor=0 (pure log) when prices are always positive; only shift
    # downward when the series contains non-positive values (e.g. ERCOT wind).
    min_lmp = float(daily["lmp_avg"].min())
    floor   = 0.0 if min_lmp > 0 else min_lmp - 1.0
    y_log   = np.log(daily["lmp_avg"] - floor)

    exog = daily[["CDD_avg", "HDD_avg", "CDD_max"]]

    model = SARIMAX(
        y_log,
        exog=exog,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, method="lbfgs", maxiter=200)
    sigma_base = float(np.std(result.resid.dropna()))
    return result, sigma_base, daily, floor


def _predict_arima(result, wx_fcast: pd.DataFrame) -> np.ndarray:
    """
    Forecast daily LMP using a fitted SARIMAX result.

    wx_fcast must have columns: time (date index), CDD_avg, HDD_avg, CDD_max.
    Returns log-scale predictions — caller must back-transform with
    np.exp(preds) + floor to get $/MWh values.
    """
    exog_fcast = wx_fcast[["CDD_avg", "HDD_avg", "CDD_max"]]
    forecast = result.forecast(steps=len(exog_fcast), exog=exog_fcast)
    return forecast.values  # log scale


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

        # Use first node only for correlation; fetch weather at that hub's location
        node    = lmp_df["Location"].iloc[0]
        wx_df   = wx_data.weather_for_hub(iso, node, start, end)
        node_df = lmp_df[lmp_df["Location"] == node].copy()
        node_df["time"] = node_df["Time"].dt.tz_localize(None).dt.floor("h") if node_df["Time"].dt.tz is not None else node_df["Time"].dt.floor("h")
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
            f"ISO: {iso} | Lookback: {lookback_days} days | Node: {node} | Weather: hub-specific\n\n"
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


def _build_hub_merged(lmp_df: pd.DataFrame, hub: str, iso: str,
                      start_hist: str, end_hist: str) -> pd.DataFrame:
    """Filter LMP to one hub and merge with hub-specific historical weather."""
    node_df = lmp_df[lmp_df["Location"] == hub].copy()
    node_df["time"] = (
        node_df["Time"].dt.tz_localize(None).dt.floor("h")
        if node_df["Time"].dt.tz is not None
        else node_df["Time"].dt.floor("h")
    )
    wx = wx_data.weather_for_hub(iso, hub, start_hist, end_hist)
    wx["time"] = wx["time"].dt.floor("h")
    return node_df.merge(wx, on="time", how="inner")


def _build_hub_fcast_daily(iso: str, hub: str,
                           horizon_days: int) -> pd.DataFrame:
    """Aggregate hub-specific hourly weather forecast to daily CDD/HDD exog DataFrame."""
    wx_fcast = wx_data.forecast_for_hub(iso, hub, horizon_days)
    daily = (
        wx_fcast.groupby(wx_fcast["time"].dt.date)
        .agg(temp_avg=("temperature_2m", "mean"),
             temp_max=("temperature_2m", "max"),
             temp_low=("temperature_2m", "min"))
        .reset_index()
    )
    daily["time"] = pd.to_datetime(daily["time"])
    daily = daily.set_index("time").asfreq("D").iloc[:horizon_days]
    return _add_cdd_hdd(daily)


@tool
def forecast_lmp(iso: str, horizon_days: int = 7) -> str:
    """
    Forecast LMP prices for each pricing hub in the given ISO using a
    per-hub SARIMAX model trained on hub-specific LMP and local weather.

    Args:
        iso: Energy market — ERCOT, PJM, or SPP
        horizon_days: Number of days to forecast (max 14)

    Returns predicted daily average LMP per hub with 95% confidence intervals.
    """
    horizon_days = min(horizon_days, 14)
    lookback = 45

    end_hist   = (date.today() - timedelta(days=1)).isoformat()
    start_hist = (date.today() - timedelta(days=lookback + 1)).isoformat()

    try:
        lmp_df = lmp_data.get_lmp(iso, start_hist, end_hist)
        hubs   = lmp_df["Location"].unique().tolist()
        tag    = _source_tag(lmp_df)

        sections = []
        for hub in hubs:
            try:
                merged = _build_hub_merged(lmp_df, hub, iso, start_hist, end_hist)
                if len(merged) < 48:
                    sections.append(f"  {hub}: insufficient data")
                    continue

                arima_result, sigma_base, _, floor = _fit_arima_model(merged)
                fcast_daily = _build_hub_fcast_daily(iso, hub, horizon_days)
                log_preds   = _predict_arima(arima_result, fcast_daily)
                aic         = round(arima_result.aic, 1)

                lines = [f"  Hub: {hub} | AIC={aic} | sigma={sigma_base:.3f} (log)"]
                for day_idx, (d, log_pred) in enumerate(
                        zip(fcast_daily.index, log_preds), start=1):
                    ci_log   = 1.96 * sigma_base * np.sqrt(day_idx)
                    avg      = round(float(np.exp(log_pred) + floor), 2)
                    ci_upper = round(float(np.exp(log_pred + ci_log) + floor), 2)
                    ci_lower = round(float(np.exp(log_pred - ci_log) + floor), 2)
                    t_lo     = round(fcast_daily.loc[d, "temp_low"], 1)
                    t_hi     = round(fcast_daily.loc[d, "temp_max"], 1)
                    lines.append(
                        f"    Day {day_idx:2d} ({d.date()}): avg ${avg}/MWh  "
                        f"95% CI [{ci_lower}, {ci_upper}]  "
                        f"(temp {t_lo}-{t_hi}°F)"
                    )
                sections.append("\n".join(lines))
            except Exception as hub_err:
                sections.append(f"  {hub}: error — {hub_err}")

        return (
            f"{tag}\n"
            f"ISO: {iso} | {horizon_days}-Day LMP Forecast by Hub\n"
            f"Model: SARIMAX(1,1,1)(1,0,1,7) + shifted-log | "
            f"hub-specific weather | asymmetric 95% CI\n\n"
            + "\n\n".join(sections)
            + "\n\n⚠ Uncertainty widens significantly beyond day 5. "
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
