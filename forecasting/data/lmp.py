"""
LMP (Locational Marginal Price) data layer.

Real data: fetched via the `gridstatus` library (no key for ERCOT/SPP;
           PJM_API_KEY env var required for PJM live data).
Fallback:  deterministic mock data that mirrors realistic market patterns.
"""
import logging
import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# ── Mock data ─────────────────────────────────────────────────────────────────

def _mock_lmp(iso: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Generates hourly mock LMP data with realistic daily/seasonal patterns.
    Seeded from the ISO name + start date for reproducibility.
    """
    seed = hash(f"{iso}{start.date()}") % (2**31)
    rng = np.random.default_rng(seed)

    hours = pd.date_range(start, end, freq="h", inclusive="left")
    n = len(hours)
    base = config.ISO_BASE_LMP.get(iso, 35.0)
    nodes = config.ISO_NODES[iso]

    records = []
    for node in nodes:
        h = hours.hour.values
        doy = hours.day_of_year.values

        # Daily shape: morning ramp + evening peak
        daily = 8 * np.sin((h - 8) * np.pi / 10) + 5 * np.sin((h - 18) * np.pi / 5)
        # Seasonal: summer cooling peak, winter heating shoulder
        seasonal = 12 * np.sin((doy - 172) * 2 * np.pi / 365)
        noise = rng.normal(0, 3, n)
        # Random price spikes (~4% of hours)
        spikes = np.where(rng.random(n) < 0.04, rng.uniform(60, 250, n), 0.0)

        lmp = base + daily + seasonal + noise + spikes

        # ERCOT can go negative during high wind periods
        if iso == "ERCOT":
            negatives = np.where(rng.random(n) < 0.03, rng.uniform(-25, -2, n), 0.0)
            lmp = lmp + negatives

        lmp = lmp.round(2)
        records.append(pd.DataFrame({
            "Time":       hours,
            "Location":   node,
            "LMP":        lmp,
            "Energy":     (lmp * 0.85).round(2),
            "Congestion": (lmp * 0.10).round(2),
            "Loss":       (lmp * 0.05).round(2),
            "source":     "mock",
        }))

    return pd.concat(records, ignore_index=True)


# ── Real data stubs ───────────────────────────────────────────────────────────

def _real_ercot(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch ERCOT day-ahead hourly LMP via gridstatus (no API key required)."""
    import gridstatus
    iso = gridstatus.Ercot()
    df = iso.get_lmp(date=start.date(), end=end.date(), market="DAY_AHEAD_HOURLY")
    return _normalize(df, "ERCOT")


def _real_pjm(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch PJM day-ahead hourly LMP via gridstatus.
    Requires PJM_API_KEY env var — raises ValueError if absent.
    """
    if not config.PJM_API_KEY:
        raise ValueError("PJM_API_KEY not set")
    import gridstatus
    iso = gridstatus.PJM(api_key=config.PJM_API_KEY)
    df = iso.get_lmp(date=start.date(), end=end.date(), market="DAY_AHEAD_HOURLY")
    return _normalize(df, "PJM")


def _real_spp(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch SPP day-ahead hourly LMP via gridstatus (no API key required)."""
    import gridstatus
    iso = gridstatus.spp()
    df = iso.get_lmp(date=start.date(), end=end.date(), market="DAY_AHEAD_HOURLY")
    return _normalize(df, "SPP")


def _normalize(df: pd.DataFrame, iso: str) -> pd.DataFrame:
    """Normalize gridstatus output to the canonical LMP schema."""
    # Priority order: first match wins for each target column
    location_candidates = ["Location Name", "Location", "Location Short Name"]
    time_candidates     = ["Interval Start", "Time"]
    lmp_candidates      = ["LMP", "LMP Total"]

    def first_present(candidates):
        return next((c for c in candidates if c in df.columns), None)

    renames = {}
    if col := first_present(time_candidates):
        renames[col] = "Time"
        # Drop any other columns that would collide with the renamed "Time"
        df = df.drop(columns=[c for c in time_candidates if c != col and c in df.columns])
    if col := first_present(location_candidates):
        renames[col] = "Location"
    if col := first_present(lmp_candidates):
        renames[col] = "LMP"
    for c in ["Energy", "Congestion", "Loss"]:
        if c in df.columns:
            renames[c] = c

    df = df.rename(columns=renames)
    for col in ["Energy", "Congestion", "Loss"]:
        if col not in df.columns:
            df[col] = np.nan
    df["source"] = "real"
    return df[["Time", "Location", "LMP", "Energy", "Congestion", "Loss", "source"]]


# ── Public API ────────────────────────────────────────────────────────────────

_REAL_FETCHERS = {
    "ERCOT": _real_ercot,
    "PJM":   _real_pjm,
    "SPP":   _real_spp,
}


def get_lmp(iso: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Return hourly LMP data for the given ISO and date range.

    Attempts real data via gridstatus; falls back to mock on any error
    (or immediately if USE_MOCK_DATA=true).

    Returns a DataFrame with columns:
      Time, Location, LMP, Energy, Congestion, Loss, source
    """
    iso = iso.upper()
    if iso not in config.ISO_REGIONS:
        raise ValueError(f"Unknown ISO: {iso}. Choose from {list(config.ISO_REGIONS)}")

    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)

    if config.USE_MOCK_DATA:
        logger.info(f"[LMP] Using mock data for {iso} (USE_MOCK_DATA=true)")
        return _mock_lmp(iso, start, end)

    try:
        df = _REAL_FETCHERS[iso](start, end)
        logger.info(f"[LMP] Fetched real data for {iso}: {len(df)} rows")
        return df
    except Exception as exc:
        logger.warning(f"[LMP] Real fetch failed for {iso} ({exc}). Falling back to mock.")
        return _mock_lmp(iso, start, end)
