from dotenv import load_dotenv
import gridstatus
import openmeteo_requests  # pip install openmeteo-requests

load_dotenv()


#pjm = gridstatus.PJM()
#pjm.get_lmp(date='2024-01-01', market='REAL_TIME_HOURLY', location_type='ZONE')

#ercot = gridstatus.Ercot()
#ercot.get_lmp(date='2024-01-01', market='REAL_TIME_15_MIN', location_type='HUB')

#spp = gridstatus.SPP()
#spp.get_lmp(date='2024-01-01', market='REAL_TIME_5_MIN', location_type='HUB')

# %%
om = openmeteo_requests.Client()

# # Historical (ERA5 reanalysis — back to 1940)
# historical = om.weather_api(
#     "https://archive-api.open-meteo.com/v1/archive",
#     params={
#         "latitude": 29.76,   "longitude": -95.37,  # Houston
#         "start_date": "2025-03-01",
#         "end_date":   "2025-03-31",
#         "hourly": ["temperature_2m", "precipitation",
#                    "wind_speed_10m", "cloud_cover",
#                    "shortwave_radiation"]
#     }
# )
# historical

# # %%

# # 16-day forecast
# forecast = om.weather_api(
#     "https://api.open-meteo.com/v1/forecast",
#     params={
#         "latitude": 29.76,   "longitude": -95.37,
#         "hourly": ["temperature_2m", "precipitation_probability",
#                    "wind_speed_10m"],
#         "forecast_days": 16
#     }
# )

# forecast[0].Current

# %%

# ── 0. Imports ────────────────────────────────────────────────────────────────
import json
import pprint
import warnings
from datetime import datetime, timedelta
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
 
warnings.filterwarnings("ignore", category=FutureWarning)
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — API CALL SETUP
# ─────────────────────────────────────────────────────────────────────────────
# return — swap in `hist_response` / `fcast_response` for the mock objects.

import openmeteo_requests
import requests_cache
from retry_requests import retry

cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
om = openmeteo_requests.Client(session=retry_session)

LOCATION = {"latitude": 29.76, "longitude": -95.37}  # Houston, TX

# Historical — ERA5 reanalysis
hist_response = om.weather_api(
    "https://archive-api.open-meteo.com/v1/archive",
    params={
        **LOCATION,
        "start_date": "2020-01-01",
        "end_date":   "2024-12-31",
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "precipitation",
            "wind_speed_10m", "wind_direction_10m",
            "shortwave_radiation", "cloud_cover",
            "et0_fao_evapotranspiration",
        ],
        "timezone": "America/Chicago",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
    }
)[0]
print(hist_response)

# 16-day Forecast
fcast_response = om.weather_api(
    "https://api.open-meteo.com/v1/forecast",
    params={
        **LOCATION,
        "hourly": [
            "temperature_2m", "precipitation_probability", "precipitation",
            "wind_speed_10m", "wind_gusts_10m",
            "cloud_cover", "shortwave_radiation",
        ],
        "daily": [
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "wind_speed_10m_max",
            "precipitation_probability_max",
        ],
        "forecast_days": 16,
        "timezone": "America/Chicago",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
    }
)[0]