import logging
import os
from dotenv import load_dotenv

load_dotenv()


class _RedactSecretsFilter(logging.Filter):
    """Redacts API keys from log records before they reach any handler."""
    _secrets: list[str] = []

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for secret in self._secrets:
            if secret and secret in msg:
                record.msg = msg.replace(secret, "***")
                record.args = ()
        return True


_redact_filter = _RedactSecretsFilter()

# Patch logging.Logger.addHandler at the class level so every handler
# added to ANY logger (root, library loggers with propagate=False, etc.)
# automatically gets the redact filter.
_orig_add_handler = logging.Logger.addHandler

def _patched_add_handler(self, handler):
    handler.addFilter(_redact_filter)
    _orig_add_handler(self, handler)

logging.Logger.addHandler = _patched_add_handler  # type: ignore[method-assign]

# Retroactively cover handlers already registered on existing loggers
# (libraries imported before config.py, or handlers added at module load)
for _existing in [logging.root] + list(logging.Logger.manager.loggerDict.values()):
    if isinstance(_existing, logging.Logger):
        for _h in _existing.handlers:
            _h.addFilter(_redact_filter)

# Geographic center of each ISO — used for Open-Meteo weather calls
ISO_REGIONS = {
    "ERCOT": {"lat": 30.27, "lon": -97.74, "label": "Austin, TX"},
    "PJM":   {"lat": 39.95, "lon": -75.16, "label": "Philadelphia, PA"},
    "SPP":   {"lat": 35.47, "lon": -97.52, "label": "Oklahoma City, OK"},
}

# Default pricing nodes per ISO
ISO_NODES = {
    "ERCOT": ["HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON"],
    "PJM":   ["EASTERN HUB", "WESTERN HUB", "AEP GEN HUB", "N ILLINOIS HUB"],
    "SPP":   ["SPP.AECI", "SPP.SOCO", "SPP.KACY"],
}

# Geographic center of each pricing hub — used for hub-specific weather calls.
# Coordinates are representative points within each hub's settlement area.
ISO_HUB_LOCATIONS = {
    "ERCOT": {
        "HB_NORTH":   {"lat": 32.78, "lon": -96.80},  # Dallas
        "HB_SOUTH":   {"lat": 29.42, "lon": -98.49},  # San Antonio
        "HB_WEST":    {"lat": 31.84, "lon": -102.37}, # Midland
        "HB_HOUSTON": {"lat": 29.76, "lon": -95.37},  # Houston
    },
    "PJM": {
        "EASTERN HUB":    {"lat": 40.06, "lon": -74.90},  # Southern NJ
        "WESTERN HUB":    {"lat": 40.44, "lon": -79.99},  # Pittsburgh
        "AEP GEN HUB":    {"lat": 38.35, "lon": -81.63},  # Charleston, WV
        "N ILLINOIS HUB": {"lat": 41.88, "lon": -87.63},  # Chicago
    },
    "SPP": {
        "SPP.AECI": {"lat": 38.63, "lon": -90.20},  # St. Louis
        "SPP.SOCO": {"lat": 33.75, "lon": -84.39},  # Atlanta
        "SPP.KACY": {"lat": 39.10, "lon": -94.58},  # Kansas City
    },
}

# Rough base LMP ($/MWh) per ISO for mock data calibration
ISO_BASE_LMP = {
    "ERCOT": 35.0,
    "PJM":   45.0,
    "SPP":   30.0,
}

# Set USE_MOCK_DATA=true in .env to force mock data (no API keys needed)
USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "false").lower() == "true"

AZURE_OPENAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY        = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
PJM_API_KEY = os.getenv("PJM_API_KEY", "")  # Optional — falls back to mock if absent

# Register all secrets so the log filter can redact them
_redact_filter._secrets = [
    s for s in [AZURE_OPENAI_KEY, PJM_API_KEY] if s
]
