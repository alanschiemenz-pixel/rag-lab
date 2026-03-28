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

# Apply to any handlers already on the root logger
for _h in logging.root.handlers:
    _h.addFilter(_redact_filter)

# Patch root.addHandler so every future handler also gets the filter
_orig_add_handler = logging.root.addHandler
def _patched_add_handler(handler):
    handler.addFilter(_redact_filter)
    _orig_add_handler(handler)
logging.root.addHandler = _patched_add_handler

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
