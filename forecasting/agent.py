"""
LangGraph agent: Claude + 6 energy-market tools via create_react_agent.
"""
import os
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from tools import (
    get_lmp_data,
    get_historical_weather,
    correlate_lmp_weather,
    get_weather_forecast,
    forecast_lmp,
    compare_markets,
)

SYSTEM_PROMPT = """You are an expert energy market analyst specializing in \
LMP (Locational Marginal Price) forecasting for US wholesale electricity markets.

You have access to data from three ISO markets:
- ERCOT (Texas grid)
- PJM (Mid-Atlantic / Midwest)
- SPP (Southwest Power Pool — Kansas, Oklahoma, surrounding states)

When answering, always:
1. State which ISO market(s) you are analyzing.
2. Disclose data provenance — check the [DATA_SOURCE: mock|real] tag in tool \
output and tell the user if you are reasoning from synthetic mock data.
3. Quantify uncertainty in forecasts — LMP forecasting beyond 3–5 days is \
inherently uncertain; say so.
4. Provide actionable insight — go beyond raw numbers to explain what the \
data means for buyers, sellers, or grid operators.
5. Suggest follow-up analyses when relevant (e.g., "Would you like to see how \
this compares to PJM?").

Use metric units internally but present temperatures in °F and prices in $/MWh \
to match US market conventions."""

_llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,
    max_tokens=2048,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

_tools = [
    get_lmp_data,
    get_historical_weather,
    correlate_lmp_weather,
    get_weather_forecast,
    forecast_lmp,
    compare_markets,
]

agent = create_react_agent(_llm, _tools, prompt=SYSTEM_PROMPT)
