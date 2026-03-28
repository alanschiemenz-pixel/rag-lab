import os
import sys
from datetime import date, timedelta

# config.py must run before any LangChain import (sets env vars from .env)
import config  # noqa: F401

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from langchain_core.messages import AIMessage, HumanMessage

from agent import agent
from data import lmp as lmp_data
from data import weather as wx_data


# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_dates():
    end   = (date.today() - timedelta(days=1)).isoformat()
    start = (date.today() - timedelta(days=8)).isoformat()
    return start, end


# ── Chat (Tab 1) ──────────────────────────────────────────────────────────────

def handle_chat(message: str, history: list, mock_toggle: bool) -> tuple:
    if not message.strip():
        yield history, ""
        return

    # Allow the UI toggle to override USE_MOCK_DATA at runtime
    os.environ["USE_MOCK_DATA"] = "true" if mock_toggle else "false"
    config.USE_MOCK_DATA = mock_toggle  # keep in-process config in sync

    lc_messages = []
    for entry in history:
        if entry["role"] == "user":
            lc_messages.append(HumanMessage(content=entry["content"]))
        else:
            lc_messages.append(AIMessage(content=entry["content"]))
    lc_messages.append(HumanMessage(content=message))

    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": ""},
    ]

    partial = ""
    for chunk, metadata in agent.stream(
        {"messages": lc_messages}, stream_mode="messages"
    ):
        if chunk.content and metadata.get("langgraph_node") == "model":
            partial += chunk.content
            history[-1] = {"role": "assistant", "content": partial}
            yield history, ""


# ── Data Explorer (Tab 2) ─────────────────────────────────────────────────────

def fetch_explorer_data(iso: str, start_date: str, end_date: str):
    """Fetch LMP + weather and return a table + plotly figure."""
    try:
        lmp_df = lmp_data.get_lmp(iso, start_date, end_date)
        wx_df  = wx_data.weather_for_iso(iso, start_date, end_date)
    except Exception as e:
        return pd.DataFrame(), None, f"Error: {e}"

    # ── Plotly dual-axis chart ──────────────────────────────────────────
    # Pivot LMP to first node for the chart
    node = lmp_df["Location"].iloc[0]
    node_df = lmp_df[lmp_df["Location"] == node].sort_values("Time")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=node_df["Time"], y=node_df["LMP"],
        name=f"LMP ({node})", line=dict(color="#2196F3"),
        yaxis="y1",
    ))

    # Align weather to same time axis
    wx_df_sorted = wx_df.sort_values("time")
    fig.add_trace(go.Scatter(
        x=wx_df_sorted["time"], y=wx_df_sorted["temperature_2m"],
        name="Temp (°F)", line=dict(color="#FF5722", dash="dot"),
        yaxis="y2",
    ))

    source_tag = lmp_df["source"].iloc[0]
    fig.update_layout(
        title=f"{iso} LMP vs Temperature — {start_date} to {end_date} [{source_tag} data]",
        xaxis=dict(title="Time"),
        yaxis=dict(title="LMP ($/MWh)", side="left"),
        yaxis2=dict(title="Temperature (°F)", side="right", overlaying="y",
                    showgrid=False),
        legend=dict(x=0.01, y=0.99),
        height=420,
        template="plotly_white",
    )

    # Summary table (daily averages, all nodes)
    table_df = lmp_df.groupby(["Location", lmp_df["Time"].dt.date])["LMP"] \
                     .agg(["mean", "min", "max"]).round(2).reset_index()
    table_df.columns = ["Node", "Date", "Avg LMP", "Min LMP", "Max LMP"]

    return table_df, fig, f"Loaded {len(lmp_df)} rows [{source_tag} data]"


# ── Forecast (Tab 3) ──────────────────────────────────────────────────────────

def run_forecast(iso: str, horizon: int):
    """Run the LMP forecast and return a chart + text summary."""
    from tools import forecast_lmp as forecast_tool

    # Call the tool function directly (not through the agent)
    summary_text = forecast_tool.invoke({"iso": iso, "horizon_days": horizon})

    # Also build the chart from raw data
    try:
        from datetime import timedelta
        import numpy as np
        from data.lmp import get_lmp as _get_lmp
        from data.weather import forecast_for_iso, weather_for_iso
        from tools import _fit_lmp_model, _predict_lmp

        lookback = 45
        end_hist   = (date.today() - timedelta(days=1)).isoformat()
        start_hist = (date.today() - timedelta(days=lookback + 1)).isoformat()

        lmp_df = _get_lmp(iso, start_hist, end_hist)
        wx_hist = weather_for_iso(iso, start_hist, end_hist)

        node_df = lmp_df[lmp_df["Location"] == lmp_df["Location"].iloc[0]].copy()
        node_df["time"] = node_df["Time"].dt.floor("h")
        wx_hist["time"] = wx_hist["time"].dt.floor("h")
        merged = node_df.merge(wx_hist, on="time", how="inner")

        coef = _fit_lmp_model(merged)

        wx_fcast = forecast_for_iso(iso, horizon)
        pred = _predict_lmp(coef, wx_fcast["temperature_2m"].values, wx_fcast["time"])

        daily = wx_fcast.copy()
        daily["predicted_lmp"] = pred

        daily_agg = daily.groupby(daily["time"].dt.date).agg(
            lmp_avg=("predicted_lmp", "mean"),
            lmp_high=("predicted_lmp", "max"),
            lmp_low=("predicted_lmp", "min"),
            temp_high=("temperature_2m", "max"),
        ).reset_index()

        dates = pd.to_datetime(daily_agg["time"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=daily_agg["lmp_high"],
            name="Forecast High", line=dict(color="#FF9800"),
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=daily_agg["lmp_avg"],
            name="Forecast Avg", line=dict(color="#2196F3"),
            fill="tonexty", fillcolor="rgba(33,150,243,0.1)",
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=daily_agg["lmp_low"],
            name="Forecast Low", line=dict(color="#4CAF50"),
            fill="tonexty", fillcolor="rgba(33,150,243,0.05)",
        ))

        source_tag = lmp_df["source"].iloc[0]
        fig.update_layout(
            title=f"{iso} {horizon}-Day LMP Forecast [{source_tag} training data]",
            xaxis_title="Date",
            yaxis_title="Predicted LMP ($/MWh)",
            height=420,
            template="plotly_white",
        )
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Chart error: {e}", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14))

    return fig, summary_text


# ── Gradio UI ─────────────────────────────────────────────────────────────────

ISO_CHOICES = ["ERCOT", "PJM", "SPP"]
default_start, default_end = _default_dates()

with gr.Blocks(title="LMP Forecasting Assistant") as demo:
    gr.Markdown("## LMP Forecasting Assistant")
    gr.Markdown(
        "Energy market LMP analysis and weather-driven price forecasting for "
        "ERCOT, PJM, and SPP. Use the **Chat** tab to ask questions in natural "
        "language, or the **Data Explorer** and **Forecast** tabs for direct access."
    )

    with gr.Tabs():

        # ── Tab 1: Chat ───────────────────────────────────────────────────────
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(height=480)

            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask about LMP prices, weather correlations, or forecasts...",
                    label="", scale=9, lines=1,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                mock_toggle = gr.Checkbox(
                    label="Use mock data (no API keys needed)",
                    value=not bool(config.AZURE_OPENAI_KEY),
                )

            gr.Examples(
                examples=[
                    ["What were ERCOT LMP prices like recently?"],
                    ["How does temperature correlate with PJM prices?"],
                    ["Forecast SPP prices for the next 7 days."],
                    ["Compare all three markets for yesterday."],
                    ["What's the weather forecast for the ERCOT region?"],
                ],
                inputs=[msg_box],
                label="Example questions",
            )

            clear_btn = gr.Button("Clear conversation")

            send_btn.click(
                handle_chat,
                inputs=[msg_box, chatbot, mock_toggle],
                outputs=[chatbot, msg_box],
            )
            msg_box.submit(
                handle_chat,
                inputs=[msg_box, chatbot, mock_toggle],
                outputs=[chatbot, msg_box],
            )
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])

        # ── Tab 2: Data Explorer ──────────────────────────────────────────────
        with gr.Tab("Data Explorer"):
            gr.Markdown("### Raw LMP + Weather Data")
            with gr.Row():
                exp_iso   = gr.Dropdown(ISO_CHOICES, value="ERCOT", label="ISO Market")
                exp_start = gr.Textbox(value=default_start, label="Start Date (YYYY-MM-DD)")
                exp_end   = gr.Textbox(value=default_end,   label="End Date (YYYY-MM-DD)")
                fetch_btn = gr.Button("Fetch Data", variant="primary")

            exp_status = gr.Textbox(label="Status", interactive=False)
            exp_chart  = gr.Plot(label="LMP vs Temperature")
            exp_table  = gr.DataFrame(label="Daily LMP Summary by Node")

            fetch_btn.click(
                fetch_explorer_data,
                inputs=[exp_iso, exp_start, exp_end],
                outputs=[exp_table, exp_chart, exp_status],
            )

        # ── Tab 3: Forecast ───────────────────────────────────────────────────
        with gr.Tab("Price Forecast"):
            gr.Markdown("### Weather-Driven LMP Price Forecast")
            gr.Markdown(
                "Trains a regression model on 45 days of historical LMP + weather, "
                "then projects forward using the 14-day Open-Meteo forecast."
            )
            with gr.Row():
                fcast_iso     = gr.Dropdown(ISO_CHOICES, value="ERCOT", label="ISO Market")
                fcast_horizon = gr.Slider(1, 14, value=7, step=1, label="Forecast Horizon (days)")
                fcast_btn     = gr.Button("Run Forecast", variant="primary")

            fcast_chart   = gr.Plot(label="Predicted LMP Range")
            fcast_summary = gr.Textbox(label="Forecast Summary", lines=20, interactive=False)

            fcast_btn.click(
                run_forecast,
                inputs=[fcast_iso, fcast_horizon],
                outputs=[fcast_chart, fcast_summary],
            )


if __name__ == "__main__":
    demo.launch(share=True)
