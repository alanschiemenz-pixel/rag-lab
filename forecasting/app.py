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

def fetch_explorer_data(iso: str, hub: str, start_date: str, end_date: str):
    """Fetch LMP + hub-specific weather and return a table + plotly figure."""
    try:
        lmp_df = lmp_data.get_lmp(iso, start_date, end_date)
        wx_df  = wx_data.weather_for_hub(iso, hub, start_date, end_date)
    except Exception as e:
        return pd.DataFrame(), None, f"Error: {e}"

    node_df = lmp_df[lmp_df["Location"] == hub].sort_values("Time")
    if node_df.empty:
        node_df = lmp_df[lmp_df["Location"] == lmp_df["Location"].iloc[0]].sort_values("Time")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=node_df["Time"], y=node_df["LMP"],
        name=f"LMP ({hub})", line=dict(color="#2196F3"),
        yaxis="y1",
    ))
    wx_df_sorted = wx_df.sort_values("time")
    fig.add_trace(go.Scatter(
        x=wx_df_sorted["time"], y=wx_df_sorted["temperature_2m"],
        name="Temp (°F)", line=dict(color="#FF5722", dash="dot"),
        yaxis="y2",
    ))

    source_tag = lmp_df["source"].iloc[0]
    fig.update_layout(
        title=f"{iso} — {hub} LMP vs Temperature — {start_date} to {end_date} [{source_tag} data]",
        xaxis=dict(title="Time"),
        yaxis=dict(title="LMP ($/MWh)", side="left"),
        yaxis2=dict(title="Temperature (°F)", side="right", overlaying="y",
                    showgrid=False),
        legend=dict(x=0.01, y=0.99),
        height=420,
        template="plotly_white",
    )

    # Summary table — selected hub only
    hub_lmp = lmp_df[lmp_df["Location"] == hub] if hub in lmp_df["Location"].values else lmp_df
    table_df = hub_lmp.groupby(hub_lmp["Time"].dt.date)["LMP"] \
                      .agg(["mean", "min", "max"]).round(2).reset_index()
    table_df.columns = ["Date", "Avg LMP", "Min LMP", "Max LMP"]

    return table_df, fig, f"Loaded {len(hub_lmp)} rows for {hub} [{source_tag} data]"


# ── Forecast (Tab 3) ──────────────────────────────────────────────────────────

def run_forecast(iso: str, hub: str, horizon: int):
    """Run the LMP forecast for a single hub and return chart, diagnostics, stats, and summary."""
    from tools import forecast_lmp as forecast_tool

    summary_text = forecast_tool.invoke({"iso": iso, "horizon_days": horizon})

    diag_fig   = go.Figure()
    stats_text = ""
    fig        = go.Figure()

    try:
        import numpy as np
        from plotly.subplots import make_subplots
        from data.lmp import get_lmp as _get_lmp
        from tools import _fit_arima_model, _predict_arima, _build_hub_merged, _build_hub_fcast_daily

        BURN_IN    = 8
        lookback   = 45
        end_hist   = (date.today() - timedelta(days=1)).isoformat()
        start_hist = (date.today() - timedelta(days=lookback + 1)).isoformat()

        lmp_df     = _get_lmp(iso, start_hist, end_hist)
        source_tag = lmp_df["source"].iloc[0]

        merged      = _build_hub_merged(lmp_df, hub, iso, start_hist, end_hist)
        arima_result, _, daily_train, floor = _fit_arima_model(merged)

        # ── Post-burn-in metrics ──────────────────────────────────────────
        y_act_full = daily_train["lmp_avg"].values
        y_fit_full = np.exp(arima_result.fittedvalues.values) + floor
        res_full   = arima_result.resid.dropna().values
        min_len    = min(len(y_act_full), len(y_fit_full), len(res_full))
        y_actual   = y_act_full[-min_len:][BURN_IN:]
        y_fitted   = y_fit_full[-min_len:][BURN_IN:]
        residuals  = res_full[-min_len:][BURN_IN:]
        td         = daily_train.index[-min_len:][BURN_IN:]
        sigma_base = float(np.std(residuals))

        err  = y_actual - y_fitted
        r2   = max(0.0, 1 - np.sum(err**2) / np.sum((y_actual - y_actual.mean())**2))
        rmse = float(np.sqrt(np.mean(err**2)))
        mae  = float(np.mean(np.abs(err)))
        aic  = round(arima_result.aic, 1)

        _ex_log = np.log(max(np.mean(y_actual) - floor, 1e-6))
        ci1_lo  = round(np.exp(_ex_log - 1.96 * sigma_base) + floor, 2)
        ci1_hi  = round(np.exp(_ex_log + 1.96 * sigma_base) + floor, 2)
        ci7_lo  = round(np.exp(_ex_log - 1.96 * sigma_base * np.sqrt(7)) + floor, 2)
        ci7_hi  = round(np.exp(_ex_log + 1.96 * sigma_base * np.sqrt(7)) + floor, 2)

        stats_text = (
            f"Hub: {hub} | ISO: {iso}\n"
            f"Training: {start_hist} to {end_hist}\n"
            f"Samples: {len(daily_train)} days, {len(y_actual)} post-burn-in "
            f"(first {BURN_IN} excluded: Kalman warm-up)\n\n"
            f"Model: SARIMAX(1,1,1)(1,0,1,7) + shifted-log\n"
            f"  Exogenous: daily avg temp + daily max temp (hub-specific weather)\n"
            f"  Floor: {floor:.2f} $/MWh  |  Transform: log(LMP - floor)\n\n"
            f"  Exogenous features (base = 65°F):\n"
            f"    CDD_avg = max(0, daily_avg_temp - 65)  [cooling demand]\n"
            f"    HDD_avg = max(0, 65 - daily_avg_temp)  [heating demand]\n"
            f"    CDD_max = max(0, daily_max_temp - 65)  [peak cooling]\n\n"
            f"  AIC  = {aic}\n"
            f"  R²   = {r2:.4f}\n"
            f"  RMSE = ${rmse:.2f}/MWh\n"
            f"  MAE  = ${mae:.2f}/MWh\n"
            f"  σ    = {sigma_base:.4f} (log scale)\n\n"
            f"Illustrative 95% CI (asymmetric):\n"
            f"  Day 1:  [{ci1_lo}, {ci1_hi}] $/MWh\n"
            f"  Day 7:  [{ci7_lo}, {ci7_hi}] $/MWh"
        )

        # ── Diagnostics figure ────────────────────────────────────────────
        diag_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"Actual vs Fitted — {hub} ($/MWh)",
                f"Log-scale Residuals — {hub}",
                "Residual Distribution (log scale)",
                "Fitted vs Actual Scatter ($/MWh)",
            ),
            vertical_spacing=0.15, horizontal_spacing=0.10,
        )
        diag_fig.add_trace(go.Scatter(
            x=td, y=y_actual, name="Actual",
            line=dict(color="#FF5722", width=1.5), opacity=0.85,
        ), row=1, col=1)
        diag_fig.add_trace(go.Scatter(
            x=td, y=y_fitted, name="Fitted",
            line=dict(color="#2196F3", width=1.5), opacity=0.85,
        ), row=1, col=1)
        diag_fig.add_trace(go.Scatter(
            x=td, y=residuals, mode="markers+lines",
            marker=dict(color="#9C27B0", size=4),
            line=dict(color="#9C27B0", width=0.5), showlegend=False,
        ), row=1, col=2)
        diag_fig.add_hline(y=0, line_color="black", line_dash="dash",
                           line_width=1, row=1, col=2)
        diag_fig.add_trace(go.Histogram(
            x=residuals, nbinsx=20, marker_color="#9C27B0",
            opacity=0.6, histnorm="probability density", showlegend=False,
        ), row=2, col=1)
        x_norm = np.linspace(residuals.min(), residuals.max(), 200)
        y_norm = (1 / (sigma_base * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (x_norm / sigma_base) ** 2)
        diag_fig.add_trace(go.Scatter(
            x=x_norm, y=y_norm,
            line=dict(color="red", width=2), showlegend=False,
        ), row=2, col=1)
        diag_fig.add_trace(go.Scatter(
            x=y_fitted, y=y_actual, mode="markers",
            marker=dict(color="#2196F3", size=6, opacity=0.5), showlegend=False,
        ), row=2, col=2)
        lim = [min(y_actual.min(), y_fitted.min()), max(y_actual.max(), y_fitted.max())]
        diag_fig.add_trace(go.Scatter(
            x=lim, y=lim, mode="lines",
            line=dict(color="red", dash="dash", width=1), showlegend=False,
        ), row=2, col=2)
        diag_fig.update_layout(
            title=f"{iso} SARIMAX Diagnostics — {hub} [{source_tag}]  AIC={aic}  RMSE=${rmse:.2f}",
            height=600, template="plotly_white", legend=dict(x=0.01, y=0.99),
        )
        diag_fig.update_xaxes(title_text="Date",           row=1, col=1)
        diag_fig.update_yaxes(title_text="LMP ($/MWh)",    row=1, col=1)
        diag_fig.update_xaxes(title_text="Date",           row=1, col=2)
        diag_fig.update_yaxes(title_text="Residual (log)", row=1, col=2)
        diag_fig.update_xaxes(title_text="Residual (log)", row=2, col=1)
        diag_fig.update_yaxes(title_text="Density",        row=2, col=1)
        diag_fig.update_xaxes(title_text="Fitted ($/MWh)", row=2, col=2)
        diag_fig.update_yaxes(title_text="Actual ($/MWh)", row=2, col=2)

        # ── Forecast figure ───────────────────────────────────────────────
        fcast_daily = _build_hub_fcast_daily(iso, hub, horizon)
        log_preds   = _predict_arima(arima_result, fcast_daily)
        dates       = fcast_daily.index
        ci_log      = 1.96 * sigma_base * np.sqrt(np.arange(1, len(dates) + 1))
        preds_orig  = np.exp(log_preds) + floor
        ci_upper    = np.exp(log_preds + ci_log) + floor
        ci_lower    = np.exp(log_preds - ci_log) + floor

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=ci_upper, line=dict(width=0), mode="lines", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=ci_lower, name="95% CI", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(33,150,243,0.15)", mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=preds_orig, name=hub,
            line=dict(color="#2196F3", width=2), mode="lines+markers",
        ))
        fig.update_layout(
            title=f"{iso} — {hub} {horizon}-Day Forecast [{source_tag} data]",
            xaxis_title="Date", yaxis_title="Predicted LMP ($/MWh)",
            height=420, template="plotly_white",
        )

    except Exception as e:
        fig.add_annotation(text=f"Chart error: {e}", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14))
        if not stats_text:
            stats_text = f"Error: {e}"

    return fig, diag_fig, stats_text, summary_text


# ── Gradio UI ─────────────────────────────────────────────────────────────────

ISO_CHOICES = ["ERCOT", "PJM", "SPP"]
default_start, default_end = _default_dates()

def _hub_choices(iso: str) -> list[str]:
    return config.ISO_NODES.get(iso.upper(), [])

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
                exp_hub   = gr.Dropdown(_hub_choices("ERCOT"), value=_hub_choices("ERCOT")[0],
                                        label="Hub")
                exp_start = gr.Textbox(value=default_start, label="Start Date (YYYY-MM-DD)")
                exp_end   = gr.Textbox(value=default_end,   label="End Date (YYYY-MM-DD)")
                fetch_btn = gr.Button("Fetch Data", variant="primary")

            exp_status = gr.Textbox(label="Status", interactive=False)
            exp_chart  = gr.Plot(label="LMP vs Temperature")
            exp_table  = gr.DataFrame(label="Daily LMP Summary")

            exp_iso.change(
                fn=lambda iso: gr.Dropdown(choices=_hub_choices(iso),
                                           value=_hub_choices(iso)[0]),
                inputs=[exp_iso],
                outputs=[exp_hub],
            )
            fetch_btn.click(
                fetch_explorer_data,
                inputs=[exp_iso, exp_hub, exp_start, exp_end],
                outputs=[exp_table, exp_chart, exp_status],
            )

        # ── Tab 3: Forecast ───────────────────────────────────────────────────
        with gr.Tab("Price Forecast"):
            gr.Markdown("### Weather-Driven LMP Price Forecast")
            gr.Markdown(
                "**Model: SARIMAX(1,1,1)(1,0,1,7)**\n\n"
                "Trains a Seasonal ARIMA with eXogenous inputs model on 45 days of "
                "historical LMP and weather, then projects forward using the 14-day "
                "Open-Meteo weather forecast.\n\n"
                "| Component | Meaning |\n"
                "|---|---|\n"
                "| AR(1) | Today's price depends on yesterday's price |\n"
                "| I(1) | First-difference to remove price trend/drift |\n"
                "| MA(1) | Smooths one lag of forecast error |\n"
                "| SAR(1), SMA(1) at lag 7 | Weekly demand cycle (Mon–Sun pattern) |\n"
                "| Exogenous | CDD_avg, HDD_avg, CDD_max (base 65°F) |\n\n"
                "Uncertainty grows as **1.96 × σ × √day** — a random-walk envelope "
                "anchored to the model's in-sample residual standard deviation (σ)."
            )
            with gr.Row():
                fcast_iso     = gr.Dropdown(ISO_CHOICES, value="ERCOT", label="ISO Market")
                fcast_hub     = gr.Dropdown(_hub_choices("ERCOT"), value=_hub_choices("ERCOT")[0],
                                            label="Hub")
                fcast_horizon = gr.Slider(1, 14, value=7, step=1, label="Forecast Horizon (days)")
                fcast_btn     = gr.Button("Run Forecast", variant="primary")

            fcast_chart   = gr.Plot(label="Predicted LMP Range")

            gr.Markdown("#### Model Diagnostics")
            gr.Markdown(
                "In-sample fit on daily training data: actual vs fitted LMP, residuals "
                "over time, residual distribution vs fitted normal (red), and "
                "fitted-vs-actual scatter. A tight scatter around the diagonal and "
                "residuals centered on zero indicate a well-specified model."
            )
            fcast_diag    = gr.Plot(label="SARIMAX Diagnostics")
            fcast_stats   = gr.Textbox(label="Model Statistics", lines=12, interactive=False)
            fcast_summary = gr.Textbox(label="Forecast Summary", lines=20, interactive=False)

            fcast_iso.change(
                fn=lambda iso: gr.Dropdown(choices=_hub_choices(iso),
                                           value=_hub_choices(iso)[0]),
                inputs=[fcast_iso],
                outputs=[fcast_hub],
            )
            fcast_btn.click(
                run_forecast,
                inputs=[fcast_iso, fcast_hub, fcast_horizon],
                outputs=[fcast_chart, fcast_diag, fcast_stats, fcast_summary],
            )


if __name__ == "__main__":
    demo.launch(share=True)
