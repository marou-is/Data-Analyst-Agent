from __future__ import annotations

import plotly.graph_objects as go

from analyst.contracts import AnomalyResult, ForecastResult


def anomaly_chart(result: AnomalyResult) -> go.Figure:
    f = result.frame
    flagged = f[f["is_anomaly"]]
    fig = go.Figure()
    fig.add_scatter(x=f.index, y=f["value"], mode="lines", name=result.target_col)
    fig.add_scatter(
        x=flagged.index, y=flagged["value"], mode="markers", name="anomaly",
        marker={"color": "crimson", "size": 7},
    )
    fig.update_layout(title=f"Anomalies in {result.target_col} ({result.method})", hovermode="x unified")
    return fig


def forecast_chart(result: ForecastResult) -> go.Figure:
    bt, fc = result.backtest, result.forecast
    fig = go.Figure()
    fig.add_scatter(x=bt.index, y=bt["y"], mode="lines", name="actual (backtest)")
    fig.add_scatter(x=bt.index, y=bt["yhat"], mode="lines", name="backtest prediction", line={"dash": "dot"})
    fig.add_scatter(x=fc.index, y=fc["yhat_upper"], mode="lines", line={"width": 0}, showlegend=False)
    fig.add_scatter(
        x=fc.index, y=fc["yhat_lower"], mode="lines", line={"width": 0}, fill="tonexty",
        fillcolor="rgba(99,110,250,0.2)", name="95% interval",
    )
    fig.add_scatter(x=fc.index, y=fc["yhat"], mode="lines", name="forecast")
    fig.update_layout(title=f"Forecast of {result.target_col} ({result.model})", hovermode="x unified")
    return fig
