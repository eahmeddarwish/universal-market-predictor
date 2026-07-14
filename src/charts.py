"""
charts.py
=========
Interactive Plotly Charts for Market Analysis

Charts produced:
  1. Candlestick + Bollinger Bands + Volume
  2. RSI panel
  3. MACD panel
  4. LSTM Backtest: Actual vs Predicted
  5. Training Loss curve

Author : Ahmed Darwish
Email  : eahmeddarwish@gmail.com
GitHub : github.com/eahmeddarwish
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Colour palette ──────────────────────────
C_GREEN     = "#00C896"
C_RED       = "#FF4B6E"
C_BLUE      = "#4B9EFF"
C_ORANGE    = "#FF9F40"
C_PURPLE    = "#B57BFF"
C_YELLOW    = "#FFD700"
C_GREY_DARK = "#1E2130"
C_GREY_MID  = "#2A2D3E"
C_TEXT      = "#E0E0E0"
C_GRID      = "#2F3347"

LAYOUT_BASE = dict(
    paper_bgcolor = C_GREY_DARK,
    plot_bgcolor  = C_GREY_MID,
    font          = dict(color=C_TEXT, family="Inter, sans-serif", size=12),
    xaxis         = dict(showgrid=True, gridcolor=C_GRID, zeroline=False),
    yaxis         = dict(showgrid=True, gridcolor=C_GRID, zeroline=False),
    hovermode     = "x unified",
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_GRID, borderwidth=1),
    margin        = dict(l=50, r=30, t=60, b=50),
)


# ─────────────────────────────────────────────
#  1.  MAIN PRICE CHART
# ─────────────────────────────────────────────

def make_price_chart(df: pd.DataFrame, ticker: str, name: str, currency: str = "USD") -> go.Figure:
    """
    4-row subplot:
      Row 1 (60%): Candlesticks + Bollinger + MA lines
      Row 2 (15%): Volume bars
      Row 3 (12%): RSI
      Row 4 (13%): MACD
    """
    show_volume = "Volume" in df.columns and df["Volume"].sum() > 0
    n_rows       = 4 if show_volume else 3
    row_heights  = [0.55, 0.15, 0.15, 0.15] if show_volume else [0.60, 0.20, 0.20]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=["", "Volume", "RSI", "MACD"][-n_rows:] if show_volume else ["", "RSI", "MACD"],
    )

    # ── Candlestick ──
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        increasing_line_color=C_GREEN,
        decreasing_line_color=C_RED,
        name="OHLC",
        showlegend=False,
    ), row=1, col=1)

    # ── Bollinger Bands ──
    if "BB_Upper" in df.columns:
        for band, label, dash in [
            ("BB_Upper", "BB Upper", "dot"),
            ("BB_Mid",   "BB Mid",   "solid"),
            ("BB_Lower", "BB Lower", "dot"),
        ]:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[band],
                name=label, mode="lines",
                line=dict(color=C_YELLOW, width=1, dash=dash),
                opacity=0.6,
            ), row=1, col=1)

        # Shaded band
        fig.add_trace(go.Scatter(
            x=pd.concat([df.index.to_series(), df.index.to_series()[::-1]]),
            y=pd.concat([df["BB_Upper"], df["BB_Lower"][::-1]]),
            fill="toself", fillcolor="rgba(255,215,0,0.06)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, name="BB Fill",
        ), row=1, col=1)

    # ── Moving Averages ──
    for col, colour, label in [
        ("MA_20",  C_BLUE,   "MA 20"),
        ("MA_50",  C_ORANGE, "MA 50"),
        ("MA_200", C_PURPLE, "MA 200"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                name=label, mode="lines",
                line=dict(color=colour, width=1.2),
            ), row=1, col=1)

    # ── Volume ──
    if show_volume:
        vol_row = 2
        colours = [C_GREEN if c >= o else C_RED
                   for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            name="Volume",
            marker_color=colours, opacity=0.7,
            showlegend=False,
        ), row=vol_row, col=1)

        if "Volume_SMA" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["Volume_SMA"],
                name="Vol SMA20", mode="lines",
                line=dict(color=C_ORANGE, width=1),
            ), row=vol_row, col=1)

    # ── RSI ──
    rsi_row = 3 if show_volume else 2
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"],
            name="RSI", mode="lines",
            line=dict(color=C_PURPLE, width=1.5),
        ), row=rsi_row, col=1)

        for level, colour in [(70, C_RED), (30, C_GREEN)]:
            fig.add_hline(y=level, line_dash="dash",
                          line_color=colour, opacity=0.5, row=rsi_row, col=1)

        # Shaded overbought / oversold zones
        fig.add_hrect(y0=70, y1=100, fillcolor=C_RED,   opacity=0.05,
                      row=rsi_row, col=1, line_width=0)
        fig.add_hrect(y0=0,  y1=30,  fillcolor=C_GREEN, opacity=0.05,
                      row=rsi_row, col=1, line_width=0)

    # ── MACD ──
    macd_row = 4 if show_volume else 3
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"],
            name="MACD", mode="lines",
            line=dict(color=C_BLUE, width=1.5),
        ), row=macd_row, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_Signal"],
            name="Signal", mode="lines",
            line=dict(color=C_ORANGE, width=1.2),
        ), row=macd_row, col=1)

        if "MACD_Hist" in df.columns:
            hist_colours = [C_GREEN if v >= 0 else C_RED for v in df["MACD_Hist"]]
            fig.add_trace(go.Bar(
                x=df.index, y=df["MACD_Hist"],
                name="Histogram", marker_color=hist_colours, opacity=0.6,
                showlegend=False,
            ), row=macd_row, col=1)

    # ── Layout ──
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=f"<b>{ticker} — {name}</b>   <sub>{currency}</sub>",
            font=dict(size=18),
        ),
        height=780,
        xaxis_rangeslider_visible=False,
    )

    # Y-axis labels
    fig.update_yaxes(title_text=f"Price ({currency})", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI",    row=rsi_row,  col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD",   row=macd_row, col=1)

    return fig


# ─────────────────────────────────────────────
#  2.  LSTM BACKTEST CHART
# ─────────────────────────────────────────────

def make_prediction_chart(
    df: pd.DataFrame,
    test_actual: np.ndarray,
    test_predicted: np.ndarray,
    next_price: float,
    ticker: str,
    currency: str = "USD",
) -> go.Figure:
    """
    Actual vs Predicted on the test set, plus next-day forecast dot.
    """
    # Align test-set dates
    n_test = len(test_actual)
    test_dates = df.index[-(n_test):]

    fig = go.Figure()

    # Historical close (training portion)
    split_idx = len(df) - n_test
    fig.add_trace(go.Scatter(
        x=df.index[:split_idx],
        y=df["Close"].values[:split_idx],
        name="Historical (Train)",
        mode="lines",
        line=dict(color=C_GREY_MID.replace("2A", "66"), width=1),
        opacity=0.5,
    ))

    # Actual test-set prices
    fig.add_trace(go.Scatter(
        x=test_dates, y=test_actual,
        name="Actual Price",
        mode="lines",
        line=dict(color=C_BLUE, width=2),
    ))

    # LSTM predictions
    fig.add_trace(go.Scatter(
        x=test_dates, y=test_predicted,
        name="LSTM Predicted",
        mode="lines",
        line=dict(color=C_ORANGE, width=2, dash="dash"),
    ))

    # Next-day forecast
    from datetime import timedelta
    last_date  = df.index[-1]
    next_date  = last_date + timedelta(days=1)
    curr_price = df["Close"].iloc[-1]

    colour    = C_GREEN if next_price >= curr_price else C_RED
    direction = "▲" if next_price >= curr_price else "▼"
    pct       = (next_price - curr_price) / curr_price * 100

    fig.add_trace(go.Scatter(
        x=[last_date, next_date],
        y=[curr_price, next_price],
        mode="lines+markers",
        name=f"Forecast {direction} {pct:+.2f}%",
        line=dict(color=colour, width=2, dash="dot"),
        marker=dict(size=[6, 14], color=colour, symbol=["circle", "star"]),
    ))

    fig.add_annotation(
        x=next_date, y=next_price,
        text=f"  <b>{next_price:,.2f}</b>",
        showarrow=False,
        font=dict(color=colour, size=14),
        xanchor="left",
    )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=f"<b>{ticker}</b> — LSTM Backtest & Next-Day Forecast",
            font=dict(size=16),
        ),
        height=480,
        yaxis_title=f"Price ({currency})",
    )

    return fig


# ─────────────────────────────────────────────
#  3.  TRAINING LOSS CURVE
# ─────────────────────────────────────────────

def make_loss_chart(train_loss: list, val_loss: list) -> go.Figure:
    epochs = list(range(1, len(train_loss) + 1))

    fig = go.Figure([
        go.Scatter(x=epochs, y=train_loss,
                   name="Train Loss", mode="lines",
                   line=dict(color=C_BLUE,   width=2)),
        go.Scatter(x=epochs, y=val_loss,
                   name="Val Loss",   mode="lines",
                   line=dict(color=C_ORANGE, width=2, dash="dash")),
    ])

    fig.update_layout(
        **LAYOUT_BASE,
        title="LSTM Training — Loss Curve",
        xaxis_title="Epoch",
        yaxis_title="Huber Loss",
        height=300,
    )
    return fig
