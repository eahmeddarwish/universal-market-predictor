"""
app.py  —  Universal Market Predictor
======================================
Gradio Space for Hugging Face

Supports:
  ✅ Any stock ticker worldwide  (NYSE / NASDAQ / LSE / Tadawul / Boursa Kuwait / EGX / ...)
  ✅ Cryptocurrencies            (BTC, ETH, SOL, BNB, XRP, ...)
  ✅ LSTM multi-feature prediction (16 technical features)
  ✅ Interactive Plotly charts  (Candlestick / Bollinger / RSI / MACD / LSTM Backtest)
  ✅ Technical signal summary

Author : Ahmed Darwish
Email  : eahmeddarwish@gmail.com
GitHub : github.com/engrdarwish
HF     : huggingface.co/engdarwish
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import spaces
import gradio as gr
import pandas as pd
import numpy as np

from src.data_fetcher  import (fetch_market_data, get_market_summary,
                                POPULAR_STOCKS, POPULAR_CRYPTO, PERIOD_OPTIONS)
from src.indicators    import add_all_indicators, get_signal_summary
from src.model         import train, cache_path, save_cache, load_cache
from src.charts        import make_price_chart, make_prediction_chart, make_loss_chart


# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────

CSS = """
/* ── Global ── */
body, .gradio-container { background: #0F1117 !important; color: #E0E0E0; }
.gr-box, .gr-form       { background: #1E2130 !important; border-radius: 12px; }

/* ── Header banner ── */
#header-banner {
    background: linear-gradient(135deg, #1A1F35 0%, #12273A 50%, #1A1F35 100%);
    border: 1px solid #2F3347;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 20px;
    text-align: center;
}
#header-banner h1 { font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #4B9EFF, #00C896, #FFD700);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
#header-banner p  { color: #9BA3BC; font-size: 0.95rem; margin-top: 6px; }

/* ── Metric cards ── */
.metric-card {
    background: #1E2130;
    border: 1px solid #2F3347;
    border-radius: 10px;
    padding: 14px 20px;
    text-align: center;
}
.metric-label  { font-size: 0.78rem; color: #9BA3BC; text-transform: uppercase; letter-spacing: .06em; }
.metric-value  { font-size: 1.5rem;  font-weight: 700; color: #E0E0E0; }
.metric-green  { color: #00C896 !important; }
.metric-red    { color: #FF4B6E !important; }

/* ── Buttons ── */
.btn-primary { background: linear-gradient(135deg, #4B9EFF, #0069E0) !important;
    color: #fff !important; border: none !important; font-weight: 600 !important; }
.btn-primary:hover { opacity: 0.88 !important; }

/* ── Signal box ── */
#signals-box {
    background: #1A1F35;
    border-left: 3px solid #4B9EFF;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.93rem;
    line-height: 1.9;
    white-space: pre-line;
}

/* ── Metrics box ── */
#metrics-box {
    background: #1A1F35;
    border-left: 3px solid #00C896;
    border-radius: 8px;
    padding: 14px 18px;
    font-family: 'Courier New', monospace;
    font-size: 0.91rem;
    line-height: 1.9;
    white-space: pre;
}

/* ── Forecast banner ── */
#forecast-banner {
    background: linear-gradient(135deg, #0D2237, #12273A);
    border: 1px solid #4B9EFF55;
    border-radius: 12px;
    padding: 20px 28px;
    text-align: center;
    font-size: 1.0rem;
}
#forecast-banner .price-big {
    font-size: 2.6rem; font-weight: 800;
}

/* ── Disclaimer ── */
#disclaimer {
    background: #1A1225;
    border: 1px solid #FF4B6E44;
    border-radius: 8px;
    padding: 12px 18px;
    font-size: 0.82rem;
    color: #9BA3BC;
    margin-top: 12px;
}
"""


# ─────────────────────────────────────────────
#  CORE PIPELINE
# ─────────────────────────────────────────────

@spaces.GPU(duration=120)
def run_analysis(ticker_input, preset_choice, period_label, epochs, use_cache, progress=gr.Progress()):
    """
    Main function wired to Gradio.
    Returns: price_chart, pred_chart, loss_chart, forecast_html, signals_html, metrics_html, status
    """
    # ── resolve ticker ──
    ticker = ticker_input.strip().upper() if ticker_input.strip() else preset_choice
    if not ticker:
        return None, None, None, "", "", "", "⚠️ Please enter or select a ticker."

    period = PERIOD_OPTIONS.get(period_label, "2y")

    try:
        # ── 1. Fetch data ──
        progress(0.05, desc=f"📡 Fetching {ticker} data…")
        raw = fetch_market_data(ticker, period)
        if raw["error"]:
            return None, None, None, "", "", "", raw["error"]

        df   = raw["df"]
        name = raw["name"]
        curr = raw["currency"]

        if len(df) < 220:
            return None, None, None, "", "", "", (
                f"⚠️ Only {len(df)} data points available for {ticker}. "
                "Need at least 220 rows. Try a longer period or different ticker."
            )

        # ── 2. Technical indicators ──
        progress(0.12, desc="📐 Computing technical indicators…")
        df_ind = add_all_indicators(df)

        # ── 3. Price chart (always shown) ──
        progress(0.20, desc="📊 Building price chart…")
        price_fig = make_price_chart(df_ind, ticker, name, curr)

        # ── 4. Summary card ──
        summary = get_market_summary(df_ind, curr)
        signals = get_signal_summary(df_ind)

        signals_html = "\n".join(f"{k}: {v}" for k, v in signals.items())

        # ── 5. Train / load LSTM ──
        c_path  = cache_path(ticker, period, epochs)
        result  = None

        if use_cache:
            progress(0.25, desc="📦 Checking model cache…")
            result = load_cache(c_path)

        if result is None:
            def _prog(frac, msg):
                progress(0.25 + frac * 0.70, desc=msg)

            result = train(df_ind, epochs=int(epochs), progress_cb=_prog)

            if use_cache:
                progress(0.97, desc="💾 Saving model to cache…")
                try:
                    save_cache(c_path, result)
                except Exception:
                    pass

        # ── 6. Build prediction charts ──
        progress(0.98, desc="🎨 Building prediction charts…")
        pred_fig = make_prediction_chart(
            df_ind, result["test_actual"], result["test_predicted"],
            result["next_price"], ticker, curr
        )
        loss_fig = make_loss_chart(
            result["history"]["train_loss"],
            result["history"]["val_loss"],
        )

        # ── 7. Forecast HTML card ──
        curr_price  = df_ind["Close"].iloc[-1]
        next_price  = result["next_price"]
        price_delta = next_price - curr_price
        pct_delta   = price_delta / curr_price * 100
        arrow       = "▲" if price_delta >= 0 else "▼"
        clr         = "#00C896" if price_delta >= 0 else "#FF4B6E"

        forecast_html = f"""
<div id="forecast-banner">
  <div style="color:#9BA3BC; font-size:.85rem; margin-bottom:6px;">
    🔮 LSTM Next-Day Forecast &nbsp;|&nbsp; {ticker} — {name}
  </div>
  <div class="price-big" style="color:{clr};">
    {next_price:,.4f} {curr}
  </div>
  <div style="font-size:1.1rem; margin-top:6px; color:{clr};">
    {arrow} {abs(price_delta):,.4f} ({pct_delta:+.2f}%)
    &nbsp;&nbsp;vs&nbsp;&nbsp;
    <span style="color:#9BA3BC;">Current: {curr_price:,.4f}</span>
  </div>
  <div style="font-size:.78rem; color:#9BA3BC; margin-top:8px;">
    Direction Accuracy: <b>{result['metrics']['Direction Accuracy']}%</b>
    &nbsp;|&nbsp; MAPE: <b>{result['metrics']['MAPE (%)']:.2f}%</b>
    &nbsp;|&nbsp; R²: <b>{result['metrics']['R²']:.4f}</b>
  </div>
  <div id="disclaimer">
    ⚠️ <b>Disclaimer:</b> This is a research / educational tool.
    Predictions are <b>not</b> financial advice. Markets involve risk.
    Always do your own research (DYOR).
  </div>
</div>
"""

        # ── 8. Metrics box ──
        m = result["metrics"]
        metrics_html = (
            f"Model Performance on Test Set\n"
            f"{'─'*36}\n"
            f"  MAE              : {m['MAE']:>12,.4f} {curr}\n"
            f"  RMSE             : {m['RMSE']:>12,.4f} {curr}\n"
            f"  MAPE             : {m['MAPE (%)']:>11.2f} %\n"
            f"  R² Score         : {m['R²']:>12.4f}\n"
            f"  Direction Acc.   : {m['Direction Accuracy']:>11.1f} %\n"
            f"{'─'*36}\n"
            f"  Train Samples    : {m['Train Samples']:>12,}\n"
            f"  Test  Samples    : {m['Test  Samples']:>12,}\n"
            f"  Features Used    : {m['Features Used']:>12}\n"
            f"  Epochs Run       : {m['Epochs Run']:>12}\n"
            f"  Look-Back Window : {'60 days':>12}\n"
            f"  Forecast Horizon : {'1 day':>12}\n"
        )

        return price_fig, pred_fig, loss_fig, forecast_html, signals_html, metrics_html, f"✅ Done — {ticker} ({name})"

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        return None, None, None, "", "", "", f"❌ Error: {str(e)}\n\n{err}"


# ─────────────────────────────────────────────
#  GRADIO UI
# ─────────────────────────────────────────────

HEADER_HTML = """
<div id="header-banner">
  <h1>📈 Universal Market Predictor</h1>
  <p>
    LSTM-powered price forecasting for <b>any stock</b> worldwide &amp; <b>cryptocurrencies</b><br>
    NYSE · NASDAQ · LSE · Tadawul · Boursa Kuwait · EGX · XETRA · TSE · NSE &amp; more
  </p>
  <p style="margin-top:10px; font-size:.82rem; color:#6B7394;">
    Built by <b style="color:#4B9EFF;">Ahmed Darwish</b> ·
    <a href="https://github.com/engrdarwish" style="color:#4B9EFF;">GitHub</a> ·
    <a href="https://huggingface.co/engdarwish" style="color:#4B9EFF;">Hugging Face</a>
  </p>
</div>
"""

ABOUT_TEXT = """
### 🔬 How It Works

1. **Data** — Fetches OHLCV history via Yahoo Finance (yfinance). Supports 60 + exchanges.
2. **Indicators** — Computes 16 technical features: RSI, MACD, Bollinger Bands, ATR, OBV, and more.
3. **Model** — A 3-layer LSTM trained on the last 60 trading days to predict the next day's close.
4. **Output** — Interactive Plotly charts + signal summary + backtest metrics.

### 📌 Ticker Format Guide

| Exchange | Example |
|---|---|
| 🇺🇸 US (NYSE / NASDAQ) | `AAPL`, `TSLA`, `NVDA` |
| 🇸🇦 Saudi (Tadawul) | `2222.SR` (Aramco) |
| 🇰🇼 Kuwait (Boursa) | `NBKK.KW` |
| 🇦🇪 UAE (ADX) | `FAB.AD` |
| 🇬🇧 London (LSE) | `HSBA.L` |
| 🇩🇪 Germany (XETRA) | `BMW.DE` |
| 🇯🇵 Japan (TSE) | `7203.T` |
| 🇨🇳 Hong Kong | `0700.HK` |
| 🇮🇳 India (NSE) | `RELIANCE.NS` |
| ₿ Crypto | `BTC-USD`, `ETH-USD` |

### ⚠️ Disclaimer
This tool is for **educational and research purposes only**.
Predictions are **not** financial advice. Always consult a qualified financial advisor.
"""

# Build the interface
with gr.Blocks(css=CSS, title="Universal Market Predictor — Ahmed Darwish") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row():
        # ── Left panel: Controls ──
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### ⚙️ Configuration")

            with gr.Tab("📈 Stocks"):
                stock_preset = gr.Dropdown(
                    choices=["(type custom ticker below)"] + list(POPULAR_STOCKS.keys()),
                    value="🇺🇸 AAPL  — Apple Inc.",
                    label="Quick Select",
                )
                stock_ticker = gr.Textbox(
                    label="Or type any ticker symbol",
                    placeholder="e.g. TSLA  /  2222.SR  /  HSBA.L  /  0700.HK",
                )
                stock_btn = gr.Button("🚀 Analyze Stock", variant="primary", elem_classes=["btn-primary"])

            with gr.Tab("₿ Crypto"):
                crypto_preset = gr.Dropdown(
                    choices=["(type custom ticker below)"] + list(POPULAR_CRYPTO.keys()),
                    value="₿  BTC-USD  — Bitcoin",
                    label="Quick Select",
                )
                crypto_ticker = gr.Textbox(
                    label="Or type any crypto ticker",
                    placeholder="e.g.  ETH-USD  /  SOL-USD  /  DOGE-USD",
                )
                crypto_btn = gr.Button("🚀 Analyze Crypto", variant="primary", elem_classes=["btn-primary"])

            gr.Markdown("---")
            period_dd = gr.Dropdown(
                choices=list(PERIOD_OPTIONS.keys()),
                value="2 Years",
                label="📅 Historical Period",
            )
            epochs_sl = gr.Slider(
                minimum=10, maximum=80, value=25, step=5,
                label="🧠 Training Epochs",
                info="More epochs = better fit, slower runtime",
            )
            use_cache = gr.Checkbox(
                value=True,
                label="💾 Use Cached Model (skip re-training)",
            )

            status_box = gr.Textbox(
                label="Status", interactive=False,
                value="Ready — select a ticker and click Analyze."
            )

            gr.Markdown("---")
            with gr.Accordion("ℹ️ How to Use & Ticker Guide", open=False):
                gr.Markdown(ABOUT_TEXT)

        # ── Right panel: Outputs ──
        with gr.Column(scale=3):
            forecast_html = gr.HTML(label="")

            with gr.Tabs():
                with gr.Tab("📊 Price Chart & Indicators"):
                    price_chart = gr.Plot(label="")

                with gr.Tab("🔮 LSTM Backtest & Prediction"):
                    pred_chart = gr.Plot(label="")

                with gr.Tab("📉 Training Loss"):
                    loss_chart = gr.Plot(label="")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📡 Technical Signals")
                    signals_box = gr.Textbox(
                        label="", lines=8, interactive=False,
                        elem_id="signals-box",
                    )
                with gr.Column():
                    gr.Markdown("#### 📐 Model Metrics")
                    metrics_box = gr.Textbox(
                        label="", lines=8, interactive=False,
                        elem_id="metrics-box",
                    )

    # ── Wire up Stock button ──
    def analyze_stock(ticker_custom, preset, period, epochs, cache):
        ticker = ticker_custom.strip() or POPULAR_STOCKS.get(preset, preset)
        return run_analysis(ticker, None, period, epochs, cache)

    stock_btn.click(
        fn=analyze_stock,
        inputs=[stock_ticker, stock_preset, period_dd, epochs_sl, use_cache],
        outputs=[price_chart, pred_chart, loss_chart, forecast_html, signals_box, metrics_box, status_box],
    )

    # ── Wire up Crypto button ──
    def analyze_crypto(ticker_custom, preset, period, epochs, cache):
        ticker = ticker_custom.strip() or POPULAR_CRYPTO.get(preset, preset)
        return run_analysis(ticker, None, period, epochs, cache)

    crypto_btn.click(
        fn=analyze_crypto,
        inputs=[crypto_ticker, crypto_preset, period_dd, epochs_sl, use_cache],
        outputs=[price_chart, pred_chart, loss_chart, forecast_html, signals_box, metrics_box, status_box],
    )

    # ── Footer ──
    gr.HTML("""
    <div style="text-align:center; padding:20px 0 8px; color:#6B7394; font-size:.82rem; border-top:1px solid #2F3347; margin-top:24px;">
      Built with ❤️ by <b style="color:#4B9EFF;">Ahmed Darwish</b> &nbsp;·&nbsp;
      <a href="https://github.com/engrdarwish" style="color:#4B9EFF;">GitHub</a> &nbsp;·&nbsp;
      <a href="https://huggingface.co/engdarwish" style="color:#4B9EFF;">Hugging Face</a>
      <br><span style="color:#4B4F69;">For educational &amp; research purposes only. Not financial advice.</span>
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )

