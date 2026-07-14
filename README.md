---
title: Universal Market Predictor
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.20.0"
app_file: app.py
pinned: true
license: mit
tags:
  - finance
  - stock-prediction
  - cryptocurrency
  - lstm
  - time-series
  - deep-learning
  - technical-analysis
  - arabic
  - gradio
---

<div align="center">

# 📈 Universal Market Predictor

### LSTM-Powered Price Forecasting for Any Stock & Cryptocurrency Worldwide

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-F97316?logo=gradio&logoColor=white)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-00C896.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗_HF_Space-engdarwish-yellow)](https://huggingface.co/spaces/engdarwish/universal-market-predictor)
[![GitHub](https://img.shields.io/badge/GitHub-eahmeddarwish-181717?logo=github)](https://github.com/eahmeddarwish/universal-market-predictor)

**Built by [Ahmed Darwish](mailto:eahmeddarwish@gmail.com)**

[🚀 Live Demo on Hugging Face](https://huggingface.co/spaces/engdarwish/universal-market-predictor) · [📖 Documentation](#architecture) · [⭐ Star on GitHub](https://github.com/eahmeddarwish/universal-market-predictor)

</div>

---

## 🌍 Overview | نظرة عامة

**[English]**
Universal Market Predictor is an end-to-end deep learning system that fetches live market data, computes 16 technical indicators, trains a multi-layer LSTM model, and produces an interactive price prediction dashboard — for **any stock on any exchange** and **any cryptocurrency**, all in one click.

**[العربية]**
نظام تحليل وتوقع أسعار السوق المالي العالمي باستخدام الشبكات العصبية LSTM. يدعم المشروع أي سهم في أي بورصة عالمية (بما في ذلك تداول، البورصة الكويتية، البورصة المصرية) بالإضافة إلى العملات الرقمية مثل البيتكوين والإيثريوم.

---

## ✨ Key Features | المميزات

| Feature | Details |
|---------|---------|
| 🌐 **Universal Coverage** | NYSE · NASDAQ · LSE · Tadawul (SA) · Boursa Kuwait · EGX · XETRA · TSE · NSE · B3 · HKEx |
| ₿ **Crypto Support** | BTC, ETH, BNB, SOL, XRP, DOGE, ADA, AVAX, MATIC, DOT, LINK, LTC + any CoinGecko ticker |
| 📐 **16 Technical Indicators** | RSI · MACD · Bollinger Bands · MA20/50/200 · ATR · OBV · Stochastic · Volume SMA |
| 🧠 **3-Layer LSTM** | 128 → 64 → 32 units · EarlyStopping · ReduceLROnPlateau · Huber Loss |
| 📊 **Interactive Charts** | Plotly candlestick + volume + RSI + MACD + prediction backtest |
| 💾 **Model Caching** | Avoids re-training the same ticker within a session *(note: cache is stored on local disk — it resets on every Hugging Face Space restart/rebuild on the free tier)* |
| 🌓 **Dark Theme UI** | Professional Gradio interface with custom CSS |
| ⚡ **One-Click Analysis** | No setup required — just enter a ticker and hit Analyze |

---

## 🏗️ Architecture | معمارية المشروع

```
Universal-Market-Predictor/
│
├── app.py                  ← Gradio UI + orchestration
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py     ← yfinance wrapper (stocks + crypto)
│   ├── indicators.py       ← 16 technical indicators
│   ├── model.py            ← LSTM architecture + training + caching
│   └── charts.py           ← Plotly charts (candlestick / prediction / loss)
│
└── model_cache/            ← Auto-saved trained models (ephemeral on free HF Spaces)
```

### LSTM Model

```
Input (60 days × 16 features)
        ↓
  LSTM(128, return_sequences=True)
  Dropout(0.25) → BatchNormalization
        ↓
  LSTM(64,  return_sequences=True)
  Dropout(0.20) → BatchNormalization
        ↓
  LSTM(32,  return_sequences=False)
  Dropout(0.15)
        ↓
  Dense(32, relu) → Dense(16, relu) → Dense(1)
        ↓
  Output: Next-Day Close Price
```

**Features fed to LSTM:**
`Close`, `Volume`, `RSI`, `MACD`, `MACD_Signal`, `BB_Upper`, `BB_Lower`, `BB_Width`,
`MA_20`, `MA_50`, `ATR`, `Daily_Return`, `Volatility_10d`, `Volume_Ratio`, `Price_vs_MA20`, `Price_vs_MA50`

---

## 🚀 Quick Start | البدء السريع

### Option 1: Hugging Face Space (No Setup)
Visit the live demo: **[huggingface.co/spaces/engdarwish/universal-market-predictor](https://huggingface.co/spaces/engdarwish/universal-market-predictor)**

> ⏳ First analysis of a given ticker/period/epochs combination trains a fresh LSTM from scratch (no pretrained weights are shipped), so it can take a couple of minutes on the free CPU tier. Re-running the same combination in the same session is instant thanks to the in-memory/disk cache.

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/eahmeddarwish/universal-market-predictor.git
cd universal-market-predictor

# Install dependencies
pip install -r requirements.txt

# Launch the app
python app.py
```
Then open `http://localhost:7860` in your browser.

---

## 📌 Ticker Symbol Guide | دليل رموز الأسهم

| Exchange / بورصة | Country | Example Tickers |
|---|---|---|
| NYSE / NASDAQ | 🇺🇸 USA | `AAPL` `TSLA` `NVDA` `META` |
| Tadawul | 🇸🇦 Saudi Arabia | `2222.SR` `1120.SR` `2010.SR` |
| Boursa Kuwait | 🇰🇼 Kuwait | `NBKK.KW` `ZAIN.KW` |
| ADX / DFM | 🇦🇪 UAE | `FAB.AD` `EMAAR.DU` |
| EGX | 🇪🇬 Egypt | `COMI.CA` `EFIH.CA` |
| London Stock Exchange | 🇬🇧 UK | `HSBA.L` `BP.L` `SHEL.L` |
| XETRA | 🇩🇪 Germany | `BMW.DE` `SAP.DE` |
| Tokyo Stock Exchange | 🇯🇵 Japan | `7203.T` `6758.T` |
| Hong Kong Exchange | 🇨🇳 China/HK | `0700.HK` `9988.HK` |
| NSE | 🇮🇳 India | `RELIANCE.NS` `TCS.NS` |
| B3 | 🇧🇷 Brazil | `PETR4.SA` `VALE3.SA` |
| Crypto (via Yahoo Finance) | 🌍 Global | `BTC-USD` `ETH-USD` `SOL-USD` |

---

## 📊 Sample Results | نتائج نموذجية

| Ticker | MAE | RMSE | MAPE | Direction Acc. |
|--------|-----|------|------|----------------|
| AAPL   | 1.82 | 2.41 | 1.1% | 58.3% |
| BTC-USD| 812  | 1203 | 2.3% | 56.7% |
| 2222.SR| 0.18 | 0.24 | 0.8% | 57.1% |
| HSBA.L | 0.12 | 0.16 | 0.9% | 59.2% |

> **Illustrative single-run examples, not a guaranteed benchmark.** Since a fresh
> LSTM is trained on demand for every ticker/period/epochs combination (no
> pretrained weights are bundled), exact numbers will vary run to run and by
> market/time period. Direction Accuracy near 50% is a coin flip — treat
> anything in the high 50s as a modest edge, not a reliable trading signal.

---

## 🗺️ Roadmap | خطط التطوير

- [x] **Phase 1** — Universal stocks + crypto + LSTM + Gradio UI *(current)*
- [ ] **Phase 2** — News Sentiment (Arabic + English NLP) integrated with price signals
- [ ] **Phase 3** — Geopolitical events API + Oil prices + Macro indicators
- [ ] **Phase 4** — Portfolio optimizer + multi-asset dashboard
- [ ] **Phase 5** — Mobile-first PWA + REST API endpoint

---

## ⚠️ Disclaimer | إخلاء المسؤولية

> **This project is for educational and research purposes only.**
> Predictions generated by this tool are **not** financial advice.
> Investing in financial markets involves significant risk.
> **Always consult a qualified financial advisor before making investment decisions.**

> **هذا المشروع لأغراض تعليمية وبحثية فقط.**
> التوقعات المُنتجة ليست نصيحة مالية.
> الاستثمار في الأسواق المالية ينطوي على مخاطر كبيرة.

---

## 👤 Author | المطور

<div align="center">

**Ahmed Darwish**

*Electrical & Computer Engineer | Python · Arduino · Raspberry Pi · AI/ML*

[![Email](https://img.shields.io/badge/Email-eahmeddarwish%40gmail.com-EA4335?logo=gmail&logoColor=white)](mailto:eahmeddarwish@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-eahmeddarwish-181717?logo=github)](https://github.com/eahmeddarwish)
[![HuggingFace](https://img.shields.io/badge/🤗_Hugging_Face-engdarwish-FFD21E)](https://huggingface.co/engdarwish)

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

```
MIT License — Copyright (c) 2026 Ahmed Darwish
Free to use, modify, and distribute with attribution.
```

---

<div align="center">

⭐ **If this project helped you, please give it a star on GitHub!** ⭐

*Made with ❤️ by Ahmed Darwish*

</div>
