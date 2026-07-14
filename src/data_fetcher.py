"""
data_fetcher.py
===============
Universal Market Data Fetcher
Supports: All World Stock Exchanges + Cryptocurrencies

Author : Ahmed Darwish
Email  : eahmeddarwish@gmail.com
GitHub : github.com/eahmeddarwish
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ─────────────────────────────────────────────
#  POPULAR PRESETS  (users can also type any ticker)
# ─────────────────────────────────────────────

POPULAR_STOCKS = {
    # 🇺🇸 United States
    "🇺🇸 AAPL  — Apple Inc.": "AAPL",
    "🇺🇸 MSFT  — Microsoft": "MSFT",
    "🇺🇸 GOOGL — Alphabet (Google)": "GOOGL",
    "🇺🇸 AMZN  — Amazon": "AMZN",
    "🇺🇸 TSLA  — Tesla": "TSLA",
    "🇺🇸 NVDA  — NVIDIA": "NVDA",
    "🇺🇸 META  — Meta Platforms": "META",
    "🇺🇸 JPM   — JPMorgan Chase": "JPM",
    # 🇸🇦 Saudi Arabia (Tadawul)
    "🇸🇦 2222.SR — Saudi Aramco": "2222.SR",
    "🇸🇦 1120.SR — Al Rajhi Bank": "1120.SR",
    "🇸🇦 2010.SR — SABIC": "2010.SR",
    # 🇰🇼 Kuwait (Boursa Kuwait)
    "🇰🇼 NBKK.KW — National Bank of Kuwait": "NBKK.KW",
    "🇰🇼 ZAIN.KW — Zain Telecom": "ZAIN.KW",
    # 🇦🇪 UAE (ADX / DFM)
    "🇦🇪 FAB.AD  — First Abu Dhabi Bank": "FAB.AD",
    "🇦🇪 EMAAR.DU — Emaar Properties": "EMAAR.DU",
    # 🇪🇬 Egypt (EGX)
    "🇪🇬 COMI.CA — Commercial Intl. Bank": "COMI.CA",
    "🇪🇬 EFIH.CA — EFG Hermes": "EFIH.CA",
    # 🇬🇧 United Kingdom (LSE)
    "🇬🇧 HSBA.L — HSBC Holdings": "HSBA.L",
    "🇬🇧 BP.L   — BP PLC": "BP.L",
    "🇬🇧 SHEL.L — Shell PLC": "SHEL.L",
    # 🇩🇪 Germany (XETRA)
    "🇩🇪 BMW.DE  — BMW Group": "BMW.DE",
    "🇩🇪 SAP.DE  — SAP SE": "SAP.DE",
    "🇩🇪 BAYN.DE — Bayer AG": "BAYN.DE",
    # 🇯🇵 Japan (TSE)
    "🇯🇵 7203.T — Toyota Motor": "7203.T",
    "🇯🇵 6758.T — Sony Group": "6758.T",
    # 🇨🇳 China (HK/Shanghai)
    "🇨🇳 0700.HK — Tencent Holdings": "0700.HK",
    "🇨🇳 9988.HK — Alibaba Group": "9988.HK",
    # 🇮🇳 India (NSE)
    "🇮🇳 RELIANCE.NS — Reliance Industries": "RELIANCE.NS",
    "🇮🇳 TCS.NS      — Tata Consultancy": "TCS.NS",
    # 🇧🇷 Brazil (B3)
    "🇧🇷 PETR4.SA — Petrobras": "PETR4.SA",
    "🇧🇷 VALE3.SA — Vale SA": "VALE3.SA",
}

POPULAR_CRYPTO = {
    "₿  BTC-USD  — Bitcoin": "BTC-USD",
    "Ξ  ETH-USD  — Ethereum": "ETH-USD",
    "🟡 BNB-USD  — BNB (Binance)": "BNB-USD",
    "◎  SOL-USD  — Solana": "SOL-USD",
    "✕  XRP-USD  — Ripple": "XRP-USD",
    "🐕 DOGE-USD — Dogecoin": "DOGE-USD",
    "🔷 ADA-USD  — Cardano": "ADA-USD",
    "🔺 AVAX-USD — Avalanche": "AVAX-USD",
    "🔵 MATIC-USD— Polygon": "MATIC-USD",
    "🟣 DOT-USD  — Polkadot": "DOT-USD",
    "🔗 LINK-USD — Chainlink": "LINK-USD",
    "Ł  LTC-USD  — Litecoin": "LTC-USD",
}

PERIOD_OPTIONS = {
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
}


# ─────────────────────────────────────────────
#  MAIN FETCHER
# ─────────────────────────────────────────────

def fetch_market_data(ticker: str, period: str = "2y") -> dict:
    """
    Fetch OHLCV data for any ticker (stock or crypto).

    Returns
    -------
    dict with keys:
        df       : pd.DataFrame  — OHLCV data
        name     : str           — full asset name
        currency : str
        sector   : str
        market   : str
        error    : str | None
    """
    result = {
        "df": None,
        "name": ticker,
        "currency": "USD",
        "sector": "—",
        "market": "—",
        "error": None,
    }

    try:
        t = yf.Ticker(ticker.upper().strip())
        df = t.history(period=period, auto_adjust=True)

        if df.empty:
            result["error"] = (
                f"❌ No data found for '{ticker}'. "
                "Check the ticker symbol and try again.\n\n"
                "Examples: AAPL, BTC-USD, 2222.SR, HSBA.L"
            )
            return result

        # Keep standard columns only
        for col in ["Dividends", "Stock Splits", "Capital Gains"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        # Fetch metadata (non-critical)
        try:
            info = t.info
            result["name"]     = info.get("longName") or info.get("shortName") or ticker
            result["currency"] = info.get("currency", "USD")
            result["sector"]   = info.get("sector") or info.get("quoteType", "—")
            result["market"]   = info.get("exchange", "—")
        except Exception:
            pass  # metadata is optional

        result["df"] = df

    except Exception as e:
        result["error"] = f"❌ Error fetching data: {str(e)}"

    return result


def get_market_summary(df: pd.DataFrame, currency: str = "USD") -> dict:
    """
    Compute quick summary statistics from the price dataframe.
    """
    close = df["Close"]
    latest   = close.iloc[-1]
    prev     = close.iloc[-2] if len(close) > 1 else latest
    change   = latest - prev
    pct      = (change / prev) * 100

    high_52w = close.tail(252).max()
    low_52w  = close.tail(252).min()

    avg_vol  = df["Volume"].tail(30).mean()

    return {
        "price"    : latest,
        "change"   : change,
        "pct"      : pct,
        "high_52w" : high_52w,
        "low_52w"  : low_52w,
        "avg_vol"  : avg_vol,
        "currency" : currency,
        "data_pts" : len(df),
        "from_date": df.index[0].strftime("%Y-%m-%d"),
        "to_date"  : df.index[-1].strftime("%Y-%m-%d"),
    }
