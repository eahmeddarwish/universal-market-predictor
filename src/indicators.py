"""
indicators.py
=============
Technical Analysis Indicators for Market Prediction

Computed indicators:
  - RSI (Relative Strength Index)
  - MACD + Signal Line + Histogram
  - Bollinger Bands (Upper / Middle / Lower)
  - Moving Averages: MA20, MA50, MA200
  - ATR (Average True Range)
  - Volume SMA
  - On-Balance Volume (OBV)

Author : Ahmed Darwish
Email  : eahmeddarwish@gmail.com
GitHub : github.com/eahmeddarwish
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
#  INDIVIDUAL INDICATORS
# ─────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return rsi.rename("RSI")


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return pd.DataFrame({
        "MACD"      : macd_line,
        "MACD_Signal": signal_line,
        "MACD_Hist" : histogram,
    })


def compute_bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    mid   = close.rolling(window=period, min_periods=period).mean()
    std   = close.rolling(window=period, min_periods=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    width = (upper - lower) / mid  # Bandwidth — useful feature
    return pd.DataFrame({
        "BB_Upper" : upper,
        "BB_Mid"   : mid,
        "BB_Lower" : lower,
        "BB_Width" : width,
    })


def compute_moving_averages(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "MA_20" : close.rolling(20,  min_periods=20).mean(),
        "MA_50" : close.rolling(50,  min_periods=50).mean(),
        "MA_200": close.rolling(200, min_periods=200).mean(),
        "EMA_12": close.ewm(span=12, adjust=False).mean(),
        "EMA_26": close.ewm(span=26, adjust=False).mean(),
    })


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr.rename("ATR")


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    obv       = (direction * volume).cumsum()
    return obv.rename("OBV")


def compute_volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume.rolling(window=period, min_periods=period).mean().rename("Volume_SMA")


def compute_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, d_period: int = 3,
) -> pd.DataFrame:
    lowest_low   = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    stoch_k      = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    stoch_d      = stoch_k.rolling(d_period).mean()
    return pd.DataFrame({"Stoch_K": stoch_k, "Stoch_D": stoch_d})


# ─────────────────────────────────────────────
#  MASTER FUNCTION — adds all indicators to df
# ─────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes an OHLCV DataFrame and returns it with all indicators appended.
    Drops rows with NaN (caused by look-back windows).
    """
    df = df.copy()

    # Core indicators
    df["RSI"]        = compute_rsi(df["Close"])
    df = df.join(compute_macd(df["Close"]))
    df = df.join(compute_bollinger(df["Close"]))
    df = df.join(compute_moving_averages(df["Close"]))
    df["ATR"]        = compute_atr(df["High"], df["Low"], df["Close"])
    df["OBV"]        = compute_obv(df["Close"], df["Volume"])
    df["Volume_SMA"] = compute_volume_sma(df["Volume"])
    df = df.join(compute_stochastic(df["High"], df["Low"], df["Close"]))

    # Derived features
    df["Price_vs_MA20"]  = (df["Close"] - df["MA_20"])  / df["MA_20"]
    df["Price_vs_MA50"]  = (df["Close"] - df["MA_50"])  / df["MA_50"]
    df["Daily_Return"]   = df["Close"].pct_change()
    df["Volatility_10d"] = df["Daily_Return"].rolling(10).std()
    df["Volume_Ratio"]   = df["Volume"] / df["Volume_SMA"]

    # Drop NaN rows introduced by look-back windows (longest = MA200)
    df = df.dropna()

    return df


# ─────────────────────────────────────────────
#  SIGNAL SUMMARY  (human-readable analysis)
# ─────────────────────────────────────────────

def get_signal_summary(df: pd.DataFrame) -> dict:
    """
    Returns a plain-language signal summary based on the latest bar.
    """
    last = df.iloc[-1]
    signals = {}

    # RSI
    rsi = last.get("RSI", 50)
    if rsi < 30:
        signals["RSI"] = f"🟢 Oversold ({rsi:.1f}) — potential buy zone"
    elif rsi > 70:
        signals["RSI"] = f"🔴 Overbought ({rsi:.1f}) — potential sell zone"
    else:
        signals["RSI"] = f"🟡 Neutral ({rsi:.1f})"

    # MACD
    macd_val  = last.get("MACD", 0)
    macd_hist = last.get("MACD_Hist", 0)
    if macd_hist > 0:
        signals["MACD"] = f"🟢 Bullish momentum (histogram +{macd_hist:.4f})"
    else:
        signals["MACD"] = f"🔴 Bearish momentum (histogram {macd_hist:.4f})"

    # Bollinger Bands
    close = last["Close"]
    bb_upper = last.get("BB_Upper", close)
    bb_lower = last.get("BB_Lower", close)
    bb_mid   = last.get("BB_Mid", close)
    if close > bb_upper:
        signals["Bollinger"] = "🔴 Price above upper band — overbought"
    elif close < bb_lower:
        signals["Bollinger"] = "🟢 Price below lower band — oversold"
    else:
        pct_b = (close - bb_lower) / (bb_upper - bb_lower + 1e-9) * 100
        signals["Bollinger"] = f"🟡 Inside bands ({pct_b:.0f}% of band)"

    # Moving Averages
    ma20  = last.get("MA_20",  close)
    ma50  = last.get("MA_50",  close)
    ma200 = last.get("MA_200", close)
    if close > ma20 > ma50 > ma200:
        signals["MA Trend"] = "🟢 Strong uptrend (Price > MA20 > MA50 > MA200)"
    elif close < ma20 < ma50 < ma200:
        signals["MA Trend"] = "🔴 Strong downtrend (Price < MA20 < MA50 < MA200)"
    elif close > ma200:
        signals["MA Trend"] = "🟡 Above MA200 — long-term bullish"
    else:
        signals["MA Trend"] = "🟡 Below MA200 — long-term bearish"

    # Overall (simple vote)
    greens = sum(1 for v in signals.values() if v.startswith("🟢"))
    reds   = sum(1 for v in signals.values() if v.startswith("🔴"))
    if greens > reds:
        signals["Overall"] = f"🟢 Bullish bias ({greens} bullish / {reds} bearish signals)"
    elif reds > greens:
        signals["Overall"] = f"🔴 Bearish bias ({greens} bullish / {reds} bearish signals)"
    else:
        signals["Overall"] = "🟡 Mixed / Neutral signals"

    return signals
