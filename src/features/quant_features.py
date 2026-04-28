"""퀀트 수치 피처 — 기술적 지표 연속값.

ta 라이브러리(pandas-ta 대체) 사용.
총 17개 피처:
    모멘텀(5) + 변동성(4) + 가격파생(5) + 거래량(3)
"""
import numpy as np
import pandas as pd
import ta


# ─────────────────────────────────────────────
# 모멘텀 (5개)
# ─────────────────────────────────────────────

def _momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd_hist"] = macd.macd_diff()

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["cci_20"] = ta.trend.CCIIndicator(high, low, close, window=20).cci()

    return df


# ─────────────────────────────────────────────
# 변동성 (4개)
# ─────────────────────────────────────────────

def _volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_pct_b"] = bb.bollinger_pband()
    df["bb_width"] = bb.bollinger_wband()

    df["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # 20일 수익률 표준편차 (연율화 없이 일별 기준)
    daily_ret = close.pct_change()
    df["volatility_20"] = daily_ret.rolling(20).std()

    return df


# ─────────────────────────────────────────────
# 가격 파생 (5개)
# ─────────────────────────────────────────────

def _price_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]

    df["return_5d"]  = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)
    df["return_60d"] = close.pct_change(60)

    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    df["price_vs_ma20"] = (close / ma20.replace(0, np.nan)) - 1
    df["price_vs_ma60"] = (close / ma60.replace(0, np.nan)) - 1

    return df


# ─────────────────────────────────────────────
# 거래량 (3개)
# ─────────────────────────────────────────────

def _volume_features(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["close"]
    volume = df["volume"]

    vol_ma20 = volume.rolling(20).mean().replace(0, np.nan)
    df["volume_ratio_20"] = volume / vol_ma20

    obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["obv_trend"] = (obv - obv.shift(5)) / (obv.shift(5).abs().replace(0, np.nan))

    df["volume_surge"] = (volume >= vol_ma20 * 2).astype("int8")

    return df


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

def compute_quant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV DataFrame을 받아 수치 피처를 추가해 반환.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    df = _momentum_features(df)
    df = _volatility_features(df)
    df = _price_features(df)
    df = _volume_features(df)
    return df


QUANT_FEATURE_COLS = [
    # 모멘텀
    "rsi_14", "macd_hist", "stoch_k", "stoch_d", "cci_20",
    # 변동성
    "bb_pct_b", "bb_width", "atr_14", "volatility_20",
    # 가격 파생
    "return_5d", "return_20d", "return_60d", "price_vs_ma20", "price_vs_ma60",
    # 거래량
    "volume_ratio_20", "obv_trend", "volume_surge",
]
