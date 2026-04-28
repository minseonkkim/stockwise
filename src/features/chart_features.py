"""차트 기법 피처 — 패턴 인코딩 (0/1).

TA-Lib 없이 순수 pandas/numpy/scipy로 구현.
총 19개 피처:
    추세(5) + 캔들(7) + 지지저항(5) + 차트패턴(4) - 일부 복잡 패턴 포함
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ─────────────────────────────────────────────
# 추세 패턴 피처 (5개)
# ─────────────────────────────────────────────

def _trend_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]

    ma5  = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    df["golden_cross"]      = ((ma5 > ma20) & (ma5.shift(1) <= ma20.shift(1))).astype("int8")
    df["dead_cross"]        = ((ma5 < ma20) & (ma5.shift(1) >= ma20.shift(1))).astype("int8")
    df["ma_alignment_bull"] = ((ma5 > ma20) & (ma20 > ma60)).astype("int8")
    df["ma_alignment_bear"] = ((ma5 < ma20) & (ma20 < ma60)).astype("int8")

    # 20일선 기울기 (20일 전 대비 변화율)
    df["trend_slope_20"] = (ma20 - ma20.shift(20)) / ma20.shift(20).replace(0, np.nan)

    return df


# ─────────────────────────────────────────────
# 캔들 패턴 피처 (7개)
# ─────────────────────────────────────────────

def _candle_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    body         = (c - o).abs()
    rng          = (h - l).replace(0, np.nan)
    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - l
    body_ratio   = body / rng

    # 망치형: 아래꼬리 >= 2×몸통, 위꼬리 작음, 양봉
    df["cdl_hammer"] = (
        (lower_shadow >= 2 * body) &
        (upper_shadow <= body * 0.5) &
        (body_ratio < 0.4) &
        (c >= o)
    ).astype("int8")

    # 도지: 몸통이 전체 범위의 10% 이내
    df["cdl_doji"] = (body_ratio <= 0.1).astype("int8")

    # 장악형: 현재 몸통이 전일 몸통을 완전히 감싼다
    prev_o, prev_c = o.shift(1), c.shift(1)
    bull_engulf = (c > o) & (prev_c < prev_o) & (c > prev_o) & (o < prev_c)
    bear_engulf = (c < o) & (prev_c > prev_o) & (c < prev_o) & (o > prev_c)
    df["cdl_engulfing"] = np.where(bull_engulf, 1, np.where(bear_engulf, -1, 0)).astype("int8")

    # 샛별형 (3캔들): 음봉 → 도지/소봉 → 양봉
    mid_small = body.shift(1) <= rng.shift(1) * 0.3
    df["cdl_morning_star"] = (
        (c.shift(2) < o.shift(2)) &   # 음봉
        mid_small &                    # 중간: 작은 몸통
        (c > o) &                      # 양봉
        (c > (o.shift(2) + c.shift(2)) / 2)  # 첫 캔들 중간 이상 회복
    ).astype("int8")

    # 저녁별형 (3캔들): 양봉 → 도지/소봉 → 음봉
    df["cdl_evening_star"] = (
        (c.shift(2) > o.shift(2)) &
        mid_small &
        (c < o) &
        (c < (o.shift(2) + c.shift(2)) / 2)
    ).astype("int8")

    # 세 백병 (연속 3 양봉, 각 고점 상승)
    df["cdl_three_soldiers"] = (
        (c > o) & (c.shift(1) > o.shift(1)) & (c.shift(2) > o.shift(2)) &
        (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    ).astype("int8")

    # 세 까마귀 (연속 3 음봉, 각 저점 하락)
    df["cdl_three_crows"] = (
        (c < o) & (c.shift(1) < o.shift(1)) & (c.shift(2) < o.shift(2)) &
        (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    ).astype("int8")

    return df


# ─────────────────────────────────────────────
# 지지/저항 피처 (5개)
# ─────────────────────────────────────────────

def _support_resistance_features(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    close  = df["close"].values
    n      = len(close)

    support_dist    = np.full(n, np.nan)
    resistance_dist = np.full(n, np.nan)
    res_breakout    = np.zeros(n, dtype="int8")

    for i in range(lookback, n):
        window = close[i - lookback : i]
        cur    = close[i]
        prev   = close[i - 1]

        peaks,   _ = find_peaks(window,  distance=5)
        troughs, _ = find_peaks(-window, distance=5)

        if len(troughs):
            sup = window[troughs]
            below = sup[sup <= cur]
            if len(below):
                nearest_sup = below.max()
                support_dist[i] = (cur - nearest_sup) / cur

        if len(peaks):
            res = window[peaks]
            above = res[res >= cur]
            if len(above):
                nearest_res = above.min()
                resistance_dist[i] = (nearest_res - cur) / cur
                # 저항선 돌파: 전일은 아래, 당일 위
                if prev <= nearest_res * 1.005 <= cur:
                    res_breakout[i] = 1

    # 52주(252거래일) 고점/저점 근접
    high_252 = df["close"].rolling(252, min_periods=60).max()
    low_252  = df["close"].rolling(252, min_periods=60).min()

    df["support_distance_pct"]    = support_dist
    df["resistance_distance_pct"] = resistance_dist
    df["resistance_breakout"]     = res_breakout
    df["near_52w_high"]           = (df["close"] >= high_252 * 0.95).astype("int8")
    df["near_52w_low"]            = (df["close"] <= low_252  * 1.05).astype("int8")

    return df


# ─────────────────────────────────────────────
# 차트 패턴 피처 (4개) — scipy peak 기반 간이 감지
# ─────────────────────────────────────────────

def _chart_pattern_features(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    close = df["close"].values
    n     = len(close)

    hs   = np.zeros(n, dtype="int8")
    dtop = np.zeros(n, dtype="int8")
    dbot = np.zeros(n, dtype="int8")
    tri  = np.zeros(n, dtype="int8")

    for i in range(window, n):
        seg = close[i - window : i]

        peaks,   _ = find_peaks(seg,  distance=8)
        troughs, _ = find_peaks(-seg, distance=8)

        # 헤드앤숄더: 고점 3개, 가운데가 최대
        if len(peaks) >= 3:
            p = peaks[-3:]
            vals = seg[p]
            if vals[1] > vals[0] * 1.02 and vals[1] > vals[2] * 1.02 and abs(vals[0] - vals[2]) / vals[1] < 0.05:
                hs[i] = 1

        # 더블탑: 고점 2개가 비슷한 수준
        if len(peaks) >= 2:
            p = peaks[-2:]
            vals = seg[p]
            if abs(vals[0] - vals[1]) / vals.mean() < 0.03 and (p[1] - p[0]) >= 10:
                dtop[i] = 1

        # 더블바텀: 저점 2개가 비슷한 수준
        if len(troughs) >= 2:
            t = troughs[-2:]
            vals = seg[t]
            if abs(vals[0] - vals[1]) / vals.mean() < 0.03 and (t[1] - t[0]) >= 10:
                dbot[i] = 1

        # 삼각수렴: 고점은 하락 + 저점은 상승
        if len(peaks) >= 2 and len(troughs) >= 2:
            ph = np.polyfit(peaks[-2:],  seg[peaks[-2:]],  1)
            pl = np.polyfit(troughs[-2:], seg[troughs[-2:]], 1)
            if ph[0] < -0.01 and pl[0] > 0.01:
                tri[i] = 1

    df["head_and_shoulders"]   = hs
    df["double_top"]           = dtop
    df["double_bottom"]        = dbot
    df["triangle_convergence"] = tri

    return df


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

def compute_chart_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV DataFrame을 받아 차트 기법 피처를 추가해 반환.
    df에는 최소한 open/high/low/close/volume 컬럼 필요.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    df = _trend_features(df)
    df = _candle_features(df)
    df = _support_resistance_features(df)
    df = _chart_pattern_features(df)
    return df


CHART_FEATURE_COLS = [
    # 추세
    "golden_cross", "dead_cross", "ma_alignment_bull", "ma_alignment_bear", "trend_slope_20",
    # 캔들
    "cdl_hammer", "cdl_doji", "cdl_engulfing", "cdl_morning_star", "cdl_evening_star",
    "cdl_three_soldiers", "cdl_three_crows",
    # 지지/저항
    "support_distance_pct", "resistance_distance_pct", "resistance_breakout",
    "near_52w_high", "near_52w_low",
    # 차트패턴
    "head_and_shoulders", "double_top", "double_bottom", "triangle_convergence",
]
