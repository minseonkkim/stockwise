"""횡단면 정규화 (Cross-Sectional Normalization).

왜 필요한가:
    RSI 70은 강세장에서 정상이지만 약세장에서는 과매수.
    절대값 대신 "오늘 전체 종목 중 상대적 위치"로 변환하면
    시장 국면과 무관하게 일관된 신호를 학습할 수 있다.

방법:
    날짜별 groupby → 백분위 순위(0~1)로 변환
    0.0 = 당일 최하위, 1.0 = 당일 최상위
    ex) rsi_14_cs = 0.85  →  오늘 전체 종목 중 RSI 상위 15%

대상:
    연속형 피처 19개만 정규화.
    이진형 피처(0/1 패턴) 19개는 변환하지 않음.
"""
import pandas as pd

# 이진/범주형 피처 — 정규화 대상 제외
BINARY_FEATURES = {
    "golden_cross", "dead_cross", "ma_alignment_bull", "ma_alignment_bear",
    "cdl_hammer", "cdl_doji", "cdl_engulfing",
    "cdl_morning_star", "cdl_evening_star", "cdl_three_soldiers", "cdl_three_crows",
    "resistance_breakout", "near_52w_high", "near_52w_low",
    "head_and_shoulders", "double_top", "double_bottom", "triangle_convergence",
    "volume_surge",
}

# 연속형 피처 — 정규화 대상
CONTINUOUS_FEATURES = [
    # 차트 연속
    "trend_slope_20", "support_distance_pct", "resistance_distance_pct",
    # 모멘텀
    "rsi_14", "macd_hist", "stoch_k", "stoch_d", "cci_20",
    # 변동성
    "bb_pct_b", "bb_width", "atr_14", "volatility_20",
    # 가격 파생
    "return_5d", "return_20d", "return_60d", "price_vs_ma20", "price_vs_ma60",
    # 거래량
    "volume_ratio_20", "obv_trend",
]


def cross_sectional_normalize(
    df: pd.DataFrame,
    features: list[str] | None = None,
    min_stocks_per_day: int = 10,
) -> pd.DataFrame:
    """날짜별 횡단면 백분위 순위 변환.

    Args:
        df: ticker, date, 피처 컬럼을 포함한 DataFrame.
        features: 정규화할 피처 목록. None이면 CONTINUOUS_FEATURES 전체.
        min_stocks_per_day: 날짜별 최소 종목 수. 미만이면 해당 날짜 스킵.

    Returns:
        정규화된 DataFrame (in-place 아님).
    """
    if features is None:
        features = CONTINUOUS_FEATURES

    available = [f for f in features if f in df.columns]
    skipped   = [f for f in features if f not in df.columns]
    if skipped:
        import warnings
        warnings.warn(f"cross_sectional_normalize: 컬럼 없음 — {skipped}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 날짜별 종목 수 확인
    date_counts = df.groupby("date")["ticker"].transform("count")
    sparse_mask = date_counts < min_stocks_per_day

    for feat in available:
        # 백분위 순위: NaN은 무시하고 순위 계산 (pandas 기본 동작)
        ranked = df.groupby("date")[feat].rank(pct=True, na_option="keep")
        # 종목 수 부족한 날짜는 원래 값 유지
        df[feat] = ranked.where(~sparse_mask, df[feat])

    return df
