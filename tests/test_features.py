"""Phase 2 피처 모듈 단위 테스트."""
import numpy as np
import pandas as pd
import pytest

from src.features.chart_features import compute_chart_features, CHART_FEATURE_COLS
from src.features.quant_features import compute_quant_features, QUANT_FEATURE_COLS
from src.features.target import add_target


def _make_ohlcv(n: int = 120) -> pd.DataFrame:
    """테스트용 OHLCV 데이터 생성."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    close = 100 * (1 + np.random.randn(n) * 0.01).cumprod()
    df = pd.DataFrame({
        "date":   dates,
        "open":   close * (1 + np.random.randn(n) * 0.003),
        "high":   close * (1 + np.abs(np.random.randn(n)) * 0.005),
        "low":    close * (1 - np.abs(np.random.randn(n)) * 0.005),
        "close":  close,
        "adj_close": close,
        "volume": np.random.randint(500_000, 5_000_000, n).astype("int64"),
    })
    return df


class TestChartFeatures:
    def test_returns_all_columns(self):
        df = compute_chart_features(_make_ohlcv())
        for col in CHART_FEATURE_COLS:
            assert col in df.columns, f"Missing: {col}"

    def test_golden_dead_cross_binary(self):
        df = compute_chart_features(_make_ohlcv())
        assert df["golden_cross"].isin([0, 1]).all()
        assert df["dead_cross"].isin([0, 1]).all()

    def test_no_simultaneous_golden_dead(self):
        df = compute_chart_features(_make_ohlcv())
        # 같은 날 골든크로스와 데드크로스가 동시에 발생할 수 없다
        assert not ((df["golden_cross"] == 1) & (df["dead_cross"] == 1)).any()

    def test_near_52w_high_binary(self):
        df = compute_chart_features(_make_ohlcv())
        assert df["near_52w_high"].isin([0, 1]).all()

    def test_engulfing_values(self):
        df = compute_chart_features(_make_ohlcv())
        assert df["cdl_engulfing"].isin([-1, 0, 1]).all()


class TestQuantFeatures:
    def test_returns_all_columns(self):
        df = compute_quant_features(_make_ohlcv())
        for col in QUANT_FEATURE_COLS:
            assert col in df.columns, f"Missing: {col}"

    def test_rsi_range(self):
        df = compute_quant_features(_make_ohlcv())
        valid = df["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_volume_surge_binary(self):
        df = compute_quant_features(_make_ohlcv())
        assert df["volume_surge"].isin([0, 1]).all()

    def test_no_inf_values(self):
        df = compute_quant_features(_make_ohlcv())
        for col in QUANT_FEATURE_COLS:
            assert not np.isinf(df[col].dropna()).any(), f"Inf in {col}"


class TestTarget:
    def test_target_binary(self):
        df = _make_ohlcv(50)
        df = add_target(df)
        valid = df["target"].dropna()
        assert valid.isin([0.0, 1.0]).all()

    def test_last_rows_nan(self):
        df = _make_ohlcv(50)
        df = add_target(df, horizon=5)
        assert df["target"].iloc[-5:].isna().all()

    def test_sufficient_non_nan(self):
        df = _make_ohlcv(50)
        df = add_target(df, horizon=5)
        assert df["target"].notna().sum() == 45


class TestFullPipeline:
    def test_combined_features(self):
        df = _make_ohlcv(150)
        df = compute_chart_features(df)
        df = compute_quant_features(df)
        df = add_target(df)

        all_cols = CHART_FEATURE_COLS + QUANT_FEATURE_COLS + ["target"]
        for col in all_cols:
            assert col in df.columns

    def test_no_future_leakage(self):
        """target이 미래 정보를 사용하므로 마지막 5행은 NaN이어야 한다."""
        df = _make_ohlcv(100)
        df = add_target(df, horizon=5)
        assert df["target"].iloc[-5:].isna().all()
        assert df["target"].iloc[:-5].notna().any()
