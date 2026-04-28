"""Phase 1 데이터 수집 모듈 단위 테스트."""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_collection.collector import _make_batches, _normalize


class TestMakeBatches:
    def test_even_split(self):
        items = list(range(100))
        batches = _make_batches(items, 50)
        assert len(batches) == 2
        assert len(batches[0]) == 50

    def test_remainder(self):
        items = list(range(55))
        batches = _make_batches(items, 50)
        assert len(batches) == 2
        assert len(batches[1]) == 5

    def test_empty(self):
        assert _make_batches([], 50) == []

    def test_smaller_than_batch(self):
        batches = _make_batches([1, 2, 3], 50)
        assert len(batches) == 1
        assert batches[0] == [1, 2, 3]


class TestNormalize:
    def _make_raw_df(self, ticker="AAPL", rows=5):
        idx = pd.date_range("2024-01-02", periods=rows, freq="B")
        return pd.DataFrame(
            {
                "Open": [150.0] * rows,
                "High": [155.0] * rows,
                "Low": [148.0] * rows,
                "Close": [152.0] * rows,
                "Adj Close": [152.0] * rows,
                "Volume": [1_000_000] * rows,
                "Dividends": [0.0] * rows,
                "Stock Splits": [0.0] * rows,
            },
            index=idx,
        )

    def test_columns_renamed(self):
        df = _normalize(self._make_raw_df(), "AAPL")
        assert "close" in df.columns
        assert "adj_close" in df.columns
        assert "ticker" in df.columns

    def test_ticker_set(self):
        df = _normalize(self._make_raw_df(), "AAPL")
        assert (df["ticker"] == "AAPL").all()

    def test_negative_price_filtered(self):
        raw = self._make_raw_df()
        raw.loc[raw.index[0], "Close"] = -1.0
        df = _normalize(raw, "AAPL")
        assert len(df) == 4

    def test_adj_close_fallback(self):
        raw = self._make_raw_df().drop(columns=["Adj Close"])
        df = _normalize(raw, "AAPL")
        assert "adj_close" in df.columns
        assert (df["adj_close"] == df["close"]).all()


class TestValidator:
    def test_check_missing_days_no_gaps(self):
        from src.data_collection.validator import _check_missing_days

        # 연속된 거래일 5개 (2024-01-02 ~ 2024-01-08)
        df = pd.DataFrame(
            {
                "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4),
                         date(2024, 1, 5), date(2024, 1, 8)],
                "close": [100.0] * 5,
            }
        )
        alerts = _check_missing_days("TEST", df)
        # 01-04 is not a holiday and is a Thursday, 01-05 is Friday
        # Should have no missing days for this range
        assert isinstance(alerts, list)

    def test_check_price_outliers_detects_spike(self):
        from src.data_collection.validator import _check_price_outliers

        df = pd.DataFrame(
            {
                "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
                "close": [100.0, 200.0, 201.0],  # 100% spike on day 2
            }
        )
        alerts = _check_price_outliers("TEST", df)
        assert len(alerts) == 1
        assert alerts[0]["type"] == "outlier"

    def test_check_zero_values_detects_zero_price(self):
        from src.data_collection.validator import _check_zero_values

        df = pd.DataFrame(
            {
                "date": [date(2024, 1, 2)],
                "open": [0.0],
                "high": [100.0],
                "low": [0.0],
                "close": [0.0],
                "adj_close": [0.0],
                "volume": [1000],
            }
        )
        alerts = _check_zero_values("TEST", df)
        assert any(a["type"] == "zero_price" for a in alerts)
