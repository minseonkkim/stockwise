"""API 공유 의존성 — 예측 DataFrame을 한 번만 로드해 캐싱."""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import pandas as pd

FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("data/models")
REPORTS_DIR = Path("data/reports")
N_TOP = 20  # 백테스트와 동일한 Top-N 기본값


@lru_cache(maxsize=1)
def get_predictions_df() -> pd.DataFrame:
    """
    test 스플릿 전체 예측 결과를 한 번만 생성해 메모리에 캐시.

    반환 컬럼: [date, ticker, prob, close, actual_return_5d, signal, rank]
    signal : Top-N 방식 — 날짜별 prob 상위 N_TOP 개만 "buy", 나머지 "hold"
    """
    # ── 1. 피처 & 모델 로드 ──────────────────────────────────────
    feature_cols_path = FEATURES_DIR / "feature_cols.pkl"
    model_path = MODELS_DIR / "xgb_model.pkl"

    if not feature_cols_path.exists():
        raise FileNotFoundError("build_features.py 를 먼저 실행하세요.")
    if not model_path.exists():
        raise FileNotFoundError("train_model.py 를 먼저 실행하세요.")

    with open(feature_cols_path, "rb") as f:
        feature_cols: list[str] = pickle.load(f)

    with open(model_path, "rb") as f:
        data = pickle.load(f)
    model = data["model"]

    # ── 2. test 파티션 로드 ───────────────────────────────────────
    test_path = FEATURES_DIR / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError("build_features.py 를 먼저 실행하세요.")

    df = pd.read_parquet(test_path)
    df["date"] = pd.to_datetime(df["date"])

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0).replace([float("inf"), float("-inf")], 0)

    # ── 3. 예측 ──────────────────────────────────────────────────
    df["prob"] = model.predict_proba(X)[:, 1]

    # ── 4. 날짜별 Top-N 신호 분류 (PLAN.md Phase 5 명시 방식) ────
    df = df.sort_values(["date", "prob"], ascending=[True, False])
    df["rank"] = df.groupby("date")["prob"].rank(ascending=False, method="first").astype(int)
    df["signal"] = df["rank"].apply(lambda r: "buy" if r <= N_TOP else "hold")

    # ── 5. 필요 컬럼만 반환 ──────────────────────────────────────
    keep = ["date", "ticker", "prob", "signal", "rank"]
    if "close" in df.columns:
        keep.append("close")

    return df[keep].reset_index(drop=True)
