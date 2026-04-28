"""모델 예측 생성 — 백테스트용 예측 확률 DataFrame 생성."""
from pathlib import Path

import pandas as pd

from src.features.dataset import load_dataset
from src.features.pipeline import FEATURES_DIR
from src.models.trainer import XGBoostTrainer
from src.utils import logger

FEATURES_ALL = FEATURES_DIR / "features_all.parquet"


def generate_predictions(
    model_name: str = "xgb_model",
    split: str = "test",
) -> pd.DataFrame:
    """학습된 모델로 예측 확률 DataFrame 생성.

    Returns:
        DataFrame with columns: [ticker, date, prob, actual_return_5d, close]
          - prob             : 모델 예측 확률 (상승 P(y=1))
          - actual_return_5d : 실제 5거래일 수익률 (백테스트 수익 계산용)
          - close            : 매수 기준가
    """
    logger.info(f"예측 생성 중... (model={model_name}, split={split})")

    # ─── 1. 모델 로드 ─────────────────────────────────────────
    trainer = XGBoostTrainer.load(model_name)

    # ─── 2. 스케일된 피처 + 메타 로드 ────────────────────────
    dataset = load_dataset()
    X = dataset[f"X_{split}"]
    meta = dataset[f"meta_{split}"].copy()

    # ─── 3. 예측 ──────────────────────────────────────────────
    meta["prob"] = trainer.predict_proba(X)
    meta["date"] = pd.to_datetime(meta["date"])

    # ─── 4. 실제 수익률 (이진 타깃이 아닌 연속값) 붙이기 ─────
    raw = pd.read_parquet(FEATURES_ALL, columns=["ticker", "date", "close"])
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["ticker", "date"])

    # 실제 5거래일 후 수익률
    raw["actual_return_5d"] = raw.groupby("ticker")["close"].transform(
        lambda s: s.shift(-5) / s - 1
    )

    pred_df = meta.merge(
        raw[["ticker", "date", "close", "actual_return_5d"]],
        on=["ticker", "date"],
        how="left",
    )

    # 미래 가격 없는 행(마지막 5거래일) 제거
    pred_df = pred_df.dropna(subset=["actual_return_5d"])

    logger.info(
        f"예측 완료: {len(pred_df):,}행 | "
        f"{pred_df['ticker'].nunique()}개 종목 | "
        f"{pred_df['date'].min().date()} ~ {pred_df['date'].max().date()}"
    )
    return pred_df
