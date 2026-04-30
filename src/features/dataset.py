"""ML 데이터셋 생성 — train/valid/test 분리 + 스케일링.

시간 순서 엄수 (미래 데이터 누수 방지):
    train : date < 2021-01-01
    valid : 2021-01-01 <= date < 2023-01-01
    test  : date >= 2023-01-01

핵심 규칙: StandardScaler는 train 기준으로만 fit,
          valid / test는 transform만 적용.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.chart_features import CHART_FEATURE_COLS
from src.features.quant_features import QUANT_FEATURE_COLS
from src.features.sector_features import SECTOR_FEATURE_COLS
from src.utils import logger

FEATURES_DIR = Path("data/features")
ALL_FEATURE_COLS = CHART_FEATURE_COLS + QUANT_FEATURE_COLS + SECTOR_FEATURE_COLS

TRAIN_END  = "2021-01-01"
VALID_END  = "2023-01-01"


def build_dataset(df: pd.DataFrame) -> dict:
    """
    전체 피처 DataFrame → train/valid/test 분리 딕셔너리 반환.

    Returns:
        {
            "X_train": pd.DataFrame, "y_train": pd.Series,
            "X_valid": pd.DataFrame, "y_valid": pd.Series,
            "X_test":  pd.DataFrame, "y_test":  pd.Series,
            "scaler":  StandardScaler,
            "feature_cols": list[str],
        }
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 유효한 피처 컬럼만 선택
    available = [c for c in ALL_FEATURE_COLS if c in df.columns]
    missing   = [c for c in ALL_FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"Missing feature cols: {missing}")

    # target NaN 제거
    df = df.dropna(subset=["target"])

    # 시간 순서 분리
    train_df = df[df["date"] <  TRAIN_END].copy()
    valid_df = df[(df["date"] >= TRAIN_END) & (df["date"] < VALID_END)].copy()
    test_df  = df[df["date"] >= VALID_END].copy()

    logger.info(
        f"Split sizes - train: {len(train_df):,}, valid: {len(valid_df):,}, test: {len(test_df):,}"
    )

    # Scaler: train 기준 fit
    scaler = StandardScaler()

    X_train = _prepare_X(train_df, available)
    X_valid = _prepare_X(valid_df, available)
    X_test  = _prepare_X(test_df,  available)

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=available, index=X_train.index
    )
    X_valid_scaled = pd.DataFrame(
        scaler.transform(X_valid), columns=available, index=X_valid.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=available, index=X_test.index
    )

    return {
        "X_train": X_train_scaled,
        "y_train": train_df["target"].values,
        "X_valid": X_valid_scaled,
        "y_valid": valid_df["target"].values,
        "X_test":  X_test_scaled,
        "y_test":  test_df["target"].values,
        "scaler":  scaler,
        "feature_cols": available,
        "meta_train": train_df[["ticker", "date"]].reset_index(drop=True),
        "meta_valid": valid_df[["ticker", "date"]].reset_index(drop=True),
        "meta_test":  test_df[["ticker", "date"]].reset_index(drop=True),
    }


def save_dataset(dataset: dict) -> None:
    """parquet + pickle로 저장."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        X = dataset[f"X_{split}"]
        y = dataset[f"y_{split}"]
        meta = dataset[f"meta_{split}"]

        combined = meta.copy()
        combined[X.columns.tolist()] = X.values
        combined["target"] = y
        combined.to_parquet(FEATURES_DIR / f"{split}.parquet", index=False, compression="snappy")

    with open(FEATURES_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(dataset["scaler"], f)

    with open(FEATURES_DIR / "feature_cols.pkl", "wb") as f:
        pickle.dump(dataset["feature_cols"], f)

    logger.info(f"Dataset saved to {FEATURES_DIR}/")


def load_dataset() -> dict:
    """저장된 데이터셋 로드."""
    result = {}
    for split in ("train", "valid", "test"):
        path = FEATURES_DIR / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Run build_features.py first: {path}")
        df = pd.read_parquet(path)

        with open(FEATURES_DIR / "feature_cols.pkl", "rb") as f:
            feature_cols = pickle.load(f)

        result[f"X_{split}"] = df[feature_cols]
        result[f"y_{split}"] = df["target"].values
        result[f"meta_{split}"] = df[["ticker", "date"]]

    with open(FEATURES_DIR / "scaler.pkl", "rb") as f:
        result["scaler"] = pickle.load(f)

    result["feature_cols"] = feature_cols
    return result


def _prepare_X(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """피처 선택 + Inf/-Inf → NaN → 0 처리."""
    X = df[cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    return X
