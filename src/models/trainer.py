"""XGBoost 분류 모델 학습 및 저장/로드."""
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

from src.utils import logger

MODELS_DIR = Path("data/models")

DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1,
    "eval_metric": "logloss",
    "early_stopping_rounds": 30,
    "random_state": 42,
    "n_jobs": -1,
}


class XGBoostTrainer:
    """XGBoost 이진 분류 모델 학습기."""

    def __init__(self, params: Optional[dict] = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_cols: Optional[list[str]] = None
        self.train_accuracy: Optional[float] = None
        self.valid_accuracy: Optional[float] = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: pd.DataFrame,
        y_valid: np.ndarray,
        early_stopping_rounds: int = 30,
    ) -> xgb.XGBClassifier:
        """모델 학습. early stopping은 valid logloss 기준."""
        self.feature_cols = X_train.columns.tolist()

        # scale_pos_weight: 클래스 불균형 보정
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        if pos > 0:
            self.params["scale_pos_weight"] = round(neg / pos, 2)
            logger.info(f"scale_pos_weight = {self.params['scale_pos_weight']} (neg={neg}, pos={pos})")

        # XGBoost 2.x: early_stopping_rounds는 생성자에 전달
        self.params["early_stopping_rounds"] = early_stopping_rounds
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=50,
        )

        self.train_accuracy = accuracy_score(y_train, self.model.predict(X_train))
        self.valid_accuracy = accuracy_score(y_valid, self.model.predict(X_valid))

        logger.info(
            f"Train acc: {self.train_accuracy:.4f} | Valid acc: {self.valid_accuracy:.4f} "
            f"| Gap: {abs(self.train_accuracy - self.valid_accuracy):.4f}"
        )
        return self.model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """클래스 1(상승)의 확률 반환."""
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. train()을 먼저 실행하세요.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, name: str = "xgb_model") -> Path:
        """모델 + 메타정보 저장."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = MODELS_DIR / f"{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "params": self.params,
                "feature_cols": self.feature_cols,
                "train_accuracy": self.train_accuracy,
                "valid_accuracy": self.valid_accuracy,
            }, f)

        logger.info(f"Model saved → {model_path}")
        return model_path

    @classmethod
    def load(cls, name: str = "xgb_model") -> "XGBoostTrainer":
        """저장된 모델 로드."""
        model_path = MODELS_DIR / f"{name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        trainer = cls()
        trainer.model = data["model"]
        trainer.params = data["params"]
        trainer.feature_cols = data["feature_cols"]
        trainer.train_accuracy = data["train_accuracy"]
        trainer.valid_accuracy = data["valid_accuracy"]

        logger.info(f"Model loaded ← {model_path}")
        return trainer
