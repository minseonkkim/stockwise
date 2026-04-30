"""LightGBM 이진 분류 모델 학습 및 저장/로드."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.utils import logger

MODELS_DIR = Path("data/models")

DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


class LGBMTrainer:
    """LightGBM 이진 분류 모델 학습기."""

    def __init__(self, params: Optional[dict] = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[lgb.LGBMClassifier] = None
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
    ) -> lgb.LGBMClassifier:
        self.feature_cols = X_train.columns.tolist()

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        if pos > 0:
            self.params["scale_pos_weight"] = round(neg / pos, 2)
            logger.info(f"[LGBM] scale_pos_weight = {self.params['scale_pos_weight']}")

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(50),
            ],
        )

        self.train_accuracy = accuracy_score(y_train, self.model.predict(X_train))
        self.valid_accuracy = accuracy_score(y_valid, self.model.predict(X_valid))

        logger.info(
            f"[LGBM] Train acc: {self.train_accuracy:.4f} | "
            f"Valid acc: {self.valid_accuracy:.4f} | "
            f"Gap: {abs(self.train_accuracy - self.valid_accuracy):.4f}"
        )
        return self.model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("train()을 먼저 실행하세요.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, name: str = "lgbm_model") -> Path:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "params": self.params,
                "feature_cols": self.feature_cols,
                "train_accuracy": self.train_accuracy,
                "valid_accuracy": self.valid_accuracy,
            }, f)
        logger.info(f"[LGBM] Model saved → {path}")
        return path

    @classmethod
    def load(cls, name: str = "lgbm_model") -> "LGBMTrainer":
        path = MODELS_DIR / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"LGBM model not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        trainer = cls()
        trainer.model = data["model"]
        trainer.params = data["params"]
        trainer.feature_cols = data["feature_cols"]
        trainer.train_accuracy = data["train_accuracy"]
        trainer.valid_accuracy = data["valid_accuracy"]
        logger.info(f"[LGBM] Model loaded ← {path}")
        return trainer
