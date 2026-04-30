"""XGBoost + LightGBM 앙상블 — 두 모델의 predict_proba 평균."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.models.lgbm_trainer import LGBMTrainer
from src.models.trainer import XGBoostTrainer
from src.utils import logger

MODELS_DIR = Path("data/models")


class EnsembleTrainer:
    """XGBoost + LightGBM 확률 평균 앙상블."""

    def __init__(
        self,
        xgb_weight: float = 0.5,
        lgbm_weight: float = 0.5,
    ):
        self.xgb_weight = xgb_weight
        self.lgbm_weight = lgbm_weight
        self.xgb: Optional[XGBoostTrainer] = None
        self.lgbm: Optional[LGBMTrainer] = None
        self.valid_accuracy: Optional[float] = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: pd.DataFrame,
        y_valid: np.ndarray,
        xgb_params: Optional[dict] = None,
        lgbm_params: Optional[dict] = None,
    ) -> "EnsembleTrainer":
        logger.info("[Ensemble] XGBoost 학습 중...")
        self.xgb = XGBoostTrainer(params=xgb_params)
        self.xgb.train(X_train, y_train, X_valid, y_valid)

        logger.info("[Ensemble] LightGBM 학습 중...")
        self.lgbm = LGBMTrainer(params=lgbm_params)
        self.lgbm.train(X_train, y_train, X_valid, y_valid)

        # 앙상블 검증 정확도
        proba = self.predict_proba(X_valid)
        preds = (proba >= 0.5).astype(int)
        self.valid_accuracy = accuracy_score(y_valid, preds)

        logger.info(
            f"[Ensemble] Valid acc: {self.valid_accuracy:.4f} "
            f"(XGB: {self.xgb.valid_accuracy:.4f}, LGBM: {self.lgbm.valid_accuracy:.4f})"
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.xgb is None or self.lgbm is None:
            raise RuntimeError("train()을 먼저 실행하세요.")
        xgb_p = self.xgb.predict_proba(X)
        lgbm_p = self.lgbm.predict_proba(X)
        return self.xgb_weight * xgb_p + self.lgbm_weight * lgbm_p

    def save(self, name: str = "ensemble") -> Path:
        """XGBoost, LightGBM 각각 저장 + 앙상블 메타 저장."""
        self.xgb.save(f"{name}_xgb")
        self.lgbm.save(f"{name}_lgbm")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        meta_path = MODELS_DIR / f"{name}_meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump({
                "xgb_weight": self.xgb_weight,
                "lgbm_weight": self.lgbm_weight,
                "valid_accuracy": self.valid_accuracy,
                "xgb_name": f"{name}_xgb",
                "lgbm_name": f"{name}_lgbm",
            }, f)
        logger.info(f"[Ensemble] Meta saved → {meta_path}")
        return meta_path

    @classmethod
    def load(cls, name: str = "ensemble") -> "EnsembleTrainer":
        meta_path = MODELS_DIR / f"{name}_meta.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"Ensemble meta not found: {meta_path}")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        trainer = cls(
            xgb_weight=meta["xgb_weight"],
            lgbm_weight=meta["lgbm_weight"],
        )
        trainer.xgb = XGBoostTrainer.load(meta["xgb_name"])
        trainer.lgbm = LGBMTrainer.load(meta["lgbm_name"])
        trainer.valid_accuracy = meta["valid_accuracy"]
        logger.info(f"[Ensemble] Loaded (valid_acc={trainer.valid_accuracy:.4f})")
        return trainer
