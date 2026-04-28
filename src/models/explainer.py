"""SHAP 기반 XGBoost 피처 중요도 분석.

분석 결과:
  - 피처별 평균 |SHAP| 값 (전역 중요도)
  - 하위 10% 제거 후보 목록
  - summary_plot PNG 저장 (선택)
"""
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap

from src.utils import logger

REPORTS_DIR = Path("data/reports")
MODELS_DIR = Path("data/models")


class SHAPExplainer:
    """TreeExplainer 기반 SHAP 분석기."""

    def __init__(self, bottom_pct: float = 0.10):
        """
        Args:
            bottom_pct: 중요도 하위 비율 (기본 10%). 이 비율 이하 피처 제거 추천.
        """
        self.bottom_pct = bottom_pct
        self.explainer: Optional[shap.TreeExplainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self.low_importance_features: list[str] = []

    def fit_transform(
        self,
        model,
        X: pd.DataFrame,
        sample_size: int = 5000,
    ) -> pd.DataFrame:
        """SHAP 값 계산. feature_importance DataFrame 반환.

        Args:
            model: 학습된 XGBClassifier.
            X: 분석할 데이터셋 (valid 권장).
            sample_size: SHAP 계산 샘플 수 (메모리 절약).

        Returns:
            feature_importance DataFrame (feature, mean_abs_shap, rank, low_importance)
        """
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
            logger.info(f"SHAP 샘플링: {len(X):,} → {sample_size:,}행")
        else:
            X_sample = X

        logger.info("SHAP TreeExplainer 초기화 중...")
        self.explainer = shap.TreeExplainer(model)

        logger.info("SHAP 값 계산 중... (수 분 소요될 수 있음)")
        self.shap_values = self.explainer.shap_values(X_sample)

        # 평균 |SHAP| 피처 중요도
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        threshold = np.percentile(mean_abs, self.bottom_pct * 100)

        self.feature_importance = pd.DataFrame({
            "feature": X_sample.columns.tolist(),
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        self.feature_importance["rank"] = range(1, len(self.feature_importance) + 1)
        self.feature_importance["low_importance"] = (
            self.feature_importance["mean_abs_shap"] <= threshold
        )

        self.low_importance_features = self.feature_importance.loc[
            self.feature_importance["low_importance"], "feature"
        ].tolist()

        logger.info(f"하위 {self.bottom_pct:.0%} 제거 후보 ({len(self.low_importance_features)}개): "
                    f"{self.low_importance_features}")

        self._log_top_features()
        return self.feature_importance

    def _log_top_features(self, n: int = 15) -> None:
        if self.feature_importance is None:
            return
        top = self.feature_importance.head(n)
        logger.info(f"=== Top {n} 피처 (SHAP 중요도) ===")
        for _, row in top.iterrows():
            bar = "█" * int(row["mean_abs_shap"] / self.feature_importance["mean_abs_shap"].max() * 20)
            logger.info(f"  {row['rank']:2d}. {row['feature']:<30s} {row['mean_abs_shap']:.5f} {bar}")

    def save_importance(self, name: str = "shap_importance") -> Path:
        """피처 중요도 CSV + JSON 저장."""
        if self.feature_importance is None:
            raise RuntimeError("fit_transform()을 먼저 실행하세요.")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        csv_path = REPORTS_DIR / f"{name}.csv"
        self.feature_importance.to_csv(csv_path, index=False)

        json_path = REPORTS_DIR / f"{name}.json"
        result = {
            "low_importance_features": self.low_importance_features,
            "bottom_pct": self.bottom_pct,
            "features": self.feature_importance.to_dict(orient="records"),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"SHAP importance saved → {csv_path}")
        return csv_path

    def save_shap_values(self, name: str = "shap_values") -> Path:
        """SHAP 값 numpy 배열 저장 (재사용 목적)."""
        if self.shap_values is None:
            raise RuntimeError("fit_transform()을 먼저 실행하세요.")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.shap_values, f)
        logger.info(f"SHAP values saved → {path}")
        return path

    def get_filtered_features(self, feature_cols: list[str]) -> list[str]:
        """하위 피처 제거 후 남은 피처 목록 반환."""
        return [f for f in feature_cols if f not in self.low_importance_features]

    def try_save_plot(self, X: pd.DataFrame, name: str = "shap_summary") -> Optional[Path]:
        """SHAP summary_plot PNG 저장. matplotlib 없으면 건너뜀."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            path = REPORTS_DIR / f"{name}.png"

            if len(X) > 2000:
                X = X.sample(n=2000, random_state=42)
                sv = self.explainer.shap_values(X)
            else:
                sv = self.shap_values

            shap.summary_plot(sv, X, show=False, max_display=30)
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"SHAP summary plot saved → {path}")
            return path
        except Exception as e:
            logger.warning(f"SHAP plot 저장 실패 (무시): {e}")
            return None
