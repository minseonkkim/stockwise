"""모델 성능 평가 모듈.

PLAN.md Phase 3 성능 기준:
  - 검증 정확도 55% 이상
  - train - valid 정확도 차이 5% 이내 (과적합 방지)
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import logger

REPORTS_DIR = Path("data/reports")


class ModelEvaluator:
    """XGBoost 모델 평가기."""

    # Phase 3 목표 기준
    MIN_VALID_ACCURACY = 0.55
    MAX_OVERFITTING_GAP = 0.05

    def __init__(self):
        self.results: dict = {}

    def evaluate(
        self,
        model,
        X: pd.DataFrame,
        y: np.ndarray,
        split_name: str = "valid",
        threshold: float = 0.5,
    ) -> dict:
        """단일 split 평가. 결과 딕셔너리 반환."""
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            "split": split_name,
            "n_samples": len(y),
            "n_positive": int(y.sum()),
            "positive_rate": round(float(y.mean()), 4),
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y, y_prob), 4) if len(np.unique(y)) > 1 else None,
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "threshold": threshold,
        }

        self.results[split_name] = metrics
        self._log_metrics(metrics)
        return metrics

    def evaluate_all(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: pd.DataFrame,
        y_valid: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        threshold: float = 0.5,
    ) -> dict:
        """train / valid / test 전체 평가."""
        self.evaluate(model, X_train, y_train, "train", threshold)
        self.evaluate(model, X_valid, y_valid, "valid", threshold)
        self.evaluate(model, X_test,  y_test,  "test",  threshold)

        self._check_phase3_criteria()
        return self.results

    def _check_phase3_criteria(self) -> None:
        """Phase 3 목표 달성 여부 로그 출력."""
        valid = self.results.get("valid", {})
        train = self.results.get("train", {})

        valid_acc = valid.get("accuracy", 0)
        train_acc = train.get("accuracy", 0)
        gap = train_acc - valid_acc

        ok_acc = valid_acc >= self.MIN_VALID_ACCURACY
        ok_gap = gap <= self.MAX_OVERFITTING_GAP

        logger.info("=" * 50)
        logger.info("[ Phase 3 성능 기준 체크 ]")
        logger.info(
            f"  검증 정확도 55% 이상:  {valid_acc:.4f} → {'✓ PASS' if ok_acc else '✗ FAIL'}"
        )
        logger.info(
            f"  과적합 gap 5% 이내:   {gap:.4f} → {'✓ PASS' if ok_gap else '✗ FAIL'}"
        )
        logger.info("=" * 50)

        if not ok_acc:
            logger.warning("검증 정확도가 목표(55%) 미달입니다. 피처/파라미터를 재검토하세요.")
        if not ok_gap:
            logger.warning("과적합 gap이 5%를 초과했습니다. 정규화 강화 또는 early stopping을 조정하세요.")

    def classification_report_str(self, split: str = "valid") -> str:
        """사람이 읽기 쉬운 분류 리포트 문자열."""
        m = self.results.get(split, {})
        lines = [
            f"=== {split.upper()} 평가 결과 ===",
            f"샘플 수:     {m.get('n_samples', '-'):,}",
            f"Positive:    {m.get('positive_rate', '-'):.2%}",
            f"Accuracy:    {m.get('accuracy', '-'):.4f}",
            f"Precision:   {m.get('precision', '-'):.4f}",
            f"Recall:      {m.get('recall', '-'):.4f}",
            f"F1 Score:    {m.get('f1', '-'):.4f}",
            f"ROC AUC:     {m.get('roc_auc', '-')}",
            f"Threshold:   {m.get('threshold', 0.5)}",
        ]
        cm = m.get("confusion_matrix")
        if cm:
            lines += [
                "Confusion Matrix (pred↓ / actual→):",
                f"  TN={cm[0][0]:>6,}  FP={cm[0][1]:>6,}",
                f"  FN={cm[1][0]:>6,}  TP={cm[1][1]:>6,}",
            ]
        return "\n".join(lines)

    def save_report(self, name: str = "evaluation") -> Path:
        """평가 결과 JSON 저장."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation report saved → {path}")
        return path

    def _log_metrics(self, m: dict) -> None:
        logger.info(
            f"[{m['split'].upper():5s}] "
            f"acc={m['accuracy']:.4f} | "
            f"prec={m['precision']:.4f} | "
            f"rec={m['recall']:.4f} | "
            f"f1={m['f1']:.4f} | "
            f"auc={m.get('roc_auc', 'N/A')}"
        )
