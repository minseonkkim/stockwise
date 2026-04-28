"""Optuna 기반 XGBoost 하이퍼파라미터 자동 최적화.

목표: validation accuracy 최대화.
과적합 방지: train-valid gap 5% 초과 시 패널티 부여.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

from src.utils import logger

MODELS_DIR = Path("data/models")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaTuner:
    """Optuna를 사용한 XGBoost 하이퍼파라미터 탐색."""

    def __init__(self, n_trials: int = 100, gap_penalty: float = 0.5):
        """
        Args:
            n_trials: Optuna 시도 횟수.
            gap_penalty: train-valid 정확도 차이가 5% 초과 시 페널티 가중치.
        """
        self.n_trials = n_trials
        self.gap_penalty = gap_penalty
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[dict] = None

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: pd.DataFrame,
        y_valid: np.ndarray,
    ) -> dict:
        """하이퍼파라미터 탐색 실행. best_params 딕셔너리 반환."""
        logger.info(f"Optuna 시작: {self.n_trials} trials")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "eval_metric": "logloss",
                "early_stopping_rounds": 20,
                "random_state": 42,
                "n_jobs": -1,
            }

            # 클래스 불균형 보정
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            if pos > 0:
                params["scale_pos_weight"] = round(neg / pos, 2)

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )

            train_acc = accuracy_score(y_train, model.predict(X_train))
            valid_acc = accuracy_score(y_valid, model.predict(X_valid))
            gap = train_acc - valid_acc

            # 과적합 페널티: gap > 5% 이면 초과분을 페널티로 차감
            penalty = max(0.0, gap - 0.05) * self.gap_penalty
            score = valid_acc - penalty

            trial.set_user_attr("train_acc", round(train_acc, 4))
            trial.set_user_attr("valid_acc", round(valid_acc, 4))
            trial.set_user_attr("gap", round(gap, 4))
            return score

        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        best = self.study.best_trial
        self.best_params = best.params
        logger.info(
            f"Best trial #{best.number}: "
            f"valid_acc={best.user_attrs.get('valid_acc')}, "
            f"train_acc={best.user_attrs.get('train_acc')}, "
            f"gap={best.user_attrs.get('gap')}"
        )
        logger.info(f"Best params: {self.best_params}")
        return self.best_params

    def save_study(self, name: str = "optuna_study") -> Path:
        """Optuna study 결과 JSON으로 저장."""
        if self.study is None:
            raise RuntimeError("optimize()를 먼저 실행하세요.")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        path = MODELS_DIR / f"{name}.json"
        trials_data = [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "train_acc": t.user_attrs.get("train_acc"),
                "valid_acc": t.user_attrs.get("valid_acc"),
                "gap": t.user_attrs.get("gap"),
            }
            for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        result = {
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(trials_data),
            "trials": trials_data,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Study saved → {path}")
        return path

    def top_trials(self, n: int = 10) -> list[dict]:
        """상위 n개 trial 반환 (valid_acc 기준 정렬)."""
        if self.study is None:
            return []
        completed = [
            t for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        ranked = sorted(completed, key=lambda t: t.user_attrs.get("valid_acc", 0), reverse=True)
        return [
            {
                "number": t.number,
                "valid_acc": t.user_attrs.get("valid_acc"),
                "train_acc": t.user_attrs.get("train_acc"),
                "gap": t.user_attrs.get("gap"),
                "params": t.params,
            }
            for t in ranked[:n]
        ]
