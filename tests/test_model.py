"""Phase 3 모델 모듈 단위 테스트."""
import numpy as np
import pandas as pd
import pytest

from src.models.evaluator import ModelEvaluator
from src.models.explainer import SHAPExplainer
from src.models.trainer import XGBoostTrainer
from src.models.tuner import OptunaTuner


def _make_dataset(n_samples: int = 500, n_features: int = 10, seed: int = 42):
    """테스트용 이진 분류 데이터셋 생성."""
    rng = np.random.default_rng(seed)
    feature_cols = [f"feat_{i}" for i in range(n_features)]
    X = pd.DataFrame(rng.standard_normal((n_samples, n_features)), columns=feature_cols)
    # 의미 있는 신호 추가
    y = (X["feat_0"] + X["feat_1"] * 0.5 + rng.standard_normal(n_samples) * 0.3 > 0).astype(int).values
    return X, y, feature_cols


class TestXGBoostTrainer:
    def test_train_returns_model(self):
        X, y, _ = _make_dataset()
        n = len(X)
        X_tr, y_tr = X.iloc[: n // 2], y[: n // 2]
        X_va, y_va = X.iloc[n // 2:], y[n // 2:]

        trainer = XGBoostTrainer()
        model = trainer.train(X_tr, y_tr, X_va, y_va)

        assert model is not None
        assert trainer.train_accuracy is not None
        assert trainer.valid_accuracy is not None

    def test_predict_proba_range(self):
        X, y, _ = _make_dataset()
        n = len(X)
        X_tr, y_tr = X.iloc[: n // 2], y[: n // 2]
        X_va, y_va = X.iloc[n // 2:], y[n // 2:]

        trainer = XGBoostTrainer()
        trainer.train(X_tr, y_tr, X_va, y_va)
        proba = trainer.predict_proba(X_va)

        assert proba.shape == (len(X_va),)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_predict_binary_output(self):
        X, y, _ = _make_dataset()
        n = len(X)
        X_tr, y_tr = X.iloc[: n // 2], y[: n // 2]
        X_va, y_va = X.iloc[n // 2:], y[n // 2:]

        trainer = XGBoostTrainer()
        trainer.train(X_tr, y_tr, X_va, y_va)
        preds = trainer.predict(X_va)

        assert set(preds).issubset({0, 1})

    def test_save_and_load(self, tmp_path, monkeypatch):
        """저장/로드 후 예측 결과 동일한지 확인."""
        import src.models.trainer as trainer_mod
        monkeypatch.setattr(trainer_mod, "MODELS_DIR", tmp_path)

        X, y, _ = _make_dataset()
        n = len(X)
        X_tr, y_tr = X.iloc[: n // 2], y[: n // 2]
        X_va, y_va = X.iloc[n // 2:], y[n // 2:]

        trainer = XGBoostTrainer()
        trainer.train(X_tr, y_tr, X_va, y_va)
        trainer.save("test_model")

        loaded = XGBoostTrainer.load("test_model")
        np.testing.assert_array_almost_equal(
            trainer.predict_proba(X_va),
            loaded.predict_proba(X_va),
            decimal=5,
        )


class TestOptunaTuner:
    def test_optimize_returns_params(self):
        X, y, _ = _make_dataset(n_samples=300)
        n = len(X)
        X_tr, y_tr = X.iloc[: n // 2], y[: n // 2]
        X_va, y_va = X.iloc[n // 2:], y[n // 2:]

        tuner = OptunaTuner(n_trials=5)
        best_params = tuner.optimize(X_tr, y_tr, X_va, y_va)

        assert isinstance(best_params, dict)
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert tuner.study is not None

    def test_top_trials_sorted(self):
        X, y, _ = _make_dataset(n_samples=300)
        n = len(X)
        X_tr, y_tr = X.iloc[: n // 2], y[: n // 2]
        X_va, y_va = X.iloc[n // 2:], y[n // 2:]

        tuner = OptunaTuner(n_trials=5)
        tuner.optimize(X_tr, y_tr, X_va, y_va)
        top = tuner.top_trials(3)

        assert len(top) <= 3
        accs = [t["valid_acc"] for t in top if t["valid_acc"] is not None]
        assert accs == sorted(accs, reverse=True)


class TestModelEvaluator:
    def _trained_trainer(self):
        X, y, _ = _make_dataset()
        n = len(X)
        X_tr, y_tr = X.iloc[: n // 2], y[: n // 2]
        X_va, y_va = X.iloc[n // 2:], y[n // 2:]
        trainer = XGBoostTrainer()
        trainer.train(X_tr, y_tr, X_va, y_va)
        return trainer, X_tr, y_tr, X_va, y_va

    def test_evaluate_keys(self):
        trainer, X_tr, y_tr, X_va, y_va = self._trained_trainer()
        ev = ModelEvaluator()
        result = ev.evaluate(trainer.model, X_va, y_va, split_name="valid")

        for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix"):
            assert key in result

    def test_accuracy_in_range(self):
        trainer, X_tr, y_tr, X_va, y_va = self._trained_trainer()
        ev = ModelEvaluator()
        result = ev.evaluate(trainer.model, X_va, y_va, "valid")
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_evaluate_all_splits(self):
        trainer, X_tr, y_tr, X_va, y_va = self._trained_trainer()
        ev = ModelEvaluator()
        results = ev.evaluate_all(
            trainer.model,
            X_tr, y_tr,
            X_va, y_va,
            X_va, y_va,  # test = valid (단위 테스트용)
        )
        assert set(results.keys()) == {"train", "valid", "test"}

    def test_save_report(self, tmp_path, monkeypatch):
        import src.models.evaluator as ev_mod
        monkeypatch.setattr(ev_mod, "REPORTS_DIR", tmp_path)

        trainer, X_tr, y_tr, X_va, y_va = self._trained_trainer()
        ev = ModelEvaluator()
        ev.evaluate(trainer.model, X_va, y_va, "valid")
        path = ev.save_report("test_eval")

        assert path.exists()


class TestSHAPExplainer:
    def test_fit_transform_returns_dataframe(self):
        X, y, _ = _make_dataset(n_samples=200, n_features=5)
        n = len(X)
        X_tr, y_tr = X.iloc[: n // 2], y[: n // 2]
        X_va, y_va = X.iloc[n // 2:], y[n // 2:]

        trainer = XGBoostTrainer()
        trainer.train(X_tr, y_tr, X_va, y_va)

        exp = SHAPExplainer(bottom_pct=0.2)
        importance = exp.fit_transform(trainer.model, X_va, sample_size=50)

        assert "feature" in importance.columns
        assert "mean_abs_shap" in importance.columns
        assert "low_importance" in importance.columns
        assert len(importance) == X_va.shape[1]

    def test_low_importance_subset(self):
        X, y, _ = _make_dataset(n_samples=200, n_features=10)
        n = len(X)
        X_tr, y_tr = X.iloc[: n // 2], y[: n // 2]
        X_va, y_va = X.iloc[n // 2:], y[n // 2:]

        trainer = XGBoostTrainer()
        trainer.train(X_tr, y_tr, X_va, y_va)

        exp = SHAPExplainer(bottom_pct=0.1)
        exp.fit_transform(trainer.model, X_va, sample_size=50)

        filtered = exp.get_filtered_features(X_va.columns.tolist())
        assert len(filtered) < len(X_va.columns)
        assert all(f not in exp.low_importance_features for f in filtered)
