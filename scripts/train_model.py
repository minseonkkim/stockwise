"""Phase 3: XGBoost 모델 학습 CLI.

실행 예:
  # 기본 학습 (Optuna 100 trials)
  python scripts/train_model.py

  # 빠른 테스트 (trials 적게)
  python scripts/train_model.py --trials 10 --no-shap

  # 튜닝 없이 기본 파라미터로 학습
  python scripts/train_model.py --no-tune

  # SHAP 분석 후 하위 피처 제거 + 재학습
  python scripts/train_model.py --retrain-after-shap
"""
import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.dataset import load_dataset
from src.models.ensemble import EnsembleTrainer
from src.models.evaluator import ModelEvaluator
from src.models.explainer import SHAPExplainer
from src.models.trainer import XGBoostTrainer
from src.models.tuner import OptunaTuner
from src.utils import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3: XGBoost 모델 학습")
    parser.add_argument("--trials", type=int, default=100, help="Optuna 시도 횟수 (기본: 100)")
    parser.add_argument("--no-tune", action="store_true", help="Optuna 튜닝 건너뜀 (기본 파라미터 사용)")
    parser.add_argument("--no-shap", action="store_true", help="SHAP 분석 건너뜀")
    parser.add_argument("--retrain-after-shap", action="store_true",
                        help="SHAP 하위 피처 제거 후 재학습")
    parser.add_argument("--ensemble", action="store_true",
                        help="XGBoost + LightGBM 앙상블 학습 (data/models/ensemble_*.pkl 저장)")
    parser.add_argument("--model-name", type=str, default="xgb_model", help="저장할 모델 이름")
    parser.add_argument("--threshold", type=float, default=0.5, help="분류 임계값 (기본: 0.5)")
    parser.add_argument("--shap-samples", type=int, default=5000, help="SHAP 계산 샘플 수")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Phase 3: XGBoost 모델 학습 시작")
    logger.info("=" * 60)

    # ─── 1. 데이터 로드 ─────────────────────────────────────────
    logger.info("[1/5] 데이터셋 로드 중...")
    dataset = load_dataset()
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_valid = dataset["X_valid"]
    y_valid = dataset["y_valid"]
    X_test  = dataset["X_test"]
    y_test  = dataset["y_test"]
    feature_cols = dataset["feature_cols"]

    logger.info(
        f"  train: {X_train.shape} | valid: {X_valid.shape} | test: {X_test.shape}"
    )
    logger.info(f"  피처 수: {len(feature_cols)}")

    # ─── 2. 하이퍼파라미터 튜닝 (Optuna) ──────────────────────────
    best_params = None
    if not args.no_tune:
        logger.info(f"[2/5] Optuna 하이퍼파라미터 튜닝 ({args.trials} trials)...")
        tuner = OptunaTuner(n_trials=args.trials)
        best_params = tuner.optimize(X_train, y_train, X_valid, y_valid)
        tuner.save_study("optuna_study")

        top = tuner.top_trials(5)
        logger.info("Top 5 trials:")
        for t in top:
            logger.info(f"  #{t['number']:3d}: valid_acc={t['valid_acc']}, gap={t['gap']}")
    else:
        logger.info("[2/5] Optuna 튜닝 건너뜀 (기본 파라미터 사용)")

    # ─── 3. 모델 학습 ───────────────────────────────────────────
    logger.info("[3/5] XGBoost 모델 학습 중...")
    trainer = XGBoostTrainer(params=best_params)
    trainer.train(X_train, y_train, X_valid, y_valid)
    trainer.save(args.model_name)

    # ─── 4. 성능 평가 ────────────────────────────────────────────
    logger.info("[4/5] 모델 성능 평가...")
    evaluator = ModelEvaluator()
    evaluator.evaluate_all(
        trainer.model,
        X_train, y_train,
        X_valid, y_valid,
        X_test,  y_test,
        threshold=args.threshold,
    )
    evaluator.save_report("evaluation")

    for split in ("train", "valid", "test"):
        print(evaluator.classification_report_str(split))
        print()

    # ─── 5. SHAP 분석 ────────────────────────────────────────────
    if not args.no_shap:
        logger.info("[5/5] SHAP 피처 중요도 분석 중...")
        explainer = SHAPExplainer(bottom_pct=0.10)
        explainer.fit_transform(trainer.model, X_valid, sample_size=args.shap_samples)
        explainer.save_importance("shap_importance")
        explainer.save_shap_values("shap_values")
        explainer.try_save_plot(X_valid, "shap_summary")

        if args.retrain_after_shap and explainer.low_importance_features:
            filtered_cols = explainer.get_filtered_features(feature_cols)
            logger.info(
                f"하위 피처 {len(explainer.low_importance_features)}개 제거 후 재학습 "
                f"({len(feature_cols)} → {len(filtered_cols)} 피처)"
            )
            trainer2 = XGBoostTrainer(params=best_params)
            trainer2.train(
                X_train[filtered_cols], y_train,
                X_valid[filtered_cols], y_valid,
            )
            trainer2.save(f"{args.model_name}_filtered")

            evaluator2 = ModelEvaluator()
            evaluator2.evaluate_all(
                trainer2.model,
                X_train[filtered_cols], y_train,
                X_valid[filtered_cols], y_valid,
                X_test[filtered_cols],  y_test,
                threshold=args.threshold,
            )
            evaluator2.save_report("evaluation_filtered")

            logger.info("재학습 완료.")
            for split in ("valid", "test"):
                print(f"[피처 필터링 후] {evaluator2.classification_report_str(split)}")
                print()
    else:
        logger.info("[5/5] SHAP 분석 건너뜀")

    # ─── 6. 앙상블 학습 (선택) ──────────────────────────────────
    if args.ensemble:
        logger.info("[6] XGBoost + LightGBM 앙상블 학습 중...")
        ensemble = EnsembleTrainer()
        ensemble.train(X_train, y_train, X_valid, y_valid, xgb_params=best_params)
        ensemble.save("ensemble")

        evaluator3 = ModelEvaluator()
        evaluator3.evaluate_all(
            type("_M", (), {"predict": lambda s, X: (ensemble.predict_proba(X) >= 0.5).astype(int),
                            "predict_proba": lambda s, X: ensemble.predict_proba(X).reshape(-1,1)})(),
            X_train, y_train, X_valid, y_valid, X_test, y_test,
            threshold=args.threshold,
        )
        evaluator3.save_report("evaluation_ensemble")
        logger.info("[Ensemble] 학습 완료 → data/models/ensemble_*.pkl")
        for split in ("valid", "test"):
            print(f"[앙상블] {evaluator3.classification_report_str(split)}")

    logger.info("=" * 60)
    logger.info("Phase 3 완료")
    logger.info(f"  모델: data/models/{args.model_name}.pkl")
    logger.info(f"  평가: data/reports/evaluation.json")
    if args.ensemble:
        logger.info("  앙상블: data/models/ensemble_*.pkl")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
