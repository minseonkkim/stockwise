"""Walk-forward 검증 — 시간 순서 슬라이딩 윈도우 교차 검증.

각 윈도우마다 XGBoost를 학습하고 검증 정확도를 측정.
윈도우별 편차가 작을수록 모델이 시장 변화에 견고하다는 의미.

실행:
    python scripts/validate_walkforward.py
    python scripts/validate_walkforward.py --train-years 3 --valid-months 6
    python scripts/validate_walkforward.py --no-tune   # 빠른 실행 (튜닝 없음)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.features.pipeline import FEATURES_DIR, load_features
from src.models.trainer import XGBoostTrainer
from src.utils import logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward 검증")
    p.add_argument("--train-years",  type=int,   default=3,  help="학습 윈도우 기간 (년, 기본: 3)")
    p.add_argument("--valid-months", type=int,   default=6,  help="검증 윈도우 기간 (월, 기본: 6)")
    p.add_argument("--no-tune",      action="store_true",    help="Optuna 튜닝 건너뜀 (빠른 실행)")
    p.add_argument("--trials",       type=int,   default=30, help="Optuna 시도 횟수 (기본: 30)")
    return p.parse_args()


def make_windows(
    dates: pd.DatetimeIndex,
    train_years: int,
    valid_months: int,
) -> list[tuple[str, str, str, str]]:
    """(train_start, train_end, valid_start, valid_end) 리스트 생성."""
    min_date = dates.min()
    max_date = dates.max()

    windows = []
    valid_start = min_date + pd.DateOffset(years=train_years)

    while True:
        train_start = valid_start - pd.DateOffset(years=train_years)
        valid_end   = valid_start + pd.DateOffset(months=valid_months)

        if valid_end > max_date:
            break

        windows.append((
            str(train_start.date()),
            str(valid_start.date()),
            str(valid_start.date()),
            str(valid_end.date()),
        ))
        valid_start = valid_end  # 다음 윈도우로 이동

    return windows


def main():
    args = parse_args()

    logger.info("=" * 62)
    logger.info(f" Walk-forward 검증  (train={args.train_years}년, valid={args.valid_months}개월)")
    logger.info("=" * 62)

    # ── 피처 로드 ────────────────────────────────────────────────
    logger.info("피처 로드 중... (features_all.parquet)")

    # feature_cols 로드
    feat_cols_path = FEATURES_DIR / "feature_cols.pkl"
    if not feat_cols_path.exists():
        raise FileNotFoundError("build_features.py 를 먼저 실행하세요.")

    import pickle
    with open(feat_cols_path, "rb") as f:
        feature_cols: list[str] = pickle.load(f)

    all_path = FEATURES_DIR / "features_all.parquet"
    if not all_path.exists():
        raise FileNotFoundError("build_features.py 를 먼저 실행하세요.")

    df = pd.read_parquet(all_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["target"])

    available = [c for c in feature_cols if c in df.columns]
    logger.info(f"전체 데이터: {len(df):,}행 | 피처: {len(available)}개")

    # ── 윈도우 생성 ──────────────────────────────────────────────
    windows = make_windows(df["date"], args.train_years, args.valid_months)
    if not windows:
        logger.error("데이터 기간이 너무 짧아 윈도우를 생성할 수 없습니다.")
        return

    logger.info(f"총 {len(windows)}개 윈도우")

    # ── 윈도우별 학습 & 평가 ─────────────────────────────────────
    results = []
    for i, (tr_s, tr_e, va_s, va_e) in enumerate(windows, 1):
        train_df = df[(df["date"] >= tr_s) & (df["date"] < tr_e)]
        valid_df = df[(df["date"] >= va_s) & (df["date"] < va_e)]

        if len(train_df) < 500 or len(valid_df) < 100:
            logger.warning(f"  Window {i}: 데이터 부족 (train={len(train_df)}, valid={len(valid_df)}), 건너뜀")
            continue

        X_train = train_df[available].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_df["target"].values
        X_valid = valid_df[available].fillna(0).replace([np.inf, -np.inf], 0)
        y_valid = valid_df["target"].values

        if not args.no_tune:
            from src.models.tuner import OptunaTuner
            tuner = OptunaTuner(n_trials=args.trials)
            best_params = tuner.optimize(X_train, y_train, X_valid, y_valid)
        else:
            best_params = None

        trainer = XGBoostTrainer(params=best_params)
        trainer.train(X_train, y_train, X_valid, y_valid)

        proba = trainer.predict_proba(X_valid)
        acc   = accuracy_score(y_valid, (proba >= 0.5).astype(int))
        try:
            auc = roc_auc_score(y_valid, proba)
        except Exception:
            auc = float("nan")

        pos_rate = y_valid.mean()
        results.append({
            "window": i,
            "train": f"{tr_s} ~ {tr_e}",
            "valid": f"{va_s} ~ {va_e}",
            "train_rows": len(train_df),
            "valid_rows": len(valid_df),
            "accuracy": round(acc, 4),
            "auc": round(auc, 4),
            "pos_rate": round(pos_rate, 4),
        })
        logger.info(
            f"  Window {i:2d} | valid {va_s}~{va_e} | "
            f"acc={acc:.4f}  auc={auc:.4f}  pos={pos_rate:.2%}"
        )

    if not results:
        logger.error("유효한 윈도우가 없습니다.")
        return

    # ── 요약 출력 ────────────────────────────────────────────────
    accs = [r["accuracy"] for r in results]
    aucs = [r["auc"] for r in results if not np.isnan(r["auc"])]

    print("\n" + "=" * 62)
    print(" Walk-forward 결과 요약")
    print("=" * 62)
    print(f"  윈도우 수        : {len(results)}")
    print(f"  정확도  평균±std : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  정확도  최소/최대: {min(accs):.4f} / {max(accs):.4f}")
    if aucs:
        print(f"  AUC     평균±std : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print("=" * 62)
    print()

    # 판정
    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    if mean_acc >= 0.55 and std_acc <= 0.03:
        print("  판정: PASS — 평균 정확도 55%↑ & 표준편차 3%↓ (견고한 모델)")
    elif mean_acc >= 0.55:
        print("  판정: PARTIAL — 평균 55%↑이지만 윈도우간 편차 큼 (불안정)")
    else:
        print("  판정: FAIL — 평균 정확도 55% 미달 (피처 보강 필요)")

    # ── 결과 저장 ─────────────────────────────────────────────────
    out_path = Path("data/reports/walkforward_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    logger.info(f"Walk-forward 결과 저장 → {out_path}")


if __name__ == "__main__":
    main()
