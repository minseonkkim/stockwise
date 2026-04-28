"""타깃 임계값 변경 + 횡단면 정규화 후 데이터셋 재구성.

features_all.parquet의 close 컬럼으로 target을 재계산하고
날짜별 횡단면 백분위 순위 정규화를 적용한다.
DB 접속이나 피처 재계산 없이 완료된다.

실행:
    python scripts/rebuild_dataset.py                       # threshold=0.02, 정규화 적용
    python scripts/rebuild_dataset.py --threshold 0.015     # +1.5% 타깃
    python scripts/rebuild_dataset.py --no-normalize        # 정규화 없이 타깃만 변경
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.features.dataset import build_dataset, save_dataset
from src.features.normalizer import CONTINUOUS_FEATURES, cross_sectional_normalize
from src.features.pipeline import FEATURES_DIR
from src.utils import logger


def recompute_target(df: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    """ticker별 시간순 정렬 후 5일 후 수익률 기준 target 재계산."""
    frames = []
    for ticker, grp in df.groupby("ticker", sort=False):
        grp = grp.sort_values("date").copy()
        future_close = grp["close"].shift(-horizon)
        grp["target"] = ((future_close / grp["close"] - 1) >= threshold).astype("float32")
        grp.loc[grp.index[-horizon:], "target"] = float("nan")
        frames.append(grp)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="타깃 변경 + 횡단면 정규화 후 데이터셋 재구성")
    parser.add_argument(
        "--threshold", type=float, default=0.02,
        help="타깃 수익률 기준 (기본: 0.02 = +2%%)"
    )
    parser.add_argument("--horizon", type=int, default=5, help="예측 기간(거래일) (기본: 5)")
    parser.add_argument("--no-normalize", action="store_true", help="횡단면 정규화 건너뜀")
    args = parser.parse_args()

    src_path = FEATURES_DIR / "features_all.parquet"
    if not src_path.exists():
        logger.error(f"features_all.parquet 없음: {src_path}")
        sys.exit(1)

    t0 = time.time()

    # ─── 1. 로드 ────────────────────────────────────────────────
    logger.info(f"[1/4] features_all.parquet 로드 중...")
    df = pd.read_parquet(src_path)
    logger.info(f"  로드 완료: {len(df):,}행, {df['ticker'].nunique()}개 종목")

    old_rate = (df["target"] == 1).sum() / df["target"].notna().sum()
    logger.info(f"  현재 positive rate: {old_rate:.2%}")

    # ─── 2. 타깃 재계산 ────────────────────────────────────────
    logger.info(f"[2/4] target 재계산 중... (threshold={args.threshold:.1%})")
    df = recompute_target(df, horizon=args.horizon, threshold=args.threshold)

    new_rate = (df["target"] == 1).sum() / df["target"].notna().sum()
    logger.info(f"  positive rate: {old_rate:.2%} -> {new_rate:.2%}")

    # ─── 3. 횡단면 정규화 ─────────────────────────────────────
    if not args.no_normalize:
        available = [f for f in CONTINUOUS_FEATURES if f in df.columns]
        logger.info(f"[3/4] 횡단면 정규화 적용 중... ({len(available)}개 연속형 피처)")
        logger.info(f"  방법: 날짜별 백분위 순위 (0=최하위, 1=최상위)")
        df = cross_sectional_normalize(df, features=available)
        logger.info(f"  완료 (소요: {time.time()-t0:.1f}초)")
    else:
        logger.info("[3/4] 횡단면 정규화 건너뜀 (--no-normalize)")

    # ─── 4. parquet 저장 + 데이터셋 분리 ─────────────────────
    logger.info("[4/4] parquet 저장 + train/valid/test 분리 중...")
    df.to_parquet(src_path, index=False, compression="snappy")

    dataset = build_dataset(df)
    save_dataset(dataset)

    elapsed = time.time() - t0
    logger.info("=" * 55)
    logger.info("데이터셋 재구성 완료")
    logger.info(f"  threshold  : {args.threshold:.1%}")
    logger.info(f"  normalize  : {'YES' if not args.no_normalize else 'NO'}")
    logger.info(f"  positive   : {old_rate:.2%} -> {new_rate:.2%}")
    logger.info(f"  train      : {len(dataset['X_train']):,}행")
    logger.info(f"  valid      : {len(dataset['X_valid']):,}행")
    logger.info(f"  test       : {len(dataset['X_test']):,}행")
    logger.info(f"  소요 시간  : {elapsed:.1f}초")
    logger.info("=" * 55)
    logger.info("다음: python scripts/train_model.py")


if __name__ == "__main__":
    main()
