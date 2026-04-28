"""Phase 2 실행 스크립트 — 피처 빌드 + 데이터셋 저장.

실행:
    python scripts/build_features.py
    python scripts/build_features.py --tickers AAPL MSFT GOOGL   # 일부만
    python scripts/build_features.py --skip-dataset               # parquet만
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time

from src.features import (
    build_all_features, build_dataset, save_dataset, save_features,
)
from src.utils import logger


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Feature Engineering")
    parser.add_argument("--tickers", nargs="*", help="특정 종목만 처리 (기본: 전체)")
    parser.add_argument("--skip-dataset", action="store_true", help="train/valid/test 분리 생략")
    parser.add_argument(
        "--threshold", type=float, default=0.02,
        help="타깃 레이블 수익률 기준 (기본: 0.02 = +2%%)"
    )
    args = parser.parse_args()

    t0 = time.time()
    logger.info(f"=== Phase 2: Feature Engineering 시작 (target threshold={args.threshold:.1%}) ===")

    # 1. 피처 계산
    logger.info("1. 피처 계산 중...")
    df = build_all_features(tickers=args.tickers, threshold=args.threshold)

    # 2. 전체 parquet 저장
    logger.info("2. features_all.parquet 저장 중...")
    save_features(df)

    # 3. train/valid/test 분리
    if not args.skip_dataset:
        logger.info("3. 데이터셋 분리 및 스케일링 중...")
        dataset = build_dataset(df)
        save_dataset(dataset)

        logger.info(f"   train: {len(dataset['X_train']):,}행")
        logger.info(f"   valid: {len(dataset['X_valid']):,}행")
        logger.info(f"   test : {len(dataset['X_test']):,}행")
        logger.info(f"   피처 수: {len(dataset['feature_cols'])}개")

    elapsed = time.time() - t0
    logger.info(f"=== Phase 2 완료 ({elapsed/60:.1f}분) ===")


if __name__ == "__main__":
    main()
