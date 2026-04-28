"""초기 전체 수집 스크립트 — 2010년~현재 전 종목 1회 적재.

실행 시간: 종목 수(~500) × 요청 딜레이 기준 약 1~2시간 소요.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.data_collection import collect_all, get_active_tickers
from src.config import settings
from src.utils import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="초기 전체 데이터 수집")
    parser.add_argument("--start", default=settings.START_DATE, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="종료일 (기본: 오늘)")
    parser.add_argument("--tickers", nargs="*", help="특정 종목만 수집 (기본: 전체)")
    args = parser.parse_args()

    tickers = args.tickers or get_active_tickers()
    logger.info(f"수집 대상: {len(tickers)}개 종목, 기간: {args.start} ~ {args.end or '오늘'}")

    counts = collect_all(tickers, start_date=args.start, end_date=args.end)
    logger.info(f"완료: {counts}")


if __name__ == "__main__":
    main()
