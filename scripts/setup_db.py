"""DB 초기화 스크립트 — Phase 1 최초 1회 실행."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import init_db
from src.data_collection import sync_universe_to_db
from src.utils import logger


def main() -> None:
    logger.info("=== Phase 1: DB 초기화 시작 ===")

    logger.info("1. 테이블 생성 중...")
    init_db()

    logger.info("2. S&P 500 종목 목록 동기화 중...")
    counts = sync_universe_to_db()
    logger.info(f"   결과: {counts}")

    logger.info("=== DB 초기화 완료 ===")


if __name__ == "__main__":
    main()
