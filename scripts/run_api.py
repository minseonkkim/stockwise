"""FastAPI 서버 실행 스크립트 — Phase 5.

사용법:
    python scripts/run_api.py              # 기본 (localhost:8000)
    python scripts/run_api.py --port 8080  # 포트 변경
    python scripts/run_api.py --reload     # 개발 모드 (코드 변경 시 자동 재시작)
"""
import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockWise FastAPI 서버")
    parser.add_argument("--host", default="0.0.0.0", help="바인딩 호스트 (기본: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="포트 번호 (기본: 8000)")
    parser.add_argument("--reload", action="store_true", help="개발 모드 (파일 변경 시 자동 재시작)")
    args = parser.parse_args()

    print(f"StockWise API 서버 시작: http://{args.host}:{args.port}")
    print(f"API 문서: http://localhost:{args.port}/docs")
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
