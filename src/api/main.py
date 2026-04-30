"""FastAPI 애플리케이션 — Phase 5 API 서버."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import backtest, features, predictions

app = FastAPI(
    title="StockWise API",
    description="ML 퀀트 투자 플랫폼 — 예측 신호 & 백테스트 REST API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["backtest"])
app.include_router(features.router, prefix="/api/features", tags=["features"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
