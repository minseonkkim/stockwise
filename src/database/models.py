from datetime import date, datetime
from sqlalchemy import (
    BigInteger, Boolean, Column, Date, DateTime, Float,
    Index, Integer, String, Text, UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Stock(Base):
    """S&P 500 종목 마스터 테이블"""
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(200))
    sector = Column(String(100))
    industry = Column(String(200))
    market_cap = Column(BigInteger)
    is_active = Column(Boolean, default=True, nullable=False)
    added_to_sp500 = Column(Date)
    removed_from_sp500 = Column(Date)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return f"<Stock {self.ticker}>"


class DailyPrice(Base):
    """일별 OHLCV 가격 데이터 (TimescaleDB hypertable)"""
    __tablename__ = "daily_prices"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    adj_close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_daily_prices_ticker_date"),
        Index("ix_daily_prices_ticker_date", "ticker", "date"),
        Index("ix_daily_prices_date", "date"),
    )

    def __repr__(self) -> str:
        return f"<DailyPrice {self.ticker} {self.date}>"


class DataCollectionLog(Base):
    """데이터 수집 이력 및 상태 추적"""
    __tablename__ = "data_collection_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    collection_date = Column(Date, nullable=False)
    status = Column(String(20), nullable=False)  # success | failed | skipped
    rows_inserted = Column(Integer, default=0)
    rows_updated = Column(Integer, default=0)
    error_message = Column(Text)
    duration_seconds = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_collection_logs_ticker_date", "ticker", "collection_date"),
        Index("ix_collection_logs_date", "collection_date"),
    )


class ValidationAlert(Base):
    """데이터 검증 이상 탐지 알림"""
    __tablename__ = "validation_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10))
    alert_date = Column(Date, nullable=False)
    alert_type = Column(String(50), nullable=False)  # missing_days | outlier | zero_price | negative_volume
    message = Column(Text, nullable=False)
    is_resolved = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_validation_alerts_date", "alert_date"),
    )
