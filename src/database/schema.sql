-- ML 퀀트 플랫폼 PostgreSQL + TimescaleDB 스키마
-- Phase 1: 데이터 수집 레이어

-- TimescaleDB 확장
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ────────────────────────────────────────────
-- 1. S&P 500 종목 마스터
-- ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS stocks (
    id                  SERIAL PRIMARY KEY,
    ticker              VARCHAR(10)  NOT NULL UNIQUE,
    name                VARCHAR(200),
    sector              VARCHAR(100),
    industry            VARCHAR(200),
    market_cap          BIGINT,
    is_active           BOOLEAN      NOT NULL DEFAULT TRUE,
    added_to_sp500      DATE,
    removed_from_sp500  DATE,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_stocks_ticker     ON stocks (ticker);
CREATE INDEX IF NOT EXISTS ix_stocks_is_active  ON stocks (is_active);

-- ────────────────────────────────────────────
-- 2. 일별 OHLCV 가격 데이터
-- ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS daily_prices (
    id          BIGSERIAL,
    ticker      VARCHAR(10)  NOT NULL,
    date        DATE         NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    adj_close   DOUBLE PRECISION NOT NULL,
    volume      BIGINT       NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_daily_prices_ticker_date UNIQUE (ticker, date)
);

-- TimescaleDB hypertable 변환 (date 파티션)
SELECT create_hypertable('daily_prices', 'date', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS ix_daily_prices_ticker_date  ON daily_prices (ticker, date DESC);
CREATE INDEX IF NOT EXISTS ix_daily_prices_date         ON daily_prices (date DESC);

-- ────────────────────────────────────────────
-- 3. 데이터 수집 이력
-- ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS data_collection_logs (
    id                  SERIAL PRIMARY KEY,
    ticker              VARCHAR(10)  NOT NULL,
    collection_date     DATE         NOT NULL,
    status              VARCHAR(20)  NOT NULL CHECK (status IN ('success', 'failed', 'skipped')),
    rows_inserted       INTEGER      DEFAULT 0,
    rows_updated        INTEGER      DEFAULT 0,
    error_message       TEXT,
    duration_seconds    DOUBLE PRECISION,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_collection_logs_ticker_date ON data_collection_logs (ticker, collection_date);
CREATE INDEX IF NOT EXISTS ix_collection_logs_date        ON data_collection_logs (collection_date DESC);

-- ────────────────────────────────────────────
-- 4. 데이터 검증 알림
-- ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS validation_alerts (
    id           SERIAL PRIMARY KEY,
    ticker       VARCHAR(10),
    alert_date   DATE        NOT NULL,
    alert_type   VARCHAR(50) NOT NULL,
    message      TEXT        NOT NULL,
    is_resolved  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_validation_alerts_date ON validation_alerts (alert_date DESC);

-- ────────────────────────────────────────────
-- 유틸리티: updated_at 자동 갱신 트리거
-- ────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_stocks_updated_at ON stocks;
CREATE TRIGGER trg_stocks_updated_at
    BEFORE UPDATE ON stocks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
