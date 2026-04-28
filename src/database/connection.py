from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings
from src.database.models import Base
from src.utils import logger


engine = create_engine(
    settings.db_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False,
)

SessionFactory = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """테이블 생성 및 TimescaleDB hypertable 설정"""
    Base.metadata.create_all(engine)
    _setup_timescaledb()
    logger.info("Database initialized successfully")


def _setup_timescaledb() -> None:
    """daily_prices를 TimescaleDB hypertable로 변환"""
    with engine.connect() as conn:
        try:
            # TimescaleDB 확장 활성화
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
            conn.commit()

            # hypertable 생성 (이미 존재하면 무시)
            result = conn.execute(
                text(
                    "SELECT 1 FROM timescaledb_information.hypertables "
                    "WHERE hypertable_name = 'daily_prices'"
                )
            )
            if not result.fetchone():
                conn.execute(
                    text(
                        "SELECT create_hypertable('daily_prices', 'date', "
                        "if_not_exists => TRUE);"
                    )
                )
                conn.commit()
                logger.info("TimescaleDB hypertable created for daily_prices")
            else:
                logger.info("TimescaleDB hypertable already exists")
        except Exception as e:
            # TimescaleDB가 없는 환경(일반 PostgreSQL)에서도 동작하도록 허용
            logger.warning(f"TimescaleDB setup skipped: {e}")
            conn.rollback()
