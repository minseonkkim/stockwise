# 실행 가이드

---

## 사전 요구사항

| 도구           | 버전      | 용도                         |
| -------------- | --------- | ---------------------------- |
| Python         | 3.10 이상 | 메인 언어                    |
| PostgreSQL     | 15 / 16   | 주가 데이터 저장             |
| Docker Desktop | 최신      | DB + Airflow 컨테이너 (선택) |
| Git            | 최신      | 버전 관리                    |

---

## 1. 프로젝트 클론

```bash
git clone https://github.com/minseonkkim/stockwise.git
cd stockwise
```

---

## 2. 환경변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 DB 비밀번호 등 필요한 값을 수정합니다.

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trade
DB_USER=trade
DB_PASSWORD=trade_secret
```

---

## 3. 의존성 설치

```bash
pip install -r requirements.txt
```

---

## 4. DB 실행

### 방법 A — Docker (권장)

```bash
cd docker
docker compose up -d
cd ..
```

컨테이너 상태 확인:

```bash
docker ps
```

### 방법 B — PostgreSQL 직접 설치

1. [PostgreSQL 다운로드](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads) 후 설치
2. SQL Shell(psql) 실행:

```sql
CREATE USER trade WITH PASSWORD 'trade_secret';
CREATE DATABASE trade OWNER trade;
\q
```

---

## 5. DB 초기화 (최초 1회)

```bash
python scripts/setup_db.py
```

- 테이블 생성 (stocks, daily_prices, data_collection_logs, validation_alerts)
- S&P 500 종목 목록 Wikipedia에서 동기화 (~503개)

---

## 6. 데이터 수집 (최초 1회 — 1~2시간 소요)

```bash
python scripts/run_initial_collection.py
```

2010년 ~ 현재까지 S&P 500 전 종목 OHLCV 수집.
약 190만 행 적재.

수집 현황 확인:

```bash
python scripts/check_data.py
```

---

## 7. 피처 빌드 (Phase 2 — 15~30분 소요)

```bash
python scripts/build_features.py
```

완료 후 생성 파일:

```
data/features/
├── features_all.parquet   # 전체 피처 (~190만 행, 38개 피처)
├── train.parquet          # 2010 ~ 2020
├── valid.parquet          # 2021 ~ 2022
├── test.parquet           # 2023 ~ 현재
└── scaler.pkl             # StandardScaler (train 기준 fit)
```

특정 종목만 테스트하려면:

```bash
python scripts/build_features.py --tickers AAPL MSFT GOOGL
```

---

## 8. 테스트 실행

```bash
pytest tests/ -v
```

---

## 일별 증분 수집 (운영)

Airflow DAG이 매일 UTC 23:00 (ET 18:00, 장 마감 후)에 자동 실행됩니다.

수동으로 증분 수집하려면:

```bash
python -c "
from src.data_collection import collect_incremental, get_active_tickers
collect_incremental(get_active_tickers())
"
```

---

## 폴더 구조

```
stockwise/
├── docs/               # 기획서, 가이드
├── src/
│   ├── config/         # 환경변수 설정
│   ├── database/       # SQLAlchemy 모델, DB 연결
│   ├── data_collection/# yfinance 수집, S&P500 유니버스, 검증
│   ├── features/       # 차트 기법 피처, 수치 피처, 파이프라인
│   └── utils/          # 로거
├── airflow/dags/       # Airflow DAG
├── docker/             # docker-compose.yml
├── scripts/            # 실행 스크립트
├── tests/              # 단위 테스트
├── data/               # 생성 데이터 (git 제외)
└── requirements.txt
```

---

## 피처 목록 (Phase 2 — 총 38개)

| 구분      | 피처                                                                                                         |
| --------- | ------------------------------------------------------------------------------------------------------------ |
| 추세      | golden_cross, dead_cross, ma_alignment_bull, ma_alignment_bear, trend_slope_20                               |
| 캔들      | cdl_hammer, cdl_doji, cdl_engulfing, cdl_morning_star, cdl_evening_star, cdl_three_soldiers, cdl_three_crows |
| 지지/저항 | support_distance_pct, resistance_distance_pct, resistance_breakout, near_52w_high, near_52w_low              |
| 차트패턴  | head_and_shoulders, double_top, double_bottom, triangle_convergence                                          |
| 모멘텀    | rsi_14, macd_hist, stoch_k, stoch_d, cci_20                                                                  |
| 변동성    | bb_pct_b, bb_width, atr_14, volatility_20                                                                    |
| 가격파생  | return_5d, return_20d, return_60d, price_vs_ma20, price_vs_ma60                                              |
| 거래량    | volume_ratio_20, obv_trend, volume_surge                                                                     |

---

## 타깃 레이블

```
target = 1  →  5거래일 후 수익률 >= +2%
target = 0  →  그 외
```

---

## 문제 해결

### pip install 인코딩 오류

```bash
python -m pip install -r requirements.txt
```

### Wikipedia 403 오류 (종목 동기화)

네트워크 문제 또는 요청 차단. 잠시 후 재시도.

### parquet 저장 오류

```bash
pip install pyarrow
```

### DB 연결 오류

`.env` 파일의 `DB_HOST`, `DB_PORT`, `DB_PASSWORD` 확인.
Docker 사용 중이라면 `docker ps`로 컨테이너 상태 확인.
