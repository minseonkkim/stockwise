# 실행 가이드

---

## 사전 요구사항

| 도구           | 버전      | 용도                         |
| -------------- | --------- | ---------------------------- |
| Python         | 3.10 이상 | 메인 언어                    |
| Node.js        | 18 이상   | Next.js 프론트엔드           |
| PostgreSQL     | 15 / 16   | 주가 데이터 저장             |
| Docker Desktop | 최신      | DB + Airflow + API 컨테이너  |
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
REDIS_URL=redis://localhost:6379/0
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
docker compose up -d db redis
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
├── features_all.parquet   # 전체 피처 (~190만 행, 41개 피처)
├── train.parquet          # 2010 ~ 2020
├── valid.parquet          # 2021 ~ 2022
├── test.parquet           # 2023 ~ 현재
├── scaler.pkl             # StandardScaler (train 기준 fit)
└── sector_map.csv         # S&P 500 섹터 매핑 (캐시)
```

주요 옵션:

```bash
# 특정 종목만 테스트
python scripts/build_features.py --tickers AAPL MSFT GOOGL

# 섹터 피처 건너뜀 (오프라인 환경 등)
python scripts/build_features.py --no-sector
```

---

## 8. 모델 학습 (Phase 3 — 30분~2시간 소요)

```bash
python scripts/train_model.py
```

Optuna 100 trials 하이퍼파라미터 튜닝 → XGBoost 학습 → SHAP 피처 중요도 분석까지 자동 수행.

완료 후 생성 파일:

```
data/
├── models/
│   └── xgb_model.pkl               # 학습된 XGBoost 모델
└── reports/
    ├── evaluation.json             # train / valid / test 성능 리포트
    ├── shap_importance.json        # SHAP 피처 중요도
    ├── shap_values.npy             # SHAP 값 (배열)
    └── shap_summary.png            # SHAP 요약 플롯
```

주요 옵션:

```bash
# 빠른 테스트 (trials 10회, SHAP 건너뜀)
python scripts/train_model.py --trials 10 --no-shap

# 튜닝 없이 기본 파라미터로 학습
python scripts/train_model.py --no-tune

# SHAP 하위 피처(하위 10%) 제거 후 재학습
python scripts/train_model.py --retrain-after-shap

# XGBoost + LightGBM 앙상블 학습 (API가 자동으로 앙상블 사용)
python scripts/train_model.py --ensemble
```

성능 목표:

- 검증 정확도 **55% 이상** (랜덤 대비 유의미)
- train − validation 정확도 차이 **5% 이내** (과적합 방지)

---

## 9. 백테스트 (Phase 4 — 수분 소요)

```bash
python scripts/run_backtest.py
```

XGBoost 예측 확률 상위 20개 종목 / 주간 리밸런싱 / 거래비용 0.2%(왕복) 전략을 테스트 기간 전체에 대해 시뮬레이션합니다.

완료 후 생성 파일:

```
data/reports/
├── backtest_result.json        # CAGR, 샤프, MDD, 승률 등 성과 지표
├── backtest_weekly.csv         # 주간 수익률 및 턴오버
├── backtest_daily_returns.csv  # 일별 포트폴리오 수익률
└── backtest_qs.html            # Quantstats 풀 HTML 리포트 (선택)
```

주요 옵션:

```bash
# 상위 30개 종목으로 실행
python scripts/run_backtest.py --n-top 30

# 검증 기간(2021~2022)으로 백테스트
python scripts/run_backtest.py --split valid

# Quantstats HTML 리포트 생략 (quantstats 미설치 시)
python scripts/run_backtest.py --no-qs

# 초기 자본·거래비용 직접 지정
python scripts/run_backtest.py --capital 50000 --cost 0.002
```

성능 목표:

| 지표             | 목표 기준          |
| ---------------- | ------------------ |
| 연 수익률 (CAGR) | S&P 500 (SPY) 초과 |
| 샤프 비율        | 1.0 이상           |
| 최대 낙폭 (MDD)  | 20% 이내           |
| 승률             | 50% 이상           |

---

## 10. Walk-forward 검증 (선택 — 모델 견고성 확인)

```bash
python scripts/validate_walkforward.py
```

3년 학습 / 6개월 검증 슬라이딩 윈도우로 시장 변화에 대한 모델 안정성을 측정합니다.

```bash
# 빠른 실행 (Optuna 튜닝 생략)
python scripts/validate_walkforward.py --no-tune

# 윈도우 크기 조정
python scripts/validate_walkforward.py --train-years 2 --valid-months 3
```

완료 후 생성 파일: `data/reports/walkforward_results.csv`

판정 기준:
- **PASS**: 윈도우 평균 정확도 55%↑ & 표준편차 3%↓
- **PARTIAL**: 평균은 달성했지만 윈도우간 편차 큼
- **FAIL**: 평균 정확도 55% 미달 (피처 보강 필요)

---

## 11. API 서버 실행 (Phase 5)

Phase 7 ~ 9 (피처 빌드, 모델 학습, 백테스트)가 완료된 후 실행합니다.

```bash
pip install fastapi uvicorn httpx   # 미설치 시
python scripts/run_api.py
```

서버가 시작되면:

| URL                                | 설명                        |
| ---------------------------------- | --------------------------- |
| `http://localhost:8000/docs`       | Swagger 자동 API 문서       |
| `http://localhost:8000/api/health` | 헬스 체크                   |
| `GET /api/predictions`             | 오늘의 전 종목 예측 신호    |
| `GET /api/predictions/{ticker}`    | 특정 종목 예측 히스토리     |
| `GET /api/backtest/summary`        | 백테스트 성과 요약          |
| `GET /api/backtest/equity`         | 일별 누적 수익률 시계열     |
| `GET /api/backtest/weekly`         | 주간 수익률 테이블          |
| `GET /api/features/{ticker}`       | 종목별 최신 피처 값         |

개발 모드 (코드 변경 시 자동 재시작):

```bash
python scripts/run_api.py --reload
```

포트 변경:

```bash
python scripts/run_api.py --port 8080
```

> **신호 분류 방식**: `prob >= 0.5` 절대 임계값을 쓰면 대부분 hold로 분류됩니다.
> API는 백테스트와 동일한 **Top-20 방식** (날짜별 prob 상위 20개 = buy)을 사용합니다.

---

## 12. 프론트엔드 실행 (Phase 5)

```bash
cd frontend
npm install       # node_modules 미설치 시
npm run dev
```

브라우저에서 `http://localhost:3000` 접속:

| 경로              | 설명                                          |
| ----------------- | --------------------------------------------- |
| `/`               | 메인 대시보드 — 오늘의 매수 신호 Top-20       |
| `/backtest`       | 백테스트 리포트 — 성과 지표 + 수익 곡선       |
| `/portfolio`      | 포트폴리오 — 현재 매수 포지션                 |
| `/stock/{ticker}` | 종목 상세 — 예측 확률 차트 + 피처 현황        |

> **주의**: 프론트엔드를 실행하기 전 API 서버(`python scripts/run_api.py`)가 먼저 실행되어 있어야 합니다.

프로덕션 빌드:

```bash
cd frontend
npm run build
npm start
```

---

## 13. Docker로 전체 스택 실행 (Phase 5 통합)

DB, Redis, Airflow, FastAPI, Celery를 한 번에 실행:

```bash
cd docker
docker compose up -d
cd ..
```

서비스별 포트:

| 서비스           | 포트   | 설명               |
| ---------------- | ------ | ------------------ |
| PostgreSQL (DB)  | 5432   | 주가 데이터 저장   |
| Redis            | 6379   | Celery 브로커/백엔드 |
| Airflow          | 8080   | DAG 관리 UI        |
| FastAPI (API)    | 8000   | REST API 서버      |

FastAPI만 빠르게 실행하려면:

```bash
cd docker
docker compose up -d db redis api
```

---

## 14. Celery 일별 파이프라인 (Phase 5 자동화)

로컬에서 수동으로 Celery worker와 scheduler를 실행:

```bash
# 터미널 1 — Worker
celery -A src.tasks.pipeline worker --loglevel=info

# 터미널 2 — Beat (스케줄러, 매일 23:00 UTC 자동 실행)
celery -A src.tasks.pipeline beat --loglevel=info
```

파이프라인 태스크를 즉시 실행:

```bash
celery -A src.tasks.pipeline call src.tasks.pipeline.daily_pipeline
```

Docker로 실행하려면:

```bash
cd docker
docker compose up -d celery-worker celery-beat
```

---

## 15. 테스트 실행

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
│   ├── models/         # XGBoost 트레이너, Optuna 튜너, 평가, SHAP
│   ├── backtest/       # Phase 4: 예측 생성, 시뮬레이터, 성과 지표, 리포터
│   ├── api/            # Phase 5: FastAPI 앱, 라우터
│   │   ├── main.py     # FastAPI 앱 진입점
│   │   ├── deps.py     # 공유 의존성 (예측 캐시)
│   │   └── routes/     # predictions / backtest / features
│   ├── tasks/          # Phase 5: Celery 태스크
│   └── utils/          # 로거
├── frontend/           # Phase 5: Next.js 대시보드
│   ├── app/            # App Router 페이지
│   │   ├── page.tsx           # / 메인 대시보드
│   │   ├── backtest/          # /backtest
│   │   ├── portfolio/         # /portfolio
│   │   └── stock/[ticker]/    # /stock/{ticker}
│   ├── components/     # 공통 컴포넌트
│   └── lib/api.ts      # API 클라이언트
├── airflow/dags/       # Airflow DAG
├── docker/             # docker-compose.yml, Dockerfile.api
├── scripts/            # 실행 스크립트
│   └── run_api.py      # Phase 5: API 서버 실행
├── tests/              # 단위 테스트
├── data/
│   ├── features/       # Phase 2 피처 데이터
│   ├── models/         # Phase 3 학습된 모델 (git 제외)
│   └── reports/        # Phase 3·4 평가 리포트 (git 제외)
└── requirements.txt
```

---

## 피처 목록 (총 41개)

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
| 섹터강도  | sector_return_20d, sector_rel_strength, sector_rank_pct                                                      |

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

### xgboost / optuna / shap 설치 오류

```bash
pip install xgboost optuna shap
```

### SHAP 플롯 저장 오류 (matplotlib 없음)

```bash
pip install matplotlib
```

### Optuna 로그가 너무 많이 출력될 때

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
```

### quantstats 설치 오류

```bash
pip install quantstats
```

설치가 어려울 경우 `--no-qs` 옵션으로 HTML 리포트를 생략하고 실행할 수 있습니다.

```bash
python scripts/run_backtest.py --no-qs
```

### 백테스트 실행 시 "모델 파일 없음" 오류

Phase 3 학습이 완료되어야 합니다. `data/models/xgb_model.pkl` 파일이 있는지 확인하세요.

```bash
python scripts/train_model.py
```

### API 서버 실행 시 "피처/모델 파일 없음" 오류

Phase 2~4가 완료되어야 합니다. 순서대로 실행했는지 확인하세요:

```bash
python scripts/build_features.py   # Phase 2
python scripts/train_model.py      # Phase 3
python scripts/run_backtest.py     # Phase 4
python scripts/run_api.py          # Phase 5
```

### fastapi / uvicorn 설치 오류

```bash
pip install fastapi uvicorn[standard] httpx
```

### 프론트엔드 npm install 오류

```bash
cd frontend
npm install --legacy-peer-deps
```

### 프론트엔드에서 API 연결 실패

`next.config.js`의 rewrite 설정이 `http://localhost:8000`으로 되어 있는지 확인합니다.
API 서버를 먼저 실행해야 합니다:

```bash
python scripts/run_api.py
```
