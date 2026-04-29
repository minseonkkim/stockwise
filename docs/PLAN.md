# ML 퀀트 투자 플랫폼 개발 기획서

> 미국 주식 대상 · 차트 기법 피처 + 퀀트 수치 피처 앙상블 · XGBoost 기반  
> v1.1 | 2026년 4월

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [피처 설계](#3-피처-설계)
4. [기술 스택](#4-기술-스택)
5. [개발 로드맵](#5-개발-로드맵)
6. [단계별 상세 계획](#6-단계별-상세-계획)
7. [리스크 및 대응 방안](#7-리스크-및-대응-방안)
8. [향후 확장 계획](#8-향후-확장-계획)

---

## 1. 프로젝트 개요

### 1.1 목적

미국 주식 시장(S&P 500)을 대상으로 **차트 기법 패턴 피처**와 **퀀트 수치 피처**를 결합한 XGBoost ML 모델로 주가 방향(상승/하락)을 예측하고, 예측 신호 기반 자동화 투자 전략을 웹 대시보드로 제공하는 플랫폼을 구축한다.

### 1.2 핵심 차별점

기존 퀀트 ML 접근법이 수치 지표(RSI 값, MACD 값)만 사용하는 것과 달리, 본 프로젝트는 **차트 기법 패턴(골든크로스 발생 여부, 캔들 패턴, 지지/저항 돌파)을 0/1로 인코딩**하여 함께 학습에 활용한다. 두 피처 군을 결합하면 모델이 더 풍부한 시장 정보를 학습할 수 있다.

| 구분 | 차트 기법 피처             | 퀀트 수치 피처   |
| ---- | -------------------------- | ---------------- |
| 형태 | 패턴 감지 (0/1)            | 연속 수치        |
| 예시 | 골든크로스 발생 = 1        | RSI = 72.3       |
| 도구 | TA-Lib `CDL*`, scipy       | Pandas, ta-lib   |
| 장점 | 사람이 보는 차트 신호 반영 | 정밀한 수치 정보 |

### 1.3 예측 대상 및 범위

- **대상 종목**: S&P 500 편입 종목 (약 500개)
- **예측 방식**: 5거래일 후 수익률 기준 이진 분류 (+2% 이상 = 1, 그 외 = 0)
- **학습 데이터**: 2010년 ~ 현재 (yfinance 기준)
- **제공 기능**: 예측 대시보드, 백테스트 리포트, 포트폴리오 추적

---

## 2. 시스템 아키텍처

### 2.1 전체 레이어 구조

```
[ 데이터 수집 ]        yfinance / Alpha Vantage / FRED API
       ↓
[ 피처 엔지니어링 ]    차트 기법 피처 + 퀀트 수치 피처  ← 핵심 차별점
       ↓
[ ML 모델 ]           XGBoost 분류 모델 (추후 LSTM 앙상블)
       ↓
[ 백테스트 ]          Backtrader + Quantstats
       ↓
[ 서비스 레이어 ]     FastAPI + Next.js 대시보드
```

### 2.2 데이터 흐름

1. yfinance로 S&P 500 전 종목 일별 OHLCV 수집
2. TA-Lib `CDL*` 함수로 캔들 패턴 감지 → 0/1 인코딩
3. scipy `find_peaks`로 고점/저점 탐지 → 지지/저항 근접도 계산
4. 기술적 지표(RSI, MACD, BB) 수치 계산
5. 타깃 레이블 생성 (5일 후 +2% 이상 = 1)
6. 시간 순서대로 train/validation/test 분리
7. XGBoost 학습 → 예측 확률 산출
8. Backtrader로 전략 시뮬레이션 및 성과 지표 계산
9. FastAPI로 예측 결과 API 노출
10. Next.js + TradingView 차트로 신호 시각화

---

## 3. 피처 설계

> **총 피처 수**: 약 60~80개 (차트 기법 30개 + 수치 지표 30~50개)

### 3.1 차트 기법 피처 (패턴 인코딩)

#### 추세 패턴

| 피처명              | 설명                     | 형태 |
| ------------------- | ------------------------ | ---- |
| `golden_cross`      | 5일선이 20일선 상향 돌파 | 0/1  |
| `dead_cross`        | 5일선이 20일선 하향 돌파 | 0/1  |
| `ma_alignment_bull` | 5일 > 20일 > 60일 정배열 | 0/1  |
| `ma_alignment_bear` | 5일 < 20일 < 60일 역배열 | 0/1  |
| `trend_slope_20`    | 20일 이동평균 기울기     | 수치 |

#### 캔들 패턴 (TA-Lib CDL 함수)

| 피처명               | 설명                 | 형태   |
| -------------------- | -------------------- | ------ |
| `cdl_hammer`         | 망치형 캔들          | 0/1    |
| `cdl_doji`           | 도지(십자선) 캔들    | 0/1    |
| `cdl_engulfing`      | 장악형 캔들          | -1/0/1 |
| `cdl_morning_star`   | 샛별형 (바닥 반전)   | 0/1    |
| `cdl_evening_star`   | 저녁별형 (천장 반전) | 0/1    |
| `cdl_three_soldiers` | 세 병사형 (상승)     | 0/1    |
| `cdl_three_crows`    | 세 까마귀형 (하락)   | 0/1    |

#### 지지/저항

| 피처명                    | 설명                     | 형태 |
| ------------------------- | ------------------------ | ---- |
| `support_distance_pct`    | 직근 지지선까지 거리 (%) | 수치 |
| `resistance_distance_pct` | 직근 저항선까지 거리 (%) | 수치 |
| `resistance_breakout`     | 저항선 상향 돌파 여부    | 0/1  |
| `near_52w_high`           | 52주 고점 5% 이내        | 0/1  |
| `near_52w_low`            | 52주 저점 5% 이내        | 0/1  |

#### 차트 패턴

| 피처명                 | 설명            | 형태 |
| ---------------------- | --------------- | ---- |
| `head_and_shoulders`   | 헤드앤숄더 감지 | 0/1  |
| `double_top`           | 더블탑 감지     | 0/1  |
| `double_bottom`        | 더블바텀 감지   | 0/1  |
| `triangle_convergence` | 삼각수렴 감지   | 0/1  |

---

### 3.2 퀀트 수치 피처

#### 모멘텀 지표

| 피처명               | 설명            |
| -------------------- | --------------- |
| `rsi_14`             | RSI (14일)      |
| `macd_hist`          | MACD 히스토그램 |
| `stoch_k`, `stoch_d` | 스토캐스틱 K, D |
| `cci_20`             | CCI (20일)      |

#### 변동성 지표

| 피처명          | 설명                      |
| --------------- | ------------------------- |
| `bb_pct_b`      | 볼린저밴드 %B 위치        |
| `bb_width`      | 볼린저밴드 폭 (수축/확장) |
| `atr_14`        | ATR (14일, 변동성 크기)   |
| `volatility_20` | 20일 수익률 표준편차      |

#### 가격 파생

| 피처명                                  | 설명                    |
| --------------------------------------- | ----------------------- |
| `return_5d`, `return_20d`, `return_60d` | 5/20/60일 수익률        |
| `price_vs_ma20`                         | 현재가 / 20일 이평 비율 |
| `price_vs_ma60`                         | 현재가 / 60일 이평 비율 |

#### 거래량

| 피처명            | 설명                            |
| ----------------- | ------------------------------- |
| `volume_ratio_20` | 거래량 / 20일 평균 거래량       |
| `obv_trend`       | OBV 5일 기울기                  |
| `volume_surge`    | 거래량 2배 이상 급증 여부 (0/1) |

#### 타깃 레이블

```
target = 1  if  close(t+5) / close(t) - 1 >= 0.02
target = 0  otherwise
```

---

## 4. 기술 스택

### 4.1 전체 스택 요약

| 레이어      | 기술                                  | 용도                            |
| ----------- | ------------------------------------- | ------------------------------- |
| 데이터 수집 | yfinance, Alpha Vantage, FRED API     | 주가/펀더멘털/거시 지표         |
| 피처 — 차트 | TA-Lib (`CDL*`), scipy (`find_peaks`) | 캔들 패턴, 지지/저항 감지       |
| 피처 — 수치 | Pandas, NumPy, pandas-ta              | 기술적 지표 계산                |
| ML 모델     | XGBoost, scikit-learn                 | 이진 분류 예측                  |
| 튜닝        | Optuna                                | 하이퍼파라미터 자동 최적화      |
| 해석        | SHAP                                  | 피처 중요도 분석                |
| 백테스트    | Backtrader, Quantstats                | 전략 시뮬레이션 및 성과 분석    |
| DB          | PostgreSQL + TimescaleDB              | 시계열 주가 데이터 저장         |
| 백엔드      | FastAPI                               | 예측 결과 REST API              |
| 비동기      | Celery + Redis                        | 모델 재학습, 데이터 수집 스케줄 |
| 프론트      | Next.js + Tailwind CSS                | 예측 대시보드 UI                |
| 차트        | TradingView Lightweight Charts        | 주가 + 신호 시각화              |
| 자동화      | Apache Airflow                        | 일별 파이프라인 DAG             |
| 배포        | Docker + Vercel + AWS EC2             | 컨테이너화 및 클라우드 배포     |

### 4.2 TA-Lib 설치 참고

```bash
# macOS
brew install ta-lib
pip install TA-Lib

# Ubuntu
sudo apt-get install libta-lib-dev
pip install TA-Lib

# 설치 어려울 경우 대안
pip install pandas-ta  # TA-Lib 없이 동일한 지표 계산 가능
```

---

## 5. 개발 로드맵

| Phase              | 기간  | 핵심 작업                            | 산출물               |
| ------------------ | ----- | ------------------------------------ | -------------------- |
| 1. 데이터 수집     | 1~2주 | yfinance 수집, DB 구축, Airflow DAG  | 자동 수집 파이프라인 |
| 2. 피처 엔지니어링 | 2주   | 차트 기법 + 수치 피처 전체 구현      | 피처 데이터셋        |
| 3. 모델 학습       | 1~2주 | XGBoost 학습, Optuna 튜닝, SHAP 분석 | 학습된 예측 모델     |
| 4. 백테스트        | 1주   | Backtrader 시뮬레이션, 성과 분석     | 백테스트 리포트      |
| 5. API & 대시보드  | 2주   | FastAPI, Next.js, TradingView 차트   | 서비스 웹 앱         |
| 6. 배포            | 1주   | Docker, Vercel, AWS EC2              | 프로덕션 배포        |

**예상 총 기간: 8~10주**

---

## 6. 단계별 상세 계획

### Phase 1 — 데이터 수집 & 저장 (1~2주)

**목표**: 안정적인 미국 주식 데이터 파이프라인 구축

#### 주요 작업

- yfinance로 S&P 500 전 종목 일별 OHLCV 수집 스크립트 작성
- PostgreSQL + TimescaleDB 스키마 설계 및 구축
- 결측치 / 상장폐지 종목 예외 처리 로직 구현
- Airflow DAG: 매일 장 마감 후(오후 5시 ET) 자동 수집
- 데이터 검증 모듈 (이상치, 누락일 슬랙 알림)

#### 완료 기준

- S&P 500 전 종목 2010년~현재 데이터 DB 적재 완료
- Airflow 대시보드에서 DAG 정상 실행 확인

> **주의**: 데이터 품질이 ML 모델 성능의 기반. 이 단계를 탄탄하게 잡아야 이후 단계가 수월하다.

---

### Phase 2 — 피처 엔지니어링 (2주)

**목표**: 차트 기법 피처 + 수치 피처 통합 데이터셋 구성

#### 차트 기법 피처 구현

```python
import talib
import scipy.signal as signal

# 캔들 패턴 (자동 감지)
df['cdl_hammer']    = talib.CDLHAMMER(o, h, l, c) / 100
df['cdl_doji']      = talib.CDLDOJI(o, h, l, c) / 100
df['cdl_engulfing'] = talib.CDLENGULFING(o, h, l, c) / 100

# 골든크로스
df['ma5']  = df['close'].rolling(5).mean()
df['ma20'] = df['close'].rolling(20).mean()
df['golden_cross'] = ((df['ma5'] > df['ma20']) &
                      (df['ma5'].shift(1) <= df['ma20'].shift(1))).astype(int)

# 고점/저점 탐지 (지지/저항)
peaks, _ = signal.find_peaks(df['close'], distance=10)
troughs, _ = signal.find_peaks(-df['close'], distance=10)
```

#### 수치 피처 구현

```python
import pandas_ta as ta

df['rsi']      = ta.rsi(df['close'], length=14)
df['macd']     = ta.macd(df['close'])['MACD_12_26_9']
df['bb_pctb']  = ta.bbands(df['close'])['BBP_5_2.0']
df['atr']      = ta.atr(df['high'], df['low'], df['close'], length=14)
```

#### 타깃 레이블 & 데이터 분리

```python
df['target'] = (df['close'].shift(-5) / df['close'] - 1 >= 0.02).astype(int)

# 시간 순서 엄수 (미래 데이터 누수 방지)
train = df[df.index < '2021-01-01']
valid = df[(df.index >= '2021-01-01') & (df.index < '2023-01-01')]
test  = df[df.index >= '2023-01-01']
```

> **핵심 주의사항**: train 기준으로만 scaler.fit(), valid/test는 transform()만 적용할 것

---

### Phase 3 — XGBoost 모델 학습 (1~2주)

**목표**: 검증 정확도 55% 이상의 안정적인 분류 모델 구축

#### 학습 전략

```python
import xgboost as xgb
import optuna

def objective(trial):
    params = {
        'n_estimators':  trial.suggest_int('n_estimators', 100, 500),
        'max_depth':     trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample':     trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return accuracy_score(y_valid, model.predict(X_valid))

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

#### SHAP 피처 중요도 분석

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_valid)
shap.summary_plot(shap_values, X_valid)
# 중요도 하위 10% 피처 제거 후 재학습
```

#### 성능 기준

- 검증 정확도 **55% 이상** (랜덤 대비 유의미)
- train - validation 정확도 차이 **5% 이내** (과적합 방지)

---

### Phase 4 — 백테스트 & 검증 (1주)

**목표**: ML 신호 기반 전략의 실제 수익성 검증

#### 백테스트 설정

- 초기 자본: $100,000 (가상)
- 거래비용: 0.1% (왕복 0.2%)
- 포지션 사이징: 동일 비중 (예측 상위 20개 종목)
- 리밸런싱: 매주 월요일

#### 필수 성과 지표

| 지표             | 목표 기준    |
| ---------------- | ------------ |
| 연 수익률 (CAGR) | S&P 500 초과 |
| 샤프 비율        | 1.0 이상     |
| 최대 낙폭 (MDD)  | 20% 이내     |
| 승률             | 50% 이상     |

```python
import quantstats as qs
qs.reports.full(strategy_returns, benchmark=sp500_returns)
```

> **주의**: 거래비용 반드시 반영. 수수료 미포함 백테스트는 실전과 큰 괴리가 생긴다.

---

### Phase 5 — API & 대시보드 (2주)

**목표**: 예측 결과를 실시간으로 확인하는 웹 서비스 구축

#### 신호 분류 기준

> **주의**: `prob >= 0.5` 절대 임계값을 쓰면 거의 모든 종목이 hold로 분류된다.
> 타깃 레이블(`5일 후 +2%`)의 실제 positive 비율이 25~35%이므로, 모델 확률 분포가
> 0.5 아래에 집중되는 것은 정상이다. 반드시 **상대적 순위(Top-N)** 방식을 사용할 것.

```python
# ❌ 잘못된 방식 — 대부분 hold로 나옴
signal = "buy" if prob >= 0.5 else "hold"

# ✅ 올바른 방식 — 백테스트와 동일한 Top-N 선택
today_df = today_df.sort_values("prob", ascending=False)
today_df["signal"] = "hold"
today_df.iloc[:N_TOP, today_df.columns.get_loc("signal")] = "buy"
# N_TOP = 20 (기본값, 백테스트 설정과 동일하게 유지)
```

#### FastAPI 엔드포인트

```
GET  /api/predictions              # 오늘자 전 종목 예측 신호 (prob + signal 포함)
GET  /api/predictions/{ticker}     # 특정 종목 예측 히스토리
GET  /api/backtest/summary         # 백테스트 성과 요약
GET  /api/features/{ticker}        # 종목별 피처 현황
```

응답 예시:
```json
{
  "date": "2026-04-29",
  "ticker": "AAPL",
  "prob": 0.63,
  "signal": "buy",
  "rank": 5
}
```

#### Next.js 페이지 구성

```
/                  메인 대시보드 — 오늘의 매수 신호 종목 리스트 (prob 순위 기준 상위 20개)
/stock/{ticker}    종목 상세 — TradingView 차트 + 예측 신호 오버레이
/backtest          백테스트 리포트 — 수익 곡선, 성과 지표 테이블
/portfolio         포트폴리오 — 현재 보유 포지션 및 수익률 추적
```

#### Celery 스케줄

```python
@app.task
def daily_pipeline():
    collect_data()      # 1. 데이터 수집
    build_features()    # 2. 피처 생성
    run_prediction()    # 3. 예측 실행
    save_signals()      # 4. DB 저장
# 매일 오후 6시(ET) 자동 실행
```

---

### Phase 6 — 배포 (1주)

#### 인프라 구성

```
프론트엔드    → Vercel (Next.js 최적화, 무료 티어 가능)
백엔드 API   → AWS EC2 t3.medium (or Railway)
DB           → AWS RDS PostgreSQL + TimescaleDB
Airflow      → AWS EC2 별도 인스턴스
Redis        → AWS ElastiCache (or 동일 EC2)
```

#### Docker Compose (로컬/스테이징)

```yaml
services:
  api: # FastAPI
  worker: # Celery Worker
  beat: # Celery Beat (스케줄러)
  redis: # Redis
  db: # PostgreSQL + TimescaleDB
  airflow: # Airflow Webserver + Scheduler
```

---

## 7. 리스크 및 대응 방안

| 리스크           | 발생 상황                          | 대응 방안                                 |
| ---------------- | ---------------------------------- | ----------------------------------------- |
| 과적합           | 백테스트 좋지만 실전 성능 저하     | 워크포워드 테스트, SHAP으로 피처 수 축소  |
| 데이터 누수      | 미래 정보가 학습에 포함            | 시간 순 분리 엄수, 파이프라인 단위 테스트 |
| TA-Lib 설치 오류 | 환경별 설치 이슈                   | pandas-ta로 대체 (pip만으로 설치 가능)    |
| API 요청 제한    | yfinance / Alpha Vantage 제한 초과 | 요청 딜레이 + Redis 캐싱 레이어 적용      |
| 시장 구조 변화   | 팬데믹 등 학습 분포와 다른 시장    | 주기적 모델 재학습, 앙상블로 안정성 확보  |
| 인프라 비용      | AWS 비용 과다                      | 초기엔 Railway + Vercel 무료 티어 활용    |

---

## 8. 향후 확장 계획

### 단기 (v1.1)

- LSTM 시계열 모델 추가 → XGBoost와 앙상블
- FinBERT 기반 뉴스 감성 분석 피처 추가

### 중기 (v2.0)

- 강화학습(RL) 기반 포지션 사이징 최적화
- WebSocket 실시간 가격 스트리밍

### 장기 (v3.0)

- 사용자 계정 기능 (관심 종목, 개인 포트폴리오)
- 브로커 API 연동 (Alpaca, Interactive Brokers)

---

> **면책 사항**: 본 플랫폼은 투자 정보 제공 목적으로 개발되며, 실제 자동 매매 연동은 법적 검토 후 별도 진행을 권장합니다. 모든 예측은 참고용이며 투자 손실에 대한 책임은 이용자에게 있습니다.
