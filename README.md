## hwkims/python-trading-bot

# 파이썬 백테스팅 & 자동 거래 봇

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![ccxt](https://img.shields.io/badge/ccxt-1.78+-green.svg)
![pandas](https://img.shields.io/badge/pandas-1.3+-orange.svg)
![pandas_ta](https://img.shields.io/badge/pandas--ta-0.3+-red.svg)
![Matplotlib](https://img.shields.io/badge/matplotlib-3.4+-purple.svg)

**주의:** 이 프로젝트는 교육 및 연구 목적으로 개발되었으며, 실제 거래 환경에서의 사용을 권장하지 않습니다. 암호화폐 거래는 높은 위험을 수반하므로, 투자 결정은 신중하게 내리셔야 합니다.

## 프로젝트 개요

본 프로젝트는 파이썬(`Python`)을 이용하여 암호화폐 시장에서 백테스팅 및 자동 거래 시뮬레이션을 수행하는 봇입니다. `ccxt`, `pandas`, `pandas_ta` 라이브러리를 활용하여 다양한 기술적 지표를 계산하고, 이를 기반으로 여러 거래 전략을 구현했습니다.

**주요 기능:**

* **다양한 거래 전략 지원:**
    * 변동성 돌파 전략 (Volatility Breakout)
    * 레버리지 전략 (Leverage Strategy - RSI, MACD 기반)
    * 스캘핑 전략 (Scalping Strategy - 단기 이동평균선 기반)
* **기술적 지표 계산:** RSI, MACD, 이동평균선 (MA), 볼린저 밴드 (Bollinger Bands), ATR (Average True Range) 등
* **위험 관리:** 포지션 사이즈 계산, 손절매 (Stop Loss), 이익 실현 (Take Profit) 설정
* **백테스팅 및 시뮬레이션:** 과거 데이터를 기반으로 거래 전략의 성능을 검증하고, 모의 거래 환경에서 전략을 테스트
* **파라미터 최적화:** 다양한 기술 지표 파라미터 조합에 대한 백테스팅을 병렬로 수행하여 최적의 파라미터 설정 탐색
* **거래 결과 시각화:** matplotlib 라이브러리를 이용하여 OHLCV 차트, 기술 지표, 거래 시점 등을 시각적으로 표시
* **데이터 관리:** Parquet 형식으로 OHLCV 데이터를 저장 및 로드하여 효율적인 데이터 관리

## 사용된 라이브러리

* **ccxt:** 다양한 암호화폐 거래소 API를 통합하여 접근할 수 있도록 지원하는 라이브러리
* **pandas:** 데이터 분석 및 조작을 위한 강력한 라이브러리, OHLCV 데이터 및 기술 지표 계산에 활용
* **pandas_ta:** 다양한 기술적 분석 지표를 쉽게 계산할 수 있도록 pandas DataFrame을 확장하는 라이브러리
* **matplotlib:** 데이터 시각화를 위한 라이브러리, 거래 결과 및 기술 지표 차트 생성에 사용
* **logging:** 프로그램 실행 과정 기록 및 디버깅을 위한 로깅 라이브러리
* **concurrent.futures:** 파라미터 최적화 과정에서 병렬 처리를 위한 라이브러리

## 거래 전략

### 1. 변동성 돌파 전략 (Volatility Breakout Strategy)

* **개념:** 볼린저 밴드 하단 돌파 및 ATR 기반 변동성 돌파를 이용하여 롱 포지션 진입, 볼린저 밴드 상단 돌파 및 ATR 기반 변동성 돌파를 이용하여 숏 포지션 진입
* **진입 조건:**
    * 롱 포지션: 현재 가격이 볼린저 밴드 하단을 상향 돌파하고, 가격 변동폭이 ATR 임계값 이상이며, 거래량이 단기 평균 거래량보다 높을 때
    * 숏 포지션: 현재 가격이 볼린저 밴드 상단을 하향 돌파하고, 가격 변동폭이 ATR 임계값 이상이며, 거래량이 단기 평균 거래량보다 높을 때
* **종료 조건:**
    * 롱 포지션: 현재 가격이 볼린저 밴드 하단을 하향 돌파할 때
    * 숏 포지션: 현재 가격이 볼린저 밴드 상단을 상향 돌파할 때
* **위험 관리:** ATR 기반 손절매 및 보상 비율 기반 이익 실현 설정

### 2. 레버리지 전략 (Leverage Strategy)

* **개념:** RSI 과매도/과매수 구간 및 MACD 골든크로스/데드크로스를 이용하여 추세 반전 시점에 진입하는 전략
* **진입 조건:**
    * 롱 포지션: RSI가 30 이하로 과매도 구간에 진입하고, MACD가 시그널 선을 상향 돌파 (골든크로스) 할 때
    * 숏 포지션: RSI가 70 이상으로 과매수 구간에 진입하고, MACD가 시그널 선을 하향 돌파 (데드크로스) 할 때
* **종료 조건:**
    * 롱 포지션: RSI가 70 이상으로 과매수 구간에 진입하거나, MACD가 시그널 선을 하향 돌파 (데드크로스) 할 때
    * 숏 포지션: RSI가 30 이하로 과매도 구간에 진입하거나, MACD가 시그널 선을 상향 돌파 (골든크로스) 할 때
* **위험 관리:** ATR 기반 손절매 및 보상 비율 기반 이익 실현 설정

### 3. 스캘핑 전략 (Scalping Strategy)

* **개념:** 단기 이동평균선 교차를 이용하여 짧은 시간 내에 작은 이익을 추구하는 전략
* **진입 조건:**
    * 롱 포지션: 단기 이동평균선이 장기 이동평균선을 상향 돌파 (골든크로스) 할 때
    * 숏 포지션: 단기 이동평균선이 장기 이동평균선을 하향 돌파 (데드크로스) 할 때
* **종료 조건:** 신호가 없을 때 (익절)
* **위험 관리:** 좁은 손절폭 설정 (현재 가격의 0.5% 손실 제한)

## 시작하기

### 1. 필수 준비물

* **Python 3.7 이상**: 파이썬 공식 웹사이트([https://www.python.org/](https://www.python.org/))에서 다운로드 및 설치
* **필요한 파이썬 라이브러리 설치**: 터미널 또는 명령 프롬프트에서 다음 명령어를 실행하여 필요한 라이브러리 설치

```bash
pip install ccxt pandas pandas_ta matplotlib
```

### 2. 프로젝트 다운로드

GitHub에서 프로젝트를 복제합니다.

```bash
git clone https://github.com/hwkims/python-trading-bot.git
cd python-trading-bot
```

### 3. 봇 실행

`main.py` 파일을 실행하여 시뮬레이션 봇을 시작합니다.

```bash
python main.py
```

봇 실행 시 다음과 같은 정보가 출력됩니다.

* **로그 메시지**: 각 전략별 진입/청산 신호, 현재 가격, 손익, 잔고 변화 등
* **백테스팅 결과**: 승률, 손익비, 총 수익, 최대 낙폭 등
* **거래 시각화**: OHLCV 차트 위에 매수/매도 시점이 표시된 그래프 (matplotlib 창으로 출력)
* **파라미터 최적화 결과**: 최적 파라미터 및 해당 파라미터에서의 승률

## 사용 방법

`SimulationBot` 클래스를 이용하여 봇을 설정하고 실행합니다.

```python
from simulation_bot import SimulationBot

exchange_id = 'binance' # 거래소 ID (ccxt 지원 거래소)
symbol = 'BTC/USDT'    # 거래 심볼
timeframe = '3m'       # 캔들 타임프레임
initial_balance = 100000 # 초기 잔고
strategies = ['volatility_breakout', 'leverage', 'scalping'] # 실행할 전략 목록

bot = SimulationBot(exchange_id, symbol, timeframe, initial_balance, strategies=strategies)
bot.run() # 시뮬레이션 시작
bot.analyze_trades() # 백테스팅 결과 분석 및 출력
bot.visualize_trades() # 거래 결과 시각화
bot.optimize_parameters() # 파라미터 최적화 실행
```

**주요 설정:**

* `exchange_id`: 사용할 거래소 ID (`ccxt`에서 지원하는 거래소 ID 사용)
* `symbol`: 거래할 심볼 (예: 'BTC/USDT', 'ETH/KRW')
* `timeframe`: 캔들 타임프레임 (예: '1m', '5m', '1h', '1d')
* `initial_balance`: 시뮬레이션 시작 시 초기 잔고
* `strategies`: 실행할 거래 전략 목록 (`volatility_breakout`, `leverage`, `scalping` 중 선택)

## 파라미터 최적화

`optimize_parameters()` 메소드를 호출하여 기술 지표 파라미터 최적화를 수행할 수 있습니다.

```python
bot.optimize_parameters(
    rsi_periods=range(10, 21),        # RSI 기간 범위
    macd_fast_periods=range(10, 15),   # MACD Fast 기간 범위
    macd_slow_periods=range(20, 30),   # MACD Slow 기간 범위
    ma_short_periods=range(5, 15),    # 단기 이동평균선 기간 범위
    ma_long_periods=range(15, 25),     # 장기 이동평균선 기간 범위
    bb_lengths=range(15, 25)         # 볼린저 밴드 기간 범위
)
```

최적화 과정은 `concurrent.futures.ThreadPoolExecutor`를 이용하여 병렬로 진행되며, 각 파라미터 조합에 대한 백테스팅을 수행하고 승률을 기준으로 최적의 파라미터를 탐색합니다.

## 코드 구조

```
python-trading-bot/
├── data/                     # OHLCV 데이터 저장 디렉토리
├── main.py                   # 메인 실행 파일 (SimulationBot 인스턴스 생성 및 실행)
├── simulation_bot.py         # SimulationBot 클래스 정의 (봇 로직 구현)
├── technical_indicator.py    # TechnicalIndicatorAnalyzer 클래스 정의 (기술 지표 계산)
├── signal_generator.py       # SignalGenerator 클래스 정의 (거래 신호 생성)
├── risk_manager.py           # RiskManager 클래스 정의 (위험 관리 로직)
├── volatility_breakout_strategy.py # VolatilityBreakoutStrategy 클래스 정의 (변동성 돌파 전략)
├── leverage_strategy.py      # LeverageStrategy 클래스 정의 (레버리지 전략)
└── scalping_strategy.py      # ScalpingStrategy 클래스 정의 (스캘핑 전략)
```

## 추가 개발 계획 (향후 업데이트 예정)

* **더 다양한 거래 전략 추가:** 추세 추종 전략, 횡보장 전략 등
* **실시간 자동 거래 기능 구현:** 거래소 API 연동을 통한 실제 자동 매매 기능 추가 (현재는 시뮬레이션만 지원)
* **위험 관리 기능 강화:** 자금 관리, 마틴게일 전략 등 고급 위험 관리 기법 적용
* **파라미터 최적화 기능 개선:** 유전 알고리즘, 강화 학습 등 고급 최적화 알고리즘 적용
* **GUI 인터페이스 개발:** 사용자 친화적인 GUI 환경 제공

## 기여하기

프로젝트에 기여하고 싶으시다면 언제든지 Pull Request를 보내주세요. 버그 수정, 기능 추가, 코드 개선 등 어떤 형태의 기여든 환영합니다.

## 라이센스

MIT License

## 문의

hwkims
