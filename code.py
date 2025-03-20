import ccxt
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime
import logging
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TechnicalIndicatorAnalyzer:
    def __init__(self, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9,
                 ma_short_period=9, ma_long_period=21, bb_length=20, bb_std=2, atr_length=14):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ma_short_period = ma_short_period
        self.ma_long_period = ma_long_period
        self.bb_length = bb_length
        self.bb_std = bb_std
        self.atr_length = atr_length

    def calculate_indicators(self, df):
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)
        macd = ta.macd(df['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        df['macd'] = macd[f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
        df['macd_signal'] = macd[f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
        df['ma_short'] = ta.sma(df['close'], length=self.ma_short_period)
        df['ma_long'] = ta.sma(df['close'], length=self.ma_long_period)
        bb = ta.bbands(df['close'], length=self.bb_length, std=self.bb_std)
        df['bb_upper'] = bb[[col for col in bb.columns if col.startswith('BBU')][0]]
        df['bb_lower'] = bb[[col for col in bb.columns if col.startswith('BBL')][0]]
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_length)
        return df


class SignalGenerator:
    def __init__(self, atr_threshold_multiplier=1.5):
        self.atr_threshold_multiplier = atr_threshold_multiplier

    def generate_signal_volatility_breakout(self, df, position):
        if df is None or df.empty:
            return None

        current_close = df['close'].iloc[-1]
        current_bb_upper = df['bb_upper'].iloc[-1]
        current_bb_lower = df['bb_lower'].iloc[-1]
        current_atr = df['atr'].iloc[-1]
        volume_mean_short = df['volume'].tail(10).mean()
        current_volume = df['volume'].iloc[-1]

        atr_threshold = current_atr * self.atr_threshold_multiplier

        signal = None

        # 변동성 돌파 롱 포지션 진입
        if current_close > current_bb_lower and current_close - df['close'].iloc[
            -2] > atr_threshold and current_volume > volume_mean_short:
            if position != 'long':
                signal = 'LONG'
        # 변동성 돌파 숏 포지션 진입
        elif current_close < current_bb_upper and df['close'].iloc[
            -2] - current_close > atr_threshold and current_volume > volume_mean_short:
            if position != 'short':
                signal = 'SHORT'
        # 롱 포지션 종료
        elif position == 'long' and current_close < current_bb_lower:
            signal = 'CLOSE_LONG'
        # 숏 포지션 종료
        elif position == 'short' and current_close > current_bb_upper:
            signal = 'CLOSE_SHORT'

        return signal

    def generate_signal_leverage(self, df, position):
        if df is None or df.empty:
            return None

        current_rsi = df['rsi'].iloc[-1]
        current_macd = df['macd'].iloc[-1]
        current_macd_signal = df['macd_signal'].iloc[-1]

        signal = None

        if current_rsi < 30 and current_macd > current_macd_signal:
            if position != 'long':
                signal = 'LONG'
        elif current_rsi > 70 and current_macd < current_macd_signal:
            if position != 'short':
                signal = 'SHORT'
        elif position == 'long' and (current_rsi > 70 or current_macd < current_macd_signal):
            signal = 'CLOSE_LONG'
        elif position == 'short' and (current_rsi < 30 or current_macd > current_macd_signal):
            signal = 'CLOSE_SHORT'

        return signal

    def generate_signal_scalping(self, df):
        # 호가창 및 거래량 분석을 위한 데이터가 없으므로 간단한 이동 평균선 기반 신호 사용
        if df is None or df.empty or len(df) < 2:
            return None

        ma_short_period = 3
        ma_long_period = 7

        df['ma_short'] = ta.sma(df['close'], length=ma_short_period)
        df['ma_long'] = ta.sma(df['close'], length=ma_long_period)

        current_ma_short = df['ma_short'].iloc[-1]
        previous_ma_short = df['ma_short'].iloc[-2]
        current_ma_long = df['ma_long'].iloc[-1]
        previous_ma_long = df['ma_long'].iloc[-2]

        signal = None

        if previous_ma_short < previous_ma_long and current_ma_short > current_ma_long:
            signal = 'LONG'
        elif previous_ma_short > previous_ma_long and current_ma_short < current_ma_long:
            signal = 'SHORT'

        return signal


class RiskManager:
    def __init__(self, initial_balance=1000, risk_per_trade=0.02, reward_ratio=1.5, stop_loss_atr_multiplier=1.5, leverage=100): # 레버리지 변수 추가 및 기본값 1
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.reward_ratio = reward_ratio
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.leverage = leverage  # 레버리지 값 저장

    def calculate_position_size(self, current_price, stop_loss):
        if stop_loss is None or stop_loss <= 0:
            return 0
        risk_amount_per_trade = self.balance * self.risk_per_trade
        position_size = (risk_amount_per_trade / abs(current_price - stop_loss)) * self.leverage  # 레버리지 적용
        return position_size

    def calculate_stop_loss(self, current_price, atr):
        return current_price - (self.stop_loss_atr_multiplier * atr) if atr > 0 else None

    def calculate_take_profit(self, current_price, stop_loss, is_long=True):
        if stop_loss is None:
            return None
        risk = abs(current_price - stop_loss)
        profit = risk * self.reward_ratio
        return current_price + profit if is_long else current_price - profit

    def update_balance(self, profit):
        self.balance += profit


class VolatilityBreakoutStrategy:
    def __init__(self, signal_generator, risk_manager):
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.position = None
        self.entry_price = None
        self.trades = []

    def run(self, df, now):
        signal = self.signal_generator.generate_signal_volatility_breakout(df, self.position)
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]

        logging.info(f"[{now}] Volatility Breakout Strategy - Current Price: {current_price:.2f}")
        if signal:
            logging.info(f"[{now}] Volatility Breakout Strategy - Signal: {signal}, Current Position: {self.position}")

        if signal == 'LONG' and self.position != 'long':
            stop_loss_price = self.risk_manager.calculate_stop_loss(current_price, atr)
            if stop_loss_price:
                take_profit_price = self.risk_manager.calculate_take_profit(current_price, stop_loss_price, is_long=True)
                position_size = self.risk_manager.calculate_position_size(current_price, stop_loss_price)
                if position_size > 0:
                    logging.info(
                        f"[{now}] Volatility Breakout Strategy - Entering LONG - Price: {current_price:.2f}, Stop Loss: {stop_loss_price:.2f}, Take Profit: {take_profit_price:.2f}, Size: {position_size:.4f}")
                    self.position = 'long'
                    self.entry_price = current_price
                    self.trades.append({'timestamp': now, 'type': 'buy', 'price': current_price, 'size': position_size,
                                        'stop_loss': stop_loss_price, 'take_profit': take_profit_price})
                else:
                    logging.warning(
                        f"[{now}] Volatility Breakout Strategy - Could not calculate valid position size for LONG.")
        elif signal == 'SHORT' and self.position != 'short':
            stop_loss_price = self.risk_manager.calculate_stop_loss(current_price, atr)
            if stop_loss_price:
                take_profit_price = self.risk_manager.calculate_take_profit(current_price, stop_loss_price,
                                                                            is_long=False)
                position_size = self.risk_manager.calculate_position_size(current_price, stop_loss_price)
                if position_size > 0:
                    logging.info(
                        f"[{now}] Volatility Breakout Strategy - Entering SHORT - Price: {current_price:.2f}, Stop Loss: {stop_loss_price:.2f}, Take Profit: {take_profit_price:.2f}, Size: {position_size:.4f}")
                    self.position = 'short'
                    self.entry_price = current_price
                    self.trades.append({'timestamp': now, 'type': 'sell', 'price': current_price, 'size': position_size,
                                        'stop_loss': stop_loss_price, 'take_profit': take_profit_price})
                else:
                    logging.warning(
                        f"[{now}] Volatility Breakout Strategy - Could not calculate valid position size for SHORT.")
        elif signal == 'CLOSE_LONG' and self.position == 'long':
            position_size = next((trade['size'] for trade in self.trades if trade['type'] == 'buy' and 'size' in trade),
                                 0)
            if position_size > 0:
                profit = (current_price - self.entry_price) * position_size
                self.risk_manager.update_balance(profit)
                logging.info(
                    f"[{now}] Volatility Breakout Strategy - Closing LONG position - Price: {current_price:.2f}, Profit: {profit:.2f}, Current Balance: {self.risk_manager.balance:.2f}")
                self.position = None
                self.entry_price = None
                self.trades.append({'timestamp': now, 'type': 'close_buy', 'price': current_price, 'profit': profit})
            else:
                logging.warning(
                    f"[{now}] Volatility Breakout Strategy - Could not find initial position size for closing LONG.")
        elif signal == 'CLOSE_SHORT' and self.position == 'short':
            position_size = next(
                (trade['size'] for trade in self.trades if trade['type'] == 'sell' and 'size' in trade), 0)
            if position_size > 0:
                profit = (self.entry_price - current_price) * position_size
                self.risk_manager.update_balance(profit)
                logging.info(
                    f"[{now}] Volatility Breakout Strategy - Closing SHORT position - Price: {current_price:.2f}, Profit: {profit:.2f}, Current Balance: {self.risk_manager.balance:.2f}")
                self.position = None
                self.entry_price = None
                self.trades.append({'timestamp': now, 'type': 'close_sell', 'price': current_price, 'profit': profit})
            else:
                logging.warning(
                    f"[{now}] Volatility Breakout Strategy - Could not find initial position size for closing SHORT.")
        else:
            logging.info(f"[{now}] Volatility Breakout Strategy - No signal, Current Position: {self.position}")


class LeverageStrategy:
    def __init__(self, signal_generator, risk_manager):
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.position = None
        self.entry_price = None
        self.trades = []

    def run(self, df, now):
        signal = self.signal_generator.generate_signal_leverage(df, self.position)
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]

        logging.info(
            f"[{now}] Leverage Strategy - Current Price: {current_price:.2f}, RSI: {df['rsi'].iloc[-1]:.2f}, MACD: {df['macd'].iloc[-1]:.2f}, MACD Signal: {df['macd_signal'].iloc[-1]:.2f}")
        if signal:
            logging.info(f"[{now}] Leverage Strategy - Signal: {signal}, Current Position: {self.position}")

        if signal == 'LONG' and self.position != 'long':
            stop_loss_price = self.risk_manager.calculate_stop_loss(current_price, atr)
            if stop_loss_price:
                take_profit_price = self.risk_manager.calculate_take_profit(current_price, stop_loss_price,
                                                                            is_long=True)
                position_size = self.risk_manager.calculate_position_size(current_price, stop_loss_price)
                if position_size > 0:
                    logging.info(
                        f"[{now}] Leverage Strategy - Entering LONG - Price: {current_price:.2f}, Stop Loss: {stop_loss_price:.2f}, Take Profit: {take_profit_price:.2f}, Size: {position_size:.4f}")
                    self.position = 'long'
                    self.entry_price = current_price
                    self.trades.append({'timestamp': now, 'type': 'buy', 'price': current_price, 'size': position_size,
                                        'stop_loss': stop_loss_price, 'take_profit': take_profit_price})
                else:
                    logging.warning(f"[{now}] Leverage Strategy - Could not calculate valid position size for LONG.")
        elif signal == 'SHORT' and self.position != 'short':
            stop_loss_price = self.risk_manager.calculate_stop_loss(current_price, atr)
            if stop_loss_price:
                take_profit_price = self.risk_manager.calculate_take_profit(current_price, stop_loss_price,
                                                                            is_long=False)
                position_size = self.risk_manager.calculate_position_size(current_price, stop_loss_price)
                if position_size > 0:
                    logging.info(
                        f"[{now}] Leverage Strategy - Entering SHORT - Price: {current_price:.2f}, Stop Loss: {stop_loss_price:.2f}, Take Profit: {take_profit_price:.2f}, Size: {position_size:.4f}")
                    self.position = 'short'
                    self.entry_price = current_price
                    self.trades.append({'timestamp': now, 'type': 'sell', 'price': current_price, 'size': position_size,
                                        'stop_loss': stop_loss_price, 'take_profit': take_profit_price})
                else:
                    logging.warning(f"[{now}] Leverage Strategy - Could not calculate valid position size for SHORT.")
        elif signal == 'CLOSE_LONG' and self.position == 'long':
            position_size = next((trade['size'] for trade in self.trades if trade['type'] == 'buy' and 'size' in trade),
                                 0)
            if position_size > 0:
                profit = (current_price - self.entry_price) * position_size
                self.risk_manager.update_balance(profit)
                logging.info(
                    f"[{now}] Leverage Strategy - Closing LONG position - Price: {current_price:.2f}, Profit: {profit:.2f}, Current Balance: {self.risk_manager.balance:.2f}")
                self.position = None
                self.entry_price = None
                self.trades.append({'timestamp': now, 'type': 'close_buy', 'price': current_price, 'profit': profit})
            else:
                logging.warning(f"[{now}] Leverage Strategy - Could not find initial position size for closing LONG.")
        elif signal == 'CLOSE_SHORT' and self.position == 'short':
            position_size = next(
                (trade['size'] for trade in self.trades if trade['type'] == 'sell' and 'size' in trade), 0)
            if position_size > 0:
                profit = (self.entry_price - current_price) * position_size
                self.risk_manager.update_balance(profit)
                logging.info(
                    f"[{now}] Leverage Strategy - Closing SHORT position - Price: {current_price:.2f}, Profit: {profit:.2f}, Current Balance: {self.risk_manager.balance:.2f}")
                self.position = None
                self.entry_price = None
                self.trades.append({'timestamp': now, 'type': 'close_sell', 'price': current_price, 'profit': profit})
            else:
                logging.warning(f"[{now}] Leverage Strategy - Could not find initial position size for closing SHORT.")
        else:
            logging.info(f"[{now}] Leverage Strategy - No signal, Current Position: {self.position}")


class ScalpingStrategy:
    def __init__(self, signal_generator, risk_manager):
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.position = None
        self.entry_price = None
        self.trades = []

    def run(self, df, now):
        signal = self.signal_generator.generate_signal_scalping(df)
        current_price = df['close'].iloc[-1]

        logging.info(f"[{now}] Scalping Strategy - Current Price: {current_price:.2f}")

        if signal:
            logging.info(f"[{now}] Scalping Strategy - Signal: {signal}, Current Position: {self.position}")

        if signal == 'LONG' and self.position != 'long':
            position_size = self.risk_manager.calculate_position_size(current_price,
                                                                      current_price * 0.005)  # 스캘핑 특성상 작은 손절폭 설정
            if position_size > 0:
                logging.info(
                    f"[{now}] Scalping Strategy - Entering LONG - Price: {current_price:.2f}, Size: {position_size:.4f}")
                self.position = 'long'
                self.entry_price = current_price
                self.trades.append({'timestamp': now, 'type': 'buy', 'price': current_price, 'size': position_size})
            else:
                logging.warning(f"[{now}] Scalping Strategy - Could not calculate valid position size for LONG.")
        elif signal == 'SHORT' and self.position != 'short':
            position_size = self.risk_manager.calculate_position_size(current_price, current_price * 0.005)
            if position_size > 0:
                logging.info(
                    f"[{now}] Scalping Strategy - Entering SHORT - Price: {current_price:.2f}, Size: {position_size:.4f}")
                self.position = 'short'
                self.entry_price = current_price
                self.trades.append({'timestamp': now, 'type': 'sell', 'price': current_price, 'size': position_size})
            else:
                logging.warning(f"[{now}] Scalping Strategy - Could not calculate valid position size for SHORT.")
        elif signal is None and self.position == 'long':  # 매수 후 신호가 없을 때 익절
            position_size = next((trade['size'] for trade in self.trades if trade['type'] == 'buy' and 'size' in trade),
                                 0)
            if position_size > 0:
                profit = (current_price - self.entry_price) * position_size
                self.risk_manager.update_balance(profit)
                logging.info(
                    f"[{now}] Scalping Strategy - Closing LONG position - Price: {current_price:.2f}, Profit: {profit:.2f}, Current Balance: {self.risk_manager.balance:.2f}")
                self.position = None
                self.entry_price = None
                self.trades.append({'timestamp': now, 'type': 'close_buy', 'price': current_price, 'profit': profit})
            else:
                logging.warning(f"[{now}] Scalping Strategy - Could not find initial position size for closing LONG.")
        elif signal is None and self.position == 'short':
            position_size = next(
                (trade['size'] for trade in self.trades if trade['type'] == 'sell' and 'size' in trade), 0)
            if position_size > 0:
                profit = (self.entry_price - current_price) * position_size
                self.risk_manager.update_balance(profit)
                logging.info(
                    f"[{now}] Scalping Strategy - Closing SHORT position - Price: {current_price:.2f}, Profit: {profit:.2f}, Current Balance: {self.risk_manager.balance:.2f}")
                self.position = None
                self.entry_price = None
                self.trades.append({'timestamp': now, 'type': 'close_sell', 'price': current_price, 'profit': profit})
            else:
                logging.warning(f"[{now}] Scalping Strategy - Could not find initial position size for closing SHORT.")
        else:
            logging.info(f"[{now}] Scalping Strategy - No signal, Current Position: {self.position}")


class SimulationBot:
    def __init__(self, exchange_id, symbol, timeframe, initial_balance=1000, data_dir='data',
                 strategies=['volatility_breakout', 'leverage', 'scalping']):
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = getattr(ccxt, self.exchange_id)()
        self.indicator_analyzer = TechnicalIndicatorAnalyzer()
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager(initial_balance=initial_balance)
        self.data_dir = data_dir
        self.strategies = strategies

        os.makedirs(self.data_dir, exist_ok=True)
        self.ohlcv_data = self.load_data()
        self.trade_logic_interval = 5 * 60
        self.last_processed_time = 0
        self.data_fetch_limit = 1000
        self.min_data_length = max(self.indicator_analyzer.rsi_period, self.indicator_analyzer.macd_slow,
                                   self.indicator_analyzer.ma_long_period, self.indicator_analyzer.bb_length) + 20

        self.volatility_breakout_strategy = None
        self.leverage_strategy = None
        self.scalping_strategy = None

        if 'volatility_breakout' in self.strategies:
            self.volatility_breakout_strategy = self._create_volatility_breakout_strategy()
        if 'leverage' in self.strategies:
            self.leverage_strategy = self._create_leverage_strategy()
        if 'scalping' in self.strategies:
            self.scalping_strategy = self._create_scalping_strategy()

    def _create_volatility_breakout_strategy(self):
        return VolatilityBreakoutStrategy(self.signal_generator, self.risk_manager)

    def _create_leverage_strategy(self):
        return LeverageStrategy(self.signal_generator, self.risk_manager)

    def _create_scalping_strategy(self):
        return ScalpingStrategy(self.signal_generator, self.risk_manager)

    def load_data(self):
        filepath = os.path.join(self.data_dir, f'{self.symbol.replace("/", "")}_{self.timeframe}.parquet')
        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        return pd.DataFrame()

    def save_data(self):
        filepath = os.path.join(self.data_dir, f'{self.symbol.replace("/", "")}_{self.timeframe}.parquet')
        self.ohlcv_data.to_parquet(filepath)

    def fetch_data(self):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=self.data_fetch_limit)
            if not ohlcv:
                logging.warning(f"No data fetched from {self.exchange_id} for {self.symbol} {self.timeframe}")
                return False
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            self.ohlcv_data = pd.concat([self.ohlcv_data, df]).drop_duplicates(subset=['timestamp']).sort_values(
                'timestamp').reset_index(drop=True)
            self.save_data()
            return True
        except ccxt.NetworkError as e:
            logging.error(f"Network error during data fetch: {e}")
            return False
        except ccxt.ExchangeError as e:
            logging.error(f"Exchange error during data fetch: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error during data fetch: {e}")
            return False

    def run_strategy(self, strategy_name, df, now):
        if strategy_name == 'volatility_breakout' and self.volatility_breakout_strategy:
            self.volatility_breakout_strategy.run(df, now)
        elif strategy_name == 'leverage' and self.leverage_strategy:
            self.leverage_strategy.run(df, now)
        elif strategy_name == 'scalping' and self.scalping_strategy:
            self.scalping_strategy.run(df, now)

    def run(self):
        print("Starting simulation...")
        while True:
            now_timestamp = time.time()
            if now_timestamp - self.last_processed_time >= self.trade_logic_interval:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{now}] Running trade logic... Current Balance: {self.risk_manager.balance:.2f}")
                self.last_processed_time = now_timestamp

                if self.fetch_data():
                    if len(self.ohlcv_data) > self.min_data_length:
                        try:
                            df_analyzed = self.indicator_analyzer.calculate_indicators(self.ohlcv_data.copy())

                            for strategy in self.strategies:
                                self.run_strategy(strategy, df_analyzed, now)

                        except Exception as e:
                            logging.error(f"Error during indicator calculation or signal generation: {e}")
                    else:
                        logging.info(
                            f"[{now}] Insufficient data for analysis. Data Length: {len(self.ohlcv_data)}, Minimum Required Length: {self.min_data_length}")
                else:
                    logging.info(f"[{now}] Failed to fetch data.")
            time.sleep(1)

    def analyze_trades(self):
        all_trades = []
        if self.volatility_breakout_strategy:
            all_trades.extend(self.volatility_breakout_strategy.trades)
        if self.leverage_strategy:
            all_trades.extend(self.leverage_strategy.trades)
        if self.scalping_strategy:
            all_trades.extend(self.scalping_strategy.trades)

        if not all_trades:
            print("No trades to analyze.")
            return

        wins = 0
        losses = 0
        total_profit = 0

        for trade in all_trades:
            if 'profit' in trade:
                if trade['profit'] > 0:
                    wins += 1
                else:
                    losses += 1
                total_profit += trade['profit']
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        profit_factor = sum(trade['profit'] for trade in all_trades if 'profit' in trade and trade['profit'] > 0) / abs(
            sum(trade['profit'] for trade in all_trades if 'profit' in trade and trade['profit'] < 0)) if sum(
            trade['profit'] for trade in all_trades if 'profit' in trade and trade['profit'] < 0) != 0 else float('inf')

        balance_history = [self.risk_manager.initial_balance]
        current_balance = self.risk_manager.initial_balance
        max_balance = self.risk_manager.initial_balance
        max_drawdown = 0

        for trade in all_trades:
            if 'profit' in trade:
                current_balance += trade['profit']
                balance_history.append(current_balance)
                max_balance = max(max_balance, current_balance)
                drawdown = (max_balance - current_balance) / max_balance * 100 if max_balance > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)

        print(f"\n--- Backtesting Results ---")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Total Profit: {total_profit:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")

    def visualize_trades(self):
        if self.ohlcv_data.empty:
            print("No OHLCV data available.")
            return

        plt.figure(figsize=(16, 8))
        plt.plot(self.ohlcv_data['timestamp'], self.ohlcv_data['close'], label='Close Price')
        plt.plot(self.ohlcv_data['timestamp'], self.ohlcv_data['ma_short'],
                 label=f'{self.indicator_analyzer.ma_short_period} MA')
        plt.plot(self.ohlcv_data['timestamp'], self.ohlcv_data['ma_long'],
                 label=f'{self.indicator_analyzer.ma_long_period} MA')
        plt.plot(self.ohlcv_data['timestamp'], self.ohlcv_data['bb_upper'], label=f'BB Upper')
        plt.plot(self.ohlcv_data['timestamp'], self.ohlcv_data['bb_lower'], label='BB Lower')

        all_trades = []
        if self.volatility_breakout_strategy:
            all_trades.extend(self.volatility_breakout_strategy.trades)
        if self.leverage_strategy:
            all_trades.extend(self.leverage_strategy.trades)
        if self.scalping_strategy:
            all_trades.extend(self.scalping_strategy.trades)

        buy_trades = [trade for trade in all_trades if trade['type'] == 'buy']
        sell_trades = [trade for trade in all_trades if trade['type'] == 'sell']
        close_buy_trades = [trade for trade in all_trades if trade['type'] == 'close_buy']
        close_sell_trades = [trade for trade in all_trades if trade['type'] == 'close_sell']

        plt.scatter([t['timestamp'] for t in buy_trades], [t['price'] for t in buy_trades], color='green', marker='^',
                    s=100, label='Long Entry')
        plt.scatter([t['timestamp'] for t in sell_trades], [t['price'] for t in sell_trades], color='red', marker='v',
                    s=100, label='Short Entry')
        plt.scatter([t['timestamp'] for t in close_buy_trades], [t['price'] for t in close_buy_trades], color='lime',
                    marker='o', s=50, label='Long Exit')
        plt.scatter([t['timestamp'] for t in close_sell_trades], [t['price'] for t in close_sell_trades],
                    color='salmon', marker='o', s=50, label='Short Exit')
        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.title(f'{self.symbol} Trading Signals')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _optimize_parameter_set(self, rsi_period, macd_fast, macd_slow, ma_short, ma_long, bb_length):
        """개별 파라미터 세트에 대한 최적화 로직을 실행합니다."""
        print(
            f"Testing parameters: RSI={rsi_period}, MACD=({macd_fast},{macd_slow}), MA=({ma_short},{ma_long}), BB={bb_length}")
        temp_analyzer = TechnicalIndicatorAnalyzer(rsi_period=rsi_period, macd_fast=macd_fast, macd_slow=macd_slow,
                                                   ma_short_period=ma_short, ma_long_period=ma_long,
                                                   bb_length=bb_length)
        temp_bot = SimulationBot(self.exchange_id, self.symbol, self.timeframe)
        temp_bot.indicator_analyzer = temp_analyzer
        temp_bot.ohlcv_data = self.ohlcv_data.copy()
        temp_bot.risk_manager = RiskManager(initial_balance=self.risk_manager.initial_balance)
        temp_bot.strategies = self.strategies
        temp_bot.volatility_breakout_strategy = temp_bot._create_volatility_breakout_strategy()
        temp_bot.leverage_strategy = temp_bot._create_leverage_strategy()
        temp_bot.scalping_strategy = temp_bot._create_scalping_strategy()

        temp_bot.min_data_length = max(temp_bot.indicator_analyzer.rsi_period, temp_bot.indicator_analyzer.macd_slow,
                                       temp_bot.indicator_analyzer.ma_long_period,
                                       temp_bot.indicator_analyzer.bb_length) + 20

        all_trades = []
        for i in range(len(temp_bot.ohlcv_data)):
            if i > temp_bot.min_data_length:
                df_slice = temp_bot.ohlcv_data.iloc[:i + 1].copy()
                df_analyzed = temp_bot.indicator_analyzer.calculate_indicators(df_slice)
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                for strategy in self.strategies:
                    temp_bot.run_strategy(strategy, df_analyzed, now)

        if temp_bot.volatility_breakout_strategy:
            all_trades.extend(temp_bot.volatility_breakout_strategy.trades)
        if temp_bot.leverage_strategy:
            all_trades.extend(temp_bot.leverage_strategy.trades)
        if temp_bot.scalping_strategy:
            all_trades.extend(temp_bot.scalping_strategy.trades)

        if all_trades:
            wins = sum(1 for trade in all_trades if 'profit' in trade and trade['profit'] > 0)
            losses = sum(1 for trade in all_trades if 'profit' in trade and trade['profit'] <= 0)
            win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
            return win_rate, {
                'rsi_period': rsi_period,
                'macd_fast': macd_fast,
                'macd_slow': macd_slow,
                'ma_short': ma_short,
                'ma_long': ma_long,
                'bb_length': bb_length
            }
        return 0, None

    def optimize_parameters(self, rsi_periods=range(10, 21), macd_fast_periods=range(10, 15),
                            macd_slow_periods=range(20, 30),
                            ma_short_periods=range(5, 15), ma_long_periods=range(15, 25), bb_lengths=range(15, 25)):
        best_win_rate = 0
        best_params = None
        tasks = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            for rsi_period in rsi_periods:
                for macd_fast in macd_fast_periods:
                    for macd_slow in macd_slow_periods:
                        if macd_fast >= macd_slow:
                            continue
                        for ma_short in ma_short_periods:
                            for ma_long in ma_long_periods:
                                if ma_short >= ma_long:
                                    continue
                                for bb_length in bb_lengths:
                                    tasks.append(
                                        executor.submit(self._optimize_parameter_set, rsi_period, macd_fast, macd_slow,
                                                        ma_short, ma_long, bb_length))

            for future in as_completed(tasks):
                win_rate, params = future.result()
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_params = params
                    print(f"New best parameters found: {best_params} with Win Rate: {best_win_rate:.2f}%")

        print(f"\nBest parameters found: {best_params} with Win Rate: {best_win_rate:.2f}%")



if __name__ == "__main__":
        exchange_id = 'binance'
        symbol = 'BTC/USDT'
        timeframe = '3m'
        initial_balance = 100000

        # 실행할 전략 선택 (주석 처리로 선택/해제 가능)
        strategies = ['volatility_breakout', 'leverage', 'scalping']
        # strategies = ['volatility_breakout']
        # strategies = ['leverage']
        # strategies = ['scalping']

        bot = SimulationBot(exchange_id, symbol, timeframe, initial_balance, strategies=strategies)
        bot.run()
        bot.analyze_trades()
        bot.visualize_trades()
        bot.optimize_parameters()
