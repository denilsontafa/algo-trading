import numpy as np
import pandas as pd
import ta
from typing import Dict, Union

class BaseStrategy:
    def __init__(self):
        # Define strategy parameters
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.ema_short = 20
        self.ema_long = 50
        self.atr_period = 14
        self.bb_period = 20
        self.bb_std = 2

    def calculate_signals(self, df: pd.DataFrame) -> Dict[str, Union[float, int]]:
        """
        Calculate trading signals based on technical indicators
        Returns a dictionary of signals and their strengths
        """
        # Validate input data
        if df.empty:
            return {
                'rsi_signal': 0,
                'macd_signal': 0,
                'trend_signal': 0,
                'bb_signal': 0,
                'volatility': 0.5,
                'overall_signal': 0
            }
            
        # Ensure we have enough data for calculations
        if len(df) < max(self.ema_long, 100):  # 100 for volatility window
            print(f"Warning: Not enough data points. Have {len(df)}, need at least {max(self.ema_long, 100)}")
            return {
                'rsi_signal': 0,
                'macd_signal': 0,
                'trend_signal': 0,
                'bb_signal': 0,
                'volatility': 0.5,
                'overall_signal': 0
            }

        signals = {}

        # Calculate indicators
        self._calculate_rsi(df)
        self._calculate_macd(df)
        self._calculate_ema(df)
        self._calculate_bollinger(df)
        self._calculate_atr(df)

        # Get latest values
        latest = df.iloc[-1]

        # RSI signals
        signals['rsi_signal'] = self._get_rsi_signal(latest.RSI)
        
        # MACD signals
        signals['macd_signal'] = self._get_macd_signal(
            latest.MACD,
            latest.MACD_Signal,
            latest.MACD_Hist
        )

        # EMA signals
        signals['trend_signal'] = self._get_trend_signal(
            latest.EMA_short,
            latest.EMA_long,
            latest.close
        )

        # Bollinger Bands signals
        signals['bb_signal'] = self._get_bb_signal(
            latest.close,
            latest.BB_upper,
            latest.BB_lower
        )

        # Volatility signal from ATR
        signals['volatility'] = self._get_volatility_signal(latest.ATR, df.ATR)

        # Calculate overall signal
        signals['overall_signal'] = self._calculate_overall_signal(signals)

        return signals

    def _calculate_rsi(self, df: pd.DataFrame) -> None:
        """Calculate RSI indicator"""
        df['RSI'] = ta.momentum.rsi(df.close, window=self.rsi_period)

    def _calculate_macd(self, df: pd.DataFrame) -> None:
        """Calculate MACD indicator"""
        macd = ta.trend.MACD(
            df.close,
            window_slow=self.macd_slow,
            window_fast=self.macd_fast,
            window_sign=self.macd_signal
        )
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()

    def _calculate_ema(self, df: pd.DataFrame) -> None:
        """Calculate EMA indicators"""
        df['EMA_short'] = ta.trend.ema_indicator(df.close, window=self.ema_short)
        df['EMA_long'] = ta.trend.ema_indicator(df.close, window=self.ema_long)

    def _calculate_bollinger(self, df: pd.DataFrame) -> None:
        """Calculate Bollinger Bands"""
        bollinger = ta.volatility.BollingerBands(
            df.close,
            window=self.bb_period,
            window_dev=self.bb_std
        )
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_lower'] = bollinger.bollinger_lband()
        df['BB_middle'] = bollinger.bollinger_mavg()

    def _calculate_atr(self, df: pd.DataFrame) -> None:
        """Calculate Average True Range"""
        df['ATR'] = ta.volatility.average_true_range(
            df.high,
            df.low,
            df.close,
            window=self.atr_period
        )

    def _get_rsi_signal(self, rsi: float) -> float:
        """Generate RSI signal between -1 and 1"""
        if pd.isna(rsi):
            return 0
        if rsi > 70:
            return -1 * (rsi - 70) / 30  # Stronger sell signal as RSI increases
        elif rsi < 30:
            return (30 - rsi) / 30  # Stronger buy signal as RSI decreases
        elif rsi > 60:  # Added medium sell zone
            return -0.5 * (rsi - 60) / 10
        elif rsi < 40:  # Added medium buy zone
            return 0.5 * (40 - rsi) / 10
        return 0

    def _get_macd_signal(self, macd: float, signal: float, hist: float) -> float:
        """Generate MACD signal between -1 and 1"""
        if pd.isna(macd) or pd.isna(signal) or pd.isna(hist):
            return 0
        # Use both histogram and MACD line crossing
        signal_strength = hist / 0.001  # Adjusted sensitivity
        macd_cross = (macd - signal) / 0.001
        combined_signal = 0.7 * signal_strength + 0.3 * macd_cross
        return np.clip(combined_signal, -1, 1)

    def _get_trend_signal(self, ema_short: float, ema_long: float, price: float) -> float:
        """Generate trend signal between -1 and 1"""
        if pd.isna(ema_short) or pd.isna(ema_long) or pd.isna(price):
            return 0
        # Calculate percentage differences
        short_diff = (price - ema_short) / ema_short
        long_diff = (price - ema_long) / ema_long
        
        # Amplify the signal for smaller moves
        return np.clip((0.7 * short_diff + 0.3 * long_diff) * 200, -1, 1)

    def _get_bb_signal(self, price: float, upper: float, lower: float) -> float:
        """Generate Bollinger Bands signal between -1 and 1"""
        if pd.isna(price) or pd.isna(upper) or pd.isna(lower):
            return 0
        bb_range = upper - lower
        if bb_range == 0:
            return 0
        
        # Calculate where price is within the bands
        position = (price - lower) / bb_range
        # Convert to signal between -1 and 1 with increased sensitivity
        normalized = 2 * (position - 0.5)
        # Apply non-linear transformation to increase sensitivity near the bands
        return np.sign(normalized) * (np.abs(normalized) ** 0.7)

    def _get_volatility_signal(self, current_atr: float, atr_series: pd.Series) -> float:
        """Generate volatility signal between 0 and 1"""
        try:
            # Handle NaN values in ATR
            if pd.isna(current_atr) or atr_series.isna().all():
                return 0.5  # Return neutral signal if no valid ATR data
            
            # Use only valid ATR values
            valid_atr = atr_series.dropna()
            if len(valid_atr) < 100:  # If we have less than 100 valid points
                return 0.5
            
            # Calculate rolling min/max on valid data
            atr_min = valid_atr.rolling(window=100).min().iloc[-1]
            atr_max = valid_atr.rolling(window=100).max().iloc[-1]
            
            # Handle case where min equals max
            if atr_max == atr_min:
                return 0.5
                
            # Calculate normalized volatility
            volatility = (current_atr - atr_min) / (atr_max - atr_min)
            
            # Ensure result is between 0 and 1
            return np.clip(volatility, 0, 1)
            
        except Exception as e:
            print(f"Error calculating volatility signal: {str(e)}")
            return 0.5  # Return neutral signal on error

    def _calculate_overall_signal(self, signals: Dict[str, float]) -> float:
        """
        Calculate overall signal between -1 and 1
        Positive values suggest bullish signals, negative values suggest bearish signals
        """
        # Adjusted weights to emphasize trend and momentum
        weights = {
            'rsi_signal': 0.25,      # Increased RSI weight
            'macd_signal': 0.25,     # Increased MACD weight
            'trend_signal': 0.30,    # Kept trend as primary signal
            'bb_signal': 0.15,       # Reduced BB weight
            'volatility': 0.05       # Reduced volatility impact
        }

        # Calculate base signal without volatility
        base_signals = {k: v for k, v in signals.items() if k != 'volatility'}
        overall_signal = sum(
            base_signals[signal] * weights[signal]
            for signal in base_signals
        )
        
        # Get volatility factor (default to 0.5 if not available)
        volatility_factor = signals.get('volatility', 0.5)
        if pd.isna(volatility_factor):
            volatility_factor = 0.5
            
        # Adjust volatility impact
        volatility_adjusted = overall_signal * (0.8 + 0.4 * volatility_factor)
        
        # Return clipped signal
        return np.clip(volatility_adjusted, -1, 1)