from strategy.base_strategy import BaseStrategy
from oanda_api import OandaAPI
import pandas as pd
pd.set_option('display.max_columns', None)

def test_strategy():
    # Initialize
    api = OandaAPI()
    strategy = BaseStrategy()
    
    # Get historical data
    pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
    
    for pair in pairs:
        print(f"\nTesting strategy for {pair}")
        print("=" * 50)
        
        # Get candles
        df = api.get_candles(pair)
        
        if df.empty:
            print(f"No data available for {pair}")
            continue
            
        # Calculate signals
        signals = strategy.calculate_signals(df)
        
        # Get current price and key levels
        current_price = df.close.iloc[-1]
        high_24h = df.high[-96:].max()  # Last 24 hours (96 15-min candles)
        low_24h = df.low[-96:].min()
        
        # Calculate price metrics
        price_range = high_24h - low_24h
        risk_reward = abs(signals['overall_signal']) * price_range  # Potential move size
        
        print("\nPrice Analysis:")
        print(f"Current Price: {current_price:.5f}")
        print(f"24h High: {high_24h:.5f}")
        print(f"24h Low: {low_24h:.5f}")
        print(f"24h Range: {price_range:.5f}")
        
        print("\nTechnical Signals:")
        for signal, value in signals.items():
            print(f"{signal}: {value:.4f}")
        
        overall_signal = signals['overall_signal']
        
        # More sensitive thresholds with risk assessment
        if overall_signal > 0.1:
            direction = "BULLISH"
            strength = "Strong" if overall_signal > 0.3 else "Moderate"
            target = current_price + (risk_reward * 1.5)
            stop = current_price - (risk_reward * 0.5)
        elif overall_signal < -0.1:
            direction = "BEARISH"
            strength = "Strong" if overall_signal < -0.3 else "Moderate"
            target = current_price - (risk_reward * 1.5)
            stop = current_price + (risk_reward * 0.5)
        else:
            direction = "NEUTRAL"
            strength = "Weak"
            target = current_price
            stop = current_price
            
        print(f"\nSignal Analysis:")
        print(f"Direction: {direction} ({strength})")
        print(f"Signal Strength: {overall_signal:.4f}")
        
        if direction != "NEUTRAL":
            print(f"Suggested Entry: {current_price:.5f}")
            print(f"Target Price: {target:.5f}")
            print(f"Stop Loss: {stop:.5f}")
            print(f"Risk/Reward Ratio: 1:{abs((target-current_price)/(current_price-stop)):.2f}")
        
        print("\nDetailed Analysis:")
        if abs(signals['trend_signal']) > 0.1:
            trend_strength = "Strong" if abs(signals['trend_signal']) > 0.3 else "Moderate"
            print(f"Trend: {'Upward' if signals['trend_signal'] > 0 else 'Downward'} ({trend_strength})")
        if abs(signals['rsi_signal']) > 0.2:
            print(f"RSI: {'Oversold' if signals['rsi_signal'] > 0 else 'Overbought'}")
        if abs(signals['macd_signal']) > 0.2:
            print(f"Momentum: {'Increasing' if signals['macd_signal'] > 0 else 'Decreasing'}")
        if signals['volatility'] > 0.6:
            print("Warning: High volatility - consider reducing position size")
        elif signals['volatility'] < 0.2:
            print("Note: Low volatility - potential breakout setup")

if __name__ == "__main__":
    test_strategy() 