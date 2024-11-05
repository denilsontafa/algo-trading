from oanda_data import OandaDataFetcher
from models.data_processor import ForexDataset, update_saved_scalers
from models.lstm_model import ForexLSTM
from models.trainer import ModelTrainer
from sentiment.news_manager import NewsManager
from strategy.base_strategy import BaseStrategy
import torch
from torch.utils.data import DataLoader, random_split
from tabulate import tabulate
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import ta
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import pickle
import schedule
import time
from models.reinforcement_trainer import ReinforcementTrainer, update_model_with_latest_data
from typing import Dict, List
import config

class PositionManager:
    def __init__(self, oanda_client):
        self.oanda_client = oanda_client
        self.open_positions: Dict[str, Dict] = {}
        self.max_positions = config.MAX_POSITIONS
        self.position_size = config.POSITION_SIZE
        self.profit_target_pct = config.PROFIT_TARGET_PCT
        self.stop_loss_pct = config.STOP_LOSS_PCT
        self.max_hold_time = timedelta(hours=config.MAX_HOLD_TIME_HOURS)
    
    def manage_positions(self, analyses: List[dict]) -> None:
        """Manage all trading positions"""
        # First, update and close positions if needed
        self._update_positions()
        
        # Then, open new position if conditions are met
        self._open_new_position(analyses)
    
    def _update_positions(self) -> None:
        """Update and manage existing positions"""
        positions_to_close = []
        
        for pair, position in self.open_positions.items():
            current_price = float(self.oanda_client.get_current_price(pair))
            entry_price = position['entry_price']
            
            # Calculate profit/loss
            if position['direction'] == 'BUY':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SELL
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Update position info
            position['current_price'] = current_price
            position['pnl_pct'] = pnl_pct
            position['hold_time'] = datetime.now() - position['open_time']
            
            # Check closing conditions
            should_close = self._should_close_position(position)
            
            if should_close:
                positions_to_close.append(pair)
                print(f"\nClosing position for {pair}:")
                print(f"Entry: {entry_price:.5f}")
                print(f"Exit: {current_price:.5f}")
                print(f"P/L: {pnl_pct:.2%}")
                print(f"Hold time: {position['hold_time']}")
        
        # Close positions
        for pair in positions_to_close:
            self._close_position(pair)
    
    def _should_close_position(self, position: Dict) -> bool:
        """Determine if a position should be closed"""
        try:
            # Profit target reached
            if position['pnl_pct'] >= self.profit_target_pct:
                print(f"Closing position: Profit target {self.profit_target_pct:.1%} reached")
                return True
            
            # Stop loss hit
            if position['pnl_pct'] <= -self.stop_loss_pct:
                print(f"Closing position: Stop loss {self.stop_loss_pct:.1%} hit")
                return True
            
            # Maximum hold time exceeded
            if position['hold_time'] >= self.max_hold_time:
                print(f"Closing position: Max hold time {self.max_hold_time} exceeded")
                return True
            
            # Trailing stop loss when in profit
            if position['pnl_pct'] > 0.001:  # If in profit (0.1%)
                # Calculate trailing stop distance based on profit
                trailing_stop = min(position['pnl_pct'] * 0.5, self.stop_loss_pct)  # 50% of current profit
                
                # Calculate price movement since last high/low
                if position['direction'] == 'BUY':
                    if position.get('highest_price', 0) < position['current_price']:
                        position['highest_price'] = position['current_price']
                    price_from_high = (position['highest_price'] - position['current_price']) / position['highest_price']
                    if price_from_high > trailing_stop:
                        print(f"Closing position: Trailing stop triggered")
                        print(f"Highest: {position['highest_price']:.5f}")
                        print(f"Current: {position['current_price']:.5f}")
                        print(f"Drop: {price_from_high:.2%}")
                        return True
                else:  # SELL
                    if position.get('lowest_price', float('inf')) > position['current_price']:
                        position['lowest_price'] = position['current_price']
                    price_from_low = (position['current_price'] - position['lowest_price']) / position['lowest_price']
                    if price_from_low > trailing_stop:
                        print(f"Closing position: Trailing stop triggered")
                        print(f"Lowest: {position['lowest_price']:.5f}")
                        print(f"Current: {position['current_price']:.5f}")
                        print(f"Rise: {price_from_low:.2%}")
                        return True
            
            return False
        
        except Exception as e:
            print(f"Error in should_close_position: {str(e)}")
            return False
    
    def _open_new_position(self, analyses: List[dict]) -> None:
        """Open new position if conditions are met"""
        if len(self.open_positions) >= self.max_positions:
            print("\nMaximum positions reached, not opening new positions")
            return
        
        # Find the highest confidence signal
        best_signal = None
        highest_confidence = 0.45  # Lowered threshold
        
        print("\nAnalyzing signals for new positions:")
        for analysis in analyses:
            print(f"{analysis['pair']}: Confidence = {analysis['confidence']:.2f}")
            if analysis['confidence'] > highest_confidence:
                highest_confidence = analysis['confidence']
                best_signal = analysis
        
        if best_signal:
            pair = best_signal['pair']
            direction = 'BUY' if best_signal['predicted_change'] > 0 else 'SELL'
            
            print(f"\nAttempting to open {direction} position for {pair}")
            print(f"Confidence: {best_signal['confidence']:.2f}")
            print(f"Predicted change: {best_signal['predicted_change']:.2%}")
            
            # Open position
            try:
                response = self.oanda_client.create_order(
                    instrument=pair,
                    units=self.position_size if direction == 'BUY' else -self.position_size,
                    type='MARKET'
                )
                
                print(f"Order response: {response}")
                
                # Check for order fill
                if response and 'orderFillTransaction' in response:
                    fill_transaction = response['orderFillTransaction']
                    price = float(fill_transaction['price'])
                    
                    self.open_positions[pair] = {
                        'direction': direction,
                        'entry_price': price,
                        'open_time': datetime.now(),
                        'confidence': best_signal['confidence'],
                        'target_price': price * (1 + self.profit_target_pct if direction == 'BUY' else 1 - self.profit_target_pct),
                        'stop_loss': price * (1 - self.stop_loss_pct if direction == 'BUY' else 1 + self.stop_loss_pct),
                        'trade_id': fill_transaction['tradeOpened']['tradeID'],
                        'highest_price': price if direction == 'BUY' else float('inf'),
                        'lowest_price': price if direction == 'SELL' else 0,
                    }
                    
                    print(f"\nSuccessfully opened {direction} position for {pair}:")
                    print(f"Entry: {price:.5f}")
                    print(f"Target: {self.open_positions[pair]['target_price']:.5f}")
                    print(f"Stop: {self.open_positions[pair]['stop_loss']:.5f}")
                    print(f"Confidence: {best_signal['confidence']:.2f}")
                    print(f"Trade ID: {self.open_positions[pair]['trade_id']}")
                else:
                    print(f"Failed to open position: Invalid response format")
                    print(f"Response keys: {response.keys()}")
            
            except Exception as e:
                print(f"Error opening position for {pair}: {str(e)}")
        else:
            print(f"\nNo signals meet the minimum confidence threshold ({highest_confidence:.2f})")
    
    def _close_position(self, pair: str) -> None:
        """Close a specific position"""
        try:
            position = self.open_positions[pair]
            response = self.oanda_client.create_order(
                instrument=pair,
                units=-self.position_size if position['direction'] == 'BUY' else self.position_size,
                type='MARKET'
            )
            
            if response.get('orderFilled'):
                del self.open_positions[pair]
        
        except Exception as e:
            print(f"Error closing position for {pair}: {str(e)}")

class ForexAnalyzer:
    def __init__(self):
        self.data_fetcher = OandaDataFetcher()
        self.position_manager = PositionManager(self.data_fetcher)
        self.news_manager = NewsManager()
        self.strategy = BaseStrategy()
        self.previous_predictions = {}
        self.trainers = {
            pair: ReinforcementTrainer(pair) 
            for pair in config.CURRENCY_PAIRS
        }
        
    def analyze_pair(self, pair: str) -> dict:
        """Analyze a currency pair"""
        try:
            # Get historical data
            data = self.data_fetcher.fetch_historical_data(pair)
            if data is None:
                return None
                
            # Get current price and prediction
            current_price = data.close.iloc[-1]
            predicted_price = self._get_model_prediction(pair, data)
            predicted_price = self._verify_prediction(current_price, predicted_price)
            
            # Calculate predicted change
            predicted_change = ((predicted_price - current_price) / current_price) * 100
            
            # Get technical signal
            technical_signal = self._calculate_technical_signal(data)
            
            # Get sentiment
            sentiment_score = self._calculate_sentiment(pair)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                technical_signal,
                sentiment_score,
                predicted_change
            )
            
            # Determine trend based on predicted price direction
            trend = "NEUTRAL"
            if confidence >= 0.2:  # Only show trend if confidence is meaningful
                if predicted_change > 0:
                    trend = "BUY"
                elif predicted_change < 0:
                    trend = "SELL"
            
            print(f"\nTrend analysis for {pair}:")
            print(f"Current price: {current_price:.5f}")
            print(f"Predicted price: {predicted_price:.5f}")
            print(f"Predicted change: {predicted_change:.2f}%")
            print(f"Trend: {trend}")
            
            # Calculate volatility
            volatility = self._calculate_volatility(data)
            
            # Get news impact
            news_impact = self.news_manager.get_news_impact(pair)
            
            return {
                'pair': pair,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_change': predicted_change,
                'technical_signal': technical_signal,
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'trend': trend,
                'volatility': volatility,
                'news_impact': news_impact
            }
            
        except Exception as e:
            print(f"Error analyzing {pair}: {str(e)}")
            return None
    
    def _get_model_prediction(self, pair: str, data: pd.DataFrame) -> float:
        """Get price prediction from the model"""
        try:
            # Load model
            model_path = f'models/base_models/{pair}_model.pth'
            scaler_path = f'models/scalers/{pair}_scaler.pkl'
            
            print(f"Looking for model at: {model_path}")
            print(f"Looking for scaler at: {scaler_path}")
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                print(f"Missing files for {pair}")
                return data.close.iloc[-1]
                
            try:
                # Load the scaler
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                
                # Load the model
                checkpoint = torch.load(
                    model_path,
                    map_location=torch.device('cpu')
                )
                
                # Get model configuration
                model_config = checkpoint.get('model_config', {
                    'input_size': len(config.TECHNICAL_INDICATORS),
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.3
                })
                
                # Initialize model
                model = ForexLSTM(
                    input_size=model_config['input_size'],
                    hidden_size=model_config['hidden_size'],
                    num_layers=model_config['num_layers'],
                    dropout=model_config.get('dropout', 0.3)
                )
                
                # Load the state dict
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                # Process features
                features = self._prepare_features(data)
                
                # Convert to tensor and add batch dimension
                x = torch.FloatTensor(features[-60:]).unsqueeze(0)  # Last 60 timesteps
                
                # Get prediction
                with torch.no_grad():
                    scaled_prediction = model(x)
                    
                    # Inverse transform the prediction
                    prediction = scaler.inverse_transform(
                        scaled_prediction.numpy().reshape(-1, 1)
                    )[0][0]
                    
                    print(f"\nPrediction details for {pair}:")
                    print(f"Current price: {data.close.iloc[-1]:.5f}")
                    print(f"Scaled prediction: {scaled_prediction.item():.5f}")
                    print(f"Unscaled prediction: {prediction:.5f}")
                    
                    return prediction
                
            except Exception as e:
                print(f"Error in prediction process for {pair}: {str(e)}")
                return data.close.iloc[-1]
            
        except Exception as e:
            print(f"Error in model prediction pipeline: {str(e)}")
            return data.close.iloc[-1]
    
    def _get_max_change_threshold(self, pair: str) -> float:
        """Get maximum allowed price change threshold for a currency pair"""
        # Define thresholds based on historical data
        thresholds = {
            'EUR_USD': 0.003,
            'GBP_USD': 0.005,
            'USD_JPY': 0.007,
        }
        return thresholds.get(pair, 0.003)
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for the model"""
        # Calculate technical indicators
        data = data.copy()
        
        # Core features (8 in total)
        data['returns'] = data['close'].pct_change()
        data['rsi'] = ta.momentum.RSIIndicator(data['close']).rsi()
        
        macd = ta.trend.MACD(data['close'])
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        
        bb = ta.volatility.BollingerBands(data['close'])
        data['bb_high'] = bb.bollinger_hband()
        data['bb_low'] = bb.bollinger_lband()
        
        data['atr'] = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close']).average_true_range()
        
        # Select features for model
        feature_columns = [
            'returns',
            'rsi',
            'macd',
            'macd_signal',
            'bb_high',
            'bb_low',
            'atr',
            'close'
        ]
        
        # Fill NaN values
        for col in feature_columns:
            data[col] = data[col].ffill().bfill()
        
        # Scale the features
        scaler = MinMaxScaler()
        features = scaler.fit_transform(data[feature_columns])
        
        return features
    
    def _calculate_confidence(self, tech_signal: float, sentiment: float, pred_change: float) -> float:
        """Calculate overall confidence score"""
        try:
            # Normalize technical signal to [0, 1] with enhanced sensitivity
            tech_conf = min(abs(tech_signal) * 2, 1.0)  # Doubled sensitivity
            
            # Normalize sentiment to [0, 1]
            sent_conf = min(abs(sentiment) * 1.5, 1.0)  # Increased sensitivity
            
            # For forex, even small changes can be significant
            # 0.2% change is considered significant
            pred_conf = min(abs(pred_change) / 0.2, 1.0)
            
            # Calculate weighted confidence
            confidence = (
                0.5 * tech_conf +      # Technical analysis (50%)
                0.1 * sent_conf +      # Sentiment analysis (10%)
                0.4 * pred_conf        # Model prediction (40%)
            )
            
            # Amplify strong signals
            if confidence > 0.4:
                confidence = 0.4 + (confidence - 0.4) * 1.5
            
            print(f"\nConfidence calculation details for {pred_change:.3f}% predicted change:")
            print(f"Technical signal: {tech_signal:.3f} -> confidence: {tech_conf:.3f}")
            print(f"Sentiment: {sentiment:.3f} -> confidence: {sent_conf:.3f}")
            print(f"Predicted change: {pred_change:.3f}% -> confidence: {pred_conf:.3f}")
            print(f"Raw confidence: {confidence:.3f}")
            
            return confidence
            
        except Exception as e:
            print(f"Error calculating confidence: {str(e)}")
            return 0.0
    
    def run_scheduled_analysis(self):
        """Full analysis including opening new positions - runs every 15 minutes"""
        try:
            print(f"\nFull Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Run analysis for all pairs
            analyses = []
            for pair in config.CURRENCY_PAIRS:
                analysis = self.analyze_pair(pair)
                if analysis:
                    analyses.append(analysis)
            
            # Display results and manage positions
            if analyses:
                self._display_analysis(analyses)
                self.position_manager._open_new_position(analyses)
                
        except Exception as e:
            print(f"Error in scheduled analysis: {str(e)}")
    
    def _display_analysis(self, analyses):
        """Display analysis results in a table"""
        headers = [
            'Pair',
            'Current',
            'Predicted',
            'Change%',
            'Signal',
            'Sentiment',
            'Confidence',
            'Trend',
            'Vol',
            'News'
        ]
        
        table_data = []
        for a in analyses:
            # Format the change percentage
            change_pct = f"{a['predicted_change']:.2f}%"
            if abs(a['predicted_change']) >= 0.5:  # Highlight significant changes
                change_pct = f"* {change_pct} *"
            
            # Format confidence with highlighting
            conf_str = f"{a['confidence']:.2f}"
            if a['confidence'] >= 0.6:
                conf_str = f"** {conf_str} **"
            
            table_data.append([
                a['pair'],
                f"{a['current_price']:.5f}",
                f"{a['predicted_price']:.5f}",
                change_pct,
                f"{a['technical_signal']:.2f}",
                f"{a['sentiment_score']:.2f}",
                conf_str,
                a['trend'],
                f"{a['volatility']:.2f}",
                f"{a['news_impact']:.2f}"
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # Print trading suggestions
        print("\nTrading Suggestions:")
        for a in analyses:
            if a['confidence'] >= 0.6:  # Only show high confidence signals
                direction = "LONG" if a['predicted_change'] > 0 else "SHORT"
                print(f"\n{a['pair']}: {direction} (Confidence: {a['confidence']:.2f})")
                print(f"Entry: {a['current_price']:.5f}")
                print(f"Target: {a['predicted_price']:.5f}")
                print(f"Expected change: {a['predicted_change']:.2f}%")
                if a['volatility'] > 0.6:
                    print("Warning: High volatility - Consider smaller position size")
    
    def _calculate_technical_signal(self, data: pd.DataFrame) -> float:
        """Calculate technical analysis signal"""
        df = data.copy()
        
        # RSI with more sensitive thresholds for forex
        rsi = ta.momentum.RSIIndicator(df['close']).rsi().iloc[-1]
        rsi_signal = 0
        if rsi > 65:  # More sensitive overbought
            rsi_signal = -1
        elif rsi < 35:  # More sensitive oversold
            rsi_signal = 1
        else:
            rsi_signal = (rsi - 50) / 15  # More sensitive scaling
        
        # MACD with enhanced sensitivity
        macd = ta.trend.MACD(df['close'])
        macd_line = macd.macd().iloc[-1]
        signal_line = macd.macd_signal().iloc[-1]
        macd_signal = (macd_line - signal_line)
        macd_std = np.std(macd.macd() - macd.macd_signal())
        macd_signal = np.clip(macd_signal / (1.5 * macd_std), -1, 1)  # Increased sensitivity
        
        # Bollinger Bands with tighter thresholds
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2.0)
        current_price = df['close'].iloc[-1]
        bb_high = bb.bollinger_hband().iloc[-1]
        bb_low = bb.bollinger_lband().iloc[-1]
        bb_mid = (bb_high + bb_low) / 2
        
        bb_range = bb_high - bb_low
        bb_signal = 0
        if bb_range != 0:
            bb_position = (current_price - bb_mid) / (bb_range / 2)
            bb_signal = -np.clip(bb_position * 1.5, -1, 1)  # Enhanced sensitivity
        
        # Moving Averages with shorter periods
        ma_10 = df['close'].rolling(10).mean().iloc[-1]  # Shorter MA
        ma_30 = df['close'].rolling(30).mean().iloc[-1]  # Shorter MA
        ma_signal = (ma_10 - ma_30) / ma_30
        ma_signal = np.clip(ma_signal * 200, -1, 1)  # Increased sensitivity
        
        # Combine signals with adjusted weights
        technical_signal = (
            0.35 * rsi_signal +    # RSI (35%)
            0.35 * macd_signal +   # MACD (35%)
            0.15 * bb_signal +     # Bollinger Bands (15%)
            0.15 * ma_signal       # Moving Averages (15%)
        )
        
        print(f"\nTechnical signals:")
        print(f"RSI ({rsi:.1f}): {rsi_signal:.3f}")
        print(f"MACD: {macd_signal:.3f}")
        print(f"BB: {bb_signal:.3f}")
        print(f"MA: {ma_signal:.3f}")
        print(f"Combined: {technical_signal:.3f}")
        
        return technical_signal
    
    def _calculate_sentiment(self, pair: str) -> float:
        """Calculate sentiment score from news"""
        try:
            # Get sentiment from news
            news_sentiment = self.news_manager.get_pair_sentiment(pair)
            
            # Scale sentiment to [-1, 1]
            scaled_sentiment = max(min(news_sentiment, 1.0), -1.0)
            
            print(f"\nSentiment analysis for {pair}:")
            print(f"Raw sentiment: {news_sentiment:.2f}")
            print(f"Scaled sentiment: {scaled_sentiment:.2f}")
            
            return scaled_sentiment
            
        except Exception as e:
            print(f"Error calculating sentiment: {str(e)}")
            return 0.0
    
    def _verify_prediction(self, current_price: float, predicted_price: float) -> float:
        """Verify prediction is within reasonable bounds"""
        try:
            # Maximum allowed price change (1% for forex)
            MAX_CHANGE_PCT = 0.01
            
            # Calculate percentage change
            pct_change = abs(predicted_price - current_price) / current_price
            
            if pct_change > MAX_CHANGE_PCT:
                print(f"\nWarning: Prediction seems unreasonable")
                print(f"Current price: {current_price:.5f}")
                print(f"Predicted price: {predicted_price:.5f}")
                print(f"Percentage change: {pct_change:.2%}")
                print(f"Maximum allowed change: {MAX_CHANGE_PCT:.2%}")
                print("Using current price instead")
                return current_price
            
            return predicted_price
            
        except Exception as e:
            print(f"Error verifying prediction: {str(e)}")
            return current_price
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate volatility score based on ATR and historical std dev"""
        try:
            # Calculate ATR (Average True Range)
            atr = ta.volatility.AverageTrueRange(
                data['high'], 
                data['low'], 
                data['close'],
                window=14
            ).average_true_range().iloc[-1]
            
            # Normalize ATR by current price
            normalized_atr = atr / data['close'].iloc[-1]
            
            # Calculate rolling standard deviation of returns
            returns = data['close'].pct_change()
            std_dev = returns.rolling(window=20).std().iloc[-1]
            
            # Combine ATR and std dev for volatility score
            volatility = (0.7 * normalized_atr + 0.3 * std_dev) * 100
            
            # Scale to [0, 1] range
            # Typical forex volatility ranges from 0.1% to 1%
            scaled_volatility = min(volatility / 0.01, 1.0)
            
            print(f"\nVolatility calculation:")
            print(f"Normalized ATR: {normalized_atr:.6f}")
            print(f"20-day StdDev: {std_dev:.6f}")
            print(f"Raw volatility: {volatility:.6f}")
            print(f"Scaled volatility: {scaled_volatility:.2f}")
            
            return scaled_volatility
            
        except Exception as e:
            print(f"Error calculating volatility: {str(e)}")
            return 0.5  # Return moderate volatility on error

    def check_positions(self):
        """Only check and manage existing positions"""
        try:
            print(f"\nPosition Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if not self.position_manager.open_positions:
                print("No open positions to check")
                return
            
            print("\nChecking open positions:")
            for pair, pos in self.position_manager.open_positions.items():
                current_price = float(self.data_fetcher.get_current_price(pair))
                entry_price = pos['entry_price']
                
                # Calculate profit/loss
                if pos['direction'] == 'BUY':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:  # SELL
                    pnl_pct = (entry_price - current_price) / entry_price
                
                # Update position info
                pos['current_price'] = current_price
                pos['pnl_pct'] = pnl_pct
                pos['hold_time'] = datetime.now() - pos['open_time']
                
                print(f"\n{pair} {pos['direction']}:")
                print(f"Entry: {entry_price:.5f}")
                print(f"Current: {current_price:.5f}")
                print(f"P/L: {pnl_pct:.2%}")
                print(f"Hold time: {pos['hold_time']}")
                
                # Check if position should be closed
                if self.position_manager._should_close_position(pos):
                    print(f"Closing position for {pair}")
                    self.position_manager._close_position(pair)
            
        except Exception as e:
            print(f"Error checking positions: {str(e)}")

def main():
    # Update saved scalers to current version
    update_saved_scalers()
    
    analyzer = ForexAnalyzer()
    
    # Schedule full analysis every 15 minutes
    schedule.every(config.ANALYSIS_INTERVAL).minutes.do(analyzer.run_scheduled_analysis)
    
    # Schedule position checks every 5 minutes
    schedule.every(5).minutes.do(analyzer.check_positions)
    
    # Run full analysis immediately on start
    analyzer.run_scheduled_analysis()
    
    print(f"\nScheduler started:")
    print(f"- Full analysis and new positions every {config.ANALYSIS_INTERVAL} minutes")
    print(f"- Position checks every 5 minutes")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nScheduler stopped by user")
    except Exception as e:
        print(f"\nError in scheduler: {str(e)}")

if __name__ == "__main__":
    main() 