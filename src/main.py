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
        # Profit target reached
        if position['pnl_pct'] >= self.profit_target_pct:
            return True
        
        # Stop loss hit
        if position['pnl_pct'] <= -self.stop_loss_pct:
            return True
        
        # Maximum hold time exceeded
        if position['hold_time'] >= self.max_hold_time:
            return True
        
        return False
    
    def _open_new_position(self, analyses: List[dict]) -> None:
        """Open new position if conditions are met"""
        if len(self.open_positions) >= self.max_positions:
            print("\nMaximum positions reached, not opening new positions")
            return
        
        # Find the highest confidence signal
        best_signal = None
        highest_confidence = 0.6  # Minimum confidence threshold
        
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
                
                if response and response.get('orderFilled'):
                    price = float(response['orderFilled']['price'])
                    self.open_positions[pair] = {
                        'direction': direction,
                        'entry_price': price,
                        'open_time': datetime.now(),
                        'confidence': best_signal['confidence'],
                        'target_price': price * (1 + self.profit_target_pct if direction == 'BUY' else 1 - self.profit_target_pct),
                        'stop_loss': price * (1 - self.stop_loss_pct if direction == 'BUY' else 1 + self.stop_loss_pct)
                    }
                    
                    print(f"\nSuccessfully opened {direction} position for {pair}:")
                    print(f"Entry: {price:.5f}")
                    print(f"Target: {self.open_positions[pair]['target_price']:.5f}")
                    print(f"Stop: {self.open_positions[pair]['stop_loss']:.5f}")
                    print(f"Confidence: {best_signal['confidence']:.2f}")
                else:
                    print(f"Failed to open position: No order fill confirmation")
            
            except Exception as e:
                print(f"Error opening position for {pair}: {str(e)}")
        else:
            print("\nNo signals meet the minimum confidence threshold (0.6)")
    
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
        
    def analyze_pair(self, currency_pair: str) -> dict:
        """Analyze a currency pair using all components"""
        # Get current market data
        current_data = self.data_fetcher.fetch_historical_data(currency_pair)
        
        if current_data is None:
            return None
            
        # 1. Get model prediction
        model_prediction = self._get_model_prediction(currency_pair, current_data)
        
        # 2. Get technical signals
        tech_signals = self.strategy.calculate_signals(current_data)
        
        # 3. Get sentiment analysis
        sentiment_data = self.news_manager.get_sentiment_features(currency_pair)
        
        # 4. Calculate combined signals and confidence
        current_price = current_data.close.iloc[-1]
        predicted_change = ((model_prediction - current_price) / current_price) * 100
        
        # Determine signal direction
        if tech_signals['trend_signal'] > 0.1:
            signal_direction = "BUY"
        elif tech_signals['trend_signal'] < -0.1:
            signal_direction = "SELL"
        else:
            signal_direction = "NEUTRAL"
        
        # Combine all analysis
        return {
            'pair': currency_pair,
            'current_price': current_price,
            'predicted_price': model_prediction,
            'predicted_change': predicted_change,
            'technical_signal': tech_signals['overall_signal'],
            'sentiment_score': sentiment_data['latest_sentiment'],
            'confidence': self._calculate_confidence(
                tech_signals['overall_signal'],
                sentiment_data['latest_sentiment'],
                predicted_change
            ),
            'volatility': tech_signals['volatility'],
            'trend': signal_direction,
            'news_impact': sentiment_data['max_impact']
        }
    
    def _get_model_prediction(self, pair: str, data: pd.DataFrame) -> float:
        """Get price prediction from the model"""
        try:
            # Load model
            model_path = f'models/model_{pair}.pth'
            if not os.path.exists(model_path):
                print(f"No model found for {pair}")
                return data.close.iloc[-1]
                
            try:
                # Load model with weights_only=True
                state_dict = torch.load(
                    model_path,
                    map_location=torch.device('cpu'),
                    weights_only=True  # Add this parameter
                )
                
                model = ForexLSTM(
                    input_size=len(config.TECHNICAL_INDICATORS),
                    hidden_size=64,
                    num_layers=2
                )
                model.load_state_dict(state_dict)
                model.eval()
                
                # Prepare features
                features = self._prepare_features(data)
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                
                # Make prediction
                with torch.no_grad():
                    prediction = model(features_tensor)
                    
                predicted_price = prediction.item()
                current_price = data.close.iloc[-1]
                
                # Calculate direction and maximum allowed change
                direction = np.sign(predicted_price - current_price)
                max_change = self._get_max_change_threshold(pair)
                
                # Limit the prediction to maximum allowed change
                if abs(predicted_price - current_price) / current_price > max_change:
                    predicted_price = current_price * (1 + direction * max_change)
                
                return predicted_price
                
            except Exception as e:
                print(f"Error during model prediction: {str(e)}")
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
        tech_conf = abs(tech_signal)
        sent_conf = abs(sentiment)
        pred_conf = min(abs(pred_change) / 0.5, 1.0)
        
        # Weight the components
        confidence = (
            0.4 * tech_conf +
            0.3 * sent_conf +
            0.3 * pred_conf
        )
        return min(confidence, 1.0)
    
    def run_scheduled_analysis(self):
        """Run analysis and update models"""
        try:
            print(f"\nForex Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Run analysis for all pairs
            analyses = []
            for pair in config.CURRENCY_PAIRS:
                analysis = self.analyze_pair(pair)
                if analysis:
                    analyses.append(analysis)
                    
                    # Update model if we have previous predictions
                    if pair in self.previous_predictions:
                        prev_pred = self.previous_predictions[pair]
                        results = update_model_with_latest_data(
                            currency_pair=pair,
                            oanda_fetcher=self.data_fetcher,
                            data_processor=self
                        )
                        
                        if results:
                            print(f"\nModel Update - {pair}:")
                            print(f"Previous Prediction: {prev_pred['price']:.5f}")
                            print(f"Actual Price: {analysis['current_price']:.5f}")
                            print(f"Reward: {results['reward']:.4f}")
                            print(f"Loss: {results['loss']:.6f}")
                    
                    # Store current prediction for next update
                    self.previous_predictions[pair] = {
                        'price': analysis['predicted_price'],
                        'timestamp': datetime.now()
                    }
            
            # Display results
            self._display_analysis(analyses)
            
            # Manage trading positions
            if analyses:  # Only manage positions if we have analyses
                self.position_manager.manage_positions(analyses)
                
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
            table_data.append([
                a['pair'],
                f"{a['current_price']:.5f}",
                f"{a['predicted_price']:.5f}",
                f"{a['predicted_change']:.2f}%",
                f"{a['technical_signal']:.2f}",
                f"{a['sentiment_score']:.2f}",
                f"{a['confidence']:.2f}",
                a['trend'],
                f"{a['volatility']:.2f}",
                f"{a['news_impact']:.2f}"
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # Print trading suggestions
        print("\nTrading Suggestions:")
        for a in analyses:
            if a['confidence'] > 0.6:  # Only show high confidence signals
                direction = "LONG" if a['predicted_change'] > 0 else "SHORT"
                print(f"\n{a['pair']}: {direction}")
                print(f"Entry: {a['current_price']:.5f}")
                print(f"Target: {a['predicted_price']:.5f}")
                print(f"Confidence: {a['confidence']:.2f}")
                if a['volatility'] > 0.6:
                    print("Warning: High volatility - Consider smaller position size")

def main():
    # Update saved scalers to current version
    update_saved_scalers()
    
    analyzer = ForexAnalyzer()
    
    # Schedule analysis using config interval
    schedule.every(config.ANALYSIS_INTERVAL).minutes.do(analyzer.run_scheduled_analysis)
    
    # Run immediately on start
    analyzer.run_scheduled_analysis()
    
    print(f"\nScheduler started with {config.ANALYSIS_INTERVAL} minute interval. Press Ctrl+C to stop.")
    
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