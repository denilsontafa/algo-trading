import torch
import numpy as np
from datetime import datetime, timedelta
import torch.nn as nn
from models.model_manager import ModelManager

class ReinforcementTrainer:
    def __init__(self, currency_pair, learning_rate=0.0001):
        self.currency_pair = currency_pair
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_manager = ModelManager()
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        
    def calculate_reward(self, predicted_price, actual_price, previous_price):
        """
        Calculate reward based on prediction accuracy and direction
        Returns higher reward for correct direction prediction and accuracy
        """
        # Direction prediction reward
        predicted_direction = np.sign(predicted_price - previous_price)
        actual_direction = np.sign(actual_price - previous_price)
        direction_reward = 1 if predicted_direction == actual_direction else -1
        
        # Accuracy reward (negative MSE scaled)
        accuracy_error = ((predicted_price - actual_price) ** 2)
        accuracy_reward = -accuracy_error
        
        # Combine rewards
        total_reward = direction_reward + accuracy_reward
        return total_reward
    
    def reinforce_model(self, model, latest_data, actual_outcome):
        """
        Update the model based on the latest market data and actual outcome
        
        Args:
            model: The current model
            latest_data: The input data used for the last prediction
            actual_outcome: The actual price that occurred
        """
        model.train()
        
        # Move data to device
        latest_data = latest_data.to(self.device)
        actual_outcome = torch.tensor(actual_outcome, device=self.device)
        
        # Get the previous prediction
        with torch.no_grad():
            previous_prediction = model(latest_data)
        
        # Calculate reward
        reward = self.calculate_reward(
            predicted_price=previous_prediction.item(),
            actual_price=actual_outcome.item(),
            previous_price=latest_data[-1][-1].item()  # Last known price
        )
        
        # Create optimizer for this update
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Forward pass
        prediction = model(latest_data)
        
        # Calculate loss with reward weighting
        loss = self.criterion(prediction, actual_outcome) * (1 - torch.tanh(torch.tensor(reward)))
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Save updated model
        metrics = {
            'last_reinforcement': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'reward': reward,
            'loss': loss.item()
        }
        self.model_manager.save_model(model, self.currency_pair, metrics)
        
        return {
            'reward': reward,
            'loss': loss.item(),
            'prediction': prediction.item(),
            'actual': actual_outcome.item()
        }

def update_model_with_latest_data(currency_pair, oanda_fetcher, data_processor):
    """Function to update model with data from the last 15 minutes"""
    try:
        # Calculate time window
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=15)
        
        print(f"\nUpdating model for {currency_pair}")
        print(f"Time window: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get training window data (60 periods for context + last 15 min)
        training_data = oanda_fetcher.fetch_historical_data(
            instrument=currency_pair,
            count=65,  # 60 for context + 5 for last 15 min (M3 granularity)
            granularity="M3"  # Use M3 granularity for more precise updates
        )
        
        if training_data is None or len(training_data) < 65:
            print(f"Insufficient training data for {currency_pair}")
            return None
            
        # Split data into context and recent
        context_data = training_data[:-5]  # First 60 periods
        recent_data = training_data[-5:]   # Last 5 periods (15 minutes)
        
        # Initialize trainer and load model
        rl_trainer = ReinforcementTrainer(currency_pair)
        model_manager = ModelManager()
        model, metrics = model_manager.load_model(currency_pair)
        
        if model is None:
            print(f"No existing model found for {currency_pair}")
            return None
            
        model.to(rl_trainer.device)
        
        # Process context data
        processed_context = data_processor.process_data(context_data)
        
        # Get actual outcomes from recent data
        actual_outcomes = recent_data['close'].values
        
        # Perform reinforcement learning updates
        results = []
        for i, actual_price in enumerate(actual_outcomes):
            # Make prediction using context
            with torch.no_grad():
                prediction = model(processed_context)
                
            # Calculate reward and update model
            result = rl_trainer.reinforce_model(
                model=model,
                latest_data=processed_context,
                actual_outcome=actual_price
            )
            results.append(result)
            
            # Update context for next prediction
            if i < len(actual_outcomes) - 1:
                new_data = recent_data.iloc[i:i+1]
                processed_new = data_processor.process_data(new_data)
                processed_context = torch.cat([processed_context[1:], processed_new], dim=0)
        
        # Calculate average metrics
        avg_results = {
            'reward': np.mean([r['reward'] for r in results]),
            'loss': np.mean([r['loss'] for r in results]),
            'prediction_error': np.mean([abs(r['prediction'] - r['actual']) for r in results])
        }
        
        print(f"\nModel Update Results for {currency_pair}:")
        print(f"Updates performed: {len(results)}")
        print(f"Average Reward: {avg_results['reward']:.6f}")
        print(f"Average Loss: {avg_results['loss']:.6f}")
        print(f"Average Prediction Error: {avg_results['prediction_error']:.6f}")
        
        return avg_results
        
    except Exception as e:
        print(f"Error updating model for {currency_pair}: {str(e)}")
        return None 