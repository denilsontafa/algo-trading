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
    """
    Function to be called every 15 minutes to update the model
    """
    try:
        # Initialize reinforcement trainer
        rl_trainer = ReinforcementTrainer(currency_pair)
        
        # Load the current model
        model_manager = ModelManager()
        model, metrics = model_manager.load_model(currency_pair)
        
        if model is None:
            print(f"No existing model found for {currency_pair}")
            return None
            
        model.to(rl_trainer.device)
        
        # Get latest data
        latest_data = oanda_fetcher.fetch_historical_data(
            instrument=currency_pair,
            count=61  # 60 for sequence + 1 for the actual outcome
        )
        
        if latest_data is None or len(latest_data) < 61:
            raise ValueError("Insufficient data fetched")
        
        # Process the data
        processed_data = data_processor.process_data(latest_data[:-1])  # All except last point
        actual_outcome = latest_data.iloc[-1]['close']  # Last point is the actual outcome
        
        # Reinforce the model
        results = rl_trainer.reinforce_model(
            model=model,
            latest_data=processed_data,
            actual_outcome=actual_outcome
        )
        
        print(f"\nModel Update Results for {currency_pair}:")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Reward: {results['reward']:.6f}")
        print(f"Loss: {results['loss']:.6f}")
        print(f"Prediction Error: {abs(results['prediction'] - results['actual']):.6f}")
        
        return results
        
    except Exception as e:
        print(f"Error updating model for {currency_pair}: {str(e)}")
        return None 