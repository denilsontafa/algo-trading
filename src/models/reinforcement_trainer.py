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
        
        # Calculate percentage error instead of absolute error
        percentage_error = abs((predicted_price - actual_price) / actual_price)
        # Scale the accuracy reward to be between -1 and 0
        accuracy_reward = -np.tanh(percentage_error)
        
        # Combine rewards with proper scaling
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
        
        # Ensure data is in correct shape and type
        if len(latest_data.shape) == 2:
            latest_data = latest_data.unsqueeze(0)
        
        # Convert to float32 and move to device
        latest_data = latest_data.float().to(self.device)
        actual_outcome = torch.tensor([[actual_outcome]], dtype=torch.float32, device=self.device)
        
        # Get the previous prediction
        with torch.no_grad():
            previous_prediction = model(latest_data)
        
        # Calculate reward
        reward = self.calculate_reward(
            predicted_price=previous_prediction.item(),
            actual_price=actual_outcome.item(),
            previous_price=latest_data[0, -1, -1].item()
        )
        
        # Create optimizer for this update
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Forward pass
        prediction = model(latest_data)
        
        # Calculate percentage error for loss
        percentage_error = torch.abs((prediction - actual_outcome) / actual_outcome)
        loss = torch.tanh(percentage_error) * (1 - torch.tanh(torch.tensor(reward)))
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Save updated model
        metrics = {
            'last_reinforcement': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'reward': float(reward),
            'loss': float(loss.item())
        }
        self.model_manager.save_model(model, self.currency_pair, metrics)
        
        return {
            'reward': float(reward),
            'loss': float(loss.item()),
            'prediction': float(prediction.item()),
            'actual': float(actual_outcome.item())
        }

    def calculate_loss(self, predicted_values, actual_values, rewards):
        """Calculate loss with normalized values"""
        try:
            # Convert to tensors if they aren't already
            predicted_values = torch.tensor(predicted_values, dtype=torch.float32)
            actual_values = torch.tensor(actual_values, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32)

            # Normalize the values
            predicted_norm = (predicted_values - predicted_values.mean()) / (predicted_values.std() + 1e-8)
            actual_norm = (actual_values - actual_values.mean()) / (actual_values.std() + 1e-8)
            
            # Calculate MSE loss on normalized values
            mse_loss = nn.MSELoss()(predicted_norm, actual_norm)
            
            # Scale rewards to be in a similar range (-1 to 1)
            rewards_norm = torch.tanh(rewards / 100.0)  # Divide by 100 to reduce magnitude
            
            # Combine losses with appropriate weights
            total_loss = mse_loss - 0.1 * rewards_norm.mean()  # Reduced weight on rewards
            
            return total_loss.item()

        except Exception as e:
            print(f"Error calculating loss: {str(e)}")
            return 0.0

def update_model_with_latest_data(currency_pair, oanda_fetcher, data_processor):
    """Function to update model with data from the last 15 minutes"""
    try:
        # Calculate time window
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=15)
        
        print(f"\nUpdating model for {currency_pair}")
        print(f"Time window: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get training window data
        training_data = oanda_fetcher.fetch_historical_data(
            instrument=currency_pair,
            count=200,
            granularity="M5"
        )
        
        if training_data is None or len(training_data) < 60:
            print(f"Insufficient training data for {currency_pair}")
            return None
            
        # Process all data at once to ensure consistent scaling
        processed_data = data_processor.process_data(training_data)
        
        # Split processed data into context and recent
        context_data = processed_data[:, :-3, :]  # All but last 3 timesteps
        recent_prices = training_data['close'].values[-3:]  # Last 3 actual prices
        
        # Initialize trainer and load model
        rl_trainer = ReinforcementTrainer(currency_pair)
        model_manager = ModelManager()
        model, metrics = model_manager.load_model(currency_pair)
        
        if model is None:
            print(f"No existing model found for {currency_pair}")
            return None
            
        model.to(rl_trainer.device)
        
        # Perform reinforcement learning updates
        results = []
        current_context = context_data.clone()
        
        for i, actual_price in enumerate(recent_prices):
            # Make prediction using current context
            result = rl_trainer.reinforce_model(
                model=model,
                latest_data=current_context,
                actual_outcome=actual_price
            )
            results.append(result)
            
            # Update context for next prediction if not last iteration
            if i < len(recent_prices) - 1:
                # Shift context window forward
                if i < processed_data.size(1) - 1:
                    current_context = processed_data[:, i+1:i+1+current_context.size(1), :]
        
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