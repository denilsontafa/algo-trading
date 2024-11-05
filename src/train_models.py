import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import ForexLSTM
from oanda_data import OandaDataFetcher
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ta
import pickle
import config

class ForexDataset(Dataset):
    def __init__(self, data, sequence_length=60):
        self.sequence_length = sequence_length
        self.features = self._prepare_features(data)
        self.targets, self.target_scaler = self._prepare_targets(data)
        
    def __len__(self):
        return len(self.features) - self.sequence_length
        
    def __getitem__(self, idx):
        X = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return torch.FloatTensor(X), torch.FloatTensor([y])
    
    def _prepare_features(self, data):
        df = data.copy()
        
        # Core features (8 in total)
        df['returns'] = df['close'].pct_change()
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
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
            df[col] = df[col].ffill().bfill()
        
        # Scale features
        scaler = MinMaxScaler()
        features = scaler.fit_transform(df[feature_columns])
        
        return features
    
    def _prepare_targets(self, data):
        # Normalize the target values (close prices)
        target_scaler = MinMaxScaler()
        targets = target_scaler.fit_transform(data['close'].values.reshape(-1, 1))
        return targets.flatten(), target_scaler

def train_model(currency_pair, data, epochs=100, batch_size=32):
    """Train a model for a specific currency pair"""
    # Create dataset
    dataset = ForexDataset(data)
    
    # Save the target scaler for later use
    os.makedirs('models/scalers', exist_ok=True)
    with open(f'models/scalers/{currency_pair}_scaler.pkl', 'wb') as f:
        pickle.dump(dataset.target_scaler, f)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    input_size = dataset[0][0].shape[1]  # Number of features
    model = ForexLSTM(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop with early stopping
    early_stopping_patience = 20
    best_val_loss = float('inf')
    no_improve_count = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                outputs = model(X)
                val_loss += criterion(outputs, y).item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save best model and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0
            
            # Save model checkpoint
            os.makedirs('models/base_models', exist_ok=True)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_size': input_size,
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.3
                },
                'metrics': {
                    'best_val_loss': best_val_loss,
                    'epochs_trained': epoch + 1
                }
            }
            torch.save(checkpoint, f'models/base_models/{currency_pair}_model.pth')
        else:
            no_improve_count += 1
            if no_improve_count >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return model, {'best_val_loss': best_val_loss}

def check_existing_model(currency_pair):
    """Check if a trained model already exists"""
    model_path = f'models/base_models/{currency_pair}_model.pth'
    scaler_path = f'models/scalers/{currency_pair}_scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            # Try to load the model to verify it's valid
            checkpoint = torch.load(model_path)
            print(f"✓ Found existing model for {currency_pair}")
            print(f"  Trained for {checkpoint['metrics']['epochs_trained']} epochs")
            print(f"  Best validation loss: {checkpoint['metrics']['best_val_loss']:.6f}")
            return True
        except Exception as e:
            print(f"! Found corrupted model for {currency_pair}: {str(e)}")
            return False
    return False

def main():
    # Initialize data fetcher
    data_fetcher = OandaDataFetcher()
    
    # Ensure directories exist
    os.makedirs('models/base_models', exist_ok=True)
    os.makedirs('models/scalers', exist_ok=True)
    
    # Get list of currency pairs from config
    pairs = config.CURRENCY_PAIRS
    
    print("\nChecking existing models...")
    for pair in pairs:
        if check_existing_model(pair):
            continue
            
        print(f"\nTraining new model for {pair}")
        try:
            # Get historical data
            data = data_fetcher.fetch_historical_data(pair)
            if data is not None:
                # Train model
                model, metrics = train_model(pair, data)
                print(f"✓ Successfully trained model for {pair}")
                print(f"  Best validation loss: {metrics['best_val_loss']:.6f}")
            else:
                print(f"✗ Failed to fetch data for {pair}")
                
        except Exception as e:
            print(f"✗ Error training model for {pair}: {str(e)}")

if __name__ == "__main__":
    main()