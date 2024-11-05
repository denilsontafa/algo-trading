import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import ta
import numpy as np
import config

class ForexDataset(Dataset):
    def __init__(self, data=None, sequence_length=60):
        self.sequence_length = sequence_length
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        if data is not None:
            # Calculate features first
            features_df = self._calculate_features(data)
            
            # Fit scalers
            self.feature_scaler.fit(features_df)
            self.target_scaler.fit(data['close'].values.reshape(-1, 1))
            
            # Transform and store data
            self.features = self.feature_scaler.transform(features_df)
            self.targets = self.target_scaler.transform(data['close'].values.reshape(-1, 1)).flatten()
    
    def _calculate_features(self, data):
        """Calculate technical indicators"""
        df = data.copy()
        
        # Calculate features
        df['returns'] = df['close'].pct_change()
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close']
        ).average_true_range()
        
        # Fill NaN values
        for col in config.TECHNICAL_INDICATORS:
            df[col] = df[col].ffill().bfill()
            
        return df[config.TECHNICAL_INDICATORS]
    
    def process_data(self, data):
        """Process new data using fitted scalers"""
        if not hasattr(self.feature_scaler, 'n_features_in_'):
            raise ValueError("Scalers not fitted. Initialize with data first.")
            
        # Calculate features
        features_df = self._calculate_features(data)
        
        # Transform features
        features = self.feature_scaler.transform(features_df)
        
        # Reshape for LSTM input: [batch_size, sequence_length, n_features]
        if len(features.shape) == 2:
            features = features.reshape(1, features.shape[0], features.shape[1])
            
        # Convert to float32
        return torch.FloatTensor(features).float()
    
    def __len__(self):
        return len(self.features) - self.sequence_length if hasattr(self, 'features') else 0
        
    def __getitem__(self, idx):
        if not hasattr(self, 'features') or not hasattr(self, 'targets'):
            raise RuntimeError("Dataset not properly initialized with data")
            
        X = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return torch.FloatTensor(X), torch.FloatTensor([y])

def update_saved_scalers():
    """Update saved scalers to current version if needed"""
    print("Checking saved scalers...")
    # Implementation for updating scalers if needed
    pass