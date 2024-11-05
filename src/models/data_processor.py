import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

class ForexDataset(Dataset):
    def __init__(self, data=None, sequence_length=60):
        self.sequence_length = sequence_length
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        if data is not None:
            self.fit_scalers(data)
            self.prepare_sequences(data)
    
    def fit_scalers(self, data):
        """Fit scalers with data"""
        processed_data = self._prepare_data(data)
        self.feature_scaler.fit(processed_data)
        self.target_scaler.fit(processed_data[:, 3:4])  # Close price
        
    def prepare_sequences(self, data):
        """Prepare sequences for training"""
        processed_data = self._prepare_data(data)
        self.scaled_data = self.feature_scaler.transform(processed_data)
        self.scaled_targets = self.target_scaler.transform(processed_data[:, 3:4])
        
        self.sequences = []
        self.targets = []
        
        for i in range(len(self.scaled_data) - self.sequence_length):
            self.sequences.append(self.scaled_data[i:(i + self.sequence_length)])
            self.targets.append(self.scaled_targets[i + self.sequence_length])
    
    def process_data(self, data):
        """Process new data using fitted scalers"""
        if data is None:
            raise ValueError("No data provided for processing")
            
        processed_data = self._prepare_data(data)
        scaled_data = self.feature_scaler.transform(processed_data)
        return torch.FloatTensor(scaled_data).unsqueeze(0)  # Add batch dimension
    
    def _prepare_data(self, data):
        """Calculate technical indicators"""
        df = data.copy()
        
        # Calculate basic indicators
        df['returns'] = df['close'].pct_change()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Fill NaN values with 0
        df = df.fillna(0)
        
        # Select features for training
        features = ['open', 'high', 'low', 'close', 'volume', 
                   'returns', 'sma_20', 'rsi']
        
        return df[features].values
    
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI value
    
    def __len__(self):
        if hasattr(self, 'sequences'):
            return len(self.sequences)
        return 0
    
    def __getitem__(self, idx):
        if hasattr(self, 'sequences'):
            return (torch.FloatTensor(self.sequences[idx]), 
                   torch.FloatTensor(self.targets[idx]))
        raise IndexError("Dataset not initialized with data")