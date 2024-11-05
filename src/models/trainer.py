import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_manager import ModelManager
from models.lstm_model import ForexLSTM
from models.data_processor import ForexDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from datetime import datetime
import json

class ModelTrainer:
    def __init__(self, currency_pair: str):
        self.currency_pair = currency_pair
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_manager = ModelManager()
        
    def train_model(self, dataset: ForexDataset, epochs: int = 100, batch_size: int = 32):
        """Train a new model or continue training existing model"""
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        input_size = dataset[0][0].shape[1]
        model = ForexLSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        best_val_loss = float('inf')
        metrics = {
            'train_losses': [],
            'val_losses': [],
            'best_epoch': 0
        }
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                
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
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = model(X)
                    val_loss += criterion(outputs, y).item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Save metrics
            metrics['train_losses'].append(avg_train_loss)
            metrics['val_losses'].append(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                metrics['best_epoch'] = epoch
                
                # Save model checkpoint
                model_path = f"models/base_models/{self.currency_pair}_model.pth"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'model_config': {
                        'input_size': input_size,
                        'hidden_size': 64,
                        'num_layers': 2,
                        'dropout': 0.2
                    },
                    'metrics': metrics
                }
                
                torch.save(checkpoint, model_path)
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}')
        
        return model, metrics

if __name__ == "__main__":
    # Test training
    from data_processor import ForexDataset
    import pandas as pd
    
    # Load your data
    # Replace this with your actual data loading code
    data = pd.read_csv("path/to/your/data.csv")
    dataset = ForexDataset(data)
    
    trainer = ModelTrainer("EUR_USD")
    model, metrics = trainer.train_model(dataset)
    print("Training completed!")