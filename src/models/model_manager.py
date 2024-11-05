import os
import json
import torch
from models.lstm_model import ForexLSTM

class ModelManager:
    def __init__(self, base_path="models/base_models"):
        self.base_path = base_path
        if not os.path.exists(base_path):
            os.makedirs(base_path)
    
    def save_model(self, model, currency_pair, metrics=None):
        """Save model and its metrics for a currency pair"""
        model_path = os.path.join(self.base_path, f"{currency_pair}_model.pth")
        
        # Get model configuration
        model_config = {
            'input_size': model.lstm.input_size,
            'hidden_size': model.lstm.hidden_size,
            'num_layers': model.lstm.num_layers
        }
        
        # Save model state and configuration
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'metrics': metrics
        }, model_path)
        
        print(f"Model saved for {currency_pair}")
        if metrics:
            print("Metrics:", metrics)
    
    def load_model(self, currency_pair):
        """Load model for a given currency pair"""
        model_path = os.path.join(self.base_path, f"{currency_pair}_model.pth")
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path)
                
                # Try to get model configuration
                if 'model_config' in checkpoint:
                    config = checkpoint['model_config']
                else:
                    # Default configuration if not found
                    config = {
                        'input_size': 8,  # Default for our feature set
                        'hidden_size': 100,
                        'num_layers': 2
                    }
                    print("Using default model configuration")
                
                # Create model with configuration
                model = ForexLSTM(
                    input_size=config['input_size'],
                    hidden_size=config['hidden_size'],
                    num_layers=config['num_layers']
                )
                
                model.load_state_dict(checkpoint['model_state_dict'])
                return model, checkpoint.get('metrics')
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                return None, None
        
        return None, None
    def _create_directory_structure(self):
        """Create the directory structure for storing models"""
        # Create base models directory
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
            
        # Create subdirectories for different purposes
        self.paths = {
            'base_models': os.path.join(self.base_path, 'base_models'),
            'ensemble_models': os.path.join(self.base_path, 'ensemble_models'),
            'metadata': os.path.join(self.base_path, 'metadata')
        }
        
        for path in self.paths.values():
            if not os.path.exists(path):
                os.makedirs(path)
    
    
    def get_model_metrics(self, currency_pair):
        """Get metrics for all models of a currency pair"""
        metrics_list = []
        metadata_files = os.listdir(self.paths['metadata'])
        
        for file in metadata_files:
            if file.startswith(currency_pair):
                with open(os.path.join(self.paths['metadata'], file), 'r') as f:
                    metadata = json.load(f)
                    metrics_list.append(metadata)
        
        return metrics_list 