import torch
from models.lstm_model import ForexLSTM
import os

def test_model_loading():
    # Create a test model
    model = ForexLSTM(input_size=12, hidden_size=64, num_layers=2, dropout=0.2)
    
    # Create a test checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': 12,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2
        },
        'metrics': {}
    }
    
    # Save the test model
    os.makedirs('models/base_models', exist_ok=True)
    torch.save(checkpoint, 'models/base_models/test_model.pth')
    
    # Try loading the model
    loaded_checkpoint = torch.load('models/base_models/test_model.pth')
    
    # Create new model instance
    new_model = ForexLSTM(input_size=12, hidden_size=64, num_layers=2, dropout=0.2)
    new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    
    print("Model loaded successfully!")
    print("\nModel state dict keys:", new_model.state_dict().keys())
    print("\nCheckpoint keys:", loaded_checkpoint.keys())
    
    # Clean up
    os.remove('models/base_models/test_model.pth')

if __name__ == "__main__":
    test_model_loading() 