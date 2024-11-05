from oanda_data import OandaDataFetcher
from models.data_processor import ForexDataset
from models.reinforcement_trainer import update_model_with_latest_data
import time

def test_reinforcement_learning():
    currency_pair = "EUR_USD"
    
    # Initialize required components
    oanda_fetcher = OandaDataFetcher()
    
    # First fetch some initial data to fit the scalers
    initial_data = oanda_fetcher.fetch_historical_data(
        instrument=currency_pair,
        count=100  # Fetch enough data to fit scalers
    )
    
    # Initialize data processor with initial data
    data_processor = ForexDataset(initial_data)
    
    print(f"Testing reinforcement learning for {currency_pair}")
    print("Will perform 3 updates to simulate periodic learning...")
    
    for i in range(3):
        print(f"\nUpdate {i+1}:")
        results = update_model_with_latest_data(
            currency_pair=currency_pair,
            oanda_fetcher=oanda_fetcher,
            data_processor=data_processor
        )
        
        if results:
            print("Update successful")
        else:
            print("Update failed")
            
        # Wait for 1 minute before next update (in real implementation this would be 15 minutes)
        if i < 2:  # Don't wait after the last iteration
            print("Waiting for 1 minute...")
            time.sleep(60)

if __name__ == "__main__":
    test_reinforcement_learning() 