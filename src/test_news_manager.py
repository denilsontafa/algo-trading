from sentiment.news_manager import NewsManager
from datetime import datetime
import time

def test_news_manager():
    manager = NewsManager()
    
    currency_pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
    
    for pair in currency_pairs:
        print(f"\nTesting news manager for {pair}")
        print("=" * 50)
        
        # Get news and sentiment
        news_data = manager.get_news_sentiment(pair)
        
        if news_data:
            print(f"\nTimestamp: {news_data['timestamp']}")
            print(f"Overall Sentiment: {news_data['overall_sentiment']:.4f}")
            print(f"Number of articles: {len(news_data['articles'])}")
            
            # Print latest 3 articles
            print("\nLatest articles:")
            for article in news_data['articles'][:3]:
                print(f"\nTitle: {article['title']}")
                print(f"Published: {article['published_at']}")
                print(f"Sentiment Score: {article['sentiment_score']:.4f}")
                print(f"Impact Weight: {article['impact_weight']:.4f}")
        
        # Get features that will be used for prediction
        features = manager.get_sentiment_features(pair)
        print("\nSentiment Features for Prediction:")
        for key, value in features.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        if pair != currency_pairs[-1]:
            print("\nWaiting 5 seconds before next pair...")
            time.sleep(5)

if __name__ == "__main__":
    test_news_manager() 