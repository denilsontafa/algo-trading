import json
import os
from datetime import datetime, timedelta
import pandas as pd
from .news_scraper import InvestingNewsScraper
from .sentiment_analyzer import ForexSentimentAnalyzer
import numpy as np

class NewsManager:
    def __init__(self, storage_dir='data/news'):
        self.storage_dir = storage_dir
        self.scraper = InvestingNewsScraper()
        self.analyzer = ForexSentimentAnalyzer()
        
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    def _get_news_file_path(self, currency_pair):
        return os.path.join(self.storage_dir, f"{currency_pair}_news.json")

    def _load_latest_news(self, currency_pair):
        """Load latest news from JSON file"""
        file_path = self._get_news_file_path(currency_pair)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data
            except Exception as e:
                print(f"Error loading news data: {str(e)}")
        return None

    def _save_news_data(self, currency_pair, news_data):
        """Save latest news data to JSON file"""
        file_path = self._get_news_file_path(currency_pair)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(news_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving news data: {str(e)}")

    def get_news_sentiment(self, currency_pair):
        """Get latest news and their sentiment analysis"""
        current_time = datetime.now()
        existing_data = self._load_latest_news(currency_pair)
        
        # Check if we need to fetch new data
        should_fetch_new = True
        if existing_data:
            last_update = datetime.fromisoformat(existing_data['timestamp'])
            if (current_time - last_update).total_seconds() < 15 * 60:  # 15 minutes
                return existing_data
        
        # Fetch new news
        news_df = self.scraper.fetch_news(currency_pair)
        
        if not news_df.empty:
            # Process each article
            articles_data = []
            total_sentiment = 0
            total_weight = 0
            
            for _, article in news_df.iterrows():
                # Analyze sentiment
                sentiment = self.analyzer._analyze_article(article, currency_pair)
                
                # Calculate weighted sentiment
                weight = sentiment['time_weight'] * sentiment['impact_weight']
                total_sentiment += sentiment['sentiment_score'] * weight
                total_weight += weight
                
                # Store article data
                articles_data.append({
                    'title': article['title'],
                    'published_at': article['published_at'].isoformat(),
                    'sentiment_score': sentiment['sentiment_score'],
                    'impact_weight': sentiment['impact_weight']
                })
            
            # Create news data structure
            news_data = {
                'timestamp': current_time.isoformat(),
                'articles': articles_data,
                'overall_sentiment': total_sentiment / total_weight if total_weight > 0 else 0
            }
            
            # Save the data
            self._save_news_data(currency_pair, news_data)
            return news_data
        
        # If no new articles, return existing data or empty structure
        return existing_data or {
            'timestamp': current_time.isoformat(),
            'articles': [],
            'overall_sentiment': 0
        }

    def get_sentiment_features(self, currency_pair):
        """Get sentiment features for price prediction"""
        news_data = self.get_news_sentiment(currency_pair)
        
        if news_data and news_data['articles']:
            # Calculate features
            sentiments = [article['sentiment_score'] for article in news_data['articles']]
            impacts = [article['impact_weight'] for article in news_data['articles']]
            
            return {
                'latest_sentiment': news_data['overall_sentiment'],
                'sentiment_std': np.std(sentiments),
                'max_impact': max(impacts),
                'article_count': len(news_data['articles']),
                'last_update': news_data['timestamp']
            }
        
        return {
            'latest_sentiment': 0,
            'sentiment_std': 0,
            'max_impact': 0,
            'article_count': 0,
            'last_update': datetime.now().isoformat()
        }