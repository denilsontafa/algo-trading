import json
import os
from datetime import datetime, timedelta
import pandas as pd
from .news_scraper import InvestingNewsScraper
from .sentiment_analyzer import ForexSentimentAnalyzer
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import List, Dict
import config

class NewsManager:
    def __init__(self, storage_dir='data/news'):
        self.storage_dir = storage_dir
        self.scraper = InvestingNewsScraper()
        self.analyzer = ForexSentimentAnalyzer()
        
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

        nltk.download('vader_lexicon', quiet=True)
        self.sia = SentimentIntensityAnalyzer()
        self.news_cache: Dict[str, List[Dict]] = {}
        self.cache_expiry = timedelta(minutes=15)
        self.last_update = datetime.now() - self.cache_expiry

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

    def get_pair_sentiment(self, pair: str) -> float:
        """Calculate sentiment score for a currency pair"""
        try:
            # Get relevant news
            news_items = self._get_relevant_news(pair)
            if not news_items:
                return 0.0
            
            # Calculate sentiment for each news item
            sentiments = []
            for news in news_items:
                if 'title' in news:
                    sentiment = self.sia.polarity_scores(news['title'])
                    sentiments.append(sentiment['compound'])
            
            # Return average sentiment if we have any
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                print(f"\nNews sentiment for {pair}:")
                print(f"Number of news items: {len(sentiments)}")
                print(f"Average sentiment: {avg_sentiment:.3f}")
                return avg_sentiment
            
            return 0.0
            
        except Exception as e:
            print(f"Error calculating sentiment: {str(e)}")
            return 0.0
    
    def get_news_impact(self, pair: str) -> float:
        """Calculate news impact score (0-1) based on relevance and recency"""
        try:
            news_items = self._get_relevant_news(pair)
            if not news_items:
                return 0.0
            
            # Calculate impact based on number of recent relevant news
            impact = min(len(news_items) / 10, 1.0)  # Cap at 1.0
            
            print(f"\nNews impact for {pair}:")
            print(f"Number of relevant news: {len(news_items)}")
            print(f"Impact score: {impact:.2f}")
            
            return impact
            
        except Exception as e:
            print(f"Error calculating news impact: {str(e)}")
            return 0.0
    
    def _get_relevant_news(self, pair: str) -> List[Dict]:
        """Get news relevant to the currency pair"""
        try:
            # Check if we need to update cache
            if datetime.now() - self.last_update > self.cache_expiry:
                self._update_news_cache()
            
            # Get news for the pair
            currencies = pair.split('_')
            relevant_news = []
            
            for news in self.news_cache.get(pair, []):
                # Check if news mentions either currency
                if any(curr in news.get('title', '').upper() for curr in currencies):
                    relevant_news.append(news)
            
            return relevant_news
            
        except Exception as e:
            print(f"Error getting relevant news: {str(e)}")
            return []
    
    def _update_news_cache(self):
        """Update news cache with latest forex news from real sources"""
        try:
            print("\nFetching latest news for all currency pairs...")
            
            # Initialize scraper if not already done
            if not hasattr(self, 'scraper'):
                self.scraper = InvestingNewsScraper()
            
            # Fetch and cache news for each currency pair
            self.news_cache = {}
            
            for pair in config.CURRENCY_PAIRS:
                print(f"\nFetching news for {pair}...")
                
                # Fetch news using the scraper
                news_df = self.scraper.fetch_news(pair)
                
                if not news_df.empty:
                    # Convert DataFrame rows to dictionary format
                    pair_news = []
                    for _, row in news_df.iterrows():
                        news_item = {
                            'title': row['title'],
                            'description': row['description'],
                            'published_at': row['published_at'].isoformat(),
                            'provider': row['provider'],
                            'link': row['link']
                        }
                        pair_news.append(news_item)
                    
                    self.news_cache[pair] = pair_news
                    print(f"✓ Cached {len(pair_news)} articles for {pair}")
                else:
                    self.news_cache[pair] = []
                    print(f"! No articles found for {pair}")
            
            self.last_update = datetime.now()
            print(f"\nNews cache updated at {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Save cache to file for backup
            cache_file = os.path.join(self.storage_dir, 'news_cache.json')
            with open(cache_file, 'w', encoding='utf-8') as f:
                cache_data = {
                    'last_update': self.last_update.isoformat(),
                    'news': self.news_cache
                }
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error updating news cache: {str(e)}")
            # If error occurs, try to load from backup file
            try:
                cache_file = os.path.join(self.storage_dir, 'news_cache.json')
                if os.path.exists(cache_file):
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                        self.news_cache = cache_data['news']
                        self.last_update = datetime.fromisoformat(cache_data['last_update'])
                        print("Loaded news from backup cache file")
            except Exception as backup_error:
                print(f"Error loading backup cache: {str(backup_error)}")