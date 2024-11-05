import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from textblob import TextBlob
import config

class ForexSentimentAnalyzer:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        self.sia = SentimentIntensityAnalyzer()
        # Add forex-specific terms to VADER
        self.sia.lexicon.update(config.SENTIMENT_TERMS)
    
    def analyze_news(self, news_df, currency_pair):
        """Analyze sentiment of news articles with multiple factors"""
        if news_df.empty:
            return self._create_empty_sentiment_result()
        
        # Calculate individual scores for each article
        scores = []
        for _, article in news_df.iterrows():
            score = self._analyze_article(article, currency_pair)
            scores.append(score)
        
        # Convert to DataFrame for easier analysis
        scores_df = pd.DataFrame(scores)
        
        # Calculate weighted average sentiment
        if not scores_df.empty:
            weighted_sentiment = (
                scores_df['sentiment_score'] * 
                scores_df['time_weight'] * 
                scores_df['relevance_weight'] * 
                scores_df['impact_weight']
            ).sum() / (
                scores_df['time_weight'] * 
                scores_df['relevance_weight'] * 
                scores_df['impact_weight']
            ).sum()
            
            # Calculate sentiment strength and volatility
            sentiment_strength = abs(weighted_sentiment)
            sentiment_volatility = scores_df['sentiment_score'].std()
            
            return {
                'weighted_sentiment': weighted_sentiment,
                'sentiment_strength': sentiment_strength,
                'sentiment_volatility': sentiment_volatility,
                'bullish_count': len(scores_df[scores_df['sentiment_score'] > 0.2]),
                'bearish_count': len(scores_df[scores_df['sentiment_score'] < -0.2]),
                'neutral_count': len(scores_df[abs(scores_df['sentiment_score']) <= 0.2]),
                'total_articles': len(scores_df),
                'detailed_scores': scores_df,
                'timestamp': datetime.now()
            }
        
        return self._create_empty_sentiment_result()
    
    def _analyze_article(self, article, currency_pair):
        """Analyze a single article considering multiple factors"""
        # Basic sentiment analysis using both VADER and TextBlob
        text = f"{article['title']} {article['description']}"
        vader_scores = self.sia.polarity_scores(text)
        textblob_score = TextBlob(text).sentiment.polarity
        
        # Combine VADER and TextBlob scores (VADER has more weight)
        combined_score = vader_scores['compound'] * 0.7 + textblob_score * 0.3
        
        # Calculate time weight (more recent = more important)
        time_weight = self._calculate_time_weight(article['published_at'])
        
        # Calculate relevance weight
        relevance_weight = self._calculate_relevance_weight(text, currency_pair)
        
        # Calculate impact weight
        impact_weight = self._calculate_impact_weight(text)
        
        return {
            'title': article['title'],
            'published_at': article['published_at'],
            'sentiment_score': combined_score,
            'time_weight': time_weight,
            'relevance_weight': relevance_weight,
            'impact_weight': impact_weight,
            'vader_score': vader_scores['compound'],
            'textblob_score': textblob_score
        }
    
    def _calculate_time_weight(self, published_at):
        """Calculate time-based weight (exponential decay)"""
        hours_old = (datetime.now() - published_at).total_seconds() / 3600
        return np.exp(-0.1 * hours_old)  # Exponential decay
    
    def _calculate_relevance_weight(self, text, currency_pair):
        """Calculate relevance weight based on currency terms"""
        text_lower = text.lower()
        currencies = currency_pair.split('_')
        relevance_score = 0
        
        for currency in currencies:
            if currency in config.CURRENCY_CONFIG:
                # Check currency-specific terms
                currency_terms = config.CURRENCY_CONFIG[currency]['terms']
                matches = sum(term.lower() in text_lower for term in currency_terms)
                relevance_score += matches * 0.2
                
                # Check central bank terms (higher weight)
                bank_terms = config.CURRENCY_CONFIG[currency]['central_bank']['terms']
                matches = sum(term.lower() in text_lower for term in bank_terms)
                relevance_score += matches * 0.4
        
        return min(1.0, 0.5 + relevance_score)  # Base weight 0.5, max 1.0
    
    def _calculate_impact_weight(self, text):
        """Calculate impact weight based on economic indicators"""
        text_lower = text.lower()
        impact_score = 0.5  # Base impact
        
        # Check high-impact indicators
        for indicator in config.ECONOMIC_INDICATORS['high_impact']:
            if indicator.lower() in text_lower:
                impact_score = max(impact_score, 1.0)
                break
        
        # Check medium-impact indicators
        for indicator in config.ECONOMIC_INDICATORS['medium_impact']:
            if indicator.lower() in text_lower:
                impact_score = max(impact_score, 0.75)
                break
        
        # Check low-impact indicators
        for indicator in config.ECONOMIC_INDICATORS['low_impact']:
            if indicator.lower() in text_lower:
                impact_score = max(impact_score, 0.5)
                break
        
        return impact_score
    
    def _create_empty_sentiment_result(self):
        """Create an empty sentiment result structure"""
        return {
            'weighted_sentiment': 0,
            'sentiment_strength': 0,
            'sentiment_volatility': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'total_articles': 0,
            'detailed_scores': pd.DataFrame(),
            'timestamp': datetime.now()
        } 