from sentiment.news_scraper import InvestingNewsScraper
from sentiment.sentiment_analyzer import ForexSentimentAnalyzer
import pandas as pd
pd.set_option('display.max_columns', None)

def test_sentiment_analysis():
    scraper = InvestingNewsScraper()
    analyzer = ForexSentimentAnalyzer()
    
    currency_pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
    
    for pair in currency_pairs:
        print(f"\nAnalyzing sentiment for {pair}")
        print("=" * 50)
        
        # Fetch news
        news_df = scraper.fetch_news(currency_pair=pair)
        
        if not news_df.empty:
            # Analyze sentiment
            sentiment_results = analyzer.analyze_news(news_df, pair)
            
            # Print results
            print(f"\nSentiment Analysis Results:")
            print(f"Weighted Sentiment: {sentiment_results['weighted_sentiment']:.3f}")
            print(f"Sentiment Strength: {sentiment_results['sentiment_strength']:.3f}")
            print(f"Sentiment Volatility: {sentiment_results['sentiment_volatility']:.3f}")
            print(f"\nArticle Distribution:")
            print(f"Bullish: {sentiment_results['bullish_count']}")
            print(f"Bearish: {sentiment_results['bearish_count']}")
            print(f"Neutral: {sentiment_results['neutral_count']}")
            print(f"Total Articles: {sentiment_results['total_articles']}")
            
            # Print detailed scores for top 5 most relevant articles
            print("\nTop 5 Most Influential Articles:")
            detailed_scores = sentiment_results['detailed_scores']
            if not detailed_scores.empty:
                influence_score = (
                    detailed_scores['time_weight'] * 
                    detailed_scores['relevance_weight'] * 
                    detailed_scores['impact_weight']
                )
                detailed_scores['influence_score'] = influence_score
                top_articles = detailed_scores.nlargest(5, 'influence_score')
                
                for _, article in top_articles.iterrows():
                    print("\nTitle:", article['title'])
                    print(f"Sentiment: {article['sentiment_score']:.3f}")
                    print(f"Time Weight: {article['time_weight']:.3f}")
                    print(f"Relevance: {article['relevance_weight']:.3f}")
                    print(f"Impact: {article['impact_weight']:.3f}")
                    print(f"Influence: {article['influence_score']:.3f}")
        
        else:
            print("No news articles found")

if __name__ == "__main__":
    test_sentiment_analysis() 