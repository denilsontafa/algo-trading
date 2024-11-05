from sentiment.news_scraper import InvestingNewsScraper
import time

def test_news_scraper():
    scraper = InvestingNewsScraper()
    
    # Test general forex news
    print("\nFetching general forex news...")
    news_df = scraper.fetch_news(pages=2)
    
    if not news_df.empty:
        print(f"\nFound {len(news_df)} forex news articles")
        print("\nMost recent articles:")
        recent_articles = news_df.sort_values('published_at', ascending=False).head(3)
        
        for _, article in recent_articles.iterrows():
            print("\nTitle:", article['title'])
            print("Published:", article['published_at'])
            print("Link:", article['link'])
            if 'content' in article and article['content']:
                print("Content preview:", article['content'][:200] + "...")
            print("-" * 80)
    
    # Test currency-specific news
    currency_pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
    
    for pair in currency_pairs:
        print(f"\nFetching news relevant to {pair}...")
        
        news_df = scraper.fetch_news(currency_pair=pair, pages=2)
        
        if not news_df.empty:
            print(f"\nFound {len(news_df)} articles relevant to {pair}")
            print("\nMost recent articles:")
            recent_articles = news_df.sort_values('published_at', ascending=False).head(2)
            
            for _, article in recent_articles.iterrows():
                print("\nTitle:", article['title'])
                print("Published:", article['published_at'])
                print("Link:", article['link'])
                if 'content' in article and article['content']:
                    print("Content preview:", article['content'][:200] + "...")
                print("-" * 80)
        
        # Add delay between currency pairs
        if pair != currency_pairs[-1]:
            print("Waiting before next request...")
            time.sleep(5)

if __name__ == "__main__":
    test_news_scraper() 