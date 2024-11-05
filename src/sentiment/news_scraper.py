import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import random
import config

class InvestingNewsScraper:
    def __init__(self):
        self.base_url = "https://www.investing.com"
        self.forex_news_url = "/news/forex-news"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.cache = {}
        self.cache_duration = timedelta(minutes=config.NEWS_UPDATE_INTERVAL)

    def _get_soup(self, url):
        """Make request and get BeautifulSoup object with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                time.sleep(random.uniform(1, 3))
                
                full_url = self.base_url + url if not url.startswith('http') else url
                print(f"Fetching: {full_url}")
                
                response = requests.get(full_url, headers=self.headers)
                response.raise_for_status()
                
                return BeautifulSoup(response.text, 'html.parser')
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(random.uniform(2, 5))

    def _parse_news_item(self, article):
        """Parse a single news article element"""
        try:
            # Find title and link using the new selectors
            title_elem = article.find('a', {'data-test': 'article-title-link'})
            if not title_elem:
                return None

            title = title_elem.text.strip()
            link = title_elem.get('href', '')
            if link and not link.startswith('http'):
                link = self.base_url + link

            # Find description
            desc_elem = article.find('p', {'data-test': 'article-description'})
            description = desc_elem.text.strip() if desc_elem else ''

            # Find provider and date
            provider_elem = article.find('span', {'data-test': 'news-provider-name'})
            provider = provider_elem.text.strip() if provider_elem else ''

            # Find publish date
            date_elem = article.find('time', {'data-test': 'article-publish-date'})
            if date_elem:
                datetime_str = date_elem.get('datetime')
                if datetime_str:
                    try:
                        published_at = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        published_at = datetime.now()
                else:
                    # Handle relative time (e.g., "1 hour ago")
                    time_text = date_elem.text.strip().lower()
                    published_at = self._parse_relative_time(time_text)
            else:
                published_at = datetime.now()

            return {
                'title': title,
                'link': link,
                'description': description,
                'provider': provider,
                'published_at': published_at
            }
            
        except Exception as e:
            print(f"Error parsing news item: {str(e)}")
            return None

    def _parse_relative_time(self, time_text):
        """Parse relative time strings like '1 hour ago'"""
        now = datetime.now()
        try:
            if 'minute' in time_text:
                minutes = int(time_text.split()[0])
                return now - timedelta(minutes=minutes)
            elif 'hour' in time_text:
                hours = int(time_text.split()[0])
                return now - timedelta(hours=hours)
            elif 'day' in time_text:
                days = int(time_text.split()[0])
                return now - timedelta(days=days)
            else:
                return now
        except:
            return now

    def fetch_news(self, currency_pair=None, pages=config.NEWS_PAGES_TO_FETCH):
        """Fetch forex news and filter by currency pair if specified"""
        try:
            all_news = []
            
            for page in range(1, pages + 1):
                url = self.forex_news_url
                if page > 1:
                    url += f"/{page}"
                
                soup = self._get_soup(url)
                
                # Find all articles using the new selector
                articles = soup.find_all('article', {'data-test': 'article-item'})
                
                print(f"Found {len(articles)} articles on page {page}")
                
                for article in articles:
                    item = self._parse_news_item(article)
                    if item:
                        # Add the article if it's relevant or no currency pair filter
                        if not currency_pair or self._is_relevant_to_pair(
                            item['title'] + ' ' + item['description'], 
                            currency_pair
                        ):
                            all_news.append(item)
                
                time.sleep(random.uniform(2, 4))

            # Convert to DataFrame
            news_df = pd.DataFrame(all_news)
            if not news_df.empty:
                news_df = news_df.sort_values('published_at', ascending=False)
            
            print(f"Found {len(news_df)} relevant articles")
            return news_df

        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return pd.DataFrame()

    def _is_relevant_to_pair(self, text, currency_pair):
        """Check if the news article is relevant to the currency pair"""
        relevant_terms = self._get_relevant_terms(currency_pair)
        text_lower = text.lower()
        
        # Check currency pair specific terms
        if any(term.lower() in text_lower for term in relevant_terms):
            return True
            
        # Check high-impact economic indicators
        if any(indicator.lower() in text_lower 
              for indicator in config.ECONOMIC_INDICATORS['high_impact']):
            return True
            
        return False

    def _get_relevant_terms(self, currency_pair):
        """Get relevant terms for a currency pair"""
        currencies = currency_pair.split('_')
        terms = []
        
        for currency in currencies:
            if currency in config.CURRENCY_CONFIG:
                terms.extend(config.CURRENCY_CONFIG[currency]['terms'])
                terms.extend(config.CURRENCY_CONFIG[currency]['central_bank']['terms'])
        
        return terms