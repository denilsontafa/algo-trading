from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
from datetime import datetime, timedelta
import pandas as pd
import config

class OandaDataFetcher:
    def __init__(self):
        # Initialize OANDA API connection using config
        try:
            self.client = API(access_token=config.OANDA_API_KEY,
                          environment=config.OANDA_ENVIRONMENT)
            self.account_id = config.OANDA_ACCOUNT_ID
            print(f"Successfully connected to OANDA {config.OANDA_ENVIRONMENT} environment")
        except Exception as e:
            print(f"Error initializing OANDA API: {str(e)}")
            raise

    def get_current_price(self, instrument: str) -> float:
        """Get current price for an instrument"""
        params = {
            "count": 1,
            "granularity": "M1"
        }
        
        try:
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            self.client.request(r)
            return float(r.response['candles'][0]['mid']['c'])
        except Exception as e:
            print(f"Error getting current price for {instrument}: {str(e)}")
            return None

    def create_order(self, instrument: str, units: int, type: str = 'MARKET') -> dict:
        """Create a new order"""
        data = {
            "order": {
                "type": type,
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT"
            }
        }
        
        try:
            r = orders.OrderCreate(self.account_id, data=data)
            response = self.client.request(r)
            return response
        except Exception as e:
            print(f"Error creating order: {str(e)}")
            return None

    def close_trade(self, trade_id: str) -> dict:
        """Close a specific trade"""
        try:
            r = trades.TradeClose(self.account_id, trade_id)
            response = self.client.request(r)
            return response
        except Exception as e:
            print(f"Error closing trade: {str(e)}")
            return None

    def get_open_trades(self) -> list:
        """Get all open trades"""
        try:
            r = trades.OpenTrades(self.account_id)
            response = self.client.request(r)
            trades_list = response.get('trades', [])
            print(f"Raw trades response: {response}")  # Debug logging
            return trades_list
        except Exception as e:
            print(f"Error getting open trades: {str(e)}")
            print(f"Full error details: {type(e).__name__}: {str(e)}")
            return []
        
    def get_trade(self, trade_id: str) -> dict:
        """Get details of a specific trade"""
        try:
            # Fetch trade details using the trade ID
            r = trades.TradeDetails(self.account_id, trade_id)
            response = self.client.request(r)
            if response:
                return response
            else:
                print(f"No data received for trade ID {trade_id}")
                return None
        except Exception as e:
            print(f"Error fetching trade details for {trade_id}: {str(e)}")
            return None

    def fetch_historical_data(self, instrument: str, count: int = None, 
                             granularity: str = "M15") -> pd.DataFrame:
        """
        Fetch historical candlestick data from OANDA
        Can specify either count or time window (start_time and end_time)
        """
        params = {
            "count": count or config.CANDLE_COUNT,
            "granularity": granularity,
            "price": "M"
        }
        
        try:
            print(f"Fetching {params['count']} candles of {granularity} data for {instrument}...")
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            self.client.request(r)
            
            data = []
            for candle in r.response['candles']:
                if candle['complete']:
                    data.append({
                        'timestamp': datetime.strptime(candle['time'].split('.')[0], '%Y-%m-%dT%H:%M:%S'),
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    })
            
            df = pd.DataFrame(data)
            if len(df) == 0:
                print(f"No data received from OANDA for {instrument}")
                return None
                
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            print(f"Error fetching data for {instrument}: {str(e)}")
            return None