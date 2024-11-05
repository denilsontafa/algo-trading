from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import config

class OandaAPI:
    def __init__(self):
        self.api = API(access_token=config.OANDA_API_KEY,
                      environment=config.OANDA_ENVIRONMENT)
        
    def get_candles(self, pair, count=200, granularity="M15"):
        """
        Fetch historical candles from Oanda
        Returns pandas DataFrame with OHLCV data
        """
        try:
            # Create the request
            params = {
                "count": count,
                "granularity": granularity,
                "price": "M"  # Midpoint data
            }
            
            # Create the instruments request
            r = instruments.InstrumentsCandles(instrument=pair, params=params)
            
            # Make the request
            self.api.request(r)
            
            # Extract candle data
            candles_data = []
            for candle in r.response['candles']:
                if candle['complete']:  # Only use complete candles
                    candles_data.append({
                        'timestamp': pd.to_datetime(candle['time']),
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    })
            
            # Create DataFrame
            df = pd.DataFrame(candles_data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                print(f"Fetched {len(df)} candles for {pair}")
            
            return df
            
        except Exception as e:
            print(f"Error fetching candles for {pair}: {str(e)}")
            return pd.DataFrame() 