from datetime import timedelta

# API Configuration
OANDA_API_KEY = "c78854f02688c206c2d7ebb3db3b5f9d-f27557fdd2b1c0cc614950492d16e5e4"
OANDA_ACCOUNT_ID = "101-004-29563382-003"
OANDA_ENVIRONMENT = "practice"  # Use "practice" for demo account, "live" for real account
OANDA_DOMAIN = "api-fxpractice.oanda.com" if OANDA_ENVIRONMENT == "practice" else "api-fxtrade.oanda.com"

# Trading Parameters
INSTRUMENT = "EUR_USD"
GRANULARITY = "M15"
CANDLE_COUNT = 1000

# Database Configuration (for future use)
DB_NAME = "forex_bot"
DB_HOST = "localhost"
DB_PORT = 5432

# Model Parameters (for future use)
TRAIN_TEST_SPLIT = 0.8
PREDICTION_HORIZON = 1  # Number of candles to predict ahead

# News Configuration (for future use)
NEWS_UPDATE_INTERVAL = 3600  # in seconds
NEWS_SOURCES = ["investing.com"]

# Trading Strategy Parameters (for future use)
RISK_REWARD_RATIO = 2
MAX_RISK_PER_TRADE = 0.02  # 2% risk per trade 

# Trading pairs configuration
CURRENCY_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
# News Configuration
NEWS_LOOKBACK_HOURS = 24
NEWS_UPDATE_INTERVAL = 15  # minutes
NEWS_PAGES_TO_FETCH = 2
# Currency-specific configurations
CURRENCY_CONFIG = {
    'EUR': {
        'name': 'Euro',
        'terms': [
            'euro', 'euros', 'eurozone', 'european union', 'eu economy',
            'ecb', 'european central bank', 'christine lagarde',
            'euro area', 'european economy', 'european inflation',
            'european interest rates', 'european monetary policy',
            'european manufacturing', 'european services',
            'german economy', 'french economy', 'italian economy'
        ],
        'central_bank': {
            'name': 'European Central Bank',
            'abbreviation': 'ECB',
            'governor': 'Christine Lagarde',
            'terms': [
                'ecb meeting', 'ecb decision', 'ecb policy',
                'european central bank decision', 'lagarde speech'
            ]
        }
    },
    'USD': {
        'name': 'US Dollar',
        'terms': [
            'dollar', 'dollars', 'usd', 'us economy', 'united states economy',
            'federal reserve', 'fed', 'jerome powell', 'us inflation',
            'us interest rates', 'us monetary policy', 'us manufacturing',
            'us services', 'us employment', 'nfp', 'non-farm payrolls',
            'us treasury', 'treasury yields', 'fomc'
        ],
        'central_bank': {
            'name': 'Federal Reserve',
            'abbreviation': 'Fed',
            'governor': 'Jerome Powell',
            'terms': [
                'fed meeting', 'fomc meeting', 'fed decision', 'fomc decision',
                'fed policy', 'powell speech', 'federal reserve decision'
            ]
        }
    },
    'GBP': {
        'name': 'British Pound',
        'terms': [
            'pound', 'pounds sterling', 'sterling', 'gbp', 'british economy',
            'uk economy', 'british inflation', 'uk inflation',
            'british interest rates', 'uk interest rates',
            'british monetary policy', 'uk monetary policy',
            'british manufacturing', 'uk manufacturing',
            'british services', 'uk services'
        ],
        'central_bank': {
            'name': 'Bank of England',
            'abbreviation': 'BOE',
            'governor': 'Andrew Bailey',
            'terms': [
                'boe meeting', 'boe decision', 'boe policy',
                'bank of england decision', 'bailey speech'
            ]
        }
    },
    'JPY': {
        'name': 'Japanese Yen',
        'terms': [
            'yen', 'jpy', 'japanese economy', 'japan economy',
            'japanese inflation', 'japan inflation',
            'japanese interest rates', 'japan interest rates',
            'japanese monetary policy', 'japan monetary policy',
            'japanese manufacturing', 'japan manufacturing',
            'japanese services', 'japan services'
        ],
        'central_bank': {
            'name': 'Bank of Japan',
            'abbreviation': 'BOJ',
            'governor': 'Kazuo Ueda',
            'terms': [
                'boj meeting', 'boj decision', 'boj policy',
                'bank of japan decision', 'ueda speech'
            ]
        }
    },
    'XAU': {
        'terms': [
            'gold', 'XAU', 'precious metals', 'bullion',
            'gold price', 'gold market', 'gold trading',
            'gold futures', 'spot gold', 'gold demand',
            'gold reserves', 'gold ETF', 'gold stocks',
            'gold miners', 'gold production',
            'Federal Reserve', 'inflation', 'interest rates',
            'safe haven', 'risk aversion',
            'COMEX', 'London Gold Fix', 'Shanghai Gold Exchange'
        ],
        'central_bank': {
            'terms': [
                'central bank gold', 'gold reserves',
                'gold holdings', 'gold purchases',
                'gold sales', 'gold storage'
            ]
        }
    }
}

# Economic indicators by importance
ECONOMIC_INDICATORS = {
    'high_impact': [
        'interest rate decision', 'inflation', 'cpi', 'ppi',
        'gdp', 'employment', 'unemployment', 'non-farm payrolls',
        'retail sales', 'trade balance', 'Federal Reserve', 'inflation data',
        'NFP', 'gold reserves', 'central bank purchases'
    ],
    'medium_impact': [
        'manufacturing pmi', 'services pmi', 'industrial production',
        'consumer confidence', 'business confidence', 'housing data',
        'durable goods', 'gold production', 'mining output',
        'ETF holdings', 'jewelry demand', 'industrial demand'
    ],
    'low_impact': [
        'building permits', 'housing starts', 'capacity utilization',
        'factory orders', 'wholesale inventories', 'gold exploration',
        'mining costs', 'regional demand'
    ]
}

# Market sentiment terms
SENTIMENT_TERMS = {
    'bullish': 2.0,
    'bearish': -2.0,
    'hawkish': 1.5,
    'dovish': -1.5,
    'rally': 1.0,
    'surge': 1.5,
    'plunge': -1.5,
    'crash': -2.0,
    'strengthen': 1.0,
    'weaken': -1.0,
    'higher': 0.8,
    'lower': -0.8,
    'rise': 0.8,
    'fall': -0.8,
    'gain': 0.8,
    'loss': -0.8,
    'positive': 1.0,
    'negative': -1.0,
    'unchanged': 0.0,
    'stable': 0.2,
    'gold rally': 3.0,
    'gold surge': 3.0,
    'gold soars': 3.0,
    'gold slumps': -3.0,
    'gold drops': -3.0,
    'gold plunges': -3.0,
    'safe haven': 1.0,
    'risk aversion': -1.0,
    'COMEX': 1.0,
    'London Gold Fix': 1.0,
    'Shanghai Gold Exchange': 1.0,
    'gold demand': 2.0,
    'gold reserves': 1.5,
    'gold shortage': 2.5,
    'gold surplus': -2.0,
}

# Position Management
MIN_CONFIDENCE = 0.4
MAX_POSITIONS = 1
POSITION_SIZE = 500
PROFIT_TARGET_PCT = 0.003  # 0.3%
STOP_LOSS_PCT = 0.0015     # 0.15%
MAX_HOLD_TIME_HOURS = 4

# Feature Parameters
TECHNICAL_INDICATORS = [
    'returns',
    'rsi',
    'macd',
    'macd_signal',
    'bb_high',
    'bb_low',
    'atr',
    'close'
]

# Analysis Weights
WEIGHTS = {
    'technical': 0.4,
    'sentiment': 0.3,
    'prediction': 0.3
}

# Update Intervals
ANALYSIS_INTERVAL = 15  # minutes
MODEL_UPDATE_INTERVAL = 15  # minutes

# Risk Management Settings
FOREX_RISK_SETTINGS = {
    'EUR_USD': {
        'take_profit_pct': 0.002,  # 0.2%
        'stop_loss_pct': 0.001,    # 0.1%
    },
    'GBP_USD': {
        'take_profit_pct': 0.0025,  # 0.25%
        'stop_loss_pct': 0.0012,    # 0.12%
    },
    'USD_JPY': {
        'take_profit_pct': 0.002,   # 0.2%
        'stop_loss_pct': 0.001,     # 0.1%
    }
}

COMMODITY_RISK_SETTINGS = {
    'XAU_USD': {
        'take_profit_pct': 0.004,   # 0.4%
        'stop_loss_pct': 0.002,     # 0.2%
    }
}

TRAILING_STOP_SETTINGS = {
    'activation_profit_pct': 0.001,  # Activate at 0.1% profit
    'trailing_distance_pct': 0.0005  # 0.05% trailing distance
}

MAX_HOLD_TIMES = {
    'forex': timedelta(hours=4),
    'commodities': timedelta(hours=6)
}


# Trading pairs configuration with active trading flags
CURRENCY_PAIRS_CONFIG = {
    "EUR_USD": {
        "active_trading": True,     # Enable/disable actual trading
        "monitor_only": True,      # Still collect data but don't trade
    },
    "GBP_USD": {
        "active_trading": True,    # Don't trade this pair
        "monitor_only": True,       # But still collect data and analyze
    },
    "USD_JPY": {
        "active_trading": True,
        "monitor_only": True,
    },
    "XAU_USD": {
        "active_trading": True,
        "monitor_only": True,
    }
}

# Keep this for backwards compatibility and convenience
CURRENCY_PAIRS = list(CURRENCY_PAIRS_CONFIG.keys())

# Helper function to get active trading pairs
def get_active_trading_pairs():
    return [pair for pair, config in CURRENCY_PAIRS_CONFIG.items() 
            if config['active_trading']]

# Position Management Settings
POSITION_SETTINGS = {
    'min_confidence': 0.6,      # Minimum confidence to open position
    'position_size': 1000,      # Default position size
}