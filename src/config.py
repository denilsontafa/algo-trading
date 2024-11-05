# API Configuration
OANDA_API_KEY = "daf311de7e0581b4213ac6ce7b23395e-9d98339a3296e1dd20d1af1fd01ee915"
OANDA_ACCOUNT_ID = "101-004-29563382-001"
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
CURRENCY_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]
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
    }
}

# Economic indicators by importance
ECONOMIC_INDICATORS = {
    'high_impact': [
        'interest rate decision', 'inflation', 'cpi', 'ppi',
        'gdp', 'employment', 'unemployment', 'non-farm payrolls',
        'retail sales', 'trade balance'
    ],
    'medium_impact': [
        'manufacturing pmi', 'services pmi', 'industrial production',
        'consumer confidence', 'business confidence', 'housing data',
        'durable goods'
    ],
    'low_impact': [
        'building permits', 'housing starts', 'capacity utilization',
        'factory orders', 'wholesale inventories'
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
}

# Position Management
MAX_POSITIONS = 1
POSITION_SIZE = 1000
PROFIT_TARGET_PCT = 0.002  # 0.2%
STOP_LOSS_PCT = 0.005     # 0.5%
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
ANALYSIS_INTERVAL = 1  # minutes
MODEL_UPDATE_INTERVAL = 15  # minutes