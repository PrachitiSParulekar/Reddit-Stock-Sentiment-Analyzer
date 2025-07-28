import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # Database settings
    DATABASE_PATH = os.path.join('data', 'sentiment.db')
    
    # Data storage paths
    RAW_DATA_PATH = os.path.join('data', 'raw')
    PROCESSED_DATA_PATH = os.path.join('data', 'processed')
    
    # Reddit API settings
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'StockSentimentAnalyzer/1.0')
    
    # Target subreddits for analysis
    TARGET_SUBREDDITS = [
        'stocks',
        'investing', 
        'SecurityAnalysis',
        'StockMarket',
        'wallstreetbets'
    ]
    
    # Sentiment analysis settings
    SENTIMENT_THRESHOLD_POSITIVE = 0.1
    SENTIMENT_THRESHOLD_NEGATIVE = -0.1
    
    # Data collection settings
    MAX_POSTS_PER_SUBREDDIT = 100
    SCRAPING_INTERVAL_HOURS = 1
    
    # Flask settings
    FLASK_HOST = '0.0.0.0'
    FLASK_PORT = 5001
    FLASK_DEBUG = True
    
    # Ticker symbol patterns (common US stock tickers)
    TICKER_PATTERN = r'\b[A-Z]{1,5}\b'
    
    @staticmethod
    def create_directories():
        """Create necessary directories if they don't exist"""
        os.makedirs(Config.RAW_DATA_PATH, exist_ok=True)
        os.makedirs(Config.PROCESSED_DATA_PATH, exist_ok=True)
