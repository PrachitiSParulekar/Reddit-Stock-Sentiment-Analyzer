"""
Data Processing Module

This module handles data aggregation, processing, and storage of sentiment analysis results.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json

# Add config directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data processing, aggregation, and storage"""
    
    def __init__(self):
        """Initialize data processor"""
        self.config = Config()
        self.db_path = Config.DATABASE_PATH
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        Config.create_directories()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create posts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS posts (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    selftext TEXT,
                    author TEXT,
                    created_utc TEXT,
                    score INTEGER,
                    upvote_ratio REAL,
                    num_comments INTEGER,
                    subreddit TEXT,
                    permalink TEXT,
                    url TEXT,
                    is_self BOOLEAN,
                    collected_at TEXT,
                    cleaned_text TEXT,
                    tickers TEXT,
                    text_length INTEGER,
                    cleaned_length INTEGER,
                    textblob_polarity REAL,
                    textblob_subjectivity REAL,
                    vader_compound REAL,
                    vader_positive REAL,
                    vader_neutral REAL,
                    vader_negative REAL,
                    composite_sentiment REAL,
                    sentiment_label TEXT
                )
            ''')
            
            # Create aggregated sentiment table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_aggregated (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subreddit TEXT,
                    ticker TEXT,
                    date_hour TEXT,
                    post_count INTEGER,
                    mean_sentiment REAL,
                    median_sentiment REAL,
                    std_sentiment REAL,
                    positive_count INTEGER,
                    negative_count INTEGER,
                    neutral_count INTEGER,
                    total_score INTEGER,
                    created_at TEXT
                )
            ''')
            
            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON posts(subreddit)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_posts_created_utc ON posts(created_utc)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_subreddit ON sentiment_aggregated(subreddit)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_ticker ON sentiment_aggregated(ticker)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_date ON sentiment_aggregated(date_hour)')
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
    
    def save_posts_to_db(self, df: pd.DataFrame) -> int:
        """
        Save processed posts to database
        
        Args:
            df: DataFrame with processed posts
            
        Returns:
            Number of posts saved
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Convert tickers list to JSON string for storage
            df_copy = df.copy()
            if 'tickers' in df_copy.columns:
                df_copy['tickers'] = df_copy['tickers'].apply(
                    lambda x: json.dumps(x) if isinstance(x, list) else json.dumps([])
                )
            
            # Save to database (replace existing records with same ID)
            rows_affected = df_copy.to_sql('posts', conn, if_exists='append', index=False)
            
            conn.close()
            logger.info(f"Saved {rows_affected} posts to database")
            
            return rows_affected
            
        except Exception as e:
            logger.error(f"Failed to save posts to database: {str(e)}")
            return 0
    
    def load_posts_from_db(self, subreddit: str = None, 
                          start_date: str = None, end_date: str = None,
                          limit: int = None) -> pd.DataFrame:
        """
        Load posts from database with optional filters
        
        Args:
            subreddit: Filter by subreddit
            start_date: Start date filter (ISO format)
            end_date: End date filter (ISO format) 
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with filtered posts
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build query with filters
            query = "SELECT * FROM posts WHERE 1=1"
            params = []
            
            if subreddit:
                query += " AND subreddit = ?"
                params.append(subreddit)
                
            if start_date:
                query += " AND created_utc >= ?"
                params.append(start_date)
                
            if end_date:
                query += " AND created_utc <= ?"
                params.append(end_date)
                
            query += " ORDER BY created_utc DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            
            # Convert tickers JSON back to list
            if 'tickers' in df.columns and len(df) > 0:
                df['tickers'] = df['tickers'].apply(
                    lambda x: json.loads(x) if pd.notna(x) else []
                )
            
            conn.close()
            logger.info(f"Loaded {len(df)} posts from database")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load posts from database: {str(e)}")
            return pd.DataFrame()
    
    def aggregate_sentiment_by_hour(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment data by hour and subreddit
        
        Args:
            df: DataFrame with sentiment analysis results
            
        Returns:
            DataFrame with hourly aggregated sentiment
        """
        if df.empty or 'created_utc' not in df.columns:
            logger.warning("Empty DataFrame or missing created_utc column")
            return pd.DataFrame()
            
        try:
            # Convert created_utc to datetime and extract hour
            df['created_utc'] = pd.to_datetime(df['created_utc'])
            df['date_hour'] = df['created_utc'].dt.floor('H')
            
            # Group by subreddit and hour
            aggregated = df.groupby(['subreddit', 'date_hour']).agg({
                'composite_sentiment': ['count', 'mean', 'median', 'std'],
                'sentiment_label': lambda x: {
                    'positive': (x == 'positive').sum(),
                    'negative': (x == 'negative').sum(), 
                    'neutral': (x == 'neutral').sum()
                },
                'score': 'sum'
            }).reset_index()
            
            # Flatten column names
            aggregated.columns = [
                'subreddit', 'date_hour', 'post_count', 'mean_sentiment',
                'median_sentiment', 'std_sentiment', 'sentiment_counts', 'total_score'
            ]
            
            # Extract sentiment counts
            aggregated['positive_count'] = aggregated['sentiment_counts'].apply(lambda x: x['positive'])
            aggregated['negative_count'] = aggregated['sentiment_counts'].apply(lambda x: x['negative'])
            aggregated['neutral_count'] = aggregated['sentiment_counts'].apply(lambda x: x['neutral'])
            
            # Drop the intermediate column
            aggregated = aggregated.drop('sentiment_counts', axis=1)
            
            # Add metadata
            aggregated['ticker'] = 'ALL'  # Overall aggregation
            aggregated['created_at'] = datetime.now().isoformat()
            
            # Fill NaN values
            aggregated['std_sentiment'] = aggregated['std_sentiment'].fillna(0.0)
            
            logger.info(f"Aggregated sentiment for {len(aggregated)} hour-subreddit combinations")
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Failed to aggregate sentiment by hour: {str(e)}")
            return pd.DataFrame()
    
    def aggregate_sentiment_by_ticker(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment data by ticker symbol
        
        Args:
            df: DataFrame with sentiment analysis results
            
        Returns:
            DataFrame with ticker-aggregated sentiment
        """
        if df.empty or 'tickers' not in df.columns:
            logger.warning("Empty DataFrame or missing tickers column")
            return pd.DataFrame()
            
        try:
            # Explode tickers (one row per ticker per post)
            ticker_df = df.explode('tickers')
            ticker_df = ticker_df[ticker_df['tickers'].notna() & (ticker_df['tickers'] != '')]
            
            if ticker_df.empty:
                logger.warning("No ticker symbols found in data")
                return pd.DataFrame()
            
            # Convert created_utc to datetime and extract hour
            ticker_df['created_utc'] = pd.to_datetime(ticker_df['created_utc'])
            ticker_df['date_hour'] = ticker_df['created_utc'].dt.floor('H')
            
            # Group by ticker and hour
            aggregated = ticker_df.groupby(['tickers', 'date_hour']).agg({
                'composite_sentiment': ['count', 'mean', 'median', 'std'],
                'sentiment_label': lambda x: {
                    'positive': (x == 'positive').sum(),
                    'negative': (x == 'negative').sum(),
                    'neutral': (x == 'neutral').sum()
                },
                'score': 'sum',
                'subreddit': lambda x: x.iloc[0]  # Take first subreddit for grouping
            }).reset_index()
            
            # Flatten column names
            aggregated.columns = [
                'ticker', 'date_hour', 'post_count', 'mean_sentiment',
                'median_sentiment', 'std_sentiment', 'sentiment_counts', 'total_score', 'subreddit'
            ]
            
            # Extract sentiment counts
            aggregated['positive_count'] = aggregated['sentiment_counts'].apply(lambda x: x['positive'])
            aggregated['negative_count'] = aggregated['sentiment_counts'].apply(lambda x: x['negative'])
            aggregated['neutral_count'] = aggregated['sentiment_counts'].apply(lambda x: x['neutral'])
            
            # Drop the intermediate column
            aggregated = aggregated.drop('sentiment_counts', axis=1)
            
            # Add metadata
            aggregated['created_at'] = datetime.now().isoformat()
            
            # Fill NaN values
            aggregated['std_sentiment'] = aggregated['std_sentiment'].fillna(0.0)
            
            logger.info(f"Aggregated sentiment for {len(aggregated)} ticker-hour combinations")
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Failed to aggregate sentiment by ticker: {str(e)}")
            return pd.DataFrame()
    
    def save_aggregated_data(self, df: pd.DataFrame) -> int:
        """
        Save aggregated sentiment data to database
        
        Args:
            df: DataFrame with aggregated sentiment data
            
        Returns:
            Number of records saved
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Save to database
            rows_affected = df.to_sql('sentiment_aggregated', conn, if_exists='append', index=False)
            
            conn.close()
            logger.info(f"Saved {rows_affected} aggregated records to database")
            
            return rows_affected
            
        except Exception as e:
            logger.error(f"Failed to save aggregated data: {str(e)}")
            return 0
    
    def get_sentiment_trends(self, ticker: str = None, subreddit: str = None,
                           hours: int = 24) -> pd.DataFrame:
        """
        Get sentiment trends for the last N hours
        
        Args:
            ticker: Specific ticker symbol (None for all)
            subreddit: Specific subreddit (None for all)
            hours: Number of hours to look back
            
        Returns:
            DataFrame with sentiment trends
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate cutoff time
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            # Build query
            query = """
                SELECT * FROM sentiment_aggregated 
                WHERE date_hour >= ?
            """
            params = [cutoff_time]
            
            if ticker:
                query += " AND ticker = ?"
                params.append(ticker)
            elif ticker != 'ALL':
                query += " AND ticker != 'ALL'"
                
            if subreddit:
                query += " AND subreddit = ?"
                params.append(subreddit)
                
            query += " ORDER BY date_hour DESC"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                df['date_hour'] = pd.to_datetime(df['date_hour'])
                
            logger.info(f"Retrieved {len(df)} trend records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get sentiment trends: {str(e)}")
            return pd.DataFrame()
    
    def get_top_tickers(self, hours: int = 24, min_mentions: int = 5) -> pd.DataFrame:
        """
        Get most mentioned tickers in the last N hours
        
        Args:
            hours: Number of hours to look back
            min_mentions: Minimum number of mentions to include
            
        Returns:
            DataFrame with top tickers and their sentiment
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate cutoff time
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            query = """
                SELECT 
                    ticker,
                    SUM(post_count) as total_mentions,
                    AVG(mean_sentiment) as avg_sentiment,
                    SUM(positive_count) as total_positive,
                    SUM(negative_count) as total_negative,
                    SUM(neutral_count) as total_neutral
                FROM sentiment_aggregated 
                WHERE date_hour >= ? AND ticker != 'ALL'
                GROUP BY ticker
                HAVING total_mentions >= ?
                ORDER BY total_mentions DESC
            """
            
            df = pd.read_sql_query(query, conn, params=[cutoff_time, min_mentions])
            conn.close()
            
            if not df.empty:
                # Calculate sentiment ratios
                df['positive_ratio'] = df['total_positive'] / df['total_mentions']
                df['negative_ratio'] = df['total_negative'] / df['total_mentions']
                df['neutral_ratio'] = df['total_neutral'] / df['total_mentions']
                
            logger.info(f"Retrieved {len(df)} top tickers")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get top tickers: {str(e)}")
            return pd.DataFrame()
    
    def export_data(self, data: pd.DataFrame, filename: str, format: str = 'csv') -> str:
        """
        Export data to file
        
        Args:
            data: DataFrame to export
            filename: Output filename
            format: Export format ('csv', 'json')
            
        Returns:
            Path to exported file
        """
        try:
            Config.create_directories()
            
            if format.lower() == 'csv':
                filepath = os.path.join(Config.PROCESSED_DATA_PATH, f"{filename}.csv")
                data.to_csv(filepath, index=False)
            elif format.lower() == 'json':
                filepath = os.path.join(Config.PROCESSED_DATA_PATH, f"{filename}.json")
                data.to_json(filepath, orient='records', date_format='iso', indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            logger.info(f"Data exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export data: {str(e)}")
            return ""


def main():
    """Example usage of DataProcessor"""
    processor = DataProcessor()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'id': ['1', '2', '3'],
        'title': ['Test post 1', 'Test post 2', 'Test post 3'],
        'subreddit': ['stocks', 'investing', 'stocks'],
        'created_utc': [
            datetime.now().isoformat(),
            (datetime.now() - timedelta(hours=1)).isoformat(),
            (datetime.now() - timedelta(hours=2)).isoformat()
        ],
        'composite_sentiment': [0.5, -0.3, 0.1],
        'sentiment_label': ['positive', 'negative', 'neutral'],
        'tickers': [['AAPL'], ['TSLA'], []],
        'score': [100, 50, 75]
    })
    
    print("Sample data processing:")
    print("=" * 30)
    
    # Test aggregation
    hourly_agg = processor.aggregate_sentiment_by_hour(sample_data)
    print(f"Hourly aggregation: {len(hourly_agg)} records")
    
    ticker_agg = processor.aggregate_sentiment_by_ticker(sample_data)
    print(f"Ticker aggregation: {len(ticker_agg)} records")
    
    # Test database operations
    saved_posts = processor.save_posts_to_db(sample_data)
    print(f"Saved {saved_posts} posts to database")
    
    if len(hourly_agg) > 0:
        saved_agg = processor.save_aggregated_data(hourly_agg)
        print(f"Saved {saved_agg} aggregated records")


if __name__ == "__main__":
    main()
