"""
Utility Functions Module

This module contains helper functions and utilities used across the project.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import pandas as pd

# Add config directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if needed
    if log_file:
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def validate_reddit_credentials() -> bool:
    """
    Validate that Reddit API credentials are available
    
    Returns:
        True if credentials are valid, False otherwise
    """
    try:
        if not Config.REDDIT_CLIENT_ID or not Config.REDDIT_CLIENT_SECRET:
            logger.error("Reddit API credentials not found in environment variables")
            return False
            
        if Config.REDDIT_CLIENT_ID == 'your_client_id_here':
            logger.error("Please update your Reddit API credentials in the .env file")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating Reddit credentials: {str(e)}")
        return False


def ensure_directories_exist():
    """Create all necessary project directories"""
    directories = [
        Config.RAW_DATA_PATH,
        Config.PROCESSED_DATA_PATH,
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")


def load_json_file(filepath: str) -> Dict[str, Any]:
    """
    Load JSON data from file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary with loaded data
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON data from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON file {filepath}: {str(e)}")
        return {}


def save_json_file(data: Dict[str, Any], filepath: str) -> bool:
    """
    Save data to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.debug(f"Saved JSON data to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON file {filepath}: {str(e)}")
        return False


def get_file_size(filepath: str) -> str:
    """
    Get human-readable file size
    
    Args:
        filepath: Path to file
        
    Returns:
        Formatted file size string
    """
    try:
        size_bytes = os.path.getsize(filepath)
        
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.1f} GB"
            
    except Exception as e:
        logger.error(f"Error getting file size for {filepath}: {str(e)}")
        return "Unknown"


def format_timestamp(timestamp: datetime = None) -> str:
    """
    Format timestamp for filenames and logs
    
    Args:
        timestamp: Datetime object (uses current time if None)
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
        
    return timestamp.strftime("%Y%m%d_%H%M%S")


def calculate_percentage(value: float, total: float, decimal_places: int = 2) -> float:
    """
    Calculate percentage with error handling
    
    Args:
        value: Numerator value
        total: Denominator value
        decimal_places: Number of decimal places
        
    Returns:
        Percentage value
    """
    try:
        if total == 0:
            return 0.0
        return round((value / total) * 100, decimal_places)
    except Exception:
        return 0.0


def clean_filename(filename: str) -> str:
    """
    Clean filename by removing invalid characters
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Remove invalid characters for Windows/Unix filesystems
    invalid_chars = '<>:"/\\|?*'
    cleaned = filename
    
    for char in invalid_chars:
        cleaned = cleaned.replace(char, '_')
        
    # Remove extra spaces and dots
    cleaned = cleaned.strip(' .')
    
    # Limit length
    if len(cleaned) > 200:
        cleaned = cleaned[:200]
        
    return cleaned


def df_memory_usage(df: pd.DataFrame) -> Dict[str, str]:
    """
    Get memory usage information for a DataFrame
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Dictionary with memory usage statistics
    """
    try:
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()
        
        return {
            'total_memory': get_file_size_from_bytes(total_memory),
            'rows': len(df),
            'columns': len(df.columns),
            'memory_per_row': get_file_size_from_bytes(total_memory / len(df)) if len(df) > 0 else "0 bytes"
        }
    except Exception as e:
        logger.error(f"Error calculating DataFrame memory usage: {str(e)}")
        return {}


def get_file_size_from_bytes(size_bytes: int) -> str:
    """
    Convert bytes to human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"


def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains required columns
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if all required columns exist, False otherwise
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
        
    return True


def get_project_stats() -> Dict[str, Any]:
    """
    Get project statistics and information
    
    Returns:
        Dictionary with project statistics
    """
    stats = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'project_root': os.path.abspath('.'),
        'config': {
            'target_subreddits': Config.TARGET_SUBREDDITS,
            'max_posts_per_subreddit': Config.MAX_POSTS_PER_SUBREDDIT,
            'sentiment_thresholds': {
                'positive': Config.SENTIMENT_THRESHOLD_POSITIVE,
                'negative': Config.SENTIMENT_THRESHOLD_NEGATIVE
            }
        }
    }
    
    # Check if data directories exist and get file counts
    try:
        if os.path.exists(Config.RAW_DATA_PATH):
            raw_files = len([f for f in os.listdir(Config.RAW_DATA_PATH) if f.endswith('.csv')])
            stats['raw_data_files'] = raw_files
        else:
            stats['raw_data_files'] = 0
            
        if os.path.exists(Config.PROCESSED_DATA_PATH):
            processed_files = len([f for f in os.listdir(Config.PROCESSED_DATA_PATH) if f.endswith('.csv')])
            stats['processed_data_files'] = processed_files
        else:
            stats['processed_data_files'] = 0
            
        # Check database
        if os.path.exists(Config.DATABASE_PATH):
            stats['database_size'] = get_file_size(Config.DATABASE_PATH)
        else:
            stats['database_size'] = "Not created"
            
    except Exception as e:
        logger.error(f"Error getting project stats: {str(e)}")
        
    return stats


def print_project_info():
    """Print project information and setup status"""
    print("=" * 60)
    print("Reddit Stock Sentiment Analyzer")
    print("=" * 60)
    
    # Check setup status
    credentials_ok = validate_reddit_credentials()
    
    print(f"Reddit API Credentials: {'✓ Configured' if credentials_ok else '✗ Not configured'}")
    
    if not credentials_ok:
        print("\nTo setup Reddit API:")
        print("1. Go to https://www.reddit.com/prefs/apps")
        print("2. Create a new app (choose 'script' type)")
        print("3. Copy .env.example to .env")
        print("4. Add your client ID and secret to .env")
    
    # Project stats
    stats = get_project_stats()
    print(f"\nProject Statistics:")
    print(f"Raw data files: {stats.get('raw_data_files', 0)}")
    print(f"Processed data files: {stats.get('processed_data_files', 0)}")
    print(f"Database size: {stats.get('database_size', 'Not created')}")
    
    print("\nTarget Subreddits:")
    for subreddit in Config.TARGET_SUBREDDITS:
        print(f"  - r/{subreddit}")
    
    print("=" * 60)


def main():
    """Example usage of utility functions"""
    print_project_info()
    
    # Test other utilities
    print(f"\nCurrent timestamp: {format_timestamp()}")
    print(f"Project directories ensured")
    ensure_directories_exist()


if __name__ == "__main__":
    main()
