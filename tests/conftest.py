import pytest
import sys
import os

# Add src directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_reddit_post():
    """Sample Reddit post data for testing"""
    return {
        'id': 'test123',
        'title': 'TSLA to the moon! 🚀',
        'selftext': 'I think Tesla stock will go up significantly.',
        'score': 150,
        'upvote_ratio': 0.85,
        'num_comments': 25,
        'created_utc': 1640995200,  # 2022-01-01 00:00:00 UTC
        'subreddit': 'stocks',
        'author': 'test_user',
        'url': 'https://reddit.com/r/stocks/test123'
    }


@pytest.fixture
def sample_sentiment_data():
    """Sample sentiment analysis results"""
    return {
        'polarity': 0.5,
        'subjectivity': 0.8,
        'compound': 0.6,
        'pos': 0.3,
        'neu': 0.5,
        'neg': 0.2
    }
