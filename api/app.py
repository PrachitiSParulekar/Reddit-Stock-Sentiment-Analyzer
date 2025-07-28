"""
Flask API Application

This module provides the REST API for the Reddit Stock Sentiment Analyzer.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import logging
import os
import sys

# Add project root and src directories to path
project_root = os.path.dirname(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
config_path = os.path.join(project_root, 'config')
sys.path.append(project_root)
sys.path.append(src_path)
sys.path.append(config_path)

from data_processor import DataProcessor
from sentiment_analyzer import SentimentAnalyzer
from utils import get_project_stats, validate_reddit_credentials
from config.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Initialize components
data_processor = DataProcessor()
sentiment_analyzer = SentimentAnalyzer()


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status and configuration"""
    try:
        stats = get_project_stats()
        credentials_ok = validate_reddit_credentials()
        
        return jsonify({
            'status': 'online',
            'reddit_api_configured': credentials_ok,
            'project_stats': stats,
            'config': {
                'target_subreddits': Config.TARGET_SUBREDDITS,
                'max_posts_per_subreddit': Config.MAX_POSTS_PER_SUBREDDIT,
                'sentiment_thresholds': {
                    'positive': Config.SENTIMENT_THRESHOLD_POSITIVE,
                    'negative': Config.SENTIMENT_THRESHOLD_NEGATIVE
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({'error': 'Failed to get status'}), 500


@app.route('/api/sentiment/<subreddit>', methods=['GET'])
def get_subreddit_sentiment(subreddit):
    """
    Get sentiment data for a specific subreddit
    
    Query parameters:
    - hours: Number of hours to look back (default: 24)
    - limit: Maximum number of records (default: 100)
    """
    try:
        hours = int(request.args.get('hours', 24))
        limit = int(request.args.get('limit', 100))
        
        # Get posts from database
        posts_df = data_processor.load_posts_from_db(
            subreddit=subreddit,
            limit=limit
        )
        
        if posts_df.empty:
            return jsonify({
                'subreddit': subreddit,
                'posts': [],
                'summary': {
                    'total_posts': 0,
                    'mean_sentiment': 0.0,
                    'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0}
                }
            })
        
        # Calculate summary statistics
        summary = sentiment_analyzer.get_sentiment_statistics(posts_df)
        
        # Convert to JSON-serializable format
        posts_data = posts_df.to_dict('records')
        
        return jsonify({
            'subreddit': subreddit,
            'posts': posts_data,
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error getting sentiment for r/{subreddit}: {str(e)}")
        return jsonify({'error': 'Failed to get sentiment data'}), 500


@app.route('/api/sentiment/ticker/<ticker>', methods=['GET'])
def get_ticker_sentiment(ticker):
    """
    Get sentiment data for a specific stock ticker
    
    Query parameters:
    - hours: Number of hours to look back (default: 24)
    """
    try:
        hours = int(request.args.get('hours', 24))
        ticker = ticker.upper()
        
        # Get aggregated data for ticker
        trends_df = data_processor.get_sentiment_trends(
            ticker=ticker,
            hours=hours
        )
        
        if trends_df.empty:
            return jsonify({
                'ticker': ticker,
                'total_mentions': 0,
                'avg_sentiment': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0,
                'total_positive': 0,
                'total_negative': 0,
                'total_neutral': 0,
                'trends': []
            })
        
        # Calculate aggregated summary stats
        total_mentions = int(trends_df['post_count'].sum())
        total_positive = int(trends_df['positive_count'].sum())
        total_negative = int(trends_df['negative_count'].sum())
        total_neutral = int(trends_df['neutral_count'].sum())
        
        # Calculate weighted average sentiment
        if total_mentions > 0:
            weighted_sentiment = (trends_df['mean_sentiment'] * trends_df['post_count']).sum() / trends_df['post_count'].sum()
            positive_ratio = total_positive / total_mentions
            negative_ratio = total_negative / total_mentions
            neutral_ratio = total_neutral / total_mentions
        else:
            weighted_sentiment = 0.0
            positive_ratio = 0.0
            negative_ratio = 0.0
            neutral_ratio = 0.0
        
        # Convert trends to JSON-serializable format
        trends_data = trends_df.to_dict('records')
        
        return jsonify({
            'ticker': ticker,
            'total_mentions': total_mentions,
            'avg_sentiment': float(weighted_sentiment),
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'neutral_ratio': neutral_ratio,
            'total_positive': total_positive,
            'total_negative': total_negative,
            'total_neutral': total_neutral,
            'trends': trends_data
        })
        
    except Exception as e:
        logger.error(f"Error getting sentiment for {ticker}: {str(e)}")
        return jsonify({'error': 'Failed to get ticker sentiment'}), 500


@app.route('/api/tickers/top', methods=['GET'])
def get_top_tickers():
    """
    Get most mentioned tickers
    
    Query parameters:
    - hours: Number of hours to look back (default: 24)
    - min_mentions: Minimum mentions to include (default: 5)
    - limit: Maximum number of tickers (default: 20)
    """
    try:
        hours = int(request.args.get('hours', 24))
        min_mentions = int(request.args.get('min_mentions', 5))
        limit = int(request.args.get('limit', 20))
        
        # Get top tickers
        tickers_df = data_processor.get_top_tickers(
            hours=hours,
            min_mentions=min_mentions
        )
        
        if not tickers_df.empty:
            # Limit results
            tickers_df = tickers_df.head(limit)
            
            # Convert to JSON-serializable format
            tickers_data = tickers_df.to_dict('records')
        else:
            tickers_data = []
        
        return jsonify({
            'top_tickers': tickers_data,
            'parameters': {
                'hours': hours,
                'min_mentions': min_mentions,
                'limit': limit
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting top tickers: {str(e)}")
        return jsonify({'error': 'Failed to get top tickers'}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """
    Analyze sentiment of provided text
    
    Request body:
    {
        "text": "Text to analyze"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field in request'}), 400
        
        text = data['text']
        
        # Analyze sentiment
        result = sentiment_analyzer.analyze_text(text)
        
        return jsonify({
            'analysis': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        return jsonify({'error': 'Failed to analyze text'}), 500


@app.route('/api/stats/<period>', methods=['GET'])
def get_statistics(period):
    """
    Get aggregated statistics for a time period
    
    Periods: 'hour', 'day', 'week'
    """
    try:
        # Map period to hours
        period_hours = {
            'hour': 1,
            'day': 24,
            'week': 168
        }
        
        if period not in period_hours:
            return jsonify({'error': 'Invalid period. Use: hour, day, week'}), 400
        
        hours = period_hours[period]
        
        # Get trends for all subreddits
        trends_df = data_processor.get_sentiment_trends(hours=hours)
        
        if trends_df.empty:
            return jsonify({
                'period': period,
                'hours': hours,
                'statistics': {
                    'total_posts': 0,
                    'subreddit_breakdown': {},
                    'overall_sentiment': 0.0
                }
            })
        
        # Calculate statistics
        subreddit_stats = trends_df.groupby('subreddit').agg({
            'post_count': 'sum',
            'mean_sentiment': 'mean',
            'positive_count': 'sum',
            'negative_count': 'sum',
            'neutral_count': 'sum'
        }).to_dict('index')
        
        overall_stats = {
            'total_posts': int(trends_df['post_count'].sum()),
            'overall_sentiment': float(trends_df['mean_sentiment'].mean()),
            'total_positive': int(trends_df['positive_count'].sum()),
            'total_negative': int(trends_df['negative_count'].sum()),
            'total_neutral': int(trends_df['neutral_count'].sum())
        }
        
        return jsonify({
            'period': period,
            'hours': hours,
            'statistics': {
                **overall_stats,
                'subreddit_breakdown': subreddit_stats
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting statistics for {period}: {str(e)}")
        return jsonify({'error': 'Failed to get statistics'}), 500


@app.route('/api/subreddits', methods=['GET'])
def get_subreddits():
    """Get list of monitored subreddits"""
    return jsonify({
        'subreddits': Config.TARGET_SUBREDDITS,
        'total_count': len(Config.TARGET_SUBREDDITS)
    })


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API documentation"""
    return jsonify({
        'name': 'Reddit Stock Sentiment Analyzer API',
        'version': '1.0.0',
        'description': 'AI-powered sentiment analysis of Reddit financial discussions',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'health': {
                'url': '/api/health',
                'method': 'GET',
                'description': 'Health check endpoint'
            },
            'status': {
                'url': '/api/status',
                'method': 'GET',
                'description': 'System status and configuration'
            },
            'subreddit_sentiment': {
                'url': '/api/sentiment/<subreddit>',
                'method': 'GET',
                'description': 'Get sentiment data for a specific subreddit',
                'parameters': {
                    'hours': 'Number of hours to look back (default: 24)',
                    'limit': 'Maximum number of records (default: 100)'
                }
            },
            'ticker_sentiment': {
                'url': '/api/sentiment/ticker/<ticker>',
                'method': 'GET',
                'description': 'Get sentiment data for a specific stock ticker',
                'parameters': {
                    'hours': 'Number of hours to look back (default: 24)',
                    'limit': 'Maximum number of records (default: 100)'
                }
            },
            'top_tickers': {
                'url': '/api/tickers/top',
                'method': 'GET',
                'description': 'Get most mentioned stock tickers',
                'parameters': {
                    'limit': 'Number of top tickers to return (default: 10)',
                    'hours': 'Number of hours to look back (default: 24)'
                }
            },
            'analyze_text': {
                'url': '/api/analyze',
                'method': 'POST',
                'description': 'Analyze sentiment of provided text',
                'body': {
                    'text': 'Text to analyze (required)',
                    'extract_tickers': 'Whether to extract stock tickers (default: true)'
                }
            },
            'statistics': {
                'url': '/api/stats/<period>',
                'method': 'GET',
                'description': 'Get aggregated statistics for a time period',
                'periods': ['1h', '24h', '7d', '30d'],
                'parameters': {
                    'hours': 'Custom hours for period (optional)'
                }
            },
            'subreddits': {
                'url': '/api/subreddits',
                'method': 'GET',
                'description': 'Get list of monitored subreddits'
            }
        },
        'example_usage': {
            'health_check': f'GET http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/api/health',
            'stocks_sentiment': f'GET http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/api/sentiment/stocks?hours=12&limit=50',
            'tsla_sentiment': f'GET http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/api/sentiment/ticker/TSLA',
            'analyze_text': f'POST http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/api/analyze',
            'top_tickers': f'GET http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/api/tickers/top?limit=20',
            'daily_stats': f'GET http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/api/stats/24h'
        }
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("Starting Reddit Stock Sentiment Analyzer API...")
    print(f"API will be available at http://{Config.FLASK_HOST}:{Config.FLASK_PORT}")
    print("\nAvailable endpoints:")
    print("- GET  /api/health")
    print("- GET  /api/status")
    print("- GET  /api/sentiment/<subreddit>")
    print("- GET  /api/sentiment/ticker/<ticker>")
    print("- GET  /api/tickers/top")
    print("- POST /api/analyze")
    print("- GET  /api/stats/<period>")
    print("- GET  /api/subreddits")
    
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.FLASK_DEBUG
    )
