
from flask import Flask, render_template, request, redirect, url_for
import requests
import os
import sys
from datetime import datetime
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'stock-sentiment-analyzer-secret-key'

# Configuration
API_BASE_URL = 'http://localhost:5001/api'
WEB_UI_PORT = 5000


def make_api_request(endpoint, params=None):
    """Make a request to the backend API"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return None


@app.route('/')
def home():
    """Home page with ticker search"""
    return render_template('home.html')


@app.route('/search', methods=['POST'])
def search_ticker():
    """Handle ticker search from home page"""
    ticker = request.form.get('ticker', '').strip().upper()
    if not ticker:
        return redirect(url_for('home'))
    
    return redirect(url_for('ticker_details', ticker=ticker))


@app.route('/ticker/<ticker>')
def ticker_details(ticker):
    """Show detailed sentiment analysis for a specific ticker"""
    ticker = ticker.upper()
    hours = request.args.get('hours', 72, type=int)
    
    # Get ticker sentiment data
    sentiment_data = make_api_request(f'/sentiment/ticker/{ticker}', {'hours': hours})
    
    if not sentiment_data:
        return render_template('error.html', 
                             error_message=f"Could not fetch data for {ticker}",
                             ticker=ticker)
    
    # Extract data from current API structure
    total_mentions = sentiment_data.get('total_mentions', 0)
    avg_sentiment = sentiment_data.get('avg_sentiment', 0)
    total_positive = sentiment_data.get('total_positive', 0)
    total_negative = sentiment_data.get('total_negative', 0)
    total_neutral = sentiment_data.get('total_neutral', 0)
    positive_ratio = sentiment_data.get('positive_ratio', 0)
    negative_ratio = sentiment_data.get('negative_ratio', 0)
    neutral_ratio = sentiment_data.get('neutral_ratio', 0)
    trends = sentiment_data.get('trends', [])
    
    # Calculate sentiment percentages for display
    if total_mentions > 0:
        positive_pct = round(positive_ratio * 100, 1)
        negative_pct = round(negative_ratio * 100, 1)
        neutral_pct = round(neutral_ratio * 100, 1)
    else:
        positive_pct = negative_pct = neutral_pct = 0
    
    # Create summary dict for template compatibility
    summary = {
        'total_mentions': total_mentions,
        'avg_sentiment': avg_sentiment,
        'positive_mentions': total_positive,
        'negative_mentions': total_negative,
        'neutral_mentions': total_neutral
    }
    
    return render_template('ticker_details.html',
                         ticker=ticker,
                         sentiment_data=sentiment_data,
                         summary=summary,
                         trends=trends,
                         positive_pct=positive_pct,
                         negative_pct=negative_pct,
                         neutral_pct=neutral_pct,
                         hours=hours)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('error.html', 
                         error_message="Page not found"), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return render_template('error.html', 
                         error_message="Internal server error"), 500


if __name__ == '__main__':
    print("=" * 60)
    print("� Stock Sentiment Analyzer")
    print("=" * 60)
    print(f"Starting Web UI on http://localhost:{WEB_UI_PORT}")
    print("Make sure your API server is running on port 5001")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=WEB_UI_PORT)
