# Reddit Stock Sentiment Analyzer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0.3-orange.svg)](https://pandas.pydata.org/)
[![TextBlob](https://img.shields.io/badge/TextBlob-NLP-red.svg)](https://textblob.readthedocs.io/)
[![PRAW](https://img.shields.io/badge/PRAW-Reddit_API-ff4500.svg)](https://praw.readthedocs.io/)

An AI-powered tool that analyzes stock market sentiment from Reddit discussions using NLP.

## Quick Start

1. **Clone and setup**
   ```bash
   git clone https://github.com/PrachitiSParulekar/Reddit-Stock-Sentiment-Analyzer.git
   cd Reddit-Stock-Sentiment-Analyzer
   pip install -r requirements.txt
   ```

2. **Configure Reddit API**
   ```bash
   cp .env.example .env
   # Edit .env with your Reddit API credentials from https://www.reddit.com/prefs/apps
   ```

3. **Download required data**
   ```bash
   python -m textblob.download_corpora
   ```

4. **Run**
   ```bash
   python main.py --mode pipeline
   ```

## Usage

```bash
# Collect data from Reddit
python main.py --mode collect

# Analyze sentiment
python main.py --mode analyze --file data/raw/reddit_posts.csv

# Run full pipeline
python main.py --mode pipeline

# Start web interface
python web_ui/app.py
```

## Features

- Reddit data collection from financial subreddits
- Sentiment analysis using TextBlob and VADER
- REST API endpoints
- Web dashboard
- Stock ticker extraction

## API Endpoints

- `GET /api/sentiment/<subreddit>` - Get sentiment for subreddit
- `GET /api/sentiment/ticker/<symbol>` - Get sentiment for stock ticker

## License

MIT License - see [LICENSE](LICENSE) file.
