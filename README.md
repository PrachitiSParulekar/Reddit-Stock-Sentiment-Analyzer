# Reddit Stock Sentiment Analyzer

[![CI/CD Pipeline](https://github.com/PrachitiSParulekar/Reddit-Stock-Sentiment-Analyzer/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/PrachitiSParulekar/Reddit-Stock-Sentiment-Analyzer/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An AI-powered project that analyzes stock market sentiment from Reddit discussions using NLP and financial data.

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/PrachitiSParulekar/Reddit-Stock-Sentiment-Analyzer.git
   cd Reddit-Stock-Sentiment-Analyzer
   ```

2. **Set up environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Reddit API**
   ```bash
   cp .env.example .env
   # Edit .env with your Reddit API credentials
   ```

4. **Run the application**
   ```bash
   python main.py --mode pipeline
   ```

## ✨ Features

- 📊 **Real-time Data Collection** - Scrape Reddit posts from financial subreddits
- 🤖 **AI Sentiment Analysis** - Uses TextBlob and VADER for accurate sentiment scoring
- 📈 **Stock Ticker Extraction** - Automatically identifies stock symbols in posts
- 🌐 **REST API** - Query sentiment data programmatically
- 💻 **Web Dashboard** - Interactive web interface for data visualization
- 📅 **Scheduled Collection** - Automated data collection at regular intervals
- 📊 **Data Aggregation** - Statistical analysis and trending insights
- 🔒 **Secure Configuration** - Environment-based credential management

## Project Structure

```
finance/
├── src/                    # Core application code
│   ├── data_collector.py   # Reddit scraping logic
│   ├── sentiment_analyzer.py # NLP and sentiment analysis
│   ├── data_processor.py   # Data aggregation and processing
│   └── utils.py           # Utility functions
├── api/                   # Flask REST API
│   └── app.py            # Main Flask application
├── web_ui/               # Web User Interface
│   ├── app.py            # Flask web application
│   ├── templates/        # HTML templates
│   └── static/           # CSS and static assets
├── data/                 # Data storage
│   ├── raw/              # Raw Reddit data
│   ├── processed/        # Processed sentiment data
│   └── sentiment.db      # SQLite database
├── config/               # Configuration files
│   ├── config.py         # Application configuration
│   └── reddit_config.py  # Reddit API credentials
├── logs/                 # Application logs
└── requirements.txt      # Python dependencies
```

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Reddit API Setup**
   - Create a Reddit application at https://www.reddit.com/prefs/apps
   - Add your credentials to `config/reddit_config.py`

3. **TextBlob Setup**
   ```bash
   python -m textblob.download_corpora
   ```

## Available Commands

### Basic Commands
```bash
# Show project information and available options
python main.py --mode info

# Show help and all available arguments
python main.py --help
```

### Data Collection
```bash
# Collect from default subreddits (stocks, investing, SecurityAnalysis, StockMarket, wallstreetbets)
python main.py --mode collect

# Collect from specific subreddit(s) with custom limit
python main.py --mode collect --subreddits stocks --limit 10
python main.py --mode collect --subreddits stocks investing wallstreetbets --limit 25
```

### Sentiment Analysis
```bash
# Analyze sentiment from a specific CSV file
python main.py --mode analyze --file data/raw/reddit_posts_20250728_014313.csv

# Run full pipeline (collect + analyze + process)
python main.py --mode pipeline
python main.py --mode pipeline --subreddits stocks --limit 50
```

### Automated Collection
```bash
# Start scheduled data collection (runs automatically at intervals)
python main.py --mode schedule
```

### API Server
```bash
# Start the Flask API server
python api/app.py

# API will be available at: http://localhost:5001
```

### Web Interface
```bash
# Start the Web UI
python web_ui/app.py

# Web interface will be available at: http://localhost:5000
```

### Full Application Stack
```bash
# Start both API and Web UI (requires 2 terminals)
# Terminal 1:
python api/app.py

# Terminal 2:
python web_ui/app.py
```

### Option 4: Individual Components
```python
# Data Collection
from src.data_collector import RedditCollector
collector = RedditCollector()
collector.scrape_subreddit('stocks', limit=100)

# Sentiment Analysis
from src.sentiment_analyzer import SentimentAnalyzer
analyzer = SentimentAnalyzer()
sentiment_score = analyzer.analyze_text("TSLA to the moon!")
```

## Features

- Real-time Reddit data scraping from financial subreddits
- Sentiment analysis using TextBlob and VADER
- REST API for querying sentiment data
- Ticker symbol extraction
- Data aggregation and statistics
- Optional web dashboard

## 📊 Target Subreddits

- [r/stocks](https://reddit.com/r/stocks) - General stock discussions
- [r/investing](https://reddit.com/r/investing) - Investment strategies and advice
- [r/SecurityAnalysis](https://reddit.com/r/SecurityAnalysis) - Fundamental analysis
- [r/StockMarket](https://reddit.com/r/StockMarket) - Stock market news and discussions
- [r/wallstreetbets](https://reddit.com/r/wallstreetbets) - High-risk trading discussions

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sentiment/<subreddit>` | GET | Get sentiment analysis for a specific subreddit |
| `/api/sentiment/ticker/<symbol>` | GET | Get sentiment analysis for a stock ticker |
| `/api/stats/<period>` | GET | Get aggregated statistics for a time period |
| `/api/health` | GET | Check API health status |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This project is for educational and research purposes only. Do not use this as the sole basis for investment decisions. Always conduct your own research and consider consulting with financial advisors.

## 🙏 Acknowledgments

- [PRAW](https://praw.readthedocs.io/) - Python Reddit API Wrapper
- [TextBlob](https://textblob.readthedocs.io/) - Natural Language Processing
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) - Sentiment Analysis
- [Flask](https://flask.palletsprojects.com/) - Web Framework
