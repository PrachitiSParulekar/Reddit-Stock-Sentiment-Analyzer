"""
Sentiment Analysis Module

This module handles sentiment analysis of Reddit posts and comments using
TextBlob and VADER sentiment analysis tools.
"""

import pandas as pd
import numpy as np
import re
import sys
import os
import logging
from typing import Dict, List, Optional, Tuple
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Add config directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Handles sentiment analysis of text data"""
    
    def __init__(self):
        """Initialize sentiment analyzers"""
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.config = Config()
        
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for sentiment analysis
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove Reddit-specific formatting
        text = re.sub(r'/u/\w+', '', text)  # Remove usernames
        text = re.sub(r'/r/\w+', '', text)  # Remove subreddit mentions
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Remove bold formatting
        text = re.sub(r'\*(.+?)\*', r'\1', text)  # Remove italic formatting
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_tickers(self, text: str) -> List[str]:
        """
        Extract potential stock ticker symbols from text
        
        Args:
            text: Text to search for tickers
            
        Returns:
            List of potential ticker symbols
        """
        if not isinstance(text, str):
            return []
            
        # Convert to uppercase for ticker matching
        text_upper = text.upper()
        
        # Look for tickers in specific contexts that suggest they are stock symbols
        # Pattern 1: $TICKER format
        dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', text_upper)
        
        # Pattern 2: Ticker followed by stock-related words
        context_pattern = r'\b([A-Z]{2,5})\s+(?:STOCK|SHARES|PRICE|CALLS?|PUTS?|OPTIONS?|EARNINGS|DIVIDEND|SPLIT|MOON|ROCKET|BULL|BEAR|LONG|SHORT|BUY|SELL|HOLD|UP|DOWN|GAIN|LOSS)\b'
        context_tickers = re.findall(context_pattern, text_upper)
        
        # Pattern 3: Stock-related words followed by ticker
        reverse_context_pattern = r'\b(?:STOCK|SHARES|PRICE|CALLS?|PUTS?|OPTIONS?|EARNINGS|DIVIDEND|SPLIT|BUY|SELL|HOLD)\s+([A-Z]{2,5})\b'
        reverse_context_tickers = re.findall(reverse_context_pattern, text_upper)
        
        # Pattern 4: Traditional pattern but with stricter filtering
        all_potential_tickers = re.findall(self.config.TICKER_PATTERN, text_upper)
        
        # Combine all patterns
        potential_tickers = list(set(dollar_tickers + context_tickers + reverse_context_tickers + all_potential_tickers))
        
        # Filter out common words that might be mistaken for tickers
        excluded_words = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE',
            'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW',
            'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'USE', 'WAY', 'SHE', 'AIR', 'BIG', 'EYE',
            'FAR', 'OFF', 'PUT', 'SAY', 'SET', 'SUN', 'TOP', 'TRY', 'ASK', 'BAD', 'BAG', 'BED',
            'BOX', 'CAR', 'CAT', 'CUT', 'DOG', 'EAR', 'END', 'FEW', 'GOT', 'GUN', 'JOB', 'LEG',
            'LET', 'MAN', 'MAP', 'MOM', 'RUN', 'SIT', 'WIN', 'YES', 'YET', 'LOL', 'OMG', 'WTF',
            'EDIT', 'TLDR', 'IMO', 'IMHO', 'FYI', 'ASAP', 'ETC', 'CEO', 'CFO', 'CTO', 'IPO',
            'SEC', 'FDA', 'FED', 'GDP', 'API', 'ATH', 'ATL', 'EOD', 'AH', 'PM',
            # Additional common words that appear in Reddit posts
            'IS', 'TO', 'UP', 'DOWN', 'BUY', 'SELL', 'HOLD', 'MOON', 'GOING', 'THINK', 'WHAT',
            'DO', 'ABOUT', 'VS', 'STOCK', 'STOCKS', 'MONEY', 'GOOD', 'GREAT', 'BEST', 'WORST',
            'TODAY', 'TOMORROW', 'WEEK', 'MONTH', 'YEAR', 'TIME', 'SOME', 'MANY', 'MUCH', 'MORE',
            'LESS', 'VERY', 'JUST', 'ONLY', 'ALSO', 'EVEN', 'STILL', 'FIRST', 'LAST', 'NEXT',
            'BEEN', 'HAVE', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MIGHT', 'MUST', 'CANT', 'WONT',
            'DONT', 'ISNT', 'ARENT', 'WASNT', 'WERENT', 'HASNT', 'HAVENT', 'DIDNT', 'DOESNT',
            'APPLE', 'TESLA', 'MICROSOFT', 'AMAZON', 'GOOGLE', 'NVIDIA', 'META', 'WERE', 'THAT',
            'THIS', 'WITH', 'FROM', 'THEY', 'THEM', 'THEIR', 'THERE', 'WHERE', 'WHEN', 'WHILE',
            'THAN', 'THEN', 'THAN', 'THESE', 'THOSE', 'EACH', 'EVERY', 'BOTH', 'EITHER', 'NEITHER',
            # More prepositions and common words found in data
            'ON', 'IN', 'AT', 'BY', 'OF', 'AS', 'OR', 'IF', 'SO', 'NO', 'BE', 'IT', 'WE', 'MY',
            'HE', 'AN', 'GO', 'ME', 'US', 'UP', 'DEAL', 'TRADE', 'CASE', 'BULL', 'BEAR', 'LONG',
            'SHORT', 'CALL', 'PUTS', 'PUTS', 'MOVE', 'PLAY', 'YOLO', 'FOMO', 'HODL', 'MOON',
            'APES', 'TARD', 'RETARD', 'PUMP', 'DUMP', 'HYPE', 'NEWS', 'LEAK', 'RUMOR', 'TRUMP',
            'BIDEN', 'CHINA', 'VOTE', 'POLL', 'RATE', 'RATES', 'BANK', 'BANKS', 'LOAN', 'DEBT',
            'CASH', 'SAFE', 'RISK', 'GAIN', 'LOSS', 'BULL', 'BEAR', 'RATIO', 'WHEEL', 'WAIT',
            'WHEN', 'WHY', 'BACK', 'LOOK', 'TURN', 'SURE', 'HOPE', 'HELP', 'FIND', 'KNOW',
            'MAKE', 'TAKE', 'GIVE', 'COME', 'WANT', 'NEED', 'WORK', 'PLAY', 'KEEP', 'OPEN',
            'CLOSE', 'HIGH', 'LOW', 'SAME', 'REAL', 'TRUE', 'FALSE', 'RIGHT', 'WRONG', 'EASY',
            'HARD', 'FAST', 'SLOW', 'STOP', 'START', 'MOVE', 'STAY', 'HEAR', 'FEEL', 'SEEM',
            'LOOK', 'WANT', 'GIVE', 'TAKE', 'TELL', 'CALL', 'TRY', 'NEED', 'FEEL', 'SEEM',
            # Common words found in current database that are not tickers
            'LIKE', 'INTO', 'YEARS', 'YOUR', 'WHICH', 'AFTER', 'HTTPS', 'HERE', 'OVER', 'COM',
            'SAID', 'SHARE', 'WWW', 'ANY', 'BEING', 'DOES', 'MOST', 'TECH', 'WELL', 'DON',
            'OTHER', 'BASED', 'HAD', 'SEEMS', 'SUCH', 'UNTIL', 'DOING', 'LOVE', 'ONCE', 'AGAIN',
            'DUE', 'FREE', 'ISN', 'LARGE', 'MADE', 'MAJOR', 'OWN', 'PRICE', 'SAYS', 'SINCE',
            'SORT', 'TITLE', 'TOO', 'WON', 'AGO', 'APRIL', 'ASHX', 'CHECK', 'CHIP', 'CHIPS',
            'PLACE', 'RISES', 'GETS', 'GAINS', 'LOCK', 'WANNA', 'LIVE', 'TOTAL', 'WORTH',
            'FULL', 'COST', 'NICE', 'COOL', 'FAIR', 'HUGE', 'SMALL', 'LATE', 'EARLY', 'SOON',
            'AROUND', 'BEFORE', 'DURING', 'AFTER', 'INSIDE', 'OUTSIDE', 'ABOVE', 'BELOW',
            'BETWEEN', 'THROUGH', 'ACROSS', 'WITHIN', 'WITHOUT', 'AGAINST', 'TOWARDS', 'BEYOND',
            # Additional words found in recent analysis
            'JULY', 'DAILY', 'CALLS', 'FUNDS', 'TALKS', 'TERM', 'VALUE', 'DEEP', 'DIVE',
            'HIT', 'JAPAN', 'AUG', 'FIRM', 'GOLD', 'ROTH', 'JUNE', 'SEPT', 'OCT', 'NOV', 'DEC',
            'JAN', 'FEB', 'MAR', 'MAY', 'PUTS', 'FUND', 'BANK', 'DATA', 'TECH', 'SALE', 'SALES',
            'DEAL', 'DEALS', 'TALK', 'TALKS', 'PLAN', 'PLANS', 'TEAM', 'TEAMS', 'GAME', 'GAMES',
            'LIFE', 'LIVE', 'DIES', 'DEAD', 'KILL', 'SAVE', 'HELP', 'HURT', 'PAIN', 'GAIN',
            'FAST', 'SLOW', 'HARD', 'EASY', 'REAL', 'FAKE', 'TRUE', 'FALSE', 'RICH', 'POOR'
        }
        
        # Filter out excluded words and very short tickers
        filtered_tickers = [ticker for ticker in potential_tickers 
                          if ticker not in excluded_words and len(ticker) >= 2]
        
        # Known stock tickers for better accuracy (add more as needed)
        known_tickers = {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE',
            'CRM', 'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'ASML', 'TSM',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA', 'PYPL',
            'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'TMO', 'DHR', 'BMY', 'LLY', 'MDT',
            'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST',
            'DIS', 'CMCSA', 'T', 'VZ', 'CHTR', 'TMUS', 'ATVI', 'EA', 'TTWO', 'RBLX',
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'KMI', 'OXY', 'HAL', 'BKR',
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'IVV', 'VEA', 'VWO', 'AGG', 'BND',
            'GME', 'AMC', 'BB', 'NOK', 'PLTR', 'NIO', 'LCID', 'RIVN', 'F', 'GM'
        }
        
        # Prioritize known tickers and tickers found in specific contexts
        final_tickers = []
        for ticker in filtered_tickers:
            # Always include known tickers
            if ticker in known_tickers:
                final_tickers.append(ticker)
            # Include tickers found with $ prefix or in context
            elif ticker in (dollar_tickers + context_tickers + reverse_context_tickers):
                final_tickers.append(ticker)
            # More restrictive for unknown tickers from general pattern
            elif len(ticker) >= 3 and len(ticker) <= 5:
                final_tickers.append(ticker)
        
        return list(set(final_tickers))  # Remove duplicates
    
    def analyze_textblob_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with polarity and subjectivity scores
        """
        try:
            blob = TextBlob(text)
            return {
                'textblob_polarity': blob.sentiment.polarity,
                'textblob_subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            logger.warning(f"TextBlob analysis failed: {str(e)}")
            return {
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0
            }
    
    def analyze_vader_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with VADER scores
        """
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return {
                'vader_compound': scores['compound'],
                'vader_positive': scores['pos'],
                'vader_neutral': scores['neu'],
                'vader_negative': scores['neg']
            }
        except Exception as e:
            logger.warning(f"VADER analysis failed: {str(e)}")
            return {
                'vader_compound': 0.0,
                'vader_positive': 0.0,
                'vader_neutral': 1.0,
                'vader_negative': 0.0
            }
    
    def analyze_text(self, text: str) -> Dict[str, any]:
        """
        Perform comprehensive sentiment analysis on text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with all sentiment scores and extracted features
        """
        if not isinstance(text, str) or not text.strip():
            return self._get_empty_analysis()
            
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text:
            return self._get_empty_analysis()
            
        # Extract features
        tickers = self.extract_tickers(text)
        
        # Analyze sentiment
        textblob_scores = self.analyze_textblob_sentiment(cleaned_text)
        vader_scores = self.analyze_vader_sentiment(cleaned_text)
        
        # Combine results
        analysis = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'tickers': tickers,
            'text_length': len(text),
            'cleaned_length': len(cleaned_text),
            **textblob_scores,
            **vader_scores
        }
        
        # Add composite sentiment score
        analysis['composite_sentiment'] = self._calculate_composite_sentiment(
            textblob_scores['textblob_polarity'],
            vader_scores['vader_compound']
        )
        
        # Add sentiment label
        analysis['sentiment_label'] = self._get_sentiment_label(
            analysis['composite_sentiment']
        )
        
        return analysis
    
    def _get_empty_analysis(self) -> Dict[str, any]:
        """Return empty analysis for invalid input"""
        return {
            'original_text': '',
            'cleaned_text': '',
            'tickers': [],
            'text_length': 0,
            'cleaned_length': 0,
            'textblob_polarity': 0.0,
            'textblob_subjectivity': 0.0,
            'vader_compound': 0.0,
            'vader_positive': 0.0,
            'vader_neutral': 1.0,
            'vader_negative': 0.0,
            'composite_sentiment': 0.0,
            'sentiment_label': 'neutral'
        }
    
    def _calculate_composite_sentiment(self, textblob_polarity: float, 
                                     vader_compound: float) -> float:
        """
        Calculate composite sentiment score from TextBlob and VADER
        
        Args:
            textblob_polarity: TextBlob polarity score
            vader_compound: VADER compound score
            
        Returns:
            Composite sentiment score
        """
        # Weighted average (VADER is often better for social media text)
        composite = (0.4 * textblob_polarity) + (0.6 * vader_compound)
        return round(composite, 4)
    
    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """
        Convert sentiment score to label
        
        Args:
            sentiment_score: Numeric sentiment score
            
        Returns:
            Sentiment label ('positive', 'negative', 'neutral')
        """
        if sentiment_score > self.config.SENTIMENT_THRESHOLD_POSITIVE:
            return 'positive'
        elif sentiment_score < self.config.SENTIMENT_THRESHOLD_NEGATIVE:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'title') -> pd.DataFrame:
        """
        Analyze sentiment for all texts in a DataFrame
        
        Args:
            df: DataFrame containing text data
            text_column: Name of the column containing text to analyze
            
        Returns:
            DataFrame with added sentiment analysis columns
        """
        if text_column not in df.columns:
            logger.error(f"Column '{text_column}' not found in DataFrame")
            return df
            
        logger.info(f"Analyzing sentiment for {len(df)} texts...")
        
        # Analyze each text
        results = []
        for idx, text in enumerate(df[text_column]):
            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{len(df)} texts")
                
            analysis = self.analyze_text(str(text))
            results.append(analysis)
        
        # Convert results to DataFrame and merge with original
        results_df = pd.DataFrame(results)
        
        # Add sentiment columns to original DataFrame
        sentiment_columns = [
            'cleaned_text', 'tickers', 'text_length', 'cleaned_length',
            'textblob_polarity', 'textblob_subjectivity',
            'vader_compound', 'vader_positive', 'vader_neutral', 'vader_negative',
            'composite_sentiment', 'sentiment_label'
        ]
        
        for col in sentiment_columns:
            df[col] = results_df[col]
        
        logger.info("Sentiment analysis completed")
        return df
    
    def get_sentiment_statistics(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate sentiment statistics for a DataFrame
        
        Args:
            df: DataFrame with sentiment analysis results
            
        Returns:
            Dictionary with sentiment statistics
        """
        if 'composite_sentiment' not in df.columns:
            logger.error("DataFrame does not contain sentiment analysis results")
            return {}
            
        stats = {
            'total_posts': len(df),
            'mean_sentiment': df['composite_sentiment'].mean(),
            'median_sentiment': df['composite_sentiment'].median(),
            'std_sentiment': df['composite_sentiment'].std(),
            'min_sentiment': df['composite_sentiment'].min(),
            'max_sentiment': df['composite_sentiment'].max(),
            'sentiment_distribution': df['sentiment_label'].value_counts().to_dict(),
            'positive_ratio': (df['sentiment_label'] == 'positive').mean(),
            'negative_ratio': (df['sentiment_label'] == 'negative').mean(),
            'neutral_ratio': (df['sentiment_label'] == 'neutral').mean()
        }
        
        return stats


def main():
    """Example usage of SentimentAnalyzer"""
    analyzer = SentimentAnalyzer()
    
    # Test with sample texts
    sample_texts = [
        "TSLA to the moon! 🚀 This stock is going to make me rich!",
        "I'm very bearish on AAPL right now. Selling all my shares.",
        "The market is looking neutral today. Not much movement.",
        "Check out this DD on $GME - bullish signals everywhere!"
    ]
    
    print("Sentiment Analysis Results:")
    print("=" * 50)
    
    for text in sample_texts:
        result = analyzer.analyze_text(text)
        print(f"\nText: {text}")
        print(f"Tickers: {result['tickers']}")
        print(f"Composite Sentiment: {result['composite_sentiment']}")
        print(f"Label: {result['sentiment_label']}")
        print(f"TextBlob: {result['textblob_polarity']:.3f}")
        print(f"VADER: {result['vader_compound']:.3f}")


if __name__ == "__main__":
    main()
