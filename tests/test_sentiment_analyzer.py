"""
Tests for sentiment analyzer module
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sentiment_analyzer import SentimentAnalyzer


class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for SentimentAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = SentimentAnalyzer()
    
    def test_analyze_positive_text(self):
        """Test sentiment analysis of positive text"""
        text = "This stock is amazing! Great investment opportunity!"
        result = self.analyzer.analyze_text(text)
        
        self.assertIsInstance(result, dict)
        self.assertIn('polarity', result)
        self.assertIn('subjectivity', result)
        self.assertGreater(result['polarity'], 0)  # Should be positive
    
    def test_analyze_negative_text(self):
        """Test sentiment analysis of negative text"""
        text = "This stock is terrible! Avoid at all costs!"
        result = self.analyzer.analyze_text(text)
        
        self.assertIsInstance(result, dict)
        self.assertIn('polarity', result)
        self.assertIn('subjectivity', result)
        self.assertLess(result['polarity'], 0)  # Should be negative
    
    def test_analyze_neutral_text(self):
        """Test sentiment analysis of neutral text"""
        text = "The stock price is $100."
        result = self.analyzer.analyze_text(text)
        
        self.assertIsInstance(result, dict)
        self.assertIn('polarity', result)
        self.assertIn('subjectivity', result)
    
    def test_analyze_empty_text(self):
        """Test sentiment analysis of empty text"""
        text = ""
        result = self.analyzer.analyze_text(text)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['polarity'], 0.0)
    
    def test_extract_tickers(self):
        """Test ticker symbol extraction"""
        text = "I love $TSLA and $AAPL stocks! MSFT is also good."
        tickers = self.analyzer.extract_tickers(text)
        
        self.assertIsInstance(tickers, list)
        self.assertIn('TSLA', tickers)
        self.assertIn('AAPL', tickers)
        self.assertIn('MSFT', tickers)


if __name__ == '__main__':
    unittest.main()
