"""
Main Application Script

This script orchestrates the data collection, sentiment analysis, and processing pipeline.
"""

import sys
import os
import argparse
import logging
from datetime import datetime
import schedule
import time

# Add src and config directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

from src.data_collector import RedditCollector
from src.sentiment_analyzer import SentimentAnalyzer
from src.data_processor import DataProcessor
from src.utils import setup_logging, print_project_info, validate_reddit_credentials
from config.config import Config

# Set up logging
setup_logging('INFO', 'logs/app.log')
logger = logging.getLogger(__name__)


class SentimentAnalysisApp:
    """Main application class for sentiment analysis pipeline"""
    
    def __init__(self):
        """Initialize application components"""
        self.collector = RedditCollector()
        self.analyzer = SentimentAnalyzer()
        self.processor = DataProcessor()
        
    def run_data_collection(self, subreddits=None, limit_per_subreddit=None):
        """
        Run data collection from Reddit
        
        Args:
            subreddits: List of subreddits to scrape (uses default if None)
            limit_per_subreddit: Posts to collect per subreddit
        """
        logger.info("Starting data collection...")
        
        try:
            # Use default values if not provided
            if subreddits is None:
                subreddits = Config.TARGET_SUBREDDITS
            if limit_per_subreddit is None:
                limit_per_subreddit = Config.MAX_POSTS_PER_SUBREDDIT
            
            # Collect data
            data = self.collector.scrape_multiple_subreddits(
                subreddits=subreddits,
                limit_per_subreddit=limit_per_subreddit
            )
            
            if data.empty:
                logger.warning("No data collected")
                return None
            
            # Save raw data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_posts_{timestamp}.csv"
            filepath = self.collector.save_data(data, filename)
            
            logger.info(f"Collected {len(data)} posts and saved to {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")
            return None
    
    def run_sentiment_analysis(self, data):
        """
        Run sentiment analysis on collected data
        
        Args:
            data: DataFrame with Reddit posts
            
        Returns:
            DataFrame with sentiment analysis results
        """
        logger.info("Starting sentiment analysis...")
        
        try:
            # Analyze sentiment for titles and combine with selftext if available
            data_with_sentiment = self.analyzer.analyze_dataframe(data, 'title')
            
            # If posts have selftext, analyze that too
            if 'selftext' in data.columns:
                selftext_data = data[data['selftext'].notna() & (data['selftext'] != '')]
                if not selftext_data.empty:
                    logger.info(f"Analyzing selftext for {len(selftext_data)} posts...")
                    # For posts with selftext, combine title and body for analysis
                    combined_text = selftext_data['title'] + ' ' + selftext_data['selftext']
                    
                    for idx, text in combined_text.items():
                        analysis = self.analyzer.analyze_text(text)
                        # Update the sentiment scores for posts with selftext
                        for col in ['textblob_polarity', 'textblob_subjectivity', 'vader_compound',
                                  'vader_positive', 'vader_neutral', 'vader_negative',
                                  'composite_sentiment', 'sentiment_label']:
                            if col in analysis:
                                data_with_sentiment.loc[idx, col] = analysis[col]
            
            logger.info("Sentiment analysis completed")
            return data_with_sentiment
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return data
    
    def run_data_processing(self, data):
        """
        Process and aggregate sentiment data
        
        Args:
            data: DataFrame with sentiment analysis results
        """
        logger.info("Starting data processing...")
        
        try:
            # Save processed posts to database
            saved_posts = self.processor.save_posts_to_db(data)
            logger.info(f"Saved {saved_posts} posts to database")
            
            # Aggregate by hour and subreddit
            hourly_agg = self.processor.aggregate_sentiment_by_hour(data)
            if not hourly_agg.empty:
                saved_hourly = self.processor.save_aggregated_data(hourly_agg)
                logger.info(f"Saved {saved_hourly} hourly aggregation records")
            
            # Aggregate by ticker
            ticker_agg = self.processor.aggregate_sentiment_by_ticker(data)
            if not ticker_agg.empty:
                saved_ticker = self.processor.save_aggregated_data(ticker_agg)
                logger.info(f"Saved {saved_ticker} ticker aggregation records")
            
            # Export processed data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.processor.export_data(data, f"processed_posts_{timestamp}")
            
            logger.info("Data processing completed")
            
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
    
    def run_full_pipeline(self, subreddits=None, limit_per_subreddit=None):
        """
        Run the complete data pipeline: collection -> analysis -> processing
        
        Args:
            subreddits: List of subreddits to scrape
            limit_per_subreddit: Posts to collect per subreddit
        """
        logger.info("Starting full sentiment analysis pipeline...")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Data Collection
            data = self.run_data_collection(subreddits, limit_per_subreddit)
            if data is None or data.empty:
                logger.error("Pipeline stopped: No data collected")
                return
            
            # Step 2: Sentiment Analysis
            data_with_sentiment = self.run_sentiment_analysis(data)
            
            # Step 3: Data Processing
            self.run_data_processing(data_with_sentiment)
            
            # Calculate pipeline duration
            duration = datetime.now() - start_time
            logger.info(f"Pipeline completed successfully in {duration}")
            
            # Print summary statistics
            stats = self.analyzer.get_sentiment_statistics(data_with_sentiment)
            logger.info(f"Pipeline summary: {stats}")
            
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
    
    def schedule_data_collection(self):
        """Schedule automatic data collection"""
        logger.info("Setting up scheduled data collection...")
        
        # Schedule collection every hour
        schedule.every().hour.do(self.run_full_pipeline)
        
        logger.info("Scheduled data collection every hour")
        logger.info("Press Ctrl+C to stop the scheduler")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Reddit Stock Sentiment Analyzer')
    parser.add_argument('--mode', choices=['collect', 'analyze', 'pipeline', 'schedule', 'info'], 
                       default='info', help='Operation mode')
    parser.add_argument('--subreddits', nargs='+', help='Subreddits to scrape')
    parser.add_argument('--limit', type=int, default=50, help='Posts per subreddit')
    parser.add_argument('--file', help='CSV file to analyze (for analyze mode)')
    
    args = parser.parse_args()
    
    # Print project info
    if args.mode == 'info':
        print_project_info()
        return
    
    # Validate Reddit credentials
    if not validate_reddit_credentials():
        print("Error: Reddit API credentials not configured.")
        print("Please setup your credentials in the .env file.")
        return
    
    # Initialize application
    app = SentimentAnalysisApp()
    
    if args.mode == 'collect':
        print(f"Collecting data from {args.subreddits or Config.TARGET_SUBREDDITS}...")
        data = app.run_data_collection(args.subreddits, args.limit)
        if data is not None:
            print(f"Successfully collected {len(data)} posts")
        
    elif args.mode == 'analyze':
        if args.file:
            print(f"Analyzing sentiment in file: {args.file}")
            # Load data from file and analyze
            try:
                data = app.collector.load_data(args.file)
                if not data.empty:
                    analyzed_data = app.run_sentiment_analysis(data)
                    app.run_data_processing(analyzed_data)
                    print("Analysis completed")
                else:
                    print("No data found in file")
            except Exception as e:
                print(f"Error: {str(e)}")
        else:
            print("Please specify a file with --file option")
            
    elif args.mode == 'pipeline':
        print("Running full pipeline...")
        app.run_full_pipeline(args.subreddits, args.limit)
        print("Pipeline completed")
        
    elif args.mode == 'schedule':
        print("Starting scheduled data collection...")
        app.schedule_data_collection()


if __name__ == "__main__":
    main()
