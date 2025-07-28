"""
Reddit Data Collector Module

This module handles scraping Reddit posts and comments from financial subreddits
using the PRAW (Python Reddit API Wrapper) library.
"""

import praw
import pandas as pd
import json
import os
import sys
from datetime import datetime, timezone
from typing import List, Dict, Optional
import logging

# Add config directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedditCollector:
    """Handles Reddit data collection from financial subreddits"""
    
    def __init__(self):
        """Initialize Reddit API connection"""
        self.reddit = None
        self.config = Config()
        self._setup_reddit_api()
        
    def _setup_reddit_api(self):
        """Setup Reddit API connection using PRAW"""
        try:
            if not all([Config.REDDIT_CLIENT_ID, Config.REDDIT_CLIENT_SECRET]):
                raise ValueError("Reddit API credentials not found. Please check your .env file.")
                
            self.reddit = praw.Reddit(
                client_id=Config.REDDIT_CLIENT_ID,
                client_secret=Config.REDDIT_CLIENT_SECRET,
                user_agent=Config.REDDIT_USER_AGENT
            )
            
            # Test the connection
            self.reddit.user.me()
            logger.info("Reddit API connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Reddit API: {str(e)}")
            self.reddit = None
    
    def scrape_subreddit(self, subreddit_name: str, limit: int = 100, 
                        time_filter: str = 'day') -> List[Dict]:
        """
        Scrape posts from a specific subreddit
        
        Args:
            subreddit_name: Name of the subreddit to scrape
            limit: Maximum number of posts to fetch
            time_filter: Time filter ('hour', 'day', 'week', 'month', 'year', 'all')
            
        Returns:
            List of dictionaries containing post data
        """
        if not self.reddit:
            logger.error("Reddit API not initialized")
            return []
            
        posts_data = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get hot posts from the subreddit
            posts = subreddit.hot(limit=limit)
            
            for post in posts:
                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'selftext': post.selftext,
                    'author': str(post.author) if post.author else '[deleted]',
                    'created_utc': datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'subreddit': subreddit_name,
                    'permalink': post.permalink,
                    'url': post.url,
                    'is_self': post.is_self,
                    'collected_at': datetime.now(timezone.utc).isoformat()
                }
                posts_data.append(post_data)
                
            logger.info(f"Collected {len(posts_data)} posts from r/{subreddit_name}")
            
        except Exception as e:
            logger.error(f"Error scraping r/{subreddit_name}: {str(e)}")
            
        return posts_data
    
    def scrape_post_comments(self, post_id: str, limit: int = 50) -> List[Dict]:
        """
        Scrape comments from a specific post
        
        Args:
            post_id: Reddit post ID
            limit: Maximum number of comments to fetch
            
        Returns:
            List of dictionaries containing comment data
        """
        if not self.reddit:
            logger.error("Reddit API not initialized")
            return []
            
        comments_data = []
        
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Remove "MoreComments" objects
            
            for comment in submission.comments.list()[:limit]:
                if hasattr(comment, 'body'):  # Ensure it's a Comment object
                    comment_data = {
                        'id': comment.id,
                        'parent_id': post_id,
                        'body': comment.body,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'created_utc': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).isoformat(),
                        'score': comment.score,
                        'collected_at': datetime.now(timezone.utc).isoformat()
                    }
                    comments_data.append(comment_data)
                    
            logger.info(f"Collected {len(comments_data)} comments from post {post_id}")
            
        except Exception as e:
            logger.error(f"Error scraping comments for post {post_id}: {str(e)}")
            
        return comments_data
    
    def scrape_multiple_subreddits(self, subreddits: List[str] = None, 
                                 limit_per_subreddit: int = 100) -> pd.DataFrame:
        """
        Scrape posts from multiple subreddits
        
        Args:
            subreddits: List of subreddit names (uses default if None)
            limit_per_subreddit: Posts to fetch per subreddit
            
        Returns:
            Pandas DataFrame with all collected posts
        """
        if subreddits is None:
            subreddits = Config.TARGET_SUBREDDITS
            
        all_posts = []
        
        for subreddit in subreddits:
            posts = self.scrape_subreddit(subreddit, limit=limit_per_subreddit)
            all_posts.extend(posts)
            
        if all_posts:
            df = pd.DataFrame(all_posts)
            logger.info(f"Total posts collected: {len(df)}")
            return df
        else:
            logger.warning("No posts collected")
            return pd.DataFrame()
    
    def save_data(self, data: pd.DataFrame, filename: str = None) -> str:
        """
        Save collected data to CSV file
        
        Args:
            data: DataFrame to save
            filename: Custom filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        Config.create_directories()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_posts_{timestamp}.csv"
            
        filepath = os.path.join(Config.RAW_DATA_PATH, filename)
        data.to_csv(filepath, index=False)
        
        logger.info(f"Data saved to {filepath}")
        return filepath
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load previously saved data
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Loaded DataFrame
        """
        filepath = os.path.join(Config.RAW_DATA_PATH, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            return df
        else:
            logger.error(f"File not found: {filepath}")
            return pd.DataFrame()


def main():
    """Example usage of RedditCollector"""
    collector = RedditCollector()
    
    # Scrape data from multiple subreddits
    data = collector.scrape_multiple_subreddits(limit_per_subreddit=50)
    
    if not data.empty:
        # Save the data
        filepath = collector.save_data(data)
        print(f"Data saved to: {filepath}")
        
        # Display basic statistics
        print(f"\nCollected {len(data)} posts")
        print(f"Subreddits: {data['subreddit'].value_counts().to_dict()}")
    else:
        print("No data collected. Please check your Reddit API credentials.")


if __name__ == "__main__":
    main()
