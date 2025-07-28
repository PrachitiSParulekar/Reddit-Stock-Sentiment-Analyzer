# Reddit API Configuration
# 
# To get these credentials:
# 1. Go to https://www.reddit.com/prefs/apps
# 2. Click "Create App" or "Create Another App"
# 3. Choose "script" as the app type
# 4. Fill in the required fields
# 5. Copy the client ID and secret below

# Create a .env file in the root directory with these values:
# REDDIT_CLIENT_ID=your_client_id_here
# REDDIT_CLIENT_SECRET=your_client_secret_here
# REDDIT_USER_AGENT=StockSentimentAnalyzer/1.0

# Example .env file content:
REDDIT_CLIENT_ID = "your_client_id_here"             # Your Client ID
REDDIT_CLIENT_SECRET = "your_client_secret_here"     # Your Client Secret
REDDIT_USER_AGENT = "StockSentimentAnalyzer/1.0 by /u/yourusername"  # Your User-Agent string with your Reddit username

# Note: Never commit actual credentials to version control!
# Add .env to your .gitignore file
