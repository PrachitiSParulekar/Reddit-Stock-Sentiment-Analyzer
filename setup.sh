#!/bin/bash

# Setup script for Reddit Stock Sentiment Analyzer
# This script helps new users get started quickly

echo "🚀 Setting up Reddit Stock Sentiment Analyzer..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✓ Found Python $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "🔤 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"
python -m textblob.download_corpora

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "📝 Please edit .env file with your Reddit API credentials"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/raw data/processed logs

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your Reddit API credentials"
echo "2. Run: python main.py --mode info"
echo "3. Start collecting data: python main.py --mode pipeline"
echo ""
echo "For more information, see README.md"
