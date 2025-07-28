"""
Setup and Installation Script

This script helps set up the Reddit Stock Sentiment Analyzer project.
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✓ Python version {version.major}.{version.minor}.{version.micro} is compatible")
        return True

def install_dependencies():
    """Install Python dependencies"""
    print("\nInstalling dependencies...")
    
    # Check if pip is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("✗ pip is not available")
        return False
    
    # Install requirements
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing packages from requirements.txt"
    )
    
    if success:
        # Download TextBlob corpora
        success = run_command(
            f"{sys.executable} -m textblob.download_corpora",
            "Downloading TextBlob corpora"
        )
    
    return success

def setup_environment():
    """Set up environment file"""
    print("\nSetting up environment configuration...")
    
    env_file = ".env"
    env_example = ".env.example"
    
    if os.path.exists(env_file):
        print(f"✓ {env_file} already exists")
        return True
    
    if os.path.exists(env_example):
        try:
            # Copy example to .env
            with open(env_example, 'r') as src:
                content = src.read()
            
            with open(env_file, 'w') as dst:
                dst.write(content)
            
            print(f"✓ Created {env_file} from {env_example}")
            print("⚠️  Please edit .env file with your Reddit API credentials")
            return True
            
        except Exception as e:
            print(f"✗ Failed to create {env_file}: {str(e)}")
            return False
    else:
        print(f"✗ {env_example} not found")
        return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "logs"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"✗ Failed to create directory {directory}: {str(e)}")
            return False
    
    return True

def run_basic_test():
    """Run basic functionality test"""
    print("\nRunning basic functionality test...")
    
    try:
        # Add current directory to Python path
        sys.path.insert(0, os.getcwd())
        
        # Import and run basic test
        from tests.test_basic import run_basic_functionality_test
        run_basic_functionality_test()
        return True
        
    except ImportError as e:
        print(f"✗ Could not import test module: {str(e)}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("SETUP COMPLETED!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Edit the .env file with your Reddit API credentials:")
    print("   - Go to https://www.reddit.com/prefs/apps")
    print("   - Create a new application (choose 'script' type)")
    print("   - Copy your client ID and secret to .env")
    
    print("\n2. Test the setup:")
    print("   python main.py --mode info")
    
    print("\n3. Run data collection:")
    print("   python main.py --mode pipeline --limit 10")
    
    print("\n4. Start the API server:")
    print("   python api/app.py")
    
    print("\n5. Test the API:")
    print("   Open http://127.0.0.1:5000/api/health in your browser")
    
    print("\nFor help:")
    print("   python main.py --help")
    
    print("\n" + "="*60)

def main():
    """Main setup function"""
    print("Reddit Stock Sentiment Analyzer - Setup Script")
    print("="*60)
    print("This script will set up your development environment.")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Set up environment
    if not setup_environment():
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️  Dependency installation failed. You may need to install manually:")
        print("pip install -r requirements.txt")
        print("python -m textblob.download_corpora")
        return False
    
    # Run basic test
    if not run_basic_test():
        print("\n⚠️  Basic test failed. The setup may be incomplete.")
    
    # Print next steps
    print_next_steps()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n✗ Setup failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n✓ Setup completed successfully!")
        sys.exit(0)
