#!/usr/bin/env python3
"""
Quick start script for the Multimedia Redactor Tool
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if requirements are installed"""
    try:
        import streamlit
        import cv2
        import ffmpeg
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_environment():
    """Check environment setup"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️ .env file not found")
        print("Please copy .env.example to .env and configure your API keys")
        return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("⚠️ GOOGLE_GENERATIVE_AI_API_KEY not configured")
        print("Please set your Google Generative AI API key in .env file")
        return False
    
    return True

def main():
    """Main function to start the app"""
    print("🔐 Multimedia Redactor Tool")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("demo_app.py").exists():
        print("❌ demo_app.py not found. Please run this script from the redactor_tool directory.")
        return 1
    
    # Check requirements
    print("Checking dependencies...")
    if not check_requirements():
        return 1
    print("✅ Dependencies OK")
    
    # Check environment
    print("Checking environment...")
    if not check_environment():
        print("⚠️ Environment issues detected, but continuing...")
    else:
        print("✅ Environment OK")
    
    # Start the app
    print("\n🚀 Starting Streamlit app...")
    print("The app will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the app")
    print("-" * 40)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "demo_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting app: {e}")
        return 1
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install with: pip install streamlit")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
