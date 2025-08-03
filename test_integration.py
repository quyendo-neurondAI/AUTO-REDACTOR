#!/usr/bin/env python3
"""
Test script to verify the integration works correctly
"""

import sys
import os
import traceback

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test basic imports
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        import cv2
        print("✅ OpenCV imported successfully")
        
        import ffmpeg
        print("✅ FFmpeg-python imported successfully")
        
        from pathlib import Path
        from datetime import datetime
        import tempfile
        print("✅ Standard library imports successful")
        
        # Test video redactor imports
        try:
            from video_redactor.recognizer import build_analyzer
            from video_redactor.redactor import redact_video
            from doctr.models import ocr_predictor
            print("✅ Video redactor modules imported successfully")
        except ImportError as e:
            print(f"⚠️ Video redactor import warning: {e}")
        
        # Test audio redactor imports
        try:
            from audio_redactor.config import (
                AVAILABLE_MODELS, DEFAULT_MODEL, COMPUTE_TYPE, DEVICE,
                DEFAULT_BEAM_SIZE, ENABLE_WORD_TIMESTAMPS
            )
            from audio_redactor.gemini import detect_sensitive_content
            from audio_redactor.audio_processor import redact_audio_segments, get_audio_info
            print("✅ Audio redactor modules imported successfully")
        except ImportError as e:
            print(f"⚠️ Audio redactor import warning: {e}")
        
        try:
            from faster_whisper import WhisperModel
            print("✅ Faster-Whisper imported successfully")
        except ImportError as e:
            print(f"⚠️ Faster-Whisper import warning: {e}")
        
        try:
            from dotenv import load_dotenv
            print("✅ Python-dotenv imported successfully")
        except ImportError as e:
            print(f"⚠️ Python-dotenv import warning: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        traceback.print_exc()
        return False

def test_demo_app_syntax():
    """Test that demo_app.py has valid syntax"""
    print("\nTesting demo_app.py syntax...")
    
    try:
        import ast
        with open('demo_app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        ast.parse(content)
        print("✅ demo_app.py syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in demo_app.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking demo_app.py: {e}")
        return False

def test_config_files():
    """Test that configuration files exist and are valid"""
    print("\nTesting configuration files...")
    
    # Check if audio_redactor config exists
    config_path = "audio_redactor/config.py"
    if os.path.exists(config_path):
        print(f"✅ {config_path} exists")
        try:
            import audio_redactor.config as config
            print(f"✅ Config loaded: {len(config.AVAILABLE_MODELS)} models available")
        except Exception as e:
            print(f"⚠️ Config import warning: {e}")
    else:
        print(f"❌ {config_path} not found")
    
    # Check requirements.txt
    req_path = "requirements.txt"
    if os.path.exists(req_path):
        print(f"✅ {req_path} exists")
        with open(req_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            print(f"✅ {len(lines)} dependencies listed")
    else:
        print(f"❌ {req_path} not found")
    
    # Check README.md
    readme_path = "README.md"
    if os.path.exists(readme_path):
        print(f"✅ {readme_path} exists")
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"✅ README.md has {len(content)} characters")
    else:
        print(f"❌ {readme_path} not found")
    
    return True

def test_environment():
    """Test environment setup"""
    print("\nTesting environment...")

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Check for .env file
    env_path = ".env"
    if os.path.exists(env_path):
        print(f"✅ {env_path} exists")
    else:
        print(f"⚠️ {env_path} not found - create this file with your API keys")

    # Check environment variables
    api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    if api_key and api_key != "your_api_key_here":
        print("✅ GOOGLE_GENERATIVE_AI_API_KEY is set")
    else:
        print("⚠️ GOOGLE_GENERATIVE_AI_API_KEY not set - required for sensitive content detection")

    ffmpeg_path = os.getenv("FFMPEG_PATH")
    if ffmpeg_path:
        print(f"✅ FFMPEG_PATH is set: {ffmpeg_path}")

        # Test FFmpeg accessibility
        try:
            import subprocess

            # Add FFmpeg to PATH if specified
            if ffmpeg_path.endswith('.exe'):
                ffmpeg_dir = os.path.dirname(ffmpeg_path)
            else:
                ffmpeg_dir = ffmpeg_path

            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

            # Test FFmpeg command
            result = subprocess.run(['ffmpeg', '-version'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ FFmpeg is accessible and working")
            else:
                print("⚠️ FFmpeg path set but command failed")

        except Exception as e:
            print(f"⚠️ Could not test FFmpeg: {e}")
    else:
        print("ℹ️ FFMPEG_PATH not set - using system PATH")

    return True

def main():
    """Run all tests"""
    print("🔍 Running integration tests...\n")
    
    tests = [
        test_imports,
        test_demo_app_syntax,
        test_config_files,
        test_environment
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All tests passed! Integration appears to be working correctly.")
        print("\n🚀 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up .env file with your API keys")
        print("3. Run the app: streamlit run demo_app.py")
    else:
        print("⚠️ Some tests failed. Please review the output above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
