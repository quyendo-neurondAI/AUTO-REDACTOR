#!/usr/bin/env python3
"""
Test script to verify FFmpeg is working correctly
"""

import os
import subprocess
import sys
from dotenv import load_dotenv

def test_ffmpeg():
    """Test FFmpeg installation and configuration"""
    print("🔍 Testing FFmpeg configuration...")
    
    # Load environment variables
    load_dotenv()
    
    # Add FFmpeg to PATH if specified
    ffmpeg_path = os.getenv("FFMPEG_PATH")
    if ffmpeg_path:
        print(f"📁 FFMPEG_PATH found in .env: {ffmpeg_path}")
        
        # If FFMPEG_PATH points to the executable, get the directory
        if ffmpeg_path.endswith('.exe'):
            ffmpeg_dir = os.path.dirname(ffmpeg_path)
            print(f"📂 Using directory: {ffmpeg_dir}")
        else:
            ffmpeg_dir = ffmpeg_path
            print(f"📂 Using directory: {ffmpeg_dir}")
        
        # Add to PATH
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
        print(f"✅ Added FFmpeg directory to PATH")
    else:
        print("ℹ️ No FFMPEG_PATH set, using system PATH")
    
    # Test FFmpeg command
    try:
        print("\n🧪 Testing FFmpeg command...")
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ FFmpeg is working!")
            # Extract version info
            lines = result.stdout.split('\n')
            version_line = lines[0] if lines else "Unknown version"
            print(f"📋 Version: {version_line}")
            return True
        else:
            print(f"❌ FFmpeg command failed with return code: {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg command timed out")
        return False
    except FileNotFoundError:
        print("❌ FFmpeg not found in PATH")
        print("💡 Make sure FFmpeg is installed and FFMPEG_PATH is set correctly in .env")
        return False
    except Exception as e:
        print(f"❌ Error testing FFmpeg: {e}")
        return False

def test_ffmpeg_python():
    """Test ffmpeg-python library"""
    print("\n🔍 Testing ffmpeg-python library...")
    
    try:
        import ffmpeg
        print("✅ ffmpeg-python imported successfully")
        
        # Test probe function with a simple command
        print("🧪 Testing ffmpeg.probe...")
        
        # Create a simple test - this should fail gracefully if no input
        try:
            # This will fail but should give us a proper error, not "file not found"
            ffmpeg.probe("nonexistent_file.mp3")
        except ffmpeg.Error as e:
            if "No such file or directory" in str(e) or "cannot find the file" in str(e):
                print("✅ ffmpeg-python can execute FFmpeg (expected file not found error)")
                return True
            else:
                print(f"⚠️ Unexpected ffmpeg error: {e}")
                return False
        except Exception as e:
            if "The system cannot find the file specified" in str(e):
                print("❌ FFmpeg executable not found by ffmpeg-python")
                return False
            else:
                print(f"⚠️ Unexpected error: {e}")
                return False
                
    except ImportError:
        print("❌ ffmpeg-python not installed")
        print("💡 Install with: pip install ffmpeg-python")
        return False

def main():
    """Run FFmpeg tests"""
    print("🔧 FFmpeg Configuration Test")
    print("=" * 40)
    
    # Test basic FFmpeg
    ffmpeg_ok = test_ffmpeg()
    
    # Test ffmpeg-python library
    ffmpeg_python_ok = test_ffmpeg_python()
    
    print("\n📊 Test Results:")
    print(f"FFmpeg command: {'✅ OK' if ffmpeg_ok else '❌ FAILED'}")
    print(f"ffmpeg-python:  {'✅ OK' if ffmpeg_python_ok else '❌ FAILED'}")
    
    if ffmpeg_ok and ffmpeg_python_ok:
        print("\n🎉 All tests passed! FFmpeg should work with the audio redactor.")
    else:
        print("\n⚠️ Some tests failed. Please check your FFmpeg installation.")
        print("\n💡 Troubleshooting steps:")
        print("1. Download FFmpeg from https://ffmpeg.org/download.html")
        print("2. Extract to a folder (e.g., C:\\ffmpeg)")
        print("3. Set FFMPEG_PATH in .env file")
        print("4. Restart the application")
        
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
