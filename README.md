# 🔐 Multimedia Redactor Tool

A comprehensive multimedia redaction tool that detects and redacts sensitive information from video and audio files using advanced OCR, NLP, and AI transcription technologies.

## 🌟 Features

### Video Redaction
- **OCR-based text detection** using DocTR (Document Text Recognition)
- **PII detection** using Microsoft Presidio analyzer
- **Real-time video processing** with frame-by-frame analysis
- **Smart caching** to avoid reprocessing identical frames
- **Visual redaction** with customizable blur/rectangle overlays

### Audio Redaction
- **High-quality transcription** using OpenAI's Whisper models
- **AI-powered sensitive content detection** using Google Gemini
- **Precise audio redaction** with beep sound replacement
- **Word-level timestamp accuracy** for precise redaction
- **Multiple audio format support** (WAV, MP3, FLAC, M4A, OGG, AAC)

### Processing Modes
1. **Video Redaction Only** - Redact sensitive text from video frames
2. **Audio Redaction Only** - Transcribe and redact sensitive audio content
3. **Video + Audio Redaction** - Complete multimedia redaction
4. **Audio Transcription Only** - Generate transcripts without redaction

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (for audio/video processing)
- Google Generative AI API key (for sensitive content detection)

### System Dependencies

**🎉 NEW: Automatic FFmpeg Detection!**
The system now automatically detects FFmpeg in common locations. Simply extract FFmpeg to your project folder or any standard location - no configuration needed!

#### Windows
Option 1: Download from https://drive.google.com/file/d/1viNeLPpG3N4MoSEjU3-X-uMVzdTTXQYf/view?usp=sharing
Extract to project folder


# Option 2: Download portable version (auto-detected)
# 1. Download from https://ffmpeg.org/download.html
# 2. Extract to project folder or C:\ffmpeg
# 3. No configuration needed - automatically detected!
```

#### macOS
```bash
# Install FFmpeg using homebrew (auto-detected)
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
```

**Auto-Detection Locations:**
- Project directory: `ffmpeg*/bin/ffmpeg.exe`
- System folders: `C:/ffmpeg*/bin/ffmpeg.exe`
- Program Files: `C:/Program Files/ffmpeg*/bin/ffmpeg.exe`
- Portable builds: `ffmpeg-*-essentials-build/bin/ffmpeg.exe`

### Python Dependencies

1. **Clone the repository:**
```bash
git clone <repository-url>
cd auto-redactor
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download PII detection models:**
Download from: https://drive.google.com/file/d/1CY6yM7s2JujW0q_pTnc3z5XCdIC55Spu/view?usp=sharing
Extract to `video_redactor/models`

### Environment Configuration

Create a `.env` file in the root directory (copy from `.env.example`):
```bash
cp .env.example .env
```

Then edit `.env` with your configuration:
```env
# Google Generative AI API Key (required for sensitive content detection)
GOOGLE_GENERATIVE_AI_API_KEY=your_api_key_here

# FFmpeg path (OPTIONAL - auto-detection usually works!)
# Only set this if automatic detection fails
# IMPORTANT: Use forward slashes (/) in Windows paths to avoid escape issues
# Can be either the directory or the full path to executable
# FFMPEG_PATH="C:/path/to/ffmpeg/bin"
# or
# FFMPEG_PATH="C:/path/to/ffmpeg/bin/ffmpeg.exe"
```

#### Getting Google Generative AI API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key
5. Add it to your `.env` file

**Note:** The API has usage limits. Check [Google AI pricing](https://ai.google.dev/pricing) for current rates.

## 🚀 Usage

### Running the Application

#### Option 1: Quick Start (Recommended)
```bash
python run_app.py
```

#### Option 2: Direct Streamlit
```bash
streamlit run demo_app.py
```

#### Option 3: Test Integration First
```bash
python test_integration.py
```

The application will be available at `http://localhost:8501`

### Using Different Processing Modes

#### 1. Video Redaction Only
- Upload a video file (MP4, MOV, AVI)
- Select "Video Redaction Only" mode
- Click "Redact Video"
- Download the redacted video

#### 2. Audio Redaction Only
- Upload an audio file (WAV, MP3, FLAC, M4A, OGG, AAC)
- Select "Audio Redaction Only" mode
- Configure Whisper model settings
- Enable word timestamps
- Click "Redact Audio"
- Download the redacted audio

#### 3. Video + Audio Redaction
- Upload a video file with audio
- Select "Video + Audio Redaction" mode
- Configure audio settings
- Click "Redact Video + Audio"
- Download the fully redacted video

#### 4. Audio Transcription Only
- Upload an audio file
- Select "Audio Transcription Only" mode
- Configure transcription settings
- Click "Transcribe Audio"
- Download the transcript file

### Configuration Options

#### Whisper Model Selection
- **tiny/tiny.en** (~39 MB) - Fastest, basic accuracy
- **base/base.en** (~74 MB) - Good balance of speed/accuracy
- **small/small.en** (~244 MB) - Better accuracy
- **medium/medium.en** (~769 MB) - High accuracy, slower
- **large-v1/v2/v3** (~1550 MB) - Highest accuracy, slowest

#### Audio Settings
- **Beam Size** (1-10) - Higher values improve accuracy but take longer
- **Word Timestamps** - Required for sensitive content detection
- **Compute Type** - int8, int16, float16, float32 (affects speed/accuracy)

## 📁 Project Structure

```
redactor_tool/
├── demo_app.py              # Main Streamlit application
├── run_app.py              # Quick start script
├── test_integration.py     # Integration test script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .env                    # Environment variables (create this)
├── .env.example            # Example environment file
├── video_redactor/         # Video redaction module
│   ├── __init__.py
│   ├── redactor.py         # Video processing logic
│   ├── recognizer.py       # PII detection
│   ├── ocr.py             # OCR functionality
│   └── utils.py           # Utility functions
├── audio_redactor/         # Audio redaction module
│   ├── __init__.py
│   ├── audio_processor.py  # Audio processing logic
│   ├── gemini.py          # AI content detection
│   ├── config.py          # Configuration settings
│   └── streamlit_app.py   # Standalone audio app
└── test/                  # Test files and examples
```

## 🔧 Technical Details

### Video Processing Pipeline
1. **Frame Extraction** - Extract frames from video using OpenCV
2. **OCR Processing** - Detect text using DocTR
3. **PII Analysis** - Analyze text using Presidio
4. **Visual Redaction** - Apply redaction overlays
5. **Video Reconstruction** - Rebuild video with redacted frames

### Audio Processing Pipeline
1. **Audio Extraction** - Extract audio from video (if needed)
2. **Transcription** - Convert speech to text using Whisper
3. **Content Analysis** - Detect sensitive content using Gemini AI
4. **Audio Redaction** - Replace sensitive segments with beep sounds
5. **Audio Reconstruction** - Rebuild audio with redactions

### Supported File Formats

#### Video Input
- MP4, MOV, AVI

#### Audio Input
- WAV, MP3, FLAC, M4A, OGG, AAC

#### Output Formats
- Video: MP4
- Audio: MP3
- Transcripts: TXT

## 🔒 Privacy and Security

- **Local Processing** - All video/audio processing happens locally
- **API Usage** - Only transcripts are sent to Google Gemini for analysis
- **No Data Storage** - Files are processed in temporary directories and cleaned up
- **Configurable Sensitivity** - Adjust detection thresholds as needed

## 📋 Examples

### Example 1: Redacting a Meeting Recording

```bash
# 1. Start the application
streamlit run demo_app.py

# 2. Upload your meeting video file (meeting.mp4)
# 3. Select "Video + Audio Redaction" mode
# 4. Configure settings:
#    - Whisper Model: "base" (good balance)
#    - Beam Size: 5
#    - Word Timestamps: Enabled
# 5. Click "Redact Video + Audio"
# 6. Download the redacted file: meeting_redacted_20241203_143022.mp4
```

### Example 2: Transcribing a Podcast

```bash
# 1. Upload your podcast audio file (podcast.mp3)
# 2. Select "Audio Transcription Only" mode
# 3. Configure settings:
#    - Whisper Model: "small" (better accuracy)
#    - Beam Size: 5
#    - Word Timestamps: Enabled (for analysis)
# 4. Click "Transcribe Audio"
# 5. Review transcript and download: podcast_transcript_20241203_143022.txt
# 6. Optionally analyze for sensitive content
```

### Example 3: Processing Training Videos

```bash
# 1. Upload training video with slides (training.mp4)
# 2. Select "Video Redaction Only" mode
# 3. Click "Redact Video"
# 4. Download redacted video with blurred sensitive text
```

## 🔧 Advanced Configuration

### Custom Whisper Models

You can modify `audio_redactor/config.py` to add custom models:

```python
AVAILABLE_MODELS = [
    "tiny", "base", "small", "medium", "large-v3",
    "distil-large-v2",  # Faster large model
    # Add custom models here
]
```

### Adjusting Redaction Sensitivity

Modify the prompt in `audio_redactor/gemini.py` to customize what content gets detected:

```python
# Add custom detection criteria
PROMPT = """
Your custom redaction criteria here...
"""
```

### Video Redaction Customization

Modify `video_redactor/redactor.py` to change redaction appearance:

```python
# Change redaction color/style
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), -1)  # Red rectangles
```

## 🎯 Use Cases

### Business & Enterprise
- **Meeting Recordings** - Redact sensitive business information from recorded meetings
- **Customer Support** - Remove PII from support call recordings
- **Training Materials** - Clean training videos of confidential information
- **Compliance** - Meet GDPR, HIPAA, and other privacy regulations

### Legal & Government
- **Court Proceedings** - Redact sensitive information from legal recordings
- **Investigation Materials** - Clean evidence files for public release
- **Document Processing** - Handle confidential government communications

### Healthcare
- **Patient Consultations** - Remove PHI from medical recordings
- **Research Data** - Clean research materials for publication
- **Training Content** - Prepare educational materials without patient data

### Education
- **Lecture Recordings** - Remove student information from class recordings
- **Research Presentations** - Clean academic presentations for sharing
- **Online Courses** - Prepare content for public distribution

## 🔍 Sensitive Content Detection

The tool detects and redacts various types of sensitive information:

### Personal Identifiers
- Full names of individuals
- Email addresses
- Phone numbers
- Home and business addresses
- Social security numbers and national IDs

### Credentials & Security
- API keys and tokens
- Passwords and login credentials
- Authentication methods
- Internal IP addresses and server hostnames
- Database connection strings

### Business Confidentiality
- Internal project code names
- Client and customer names (non-public)
- Contract details and financial amounts
- Salary and compensation information
- Proprietary metrics and KPIs
- Unreleased product information
- Legal terms and negotiations

### Contextual Detection
The AI analyzes context to identify sensitive phrases like:
- "My password is..."
- "You can email me at..."
- "His salary is..."
- "The API key is..."
- "Her phone number is..."

## 🐛 Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - **Error**: `[WinError 2] The system cannot find the file specified`
   - **Solution**: The system usually auto-detects FFmpeg, but if it fails:
   - **Steps**:
     1. Download FFmpeg from https://ffmpeg.org/download.html
     2. Extract to your project folder or `C:\ffmpeg`
     3. **Try restarting the app first** - auto-detection should find it
     4. If still not found, manually set `FFMPEG_PATH` in `.env` file:
        - Use forward slashes: `FFMPEG_PATH="C:/ffmpeg/bin/ffmpeg.exe"`
     5. Test with: `ffmpeg -version` (after restarting the app)
   - **Auto-detection locations**: Project folder, C:/ffmpeg*, Program Files

2. **Google API errors**
   - Verify GOOGLE_GENERATIVE_AI_API_KEY is set correctly
   - Check API quota and billing status
   - Ensure Gemini API is enabled in Google Cloud Console

3. **Memory issues with large files**
   - Use smaller Whisper models (tiny, base instead of large)
   - Process shorter audio segments
   - Increase system RAM or use swap space
   - Close other applications during processing

4. **Slow processing**
   - Use smaller/faster Whisper models
   - Reduce beam size (try 1-3 instead of 5)
   - Use GPU acceleration (install CUDA if available)
   - Process shorter files or split large files

5. **Audio quality issues**
   - Ensure input audio is clear and high quality
   - Use appropriate Whisper model for your language
   - Check audio format compatibility
   - Verify sample rate and bit depth

6. **Transcription accuracy problems**
   - Use larger Whisper models for better accuracy
   - Ensure audio is in supported language
   - Check for background noise or poor audio quality
   - Enable word timestamps for better detection

### Performance Optimization

#### For Large Files
- Split videos/audio into smaller chunks
- Use batch processing for multiple files
- Monitor system resources during processing
- Consider cloud processing for very large files

#### For Better Accuracy
- Use higher quality input files
- Choose appropriate Whisper model size
- Increase beam size (with performance trade-off)
- Ensure clean audio without background noise

#### For Faster Processing
- Use smaller Whisper models
- Reduce beam size to 1-2
- Disable word timestamps if not needed for redaction
- Use SSD storage for temporary files

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration options

---

Built with ❤️ using Streamlit, OpenCV, Presidio, Faster-Whisper, Google Gemini AI, and FFmpeg
