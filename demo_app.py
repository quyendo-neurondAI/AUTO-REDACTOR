import streamlit as st
import tempfile
import os
import cv2
import ffmpeg
from pathlib import Path
from datetime import datetime
import traceback
from dotenv import load_dotenv

# Video redaction imports
from video_redactor.recognizer import build_analyzer
from video_redactor.redactor import redact_video
from video_redactor.object_detector import ObjectDetector
from doctr.models import ocr_predictor

# Audio redaction imports
from audio_redactor.config import (
    AVAILABLE_MODELS, DEFAULT_MODEL, COMPUTE_TYPE, DEVICE,
    DEFAULT_BEAM_SIZE, ENABLE_WORD_TIMESTAMPS
)
from audio_redactor.gemini import detect_sensitive_content
from audio_redactor.audio_processor import redact_audio_segments, get_audio_info
from faster_whisper import WhisperModel

def find_ffmpeg_automatically():
    """Automatically find FFmpeg installation in common locations"""
    import glob

    # Common FFmpeg installation patterns
    search_patterns = [
        # Current directory and subdirectories
        "ffmpeg*/bin/ffmpeg.exe",
        "ffmpeg*/ffmpeg.exe",
        "*/ffmpeg*/bin/ffmpeg.exe",
        "*/ffmpeg*/ffmpeg.exe",
        # Common installation directories
        "C:/ffmpeg*/bin/ffmpeg.exe",
        "C:/Program Files/ffmpeg*/bin/ffmpeg.exe",
        "C:/Program Files (x86)/ffmpeg*/bin/ffmpeg.exe",
        # Portable installations
        "ffmpeg-*-essentials_build/bin/ffmpeg.exe",
        "ffmpeg-*-full_build/bin/ffmpeg.exe",
        "../ffmpeg*/bin/ffmpeg.exe",
        "../../ffmpeg*/bin/ffmpeg.exe",
    ]

    # Also check relative to current working directory
    cwd = os.getcwd()
    for pattern in search_patterns[:4]:  # Only local patterns
        full_pattern = os.path.join(cwd, pattern)
        matches = glob.glob(full_pattern)
        if matches:
            # Convert to forward slashes for consistency
            return matches[0].replace('\\', '/')

    # Check system-wide patterns
    for pattern in search_patterns[4:]:
        matches = glob.glob(pattern)
        if matches:
            # Convert to forward slashes for consistency
            return matches[0].replace('\\', '/')

    return None

# Load environment variables
load_dotenv()

# Add FFmpeg to PATH if specified or found automatically
ffmpeg_path = os.getenv("FFMPEG_PATH")

# If no explicit path set, try to find FFmpeg automatically
if not ffmpeg_path:
    print("No FFMPEG_PATH set, searching for FFmpeg automatically...")
    ffmpeg_path = find_ffmpeg_automatically()
    if ffmpeg_path:
        print(f"Found FFmpeg automatically at: {ffmpeg_path}")
    else:
        print("Could not find FFmpeg automatically")

if ffmpeg_path:
    # If FFMPEG_PATH points to the executable, get the directory
    if ffmpeg_path.endswith('.exe'):
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
    else:
        ffmpeg_dir = ffmpeg_path

    # Add to PATH
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
    print(f"Added FFmpeg directory to PATH: {ffmpeg_dir}")

# -- CACHED RESOURCE LOADERS --
@st.cache_resource
def load_ocr_model():
    st.info("🔄 Loading OCR model...")
    return ocr_predictor(pretrained=True).to('cuda')

@st.cache_resource
def load_analyzer():
    st.info("🔄 Building PII analyzer...")
    return build_analyzer()

@st.cache_resource
def load_object_detector():
    """Load and cache YOLOv8 object detector"""
    try:
        st.info("🔄 Loading YOLOv8 object detector...")
        detector = ObjectDetector()
        return detector
    except Exception as e:
        st.error(f"Error loading YOLOv8 model: {e}")
        return None

@st.cache_resource
def load_whisper_model(model_size):
    """Load and cache Whisper model for audio transcription"""
    try:
        st.info(f"🔄 Loading Whisper model ({model_size})...")
        model = WhisperModel(
            model_size,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

# -- UTILITY FUNCTIONS --
def get_ffmpeg_cmd():
    """Get the ffmpeg command to use (either explicit path or 'ffmpeg')"""
    # First check environment variable
    ffmpeg_exe = os.getenv("FFMPEG_PATH")
    if not ffmpeg_exe:
        # Try to find automatically
        ffmpeg_exe = find_ffmpeg_automatically()

    if ffmpeg_exe and ffmpeg_exe.endswith('.exe') and os.path.exists(ffmpeg_exe):
        return ffmpeg_exe
    return 'ffmpeg'

def extract_audio_from_video(video_path, audio_path):
    """Extract audio from video file"""
    try:
        ffmpeg_cmd = get_ffmpeg_cmd()
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='libmp3lame', format='mp3')
            .overwrite_output()
            .run(quiet=True, cmd=ffmpeg_cmd)
        )
        return True
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return False

def combine_video_audio(video_path, audio_path, output_path):
    """Combine video with redacted audio"""
    try:
        ffmpeg_cmd = get_ffmpeg_cmd()
        video_input = ffmpeg.input(video_path)
        audio_input = ffmpeg.input(audio_path)

        (
            ffmpeg
            .output(video_input, audio_input, output_path, vcodec='copy', acodec='copy')
            .overwrite_output()
            .run(quiet=True, cmd=ffmpeg_cmd)
        )
        return True
    except Exception as e:
        st.error(f"Error combining video and audio: {e}")
        return False

def transcribe_audio(audio_file, model_size, beam_size, word_timestamps):
    """Transcribe audio file using Whisper"""
    try:
        model = load_whisper_model(model_size)
        if not model:
            return None, None

        segments, info = model.transcribe(
            audio_file,
            word_timestamps=word_timestamps,
            beam_size=beam_size
        )

        return segments, info

    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None, None

def format_transcript_output(segments, info, audio_filename, model_size):
    """Format transcription output with word-level timestamps"""
    output_lines = []
    output_lines.append(f"Transcription of: {audio_filename}")
    output_lines.append(f"Model: {model_size}")
    output_lines.append(f"Language: {info.language} (confidence: {info.language_probability:.2f})")
    output_lines.append(f"Duration: {info.duration:.2f}s")
    output_lines.append("-" * 80)
    output_lines.append("")

    for segment in segments:
        output_lines.append(f"Segment [{segment.start:.2f}s -> {segment.end:.2f}s]: {segment.text}")

        if hasattr(segment, 'words') and segment.words:
            output_lines.append("Word-level timestamps:")
            for word in segment.words:
                output_lines.append(f"  [{word.start:.2f}s -> {word.end:.2f}s] {word.word}")
        output_lines.append("")

    return "\n".join(output_lines)

# -- STREAMLIT UI SETUP --
st.set_page_config(page_title="Multimedia Redactor Demo", layout="wide")
st.title("🔐 Multimedia Redactor with Computer Vision + PII Detection")
st.markdown("Upload video or audio files to detect and redact sensitive PII using OCR, NLP, and AI transcription.")

# -- LOAD MODELS ON STARTUP --
with st.spinner("🔍 Initializing models..."):
    analyzer = load_analyzer()
    ocr_model = load_ocr_model()
    object_detector = load_object_detector()

# Inject ocr_model into the redactor module (used as a global)
import video_redactor.redactor as redactor_module_redactor
redactor_module_redactor.ocr_model = ocr_model

# -- SIDEBAR CONFIGURATION --
st.sidebar.header("⚙️ Configuration")

# Processing mode selection
processing_mode = st.sidebar.selectbox(
    "Select Processing Mode",
    [
        "Video Redaction Only",
        "Audio Redaction Only",
        "Video + Audio Redaction",
        "Audio Transcription Only"
    ],
    help="Choose what type of processing to perform"
)

# Video redaction settings (shown for video-related modes)
if processing_mode in ["Video Redaction Only", "Video + Audio Redaction"]:
    st.sidebar.subheader("Video Settings")
    
    # Text redaction options
    enable_text_redaction = st.sidebar.checkbox(
        "Enable Text Redaction (OCR + PII)",
        value=True,
        help="Detect and redact sensitive text in video"
    )
    
    # Object detection options
    enable_object_detection = st.sidebar.checkbox(
        "Enable Object Filtering",
        value=False,
        help="Detect and blur objects in video"
    )
    
    if enable_object_detection:
        # Object detection settings
        st.sidebar.subheader("Object Filtering Settings")
        
        # Object classes selection
        if object_detector:
            available_classes = object_detector.get_available_classes()
            
            # Common classes for quick selection
            common_classes = ["person", "car", "phone", "laptop", "chair", "book", "cup", "bottle"]
            
            # Create a multiselect for common classes
            selected_common_classes = st.sidebar.multiselect(
                "Common Object Classes",
                common_classes,
                default=["person"],
                help="Select common object classes to blur"
            )
            
            # Allow custom class input

            
            # Combine selected classes
            object_classes = selected_common_classes.copy()

            # Show available classes info
            if st.sidebar.button("📋 Show All Available Classes"):
                st.sidebar.write("**Available Object Classes:**")
                for i, class_name in enumerate(available_classes, 1):
                    st.sidebar.write(f"{i:3d}. {class_name}")
        
        # Object detection parameters
        object_confidence = st.sidebar.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Higher values = more accurate but fewer detections"
        )
        
        blur_strength = st.sidebar.slider(
            "Blur Strength",
            min_value=1,
            max_value=49,
            value=15,
            step=2,
            help="Higher values = stronger blur effect"
        )

# Audio model settings (shown for audio-related modes)
if processing_mode in ["Audio Redaction Only", "Video + Audio Redaction", "Audio Transcription Only"]:
    st.sidebar.subheader("Audio Settings")

    model_size = st.sidebar.selectbox(
        "Whisper Model",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in AVAILABLE_MODELS else 0,
        help="Smaller models are faster but less accurate"
    )

    beam_size = st.sidebar.slider(
        "Beam Size",
        min_value=1,
        max_value=10,
        value=DEFAULT_BEAM_SIZE,
        help="Higher values may improve accuracy but take longer"
    )

    word_timestamps = st.sidebar.checkbox(
        "Enable Word Timestamps",
        value=ENABLE_WORD_TIMESTAMPS,
        help="Required for sensitive content detection"
    )

# -- FILE UPLOAD --
file_types = []
if processing_mode in ["Video Redaction Only", "Video + Audio Redaction"]:
    file_types.extend(["mp4", "mov", "avi"])
if processing_mode in ["Audio Redaction Only", "Audio Transcription Only"]:
    file_types.extend(["wav", "mp3", "flac", "m4a", "ogg", "aac"])
if processing_mode == "Video + Audio Redaction":
    file_types.extend(["wav", "mp3", "flac", "m4a", "ogg", "aac"])

uploaded_file = st.file_uploader(
    f"📤 Upload a {'video' if 'Video' in processing_mode else 'audio'} file",
    type=file_types
)

if uploaded_file:
    # Save uploaded file temporarily
    file_suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_file_path = tmp_input.name

    # Display file info
    st.subheader("📁 File Information")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024 / 1024:.2f} MB")
        st.write(f"**Processing Mode:** {processing_mode}")

    with col2:
        # Show preview based on file type
        if processing_mode in ["Video Redaction Only", "Video + Audio Redaction"] and file_suffix.lower() in ['.mp4', '.mov', '.avi']:
            cap = cv2.VideoCapture(input_file_path)
            success, frame = cap.read()
            cap.release()
            if success:
                st.image(frame[:, :, ::-1], channels="RGB", caption="First Frame Preview", width=300)
        elif file_suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']:
            st.audio(uploaded_file)
            # Get audio info
            audio_info = get_audio_info(input_file_path)
            if audio_info:
                st.write(f"**Duration:** {audio_info['duration']:.2f} seconds")
                st.write(f"**Sample Rate:** {audio_info['sample_rate']} Hz")
                st.write(f"**Channels:** {audio_info['channels']}")

    # Processing based on selected mode
    if processing_mode == "Video Redaction Only":
        if st.button("🔧 Redact Video", type="primary"):
            st.info("Processing video for redaction...")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_output:
                output_video_path = tmp_output.name

            try:
                # Prepare object detection parameters
                object_detector_param = None
                object_classes_param = None
                object_confidence_param = 0.5
                blur_strength_param = 15
                
                if enable_object_detection and object_detector:
                    object_detector_param = object_detector
                    object_classes_param = object_classes if 'object_classes' in locals() else None
                    object_confidence_param = object_confidence if 'object_confidence' in locals() else 0.5
                    blur_strength_param = blur_strength if 'blur_strength' in locals() else 15
                
                redact_video(
                    input_file_path, 
                    output_video_path, 
                    analyzer,
                    object_detector=object_detector_param,
                    object_classes=object_classes_param,
                    object_confidence=object_confidence_param,
                    blur_strength=blur_strength_param,
                    enable_text_redaction=enable_text_redaction,
                    enable_object_redaction=enable_object_detection
                )

                with open(output_video_path, "rb") as f:
                    video_data = f.read()
                    st.success("✅ Video redaction complete!")
                    
                    # Show processing summary
                    processing_summary = []
                    if enable_text_redaction:
                        processing_summary.append("✅ Text redaction (OCR + PII)")
                    if enable_object_detection:
                        processing_summary.append("✅ Object detection (YOLOv8)")
                        if object_classes_param:
                            processing_summary.append(f"   - Classes: {', '.join(object_classes_param)}")
                            processing_summary.append(f"   - Confidence: {object_confidence_param}")
                            processing_summary.append(f"   - Blur strength: {blur_strength_param}")
                    
                    if processing_summary:
                        st.info("Processing Summary:")
                        for item in processing_summary:
                            st.write(f"   {item}")
                    
                    st.download_button(
                        "📥 Download Redacted Video",
                        video_data,
                        file_name=f"redacted_{uploaded_file.name}",
                        mime="video/mp4"
                    )

                os.remove(output_video_path)
            except Exception as e:
                st.error(f"❌ Error processing video: {e}")
                st.error(traceback.format_exc())

    elif processing_mode == "Audio Redaction Only":
        if st.button("🔧 Redact Audio", type="primary"):
            if not word_timestamps:
                st.warning("⚠️ Word timestamps are required for sensitive content detection. Please enable them in the sidebar.")
                st.stop()

            with st.spinner("Transcribing and redacting audio... This may take a while."):
                try:
                    # Step 1: Transcribe audio
                    st.info("Step 1/3: Transcribing audio...")
                    segments, info = transcribe_audio(input_file_path, model_size, beam_size, word_timestamps)

                    if not segments or not info:
                        st.error("❌ Transcription failed")
                        st.stop()

                    # Step 2: Format transcript and detect sensitive content
                    st.info("Step 2/3: Detecting sensitive content...")
                    transcript_content = format_transcript_output(segments, info, uploaded_file.name, model_size)
                    sensitive_timestamps = detect_sensitive_content(transcript_content)

                    if sensitive_timestamps:
                        st.success(f"✅ Found {len(sensitive_timestamps)} sensitive segments")

                        # Step 3: Redact audio
                        st.info("Step 3/3: Creating redacted audio...")
                        output_filename = f"{Path(uploaded_file.name).stem}_redacted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_output:
                            temp_output_path = tmp_output.name

                        success = redact_audio_segments(input_file_path, sensitive_timestamps, temp_output_path)

                        if success:
                            with open(temp_output_path, 'rb') as f:
                                redacted_audio_bytes = f.read()

                            st.success("✅ Audio redaction complete!")

                            # Show redaction stats
                            total_redacted_time = sum(end - start for start, end in sensitive_timestamps)
                            original_duration = info.duration
                            redaction_percentage = (total_redacted_time / original_duration * 100) if original_duration > 0 else 0

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Segments Redacted", len(sensitive_timestamps))
                                st.metric("Time Redacted", f"{total_redacted_time:.2f}s")
                            with col2:
                                st.metric("Original Duration", f"{original_duration:.2f}s")
                                st.metric("Redaction %", f"{redaction_percentage:.1f}%")

                            # Play redacted audio
                            st.audio(redacted_audio_bytes)

                            # Download button
                            st.download_button(
                                label="💾 Download Redacted Audio",
                                data=redacted_audio_bytes,
                                file_name=output_filename,
                                mime="audio/mpeg"
                            )

                            os.remove(temp_output_path)
                        else:
                            st.error("❌ Failed to create redacted audio")
                    else:
                        st.info("ℹ️ No sensitive content detected. Original audio unchanged.")

                        # Just copy the original file
                        with open(input_file_path, 'rb') as f:
                            original_audio_bytes = f.read()

                        st.audio(original_audio_bytes)
                        st.download_button(
                            label="💾 Download Audio (No Changes)",
                            data=original_audio_bytes,
                            file_name=uploaded_file.name,
                            mime="audio/mpeg"
                        )

                except Exception as e:
                    st.error(f"❌ Error processing audio: {e}")
                    st.error(traceback.format_exc())

    elif processing_mode == "Video + Audio Redaction":
        if st.button("🔧 Redact Video + Audio", type="primary"):
            if not word_timestamps:
                st.warning("⚠️ Word timestamps are required for audio sensitive content detection. Please enable them in the sidebar.")
                st.stop()

            with st.spinner("Processing video and audio redaction... This may take a while."):
                try:
                    # Step 1: Extract audio from video
                    st.info("Step 1/5: Extracting audio from video...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_audio:
                        temp_audio_path = tmp_audio.name

                    if not extract_audio_from_video(input_file_path, temp_audio_path):
                        st.error("❌ Failed to extract audio from video")
                        st.stop()

                    # Step 2: Redact video
                    st.info("Step 2/5: Redacting video...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                        temp_video_path = tmp_video.name

                    # Prepare object detection parameters
                    object_detector_param = None
                    object_classes_param = None
                    object_confidence_param = 0.5
                    blur_strength_param = 15
                    
                    if enable_object_detection and object_detector:
                        object_detector_param = object_detector
                        object_classes_param = object_classes if 'object_classes' in locals() else None
                        object_confidence_param = object_confidence if 'object_confidence' in locals() else 0.5
                        blur_strength_param = blur_strength if 'blur_strength' in locals() else 15

                    redact_video(
                        input_file_path, 
                        temp_video_path, 
                        analyzer,
                        object_detector=object_detector_param,
                        object_classes=object_classes_param,
                        object_confidence=object_confidence_param,
                        blur_strength=blur_strength_param,
                        enable_text_redaction=enable_text_redaction,
                        enable_object_redaction=enable_object_detection
                    )

                    # Step 3: Transcribe audio
                    st.info("Step 3/5: Transcribing audio...")
                    segments, info = transcribe_audio(temp_audio_path, model_size, beam_size, word_timestamps)

                    if not segments or not info:
                        st.error("❌ Audio transcription failed")
                        st.stop()

                    # Step 4: Detect sensitive content and redact audio
                    st.info("Step 4/5: Detecting and redacting sensitive audio content...")
                    transcript_content = format_transcript_output(segments, info, uploaded_file.name, model_size)
                    sensitive_timestamps = detect_sensitive_content(transcript_content)

                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_redacted_audio:
                        temp_redacted_audio_path = tmp_redacted_audio.name

                    success = redact_audio_segments(temp_audio_path, sensitive_timestamps, temp_redacted_audio_path)

                    if not success:
                        st.error("❌ Failed to redact audio")
                        st.stop()

                    # Step 5: Combine redacted video with redacted audio
                    st.info("Step 5/5: Combining redacted video and audio...")
                    output_filename = f"{Path(uploaded_file.name).stem}_redacted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_final:
                        temp_final_path = tmp_final.name

                    if combine_video_audio(temp_video_path, temp_redacted_audio_path, temp_final_path):
                        with open(temp_final_path, 'rb') as f:
                            final_video_bytes = f.read()

                        st.success("✅ Video + Audio redaction complete!")

                        # Show redaction stats
                        if sensitive_timestamps:
                            total_redacted_time = sum(end - start for start, end in sensitive_timestamps)
                            original_duration = info.duration
                            redaction_percentage = (total_redacted_time / original_duration * 100) if original_duration > 0 else 0

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Audio Segments Redacted", len(sensitive_timestamps))
                                st.metric("Audio Time Redacted", f"{total_redacted_time:.2f}s")
                            with col2:
                                st.metric("Original Duration", f"{original_duration:.2f}s")
                                st.metric("Audio Redaction %", f"{redaction_percentage:.1f}%")
                        else:
                            st.info("ℹ️ No sensitive audio content detected.")

                        # Download button
                        st.download_button(
                            label="💾 Download Redacted Video",
                            data=final_video_bytes,
                            file_name=output_filename,
                            mime="video/mp4"
                        )

                        # Clean up temp files
                        for temp_file in [temp_audio_path, temp_video_path, temp_redacted_audio_path, temp_final_path]:
                            try:
                                os.remove(temp_file)
                            except:
                                pass
                    else:
                        st.error("❌ Failed to combine video and audio")

                except Exception as e:
                    st.error(f"❌ Error processing video + audio: {e}")
                    st.error(traceback.format_exc())

    elif processing_mode == "Audio Transcription Only":
        if st.button("🎤 Transcribe Audio", type="primary"):
            with st.spinner("Transcribing audio... This may take a while."):
                try:
                    # Transcribe audio
                    segments, info = transcribe_audio(input_file_path, model_size, beam_size, word_timestamps)

                    if segments and info:
                        # Format transcript
                        transcript_content = format_transcript_output(segments, info, uploaded_file.name, model_size)

                        st.success("✅ Transcription completed!")

                        # Show transcription stats
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Language", info.language)
                            st.metric("Confidence", f"{info.language_probability:.2f}")
                        with col2:
                            st.metric("Duration", f"{info.duration:.2f}s")
                            st.metric("Segments", len(list(segments)))

                        # Display transcript preview
                        st.subheader("📄 Transcript Preview")
                        st.text_area("Transcript", transcript_content, height=400)

                        # Download transcript
                        output_filename = f"{Path(uploaded_file.name).stem}_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        st.download_button(
                            label="💾 Download Transcript",
                            data=transcript_content,
                            file_name=output_filename,
                            mime="text/plain"
                        )

                        # Optional: Detect sensitive content for information
                        if word_timestamps:
                            if st.button("🔍 Analyze for Sensitive Content (Preview Only)"):
                                with st.spinner("Analyzing transcript..."):
                                    sensitive_timestamps = detect_sensitive_content(transcript_content)

                                    if sensitive_timestamps:
                                        st.warning(f"⚠️ Found {len(sensitive_timestamps)} potentially sensitive segments:")
                                        for i, (start, end) in enumerate(sensitive_timestamps):
                                            st.write(f"**Segment {i+1}:** {start:.2f}s - {end:.2f}s ({end-start:.2f}s duration)")
                                        st.info("💡 Switch to 'Audio Redaction Only' mode to create a redacted version.")
                                    else:
                                        st.success("✅ No sensitive content detected in transcript.")
                    else:
                        st.error("❌ Transcription failed. Please try again.")

                except Exception as e:
                    st.error(f"❌ Error transcribing audio: {e}")
                    st.error(traceback.format_exc())

    # Clean up input file
    try:
        os.remove(input_file_path)
    except:
        pass

# -- FOOTER --
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, OpenCV, Presidio, Faster-Whisper, Google Gemini AI, YOLOv8, and FFmpeg")
