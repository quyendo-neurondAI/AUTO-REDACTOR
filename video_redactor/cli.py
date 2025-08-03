import argparse
from video_redactor.recognizer import build_analyzer
from video_redactor.redactor import redact_video

def main():
    parser = argparse.ArgumentParser(description="Video Redactor using OCR + PII Detection")
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("output", help="Path to output (redacted) video")
    args = parser.parse_args()

    analyzer = build_analyzer()
    redact_video(args.input, args.output, analyzer)

if __name__ == "__main__":
    main()
