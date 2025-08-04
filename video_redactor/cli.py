import argparse
from video_redactor.recognizer import build_analyzer
from video_redactor.redactor import redact_video
from video_redactor.object_detector import ObjectDetector


def main():
    parser = argparse.ArgumentParser(description="Video Redactor using OCR + PII Detection + YOLOv8 Object Detection")
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("output", help="Path to output (redacted) video")
    
    # Text redaction options
    parser.add_argument("--no-text-redaction", action="store_true", 
                       help="Disable text redaction (OCR + PII)")
    
    # Object detection options
    parser.add_argument("--enable-object-detection", action="store_true",
                       help="Enable YOLOv8 object detection and blurring")
    parser.add_argument("--object-classes", nargs="+", 
                       help="Object classes to blur (e.g., 'person' 'car' 'phone')")
    parser.add_argument("--object-confidence", type=float, default=0.5,
                       help="Confidence threshold for object detection (default: 0.5)")
    parser.add_argument("--blur-strength", type=int, default=15,
                       help="Blur strength for object blurring (default: 15)")
    parser.add_argument("--list-classes", action="store_true",
                       help="List all available object classes and exit")
    
    args = parser.parse_args()
    
    # List available classes if requested
    if args.list_classes:
        detector = ObjectDetector()
        classes = detector.get_available_classes()
        print("Available object classes:")
        for i, class_name in enumerate(classes, 1):
            print(f"{i:3d}. {class_name}")
        return
    
    # Initialize text analyzer
    analyzer = build_analyzer()
    
    # Initialize object detector if needed
    object_detector = None
    if args.enable_object_detection:
        object_detector = ObjectDetector()
        
        # Validate object classes if specified
        if args.object_classes:
            available_classes = object_detector.get_available_classes()
            invalid_classes = [cls for cls in args.object_classes if cls not in available_classes]
            if invalid_classes:
                print(f"Warning: Invalid object classes: {invalid_classes}")
                print("Available classes:", available_classes)
                return
    
    # Perform video redaction
    redact_video(
        args.input, 
        args.output, 
        analyzer,
        object_detector=object_detector,
        object_classes=args.object_classes,
        object_confidence=args.object_confidence,
        blur_strength=args.blur_strength,
        enable_text_redaction=not args.no_text_redaction,
        enable_object_redaction=args.enable_object_detection
    )
    
    print(f"✅ Video redaction completed: {args.output}")


if __name__ == "__main__":
    main()
