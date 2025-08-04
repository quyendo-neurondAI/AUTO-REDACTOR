import cv2
import numpy as np
from doctr.io import DocumentFile
from video_redactor.ocr import ocr_model
from video_redactor.utils import find_matching_boxes, hash_cv2_image
from video_redactor.recognizer import ENTITIES_LIST
from video_redactor.object_detector import ObjectDetector
import torch 
from typing import List, Optional


def redact_frame_from_cv2(image: np.ndarray, analyzer, 
                         object_detector: Optional[ObjectDetector] = None,
                         object_classes: Optional[List[str]] = None,
                         object_confidence: float = 0.5,
                         blur_strength: int = 15,
                         enable_text_redaction: bool = True,
                         enable_object_redaction: bool = False):
    """
    Redact text and/or objects from a frame
    
    Args:
        image: Input frame
        analyzer: Text analyzer for PII detection
        object_detector: YOLOv8 object detector
        object_classes: List of object classes to blur
        object_confidence: Confidence threshold for object detection
        blur_strength: Strength of blur effect
        enable_text_redaction: Whether to perform text redaction
        enable_object_redaction: Whether to perform object redaction
    """
    result_image = image.copy()
    joined_text = ""
    entities = []
    boxes = []
    
    # Text redaction
    if enable_text_redaction:
        height, width = image.shape[:2]
        success, encoded_image = cv2.imencode('.jpg', image)
        doc = DocumentFile.from_images([encoded_image.tobytes()])
        result = ocr_model(doc)

        boxes, full_text = [], []

        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        text = word.value
                        if text.strip():
                            (x_min, y_min), (x_max, y_max) = word.geometry
                            x = int(x_min * width)
                            y = int(y_min * height)
                            w = int((x_max - x_min) * width)
                            h = int((y_max - y_min) * height)
                            boxes.append({
                                "text": text.strip('''"'()[]{}<>.,:;!?- '''), 
                                "left": x, "top": y, "width": w, "height": h
                            })
                            full_text.append(text.strip())

        joined_text = " ".join(full_text)
        entities = analyzer.analyze(text=joined_text, entities=ENTITIES_LIST, language='en')

        redacted_indices = []
        for entity in entities:
            matched_text = joined_text[entity.start:entity.end].strip()
            if len(matched_text) <= 1 and entity.entity_type not in ["BUILDINGNUM", "NAME"]:
                continue
            redacted_indices.extend(find_matching_boxes(matched_text, boxes))

        for idx in redacted_indices:
            box = boxes[idx]
            x, y, w, h = box["left"], box["top"], box["width"], box["height"]
            shrink_x, shrink_y = 0.1, 0.5
            dx, dy = int(w * shrink_x / 2), int(h * shrink_y / 2)
            x1, y1, x2, y2 = x + dx, y + dy, x + w - dx, y + h - dy
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (200, 200, 200), -1)
    
    # Object redaction
    if enable_object_redaction and object_detector is not None:
        detections = object_detector.detect_objects(
            result_image, 
            classes_to_detect=object_classes,
            confidence_threshold=object_confidence
        )
        result_image = object_detector.blur_objects(
            result_image, 
            detections, 
            blur_strength=blur_strength
        )

    return result_image, joined_text, entities, boxes


def redact_video(input_video_path, output_video_path, analyzer,
                object_detector: Optional[ObjectDetector] = None,
                object_classes: Optional[List[str]] = None,
                object_confidence: float = 0.5,
                blur_strength: int = 15,
                enable_text_redaction: bool = True,
                enable_object_redaction: bool = False):
    """
    Redact video with both text and object detection
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to output video
        analyzer: Text analyzer for PII detection
        object_detector: YOLOv8 object detector
        object_classes: List of object classes to blur
        object_confidence: Confidence threshold for object detection
        blur_strength: Strength of blur effect
        enable_text_redaction: Whether to perform text redaction
        enable_object_redaction: Whether to perform object redaction
    """
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    last_frame_hash = None
    last_redacted_image = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_hash = hash_cv2_image(frame)

        if frame_hash == last_frame_hash:
            out.write(last_redacted_image)
        else:
            redacted_frame, *_ = redact_frame_from_cv2(
                frame, 
                analyzer,
                object_detector=object_detector,
                object_classes=object_classes,
                object_confidence=object_confidence,
                blur_strength=blur_strength,
                enable_text_redaction=enable_text_redaction,
                enable_object_redaction=enable_object_redaction
            )
            out.write(redacted_frame)
            last_redacted_image = redacted_frame
            last_frame_hash = frame_hash

    cap.release()
    out.release()
