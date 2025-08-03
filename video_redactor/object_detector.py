import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import torch


class ObjectDetector:
    """
    Object detection using YOLOv8 nano model
    """
    
    def __init__(self, model_name: str = "yolov8n.pt", device: Optional[str] = None):
        """
        Initialize the object detector
        
        Args:
            model_name: YOLOv8 model name (default: yolov8n.pt for nano)
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.model = YOLO(model_name)
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"✅ YOLOv8 model loaded on {self.device}")
        
        # Get class names
        self.class_names = self.model.names
        
    def get_available_classes(self) -> List[str]:
        """Get list of available object classes"""
        return list(self.class_names.values())
    
    def detect_objects(self, frame: np.ndarray, 
                      classes_to_detect: Optional[List[str]] = None,
                      confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects in a frame
        
        Args:
            frame: Input frame as numpy array
            classes_to_detect: List of class names to detect (None for all)
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of dictionaries with detection info
        """
        # Convert class names to indices if specified
        class_indices = None
        if classes_to_detect:
            class_indices = []
            for class_name in classes_to_detect:
                if class_name in self.class_names.values():
                    # Find the index of this class name
                    for idx, name in self.class_names.items():
                        if name == class_name:
                            class_indices.append(idx)
                            break
        
        # Run inference
        results = self.model(frame, conf=confidence_threshold, classes=class_indices, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name
                    })
        
        return detections
    
    def blur_objects(self, frame: np.ndarray, 
                    detections: List[Dict],
                    blur_strength: int = 15) -> np.ndarray:
        """
        Blur detected objects in the frame
        
        Args:
            frame: Input frame
            detections: List of object detections
            blur_strength: Strength of the blur effect
            
        Returns:
            Frame with blurred objects
        """
        result_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            
            # Extract the region to blur
            region = result_frame[y1:y2, x1:x2]
            
            if region.size > 0:  # Check if region is not empty
                # Apply Gaussian blur
                blurred_region = cv2.GaussianBlur(region, (blur_strength, blur_strength), 0)
                
                # Replace the region with blurred version
                result_frame[y1:y2, x1:x2] = blurred_region
        
        return result_frame 