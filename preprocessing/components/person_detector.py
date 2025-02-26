import torch
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
from ultralytics import YOLO

class PersonDetector:
    """Person detection using YOLOv5 for gait recognition preprocessing"""
    
    def __init__(self, 
                 model_size: str = 's',
                 confidence: float = 0.5,
                 device: Optional[str] = None):
        """
        Initialize YOLOv5 person detector
        
        Args:
            model_size: YOLOv5 model size ('n', 's', 'm', 'l', 'x')
            confidence: Detection confidence threshold
            device: Device to run model on ('cpu', 'cuda', None=auto)
        """
        self.confidence = confidence
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize YOLOv5 model
        try:
            self.model = YOLO(f'yolov5{model_size}.pt')
            logging.info(f"Loaded YOLOv5{model_size} on {self.device}")
        except Exception as e:
            logging.error(f"Error loading YOLOv5: {str(e)}")
            raise
        
        # Set model parameters
        self.model.conf = confidence
        self.model.classes = [0]  # Only detect persons (class 0)
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in frame
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of detections, each containing:
            - bbox: (x1, y1, x2, y2)
            - confidence: Detection confidence
            - class_id: Always 0 (person)
        """
        try:
            # Run inference
            results = self.model(frame)
            
            # Process detections
            detections = []
            for det in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = det
                if int(cls) == 0:  # Person class
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(conf),
                        'class_id': 0
                    })
            
            return detections
            
        except Exception as e:
            logging.error(f"Error during detection: {str(e)}")
            return []
    
    def extract_person(self, 
                      frame: np.ndarray,
                      bbox: Tuple[int, int, int, int],
                      target_size: Tuple[int, int] = (128, 64)) -> Optional[np.ndarray]:
        """
        Extract and normalize person ROI from frame
        
        Args:
            frame: Input image
            bbox: Bounding box (x1, y1, x2, y2)
            target_size: Output size (height, width)
            
        Returns:
            Normalized person ROI or None if extraction fails
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Add margin to bbox
            height = y2 - y1
            margin = int(height * 0.1)  # 10% margin
            y1 = max(0, y1 - margin)
            y2 = min(frame.shape[0], y2 + margin)
            
            # Extract ROI
            person = frame[y1:y2, x1:x2]
            
            # Resize to target size
            person = cv2.resize(person, (target_size[1], target_size[0]))
            
            return person
            
        except Exception as e:
            logging.error(f"Error extracting person: {str(e)}")
            return None
    
    def process_frame(self,
                     frame: np.ndarray,
                     target_size: Tuple[int, int] = (128, 64)) -> List[Dict]:
        """
        Detect and extract all persons from frame
        
        Args:
            frame: Input image
            target_size: Output size for extracted persons
            
        Returns:
            List of processed detections with normalized ROIs
        """
        # Detect persons
        detections = self.detect(frame)
        
        # Extract ROIs
        processed = []
        for det in detections:
            roi = self.extract_person(frame, det['bbox'], target_size)
            if roi is not None:
                det['roi'] = roi
                processed.append(det)
        
        return processed

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize detector
    detector = PersonDetector(model_size='s', confidence=0.5)
    
    # Test on sample image
    test_img = cv2.imread("test.jpg")
    if test_img is not None:
        detections = detector.process_frame(test_img)
        logging.info(f"Detected {len(detections)} persons")
        
        # Visualize detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(test_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        cv2.imwrite("test_output.jpg", test_img)
