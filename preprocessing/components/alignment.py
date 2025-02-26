import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging
from pathlib import Path
import torch
import torchvision.transforms as T

class GaitAlignment:
    """Alignment and normalization for gait sequences"""
    
    def __init__(self,
                 target_size: Tuple[int, int] = (128, 64),
                 pad_ratio: float = 0.1):
        """
        Initialize alignment processor
        
        Args:
            target_size: Output size (height, width)
            pad_ratio: Padding ratio for height
        """
        self.target_size = target_size
        self.pad_ratio = pad_ratio
        
        # Standard transforms
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def segment_body_regions(self, 
                           person_roi: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Segment person ROI into body regions
        
        Args:
            person_roi: Aligned person image
            
        Returns:
            Dict containing:
            - 'upper': Upper body region
            - 'lower': Lower body region
            - 'full': Full body
        """
        height, width = person_roi.shape[:2]
        
        # Approximate body regions
        upper_height = int(height * 0.4)  # Upper 40%
        lower_start = int(height * 0.3)   # Overlap in middle
        
        regions = {
            'upper': person_roi[:upper_height],
            'lower': person_roi[lower_start:],
            'full': person_roi
        }
        
        return regions
    
    def align_person(self,
                    frame: np.ndarray,
                    bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract and align person ROI
        
        Args:
            frame: Input frame
            bbox: Person bounding box (x1, y1, x2, y2)
            
        Returns:
            Aligned person ROI or None if alignment fails
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Add padding
            height = y2 - y1
            pad = int(height * self.pad_ratio)
            y1 = max(0, y1 - pad)
            y2 = min(frame.shape[0], y2 + pad)
            
            # Extract ROI
            person = frame[y1:y2, x1:x2]
            
            # Resize maintaining aspect ratio
            target_h, target_w = self.target_size
            h, w = person.shape[:2]
            scale = min(target_w/w, target_h/h)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            person = cv2.resize(person, (new_w, new_h))
            
            # Center pad to target size
            pad_h = (target_h - new_h) // 2
            pad_w = (target_w - new_w) // 2
            
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = person
            
            return padded
            
        except Exception as e:
            logging.error(f"Error in person alignment: {str(e)}")
            return None
    
    def process_sequence(self,
                        frames: List[np.ndarray],
                        bboxes: List[Tuple[int, int, int, int]]) -> Dict[str, np.ndarray]:
        """
        Process full sequence of frames
        
        Args:
            frames: List of input frames
            bboxes: List of bounding boxes
            
        Returns:
            Dict containing processed sequences for each body region
        """
        if len(frames) != len(bboxes):
            raise ValueError("Number of frames and bboxes must match")
            
        # Initialize lists for each region
        processed_regions: Dict[str, List[np.ndarray]] = {
            'upper': [],
            'lower': [],
            'full': []
        }
        
        # Process each frame
        for frame, bbox in zip(frames, bboxes):
            # Align person
            aligned = self.align_person(frame, bbox)
            if aligned is None:
                continue
                
            # Segment body regions
            regions = self.segment_body_regions(aligned)
            
            # Add to sequences
            for region_name, region_img in regions.items():
                processed_regions[region_name].append(region_img)
        
        # Convert lists to numpy arrays
        result: Dict[str, np.ndarray] = {}
        for region_name, frames_list in processed_regions.items():
            if frames_list:
                result[region_name] = np.stack(frames_list)
            else:
                result[region_name] = np.zeros((0, *self.target_size, 3), dtype=np.uint8)
        
        return result
    
    def normalize_sequence(self, sequence: np.ndarray) -> torch.Tensor:
        """Normalize sequence for model input"""
        # Convert to torch tensor
        if sequence.dtype == np.uint8:
            sequence = sequence.astype(np.float32) / 255.0
            
        tensor = torch.from_numpy(sequence)
        
        # Add batch dimension if needed
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
            
        # Channels first
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2)
            
        # Normalize
        tensor = self.normalize(tensor)
        
        return tensor

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test alignment
    aligner = GaitAlignment()
    
    # Load test image
    test_img = cv2.imread("test.jpg")
    if test_img is not None:
        # Test bbox
        bbox = (100, 50, 300, 450)  # Example bbox
        
        # Align person
        aligned = aligner.align_person(test_img, bbox)
        if aligned is not None:
            # Get body regions
            regions = aligner.segment_body_regions(aligned)
            
            # Save results
            cv2.imwrite("aligned_full.jpg", regions['full'])
            cv2.imwrite("aligned_upper.jpg", regions['upper'])
            cv2.imwrite("aligned_lower.jpg", regions['lower'])
