import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from .person_detector import PersonDetector
from .tracker import GaitTracker
from .alignment import GaitAlignment

class GaitPreprocessor:
    """Main preprocessing pipeline for gait recognition"""
    
    def __init__(self,
                 detector_model: str = 's',
                 target_size: Tuple[int, int] = (128, 64)):
        """
        Initialize preprocessing pipeline
        
        Args:
            detector_model: YOLOv5 model size ('n', 's', 'm', 'l', 'x')
            target_size: Output size for aligned persons
        """
        self.detector = PersonDetector(model_size=detector_model)
        self.tracker = GaitTracker()
        self.aligner = GaitAlignment(target_size=target_size)
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process single frame
        
        Returns:
            Dict containing:
            - detections: List of detected persons
            - tracks: List of tracked persons
            - aligned: Dict of aligned ROIs by track_id
        """
        # Detect persons
        detections = self.detector.process_frame(frame)
        
        # Update tracker
        tracks = self.tracker.update(frame, detections)
        
        # Align detected persons
        aligned_persons = {}
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            
            # Align person
            aligned = self.aligner.align_person(frame, bbox)
            if aligned is not None:
                # Get body regions
                regions = self.aligner.segment_body_regions(aligned)
                aligned_persons[track_id] = regions
        
        return {
            'detections': detections,
            'tracks': tracks,
            'aligned': aligned_persons
        }
    
    def process_video(self,
                     video_path: str,
                     output_dir: Optional[str] = None) -> Dict[int, Dict]:
        """
        Process video file
        
        Args:
            video_path: Path to input video
            output_dir: Optional directory to save visualizations
            
        Returns:
            Dict mapping track_id to processed sequences
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Setup output directory
        output_path: Optional[Path] = None
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Track sequences
        sequences: Dict[int, Dict] = {}
        frame_idx = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                results = self.process_frame(frame)
                
                # Update sequences
                for track in results['tracks']:
                    track_id = track['track_id']
                    if track_id not in sequences:
                        sequences[track_id] = {
                            'frames': [],
                            'bboxes': [],
                            'regions': {
                                'upper': [],
                                'lower': [],
                                'full': []
                            }
                        }
                    
                    # Add frame data
                    sequences[track_id]['frames'].append(frame_idx)
                    sequences[track_id]['bboxes'].append(track['bbox'])
                    
                    # Add aligned regions if available
                    if track_id in results['aligned']:
                        regions = results['aligned'][track_id]
                        for region_name, region_img in regions.items():
                            sequences[track_id]['regions'][region_name].append(region_img)
                
                # Save visualization if requested
                if output_path is not None:
                    viz_path = str(output_path / f"frame_{frame_idx:06d}.jpg")
                    self._visualize_frame(frame, results, viz_path)
                
                frame_idx += 1
                
        finally:
            cap.release()
        
        # Convert sequences to numpy arrays
        processed_sequences = {}
        for track_id, seq_data in sequences.items():
            # Check if sequence contains a full gait cycle
            cycle = self.tracker.detect_gait_cycle(track_id)
            if cycle is None:
                continue
                
            start_idx, end_idx = cycle
            
            # Process regions
            processed_regions = {}
            for region_name, region_frames in seq_data['regions'].items():
                if region_frames:
                    # Convert to numpy array
                    region_seq = np.stack(region_frames)
                    # Extract cycle frames
                    cycle_seq = region_seq[start_idx:end_idx+1]
                    processed_regions[region_name] = cycle_seq
            
            if processed_regions:
                processed_sequences[track_id] = {
                    'cycle': (start_idx, end_idx),
                    'regions': processed_regions
                }
        
        return processed_sequences
    
    def _visualize_frame(self,
                        frame: np.ndarray,
                        results: Dict,
                        output_path: str) -> None:
        """
        Save visualization of processed frame
        
        Args:
            frame: Input frame to visualize
            results: Detection and tracking results
            output_path: Path to save visualization
        """
        viz = frame.copy()
        
        # Draw detections and tracks
        for track in results['tracks']:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['track_id']
            
            # Draw bounding box
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw track ID
            cv2.putText(viz, f"ID: {track_id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Check for gait cycle
            cycle = self.tracker.detect_gait_cycle(track_id)
            if cycle:
                start, end = cycle
                cv2.putText(viz, f"Cycle: {start}-{end}", (x1, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.imwrite(str(output_path), viz)

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize preprocessor
    processor = GaitPreprocessor()
    
    # Test on video
    video_path = "test.mp4"
    output_dir = "output"
    
    try:
        sequences = processor.process_video(video_path, output_dir)
        logging.info(f"Processed {len(sequences)} valid gait sequences")
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
