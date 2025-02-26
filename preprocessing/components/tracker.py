import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

class GaitTracker:
    """Person tracking using DeepSORT for continuous gait sequence extraction"""
    
    def __init__(self,
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3):
        """
        Initialize DeepSORT tracker
        
        Args:
            max_age: Maximum frames to keep track of lost objects
            min_hits: Minimum hits to start tracking
            iou_threshold: IOU threshold for matching
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=min_hits,
            nms_max_overlap=iou_threshold,
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None,
            embedder=None,  # Use default feature extractor
            bgr=True
        )
        
        self.tracks = {}  # Track history
        self.frame_count = 0
    
    def _format_detection(self, detection: Dict) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
        """Convert detection dict to DeepSORT format"""
        bbox = detection['bbox']
        confidence = detection['confidence']
        
        # Convert to [x1, y1, w, h] format
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        detection_rect = np.array([bbox[0], bbox[1], width, height])
        
        return detection_rect, confidence, None  # No feature vector needed
    
    def update(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            frame: Current frame
            detections: List of detections from person detector
            
        Returns:
            List of tracked objects with track IDs
        """
        self.frame_count += 1
        
        # Format detections for DeepSORT
        detection_list = []
        for det in detections:
            rect, conf, feat = self._format_detection(det)
            detection_list.append((rect, conf, feat))
        
        # Update tracker
        tracks = self.tracker.update_tracks(detection_list, frame=frame)
        
        # Process tracks
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            
            # Update track history
            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    'frames': [],
                    'bboxes': [],
                    'start_frame': self.frame_count
                }
            
            self.tracks[track_id]['frames'].append(self.frame_count)
            self.tracks[track_id]['bboxes'].append(ltrb)
            
            # Create result dict
            result = {
                'track_id': track_id,
                'bbox': tuple(map(int, ltrb)),
                'age': len(self.tracks[track_id]['frames']),
                'start_frame': self.tracks[track_id]['start_frame']
            }
            results.append(result)
        
        return results
    
    def get_track_info(self, track_id: int) -> Optional[Dict]:
        """Get information about a specific track"""
        if track_id not in self.tracks:
            return None
            
        track = self.tracks[track_id]
        return {
            'frames': track['frames'],
            'bboxes': track['bboxes'],
            'start_frame': track['start_frame'],
            'duration': len(track['frames']),
            'is_active': track['frames'][-1] == self.frame_count
        }
    
    def clean_old_tracks(self, max_age: int = 300):
        """Remove old tracks to free memory"""
        current_frame = self.frame_count
        old_tracks = []
        
        for track_id, track in self.tracks.items():
            if current_frame - track['frames'][-1] > max_age:
                old_tracks.append(track_id)
        
        for track_id in old_tracks:
            del self.tracks[track_id]
    
    def detect_gait_cycle(self, track_id: int, min_frames: int = 30) -> Optional[Tuple[int, int]]:
        """
        Detect gait cycle start and end frames for a track
        
        Args:
            track_id: Track ID to analyze
            min_frames: Minimum frames for a valid gait cycle
            
        Returns:
            Tuple of (start_frame, end_frame) or None if no cycle detected
        """
        track = self.get_track_info(track_id)
        if track is None or track['duration'] < min_frames:
            return None
            
        # Simple gait cycle detection based on bounding box motion
        bboxes = np.array(track['bboxes'])
        centers = (bboxes[:, 0] + bboxes[:, 2]) / 2  # x-coordinates of centers
        
        # Find local minima in horizontal motion (when person changes direction)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(-centers)  # Negative to find minima
        
        if len(peaks) >= 2:
            # Return first complete cycle
            return track['frames'][peaks[0]], track['frames'][peaks[1]]
        
        return None

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize tracker
    tracker = GaitTracker()
    
    # Test tracking
    from person_detector import PersonDetector
    detector = PersonDetector()
    
    # Open video file or camera
    cap = cv2.VideoCapture(0)  # Use camera
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect persons
            detections = detector.process_frame(frame)
            
            # Update tracker
            tracks = tracker.update(frame, detections)
            
            # Visualize tracks
            for track in tracks:
                x1, y1, x2, y2 = track['bbox']
                track_id = track['track_id']
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw ID
                cv2.putText(frame, f"ID: {track_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Check for gait cycle
                cycle = tracker.detect_gait_cycle(track_id)
                if cycle:
                    start, end = cycle
                    cv2.putText(frame, f"Cycle: {start}-{end}", (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Show frame
            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
