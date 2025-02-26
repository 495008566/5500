import torch
import cv2
import numpy as np
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import yaml
import time

from models.gait_model import GaitModel
from preprocessing.components.person_detector import PersonDetector
from preprocessing.components.tracker import GaitTracker
from preprocessing.components.alignment import GaitAlignment
from utils.visualization import visualize_results

def parse_args():
    parser = argparse.ArgumentParser(description="Gait Recognition Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Model configuration")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = GaitModel(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    # Initialize preprocessing components
    detector = PersonDetector(model_size='s', confidence=0.5, device=args.device)
    tracker = GaitTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    aligner = GaitAlignment(target_size=(128, 64), pad_ratio=0.1)
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {args.video}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize output video if visualization is enabled
    if args.visualize:
        output_video_path = output_dir / "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    # Process video
    frame_count = 0
    track_data = {}
    
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect persons
            detections = detector.process_frame(frame)
            
            # Track persons
            tracks = tracker.update(frame, detections)
            
            # Process each track
            for track in tracks:
                track_id = track['track_id']
                bbox = track['bbox']
                
                # Check if track has enough frames for gait cycle
                cycle = tracker.detect_gait_cycle(track_id)
                if cycle:
                    start_frame, end_frame = cycle
                    
                    # Get track history
                    track_info = tracker.get_track_info(track_id)
                    if track_info and track_id not in track_data:
                        # Extract frames and bboxes for the gait cycle
                        cycle_frames = []
                        cycle_bboxes = []
                        
                        for i, frame_idx in enumerate(track_info['frames']):
                            if start_frame <= frame_idx <= end_frame:
                                cycle_frames.append(frame)
                                cycle_bboxes.append(track_info['bboxes'][i])
                        
                        # Process sequence
                        if cycle_frames:
                            # Align and segment body regions
                            regions = aligner.process_sequence(cycle_frames, cycle_bboxes)
                            
                            # Normalize sequences
                            normalized_regions = {}
                            for region_name, sequence in regions.items():
                                if sequence.size > 0:
                                    normalized_regions[region_name] = aligner.normalize_sequence(sequence)
                            
                            # Perform gait recognition
                            if normalized_regions:
                                # Prepare input for model
                                inputs = {
                                    'upper_body': normalized_regions.get('upper').to(device),
                                    'lower_body': normalized_regions.get('lower').to(device),
                                    'full_body': normalized_regions.get('full').to(device),
                                    'view_angle': torch.tensor([90]).to(device)  # Assume 90 degrees if unknown
                                }
                                
                                # Forward pass
                                outputs = model(inputs)
                                
                                # Get prediction
                                pred_id = outputs['logits'].argmax(dim=1).item()
                                confidence = torch.softmax(outputs['logits'], dim=1).max().item()
                                
                                # Store results
                                track_data[track_id] = {
                                    'identity': pred_id,
                                    'confidence': confidence,
                                    'bbox': bbox
                                }
            
            # Visualize results
            if args.visualize:
                # Draw bounding boxes and IDs
                for track in tracks:
                    track_id = track['track_id']
                    x1, y1, x2, y2 = track['bbox']
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw track ID
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw recognition result if available
                    if track_id in track_data:
                        identity = track_data[track_id]['identity']
                        confidence = track_data[track_id]['confidence']
                        cv2.putText(frame, f"Person: {identity} ({confidence:.2f})",
                                   (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Write frame to output video
                out.write(frame)
                
                # Display progress
                if frame_count % 10 == 0:
                    logging.info(f"Processed {frame_count} frames")
    
    # Release resources
    cap.release()
    if args.visualize:
        out.release()
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        import json
        json.dump(track_data, f, indent=4)
    
    logging.info(f"Results saved to {results_file}")
    if args.visualize:
        logging.info(f"Output video saved to {output_video_path}")

if __name__ == "__main__":
    main()
