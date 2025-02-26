#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import argparse
import pickle
from pathlib import Path
import logging
from tqdm import tqdm

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def normalize_silhouette(silhouette, target_size=(64, 128)):
    """
    Normalize silhouette by resizing and centering
    
    Args:
        silhouette: Input silhouette image
        target_size: Target size (width, height)
        
    Returns:
        Normalized silhouette
    """
    # Convert to binary if not already
    if len(silhouette.shape) > 2:
        silhouette = cv2.cvtColor(silhouette, cv2.COLOR_BGR2GRAY)
    
    # Threshold to ensure binary
    _, silhouette = cv2.threshold(silhouette, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # If no contours found, return resized empty image
        return cv2.resize(silhouette, target_size)
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Check if bounding box is valid
    if w <= 0 or h <= 0:
        return cv2.resize(silhouette, target_size)
    
    # Crop to bounding box with some margin
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)
    
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(silhouette.shape[1], x + w + margin_x)
    y2 = min(silhouette.shape[0], y + h + margin_y)
    
    cropped = silhouette[y1:y2, x1:x2]
    
    # Resize to target size
    normalized = cv2.resize(cropped, target_size)
    
    return normalized

def process_casia_b(input_path, output_path, target_size=(64, 128)):
    """
    Process CASIA-B dataset
    
    Args:
        input_path: Path to input dataset
        output_path: Path to output processed dataset
        target_size: Target size for normalized silhouettes
    """
    logging.info(f"Processing CASIA-B dataset from {input_path} to {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get list of subjects
    subjects = sorted([d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))])
    
    for subject in tqdm(subjects, desc="Processing subjects"):
        subject_path = os.path.join(input_path, subject)
        
        # Get list of walking conditions
        conditions = sorted([d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))])
        
        for condition in conditions:
            condition_path = os.path.join(subject_path, condition)
            
            # Get list of view angles
            views = sorted([d for d in os.listdir(condition_path) if os.path.isdir(os.path.join(condition_path, d))])
            
            for view in views:
                view_path = os.path.join(condition_path, view)
                
                # Get list of sequence images
                images = sorted([f for f in os.listdir(view_path) if f.endswith('.png')])
                
                if not images:
                    continue
                
                # Process sequence
                sequence = []
                for img_name in images:
                    img_path = os.path.join(view_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        logging.warning(f"Failed to read image: {img_path}")
                        continue
                    
                    # Normalize silhouette
                    normalized = normalize_silhouette(img, target_size)
                    sequence.append(normalized)
                
                if sequence:
                    # Create output directory structure
                    output_subject_dir = os.path.join(output_path, subject)
                    output_condition_dir = os.path.join(output_subject_dir, condition)
                    output_view_dir = os.path.join(output_condition_dir, view)
                    os.makedirs(output_view_dir, exist_ok=True)
                    
                    # Save as pickle file
                    output_file = os.path.join(output_view_dir, 'sequence.pkl')
                    with open(output_file, 'wb') as f:
                        pickle.dump(np.array(sequence), f)
                    
                    logging.debug(f"Saved sequence to {output_file}")

def process_oumvlp(input_path, output_path, target_size=(64, 128)):
    """
    Process OUMVLP dataset
    
    Args:
        input_path: Path to input dataset
        output_path: Path to output processed dataset
        target_size: Target size for normalized silhouettes
    """
    logging.info(f"Processing OUMVLP dataset from {input_path} to {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get list of subjects
    subjects = sorted([d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))])
    
    for subject in tqdm(subjects, desc="Processing subjects"):
        subject_path = os.path.join(input_path, subject)
        
        # Get list of view angles
        views = sorted([d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))])
        
        for view in views:
            view_path = os.path.join(subject_path, view)
            
            # Get list of sequence images
            images = sorted([f for f in os.listdir(view_path) if f.endswith('.png')])
            
            if not images:
                continue
            
            # Process sequence
            sequence = []
            for img_name in images:
                img_path = os.path.join(view_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    logging.warning(f"Failed to read image: {img_path}")
                    continue
                
                # Normalize silhouette
                normalized = normalize_silhouette(img, target_size)
                sequence.append(normalized)
            
            if sequence:
                # Create output directory structure
                output_subject_dir = os.path.join(output_path, subject)
                output_view_dir = os.path.join(output_subject_dir, view)
                os.makedirs(output_view_dir, exist_ok=True)
                
                # Save as pickle file
                output_file = os.path.join(output_view_dir, 'sequence.pkl')
                with open(output_file, 'wb') as f:
                    pickle.dump(np.array(sequence), f)
                
                logging.debug(f"Saved sequence to {output_file}")

def process_gait3d(input_path, output_path, target_size=(64, 128)):
    """
    Process Gait3D dataset
    
    Args:
        input_path: Path to input dataset
        output_path: Path to output processed dataset
        target_size: Target size for normalized silhouettes
    """
    logging.info(f"Processing Gait3D dataset from {input_path} to {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get list of subjects
    subjects = sorted([d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))])
    
    for subject in tqdm(subjects, desc="Processing subjects"):
        subject_path = os.path.join(input_path, subject)
        
        # Get list of cameras
        cameras = sorted([d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))])
        
        for camera in cameras:
            camera_path = os.path.join(subject_path, camera)
            
            # Get list of sequences
            sequences = sorted([d for d in os.listdir(camera_path) if os.path.isdir(os.path.join(camera_path, d))])
            
            for seq in sequences:
                seq_path = os.path.join(camera_path, seq)
                
                # Get list of sequence images
                images = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
                
                if not images:
                    continue
                
                # Process sequence
                sequence = []
                for img_name in images:
                    img_path = os.path.join(seq_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        logging.warning(f"Failed to read image: {img_path}")
                        continue
                    
                    # Normalize silhouette
                    normalized = normalize_silhouette(img, target_size)
                    sequence.append(normalized)
                
                if sequence:
                    # Create output directory structure
                    output_subject_dir = os.path.join(output_path, subject)
                    output_camera_dir = os.path.join(output_subject_dir, camera)
                    output_seq_dir = os.path.join(output_camera_dir, seq)
                    os.makedirs(output_seq_dir, exist_ok=True)
                    
                    # Save as pickle file
                    output_file = os.path.join(output_seq_dir, 'sequence.pkl')
                    with open(output_file, 'wb') as f:
                        pickle.dump(np.array(sequence), f)
                    
                    logging.debug(f"Saved sequence to {output_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Preprocess gait datasets')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output processed dataset')
    parser.add_argument('--dataset', type=str, default='auto', choices=['auto', 'casia_b', 'oumvlp', 'gait3d'],
                       help='Dataset type')
    parser.add_argument('--target_width', type=int, default=64, help='Target width for normalized silhouettes')
    parser.add_argument('--target_height', type=int, default=128, help='Target height for normalized silhouettes')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Determine dataset type if auto
    dataset_type = args.dataset
    if dataset_type == 'auto':
        input_path = Path(args.input_path)
        if 'CASIA' in input_path.name:
            dataset_type = 'casia_b'
        elif 'OUMVLP' in input_path.name:
            dataset_type = 'oumvlp'
        elif 'Gait3D' in input_path.name:
            dataset_type = 'gait3d'
        else:
            logging.error(f"Could not determine dataset type from path: {input_path}")
            return
    
    # Process dataset
    target_size = (args.target_width, args.target_height)
    if dataset_type == 'casia_b':
        process_casia_b(args.input_path, args.output_path, target_size)
    elif dataset_type == 'oumvlp':
        process_oumvlp(args.input_path, args.output_path, target_size)
    elif dataset_type == 'gait3d':
        process_gait3d(args.input_path, args.output_path, target_size)
    else:
        logging.error(f"Unknown dataset type: {dataset_type}")
        return
    
    logging.info(f"Preprocessing completed for {dataset_type} dataset")

if __name__ == "__main__":
    main()
