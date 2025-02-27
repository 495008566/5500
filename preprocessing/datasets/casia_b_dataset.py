#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import logging
from pathlib import Path
import random
from torchvision import transforms
from PIL import Image

class CasiaBDataset(Dataset):
    """
    Dataset class for CASIA-B gait recognition dataset
    
    CASIA-B is a multi-view gait dataset with 124 subjects, 11 views, and 3 walking conditions
    """
    
    def __init__(self, root_dir='data/casia_b', split='train', transform=None, 
                 img_size=(64, 128), sequence_length=30, view_angles=None):
        """
        Initialize the CASIA-B dataset
        
        Args:
            root_dir: Root directory of the dataset
            split: 'train' or 'test' split
            transform: Optional transform to be applied to the images
            img_size: Size of the images (height, width)
            sequence_length: Number of frames to use in each sequence
            view_angles: List of view angles to include (if None, include all)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.sequence_length = sequence_length
        
        # Define view angles (0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180)
        self.all_view_angles = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
        self.view_angles = view_angles if view_angles is not None else self.all_view_angles
        
        # Define walking conditions (normal, carrying bag, wearing coat)
        self.conditions = ['nm', 'bg', 'cl']
        
        # Create default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Load dataset
        self.data = []
        self.labels = []
        self.view_angles_list = []
        self.conditions_list = []
        
        # Split subjects into train and test
        all_subjects = list(range(1, 125))  # 124 subjects in total
        
        if split == 'train':
            # Use 80% of subjects for training
            self.subjects = all_subjects[:100]
        else:
            # Use 20% of subjects for testing
            self.subjects = all_subjects[100:]
        
        self._load_data()
        
        logging.info(f"Loaded {len(self.data)} sequences for {split} split")
    
    def _load_data(self):
        """Load data from the dataset directory"""
        # Check if the dataset directory exists
        if not self.root_dir.exists():
            logging.warning(f"Dataset directory {self.root_dir} does not exist")
            return
        
        # Iterate through subjects
        for subject_id in self.subjects:
            subject_dir = self.root_dir / f"{subject_id:03d}"
            
            if not subject_dir.exists():
                continue
            
            # Iterate through conditions
            for condition in self.conditions:
                # Find all condition directories for this subject
                condition_dirs = list(subject_dir.glob(f"{condition}-*"))
                
                for condition_dir in condition_dirs:
                    # Extract sequence number from directory name
                    sequence_num = int(condition_dir.name.split('-')[1]) if len(condition_dir.name.split('-')) > 1 else 0
                    
                    # Iterate through view angles
                    for view_angle in self.view_angles:
                        view_dir = condition_dir / f"{view_angle:03d}"
                        
                        if not view_dir.exists():
                            continue
                        
                        # Get all frame files in this view directory
                        frame_files = sorted(list(view_dir.glob("*.png")))
                        
                        if len(frame_files) == 0:
                            continue
                        
                        # If we have more frames than sequence_length, sample frames
                        if len(frame_files) > self.sequence_length:
                            # Sample frames evenly
                            indices = np.linspace(0, len(frame_files) - 1, self.sequence_length, dtype=int)
                            frame_files = [frame_files[i] for i in indices]
                        elif len(frame_files) < self.sequence_length:
                            # Pad with duplicates of the last frame
                            frame_files = frame_files + [frame_files[-1]] * (self.sequence_length - len(frame_files))
                        
                        # Add sequence to dataset
                        self.data.append(frame_files)
                        self.labels.append(subject_id - 1)  # Convert to 0-indexed
                        self.view_angles_list.append(view_angle)
                        self.conditions_list.append(condition)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        frames = self.data[idx]
        label = self.labels[idx]
        
        # Process frames
        processed_frames = []
        for frame in frames:
            # Load image from file
            img = cv2.imread(str(frame))
            if img is None:
                # If image loading fails, create a blank image
                img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
            
            # Apply transform
            if self.transform:
                img = self.transform(Image.fromarray(img))
            else:
                img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            
            processed_frames.append(img)
        
        # Stack frames to create a sequence
        sequence_tensor = torch.stack(processed_frames)
        
        return sequence_tensor, label
