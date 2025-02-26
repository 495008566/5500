#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import logging
from pathlib import Path

class SimpleDataset(Dataset):
    """A simple dataset implementation for testing."""
    
    def __init__(self, dataset_name, split='train', num_samples=100, num_classes=10):
        self.dataset_name = dataset_name
        self.split = split
        self.num_samples = num_samples
        self.num_classes = num_classes
        
        # Create synthetic data for testing
        self.data = []
        self.labels = []
        
        for i in range(num_samples):
            # Create a random sequence (64x128 silhouette with 30 frames)
            # Using 3 channels to match ResNet input requirements
            sequence = np.random.rand(30, 3, 64, 128).astype(np.float32)
            label = i % num_classes
            
            self.data.append(sequence)
            self.labels.append(label)
        
        logging.info(f"Created {len(self.data)} samples for {dataset_name} {split} split")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        label = self.labels[idx]
        
        # Convert to tensor
        sequence = torch.tensor(sequence, dtype=torch.float32)
        
        # For training, we'll use a single frame (the middle one) to simplify
        # In a real implementation, we would process the entire sequence
        middle_frame_idx = sequence.shape[0] // 2
        frame = sequence[middle_frame_idx]
        
        return frame, label
