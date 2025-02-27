#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models
import logging

class GaitModel(nn.Module):
    """
    Simplified gait recognition model for handling sequences
    """
    
    def __init__(self, config):
        super(GaitModel, self).__init__()
        self.config = config
        
        # Set up backbone network
        if config.backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            feature_dim = 512
        elif config.backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            feature_dim = 512
        elif config.backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            feature_dim = 2048
        elif config.backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=True)
            feature_dim = 2048
        elif config.backbone == 'resnet152':
            self.backbone = models.resnet152(pretrained=True)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {config.backbone}")
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimension reduction
        self.feature_reduction = nn.Sequential(
            nn.Linear(feature_dim, config.feature_dim),
            nn.BatchNorm1d(config.feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Fully connected layers
        self.fc = nn.Linear(config.feature_dim, config.num_classes)
        
        logging.info(f"Created GaitModel with backbone {config.backbone}")
    
    def forward(self, x):
        """
        Forward pass of the model
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
        
        Returns:
            features: Feature tensor of shape (batch_size, feature_dim)
            logits: Logits tensor of shape (batch_size, num_classes)
        """
        batch_size, sequence_length, channels, height, width = x.shape
        
        # Process each frame in the sequence
        frame_features = []
        for i in range(sequence_length):
            frame = x[:, i, :, :, :]  # (batch_size, channels, height, width)
            
            # Extract features using backbone
            features = self.backbone(frame)
            features = features.view(batch_size, -1)
            
            # Apply feature reduction
            features = self.feature_reduction(features)
            
            frame_features.append(features)
        
        # Aggregate features across frames (temporal pooling)
        features = torch.stack(frame_features, dim=1)  # (batch_size, sequence_length, feature_dim)
        features = torch.mean(features, dim=1)  # (batch_size, feature_dim)
        
        # Classification
        logits = self.fc(features)
        
        return features, logits
