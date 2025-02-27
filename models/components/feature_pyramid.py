#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature extraction
    """
    
    def __init__(self, in_channels, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        ])
        
        # Top-down connections
        self.top_down_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        ])
        
        # Output convolutions
        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        """
        Forward pass of the feature pyramid network
        
        Args:
            x: Input feature map
        
        Returns:
            Feature map with multi-scale features
        """
        # Lateral connection
        lateral = self.lateral_convs[0](x)
        
        # Top-down connection
        out = self.top_down_convs[0](lateral)
        
        # Output convolution
        out = self.output_conv(out)
        
        return out

class GaitFeatureExtractor(nn.Module):
    """
    Feature extractor for gait recognition with multi-scale feature fusion
    """
    
    def __init__(self, in_channels_list, out_channels):
        super(GaitFeatureExtractor, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # Top-down connections
        self.top_down_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(len(in_channels_list) - 1)
        ])
        
        # Output convolution
        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final fully connected layer
        self.fc = nn.Linear(out_channels, out_channels)
    
    def forward(self, features_list):
        """
        Forward pass of the feature extractor
        
        Args:
            features_list: List of feature maps from different stages of the backbone
        
        Returns:
            Feature vector for gait recognition
        """
        # Apply lateral connections
        laterals = [
            conv(features)
            for features, conv in zip(features_list, self.lateral_convs)
        ]
        
        # Apply top-down connections
        out = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample
            out = F.interpolate(out, size=laterals[i].shape[2:], mode='nearest')
            # Add lateral connection
            out = out + laterals[i]
            # Apply convolution
            if i > 0:
                out = self.top_down_convs[i-1](out)
        
        # Apply output convolution
        out = self.output_conv(out)
        
        # Global average pooling
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        
        # Final fully connected layer
        out = self.fc(out)
        
        return out
