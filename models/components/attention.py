#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEModule(nn.Module):
    """Squeeze-and-Excitation (SE) block implementation."""
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel attention
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        # Spatial attention
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        channel_att = self.channel_gate(x)
        channel_att = self.sigmoid(channel_att)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_gate(spatial_input)
        x = x * spatial_att
        
        return x

class ViewTransformationNetwork(nn.Module):
    """Network for transforming features across different view angles."""
    def __init__(self, feature_dim, view_angles):
        super(ViewTransformationNetwork, self).__init__()
        self.feature_dim = feature_dim
        self.view_angles = view_angles
        self.num_views = len(view_angles)
        
        # Create transformation matrices for each view pair
        self.transforms = nn.ModuleDict()
        for i, src_view in enumerate(view_angles):
            for j, tgt_view in enumerate(view_angles):
                if i != j:
                    key = f"{src_view}_{tgt_view}"
                    self.transforms[key] = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, features, src_view, tgt_view=None):
        """
        Transform features from source view to target view.
        If tgt_view is None, transform to all views.
        """
        if tgt_view is None:
            # Transform to all views
            transformed_features = []
            for view in self.view_angles:
                if view != src_view:
                    key = f"{src_view}_{view}"
                    transformed = self.transforms[key](features)
                    transformed_features.append(transformed)
                else:
                    transformed_features.append(features)
            return torch.stack(transformed_features, dim=1)
        else:
            # Transform to specific target view
            if src_view == tgt_view:
                return features
            key = f"{src_view}_{tgt_view}"
            return self.transforms[key](features)

class DynamicWeightModule(nn.Module):
    """Module for dynamically weighting features from different body parts."""
    def __init__(self, feature_dim):
        super(DynamicWeightModule, self).__init__()
        self.feature_dim = feature_dim
        
        # Weight prediction network
        self.weight_net = nn.Sequential(
            nn.Linear(feature_dim * 3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )
    
    def forward(self, upper_features, lower_features, full_features):
        """
        Dynamically weight features from upper body, lower body, and full body.
        """
        # Concatenate features
        concat_features = torch.cat([
            upper_features, lower_features, full_features
        ], dim=1)
        
        # Predict weights
        weights = self.weight_net(concat_features)
        
        # Apply weights
        weighted_features = (
            weights[:, 0:1] * upper_features +
            weights[:, 1:2] * lower_features +
            weights[:, 2:3] * full_features
        )
        
        return weighted_features, weights
