#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class SEModule(nn.Module):
    """Squeeze-and-Excitation Module."""
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CBAM(nn.Module):
    """Enhanced Convolutional Block Attention Module with improved channel and spatial attention."""
    def __init__(self, channels, reduction=8):  # Reduced ratio for more capacity
        super(CBAM, self).__init__()
        # Channel attention with both avg and max pooling paths
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for channel attention
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),  # Added batch normalization
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # Spatial attention with improved convolution
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=7, stride=1, padding=3, bias=False),  # Increased channels
            nn.BatchNorm2d(8),  # Added batch normalization
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention with shared MLP
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_gate(spatial)
        
        return x * spatial_att

class ViewTransformationNetwork(nn.Module):
    """Enhanced network for transforming features across different view angles with residual connections."""
    def __init__(self, feature_dim, view_angles):
        super(ViewTransformationNetwork, self).__init__()
        self.feature_dim = feature_dim
        self.view_angles = view_angles
        self.num_views = len(view_angles)
        
        # Create enhanced transformation modules for each view pair
        self.transforms = nn.ModuleDict()
        for i, src_view in enumerate(view_angles):
            for j, tgt_view in enumerate(view_angles):
                if i != j:
                    key = f"{src_view}_{tgt_view}"
                    # More powerful transformation with residual connection
                    self.transforms[key] = nn.Sequential(
                        nn.Linear(feature_dim, feature_dim * 2),
                        nn.BatchNorm1d(feature_dim * 2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.3),
                        nn.Linear(feature_dim * 2, feature_dim),
                        nn.BatchNorm1d(feature_dim)
                    )
        
        # Global view-invariant feature extractor
        self.global_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features, src_view, tgt_view=None):
        """
        Transform features from source view to target view with residual connections.
        If tgt_view is None, transform to all views.
        """
        # Extract view-invariant features
        global_features = self.global_transform(features)
        
        if tgt_view is None:
            # Transform to all views
            transformed_features = []
            for view in self.view_angles:
                if view != src_view:
                    key = f"{src_view}_{view}"
                    # Apply transformation and add residual connection
                    transformed = self.transforms[key](features)
                    # Add global features as a residual connection
                    transformed = transformed + global_features
                    transformed_features.append(transformed)
                else:
                    # For same view, use original features + global features
                    transformed_features.append(features + global_features)
            return torch.stack(transformed_features, dim=1)
        else:
            # Transform to specific target view
            if src_view == tgt_view:
                return features + global_features
            key = f"{src_view}_{tgt_view}"
            transformed = self.transforms[key](features)
            # Add global features as a residual connection
            return transformed + global_features

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
