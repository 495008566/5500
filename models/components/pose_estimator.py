import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

class PoseEstimator(nn.Module):
    """Pose estimation for gait feature extraction"""
    
    def __init__(self,
                 num_keypoints: int = 17,
                 pretrained: bool = True):
        """
        Initialize pose estimator using HRNet
        
        Args:
            num_keypoints: Number of keypoints to detect
            pretrained: Whether to use pretrained weights
        """
        super(PoseEstimator, self).__init__()
        
        # Initialize HRNet backbone
        self.backbone = self._create_hrnet_backbone(pretrained)
        
        # Keypoint head
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_hrnet_backbone(self, pretrained: bool) -> nn.Module:
        """Create HRNet backbone"""
        try:
            # Use torchvision's implementation
            backbone = models.hrnet_w32(pretrained=pretrained)
            # Remove classification head
            backbone = nn.Sequential(*list(backbone.children())[:-2])
            return backbone
        except Exception as e:
            logging.error(f"Error creating HRNet backbone: {str(e)}")
            raise
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Dict containing:
            - heatmaps: Keypoint heatmaps
            - features: Backbone features
        """
        # Extract features
        features = self.backbone(x)
        
        # Generate heatmaps
        heatmaps = self.keypoint_head(features)
        
        return {
            'heatmaps': heatmaps,
            'features': features
        }
    
    def extract_keypoints(self,
                         heatmaps: torch.Tensor,
                         threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract keypoint coordinates from heatmaps
        
        Args:
            heatmaps: Keypoint heatmaps (batch_size, num_keypoints, height, width)
            threshold: Confidence threshold
            
        Returns:
            Tuple of:
            - keypoints: Array of shape (batch_size, num_keypoints, 2)
            - confidences: Array of shape (batch_size, num_keypoints)
        """
        batch_size, num_keypoints, height, width = heatmaps.shape
        
        # Convert to numpy
        heatmaps_np = heatmaps.detach().cpu().numpy()
        
        # Initialize outputs
        keypoints = np.zeros((batch_size, num_keypoints, 2))
        confidences = np.zeros((batch_size, num_keypoints))
        
        # Process each sample in batch
        for b in range(batch_size):
            for k in range(num_keypoints):
                heatmap = heatmaps_np[b, k]
                
                # Find maximum location
                conf = heatmap.max()
                if conf > threshold:
                    loc = np.unravel_index(heatmap.argmax(), heatmap.shape)
                    # Convert to (x, y) format
                    keypoints[b, k] = [loc[1], loc[0]]
                    confidences[b, k] = conf
        
        return keypoints, confidences
    
    def segment_body_regions(self,
                           keypoints: np.ndarray,
                           confidences: np.ndarray,
                           conf_threshold: float = 0.5) -> Dict[str, List[int]]:
        """
        Segment body into regions based on keypoints
        
        Args:
            keypoints: Array of shape (num_keypoints, 2)
            confidences: Array of shape (num_keypoints,)
            conf_threshold: Minimum confidence for valid keypoint
            
        Returns:
            Dict mapping region names to keypoint indices
        """
        # Define keypoint indices for each region
        regions = {
            'upper_body': [5, 6, 7, 8, 9, 10],  # Shoulders, elbows, wrists
            'lower_body': [11, 12, 13, 14, 15, 16],  # Hips, knees, ankles
            'full_body': list(range(17))  # All keypoints
        }
        
        # Filter regions based on confidence
        valid_regions = {}
        for region_name, indices in regions.items():
            valid_keypoints = []
            for idx in indices:
                if confidences[idx] > conf_threshold:
                    valid_keypoints.append(idx)
            if valid_keypoints:
                valid_regions[region_name] = valid_keypoints
        
        return valid_regions

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test pose estimator
    model = PoseEstimator(pretrained=True)
    
    # Create dummy input
    x = torch.randn(1, 3, 256, 128)
    
    # Forward pass
    outputs = model(x)
    
    # Extract keypoints
    keypoints, confidences = model.extract_keypoints(outputs['heatmaps'])
    
    # Segment body regions
    regions = model.segment_body_regions(keypoints[0], confidences[0])
    
    logging.info(f"Detected regions: {regions}")
