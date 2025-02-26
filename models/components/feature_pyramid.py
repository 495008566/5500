import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import logging

class MultiScaleConv(nn.Module):
    """Multi-scale convolution block using different kernel sizes"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        """
        Initialize multi-scale convolution block
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels per scale
        """
        super(MultiScaleConv, self).__init__()
        
        # Different scale convolutions
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        
        # Batch normalization and activation
        self.bn = nn.BatchNorm2d(out_channels * 3)
        self.relu = nn.ReLU(inplace=True)
        
        # Feature fusion
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Apply different scale convolutions
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)
        feat7 = self.conv7x7(x)
        
        # Concatenate features
        multi_scale = torch.cat([feat3, feat5, feat7], dim=1)
        
        # Apply normalization and activation
        multi_scale = self.bn(multi_scale)
        multi_scale = self.relu(multi_scale)
        
        # Fuse features
        output = self.fusion(multi_scale)
        
        return output

class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""
    
    def __init__(self,
                 in_channels_list: List[int],
                 out_channels: int):
        """
        Initialize FPN
        
        Args:
            in_channels_list: List of input channels for each level
            out_channels: Number of output channels
        """
        super(FeaturePyramidNetwork, self).__init__()
        
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        # Create lateral and output convolutions
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = MultiScaleConv(out_channels, out_channels)
            
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
            
        # Initialize weights
        self._initialize_weights()
    
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
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass
        
        Args:
            features: List of feature maps from backbone network
            
        Returns:
            List of processed feature maps
        """
        last_inner = self.inner_blocks[-1](features[-1])
        results = []
        results.append(self.layer_blocks[-1](last_inner))
        
        for idx in range(len(features) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](features[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode='nearest')
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))
        
        return results

class BodyRegionNetwork(nn.Module):
    """Sub-network for body region feature extraction"""
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 256):
        """
        Initialize body region network
        
        Args:
            in_channels: Number of input channels
            hidden_channels: Number of hidden channels
        """
        super(BodyRegionNetwork, self).__init__()
        
        # Multi-scale feature extraction
        self.multi_scale = MultiScaleConv(in_channels, hidden_channels)
        
        # Additional processing
        self.process = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Extract multi-scale features
        features = self.multi_scale(x)
        
        # Additional processing
        features = self.process(features)
        
        # Global pooling
        output = self.pool(features)
        output = output.view(output.size(0), -1)
        
        return output

class GaitFeatureExtractor(nn.Module):
    """Complete feature extraction module"""
    
    def __init__(self,
                 backbone_channels: List[int],
                 fpn_channels: int = 256):
        """
        Initialize feature extractor
        
        Args:
            backbone_channels: List of channel numbers from backbone
            fpn_channels: Number of FPN channels
        """
        super(GaitFeatureExtractor, self).__init__()
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(backbone_channels, fpn_channels)
        
        # Body region networks
        self.upper_body_net = BodyRegionNetwork(fpn_channels)
        self.lower_body_net = BodyRegionNetwork(fpn_channels)
        self.full_body_net = BodyRegionNetwork(fpn_channels)
        
        # Feature fusion
        total_channels = fpn_channels * 3
        self.fusion = nn.Sequential(
            nn.Linear(total_channels, total_channels // 2),
            nn.BatchNorm1d(total_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(total_channels // 2, fpn_channels)
        )
    
    def forward(self,
                features: List[torch.Tensor],
                body_regions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            features: List of backbone features
            body_regions: Dict containing upper, lower, and full body regions
            
        Returns:
            Dict containing extracted features
        """
        # Get FPN features
        fpn_features = self.fpn(features)
        
        # Process body regions
        upper_features = self.upper_body_net(body_regions['upper'])
        lower_features = self.lower_body_net(body_regions['lower'])
        full_features = self.full_body_net(body_regions['full'])
        
        # Concatenate features
        combined = torch.cat([upper_features, lower_features, full_features], dim=1)
        
        # Fuse features
        fused = self.fusion(combined)
        
        return {
            'upper_body': upper_features,
            'lower_body': lower_features,
            'full_body': full_features,
            'fused': fused,
            'fpn_features': fpn_features
        }

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test feature extraction
    backbone_channels = [256, 512, 1024, 2048]  # Example ResNet channels
    extractor = GaitFeatureExtractor(backbone_channels)
    
    # Create dummy inputs
    features = [
        torch.randn(1, channels, 64 // (2 ** i), 32 // (2 ** i))
        for i, channels in enumerate(backbone_channels)
    ]
    
    body_regions = {
        'upper': torch.randn(1, 256, 32, 16),
        'lower': torch.randn(1, 256, 32, 16),
        'full': torch.randn(1, 256, 64, 32)
    }
    
    # Forward pass
    outputs = extractor(features, body_regions)
    
    # Print shapes
    for name, tensor in outputs.items():
        if isinstance(tensor, (list, tuple)):
            logging.info(f"{name} shapes: {[t.shape for t in tensor]}")
        else:
            logging.info(f"{name} shape: {tensor.shape}")
