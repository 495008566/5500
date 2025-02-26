import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple

class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()
        self.config = config
        
        # Initialize backbone with efficient architecture
        self.backbone = self._create_efficient_backbone()
        
        # Get the number of output channels from backbone
        self.out_channels = 160  # MobileNetV3 large last layer channels
        self.feature_dim = 256   # Target feature dimension
        
        # Feature dimension adjustment
        self.feature_adj = nn.Sequential(
            nn.Conv2d(self.out_channels, self.feature_dim, kernel_size=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.feature_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Channel attention module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.feature_dim, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.feature_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
        
        # CPU optimizations if needed
        if config.device == 'cpu':
            self._optimize_for_cpu()
    
    def _create_efficient_backbone(self):
        # Using MobileNetV3 for efficiency
        backbone = models.mobilenet_v3_large(pretrained=True)
        # Modify for our needs
        features = list(backbone.features)
        return nn.Sequential(*features[:-1])  # Remove last layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)
        
        # Adjust feature dimensions
        features = self.feature_adj(features)
        
        # Apply attention mechanisms
        spatial_weights = self.spatial_attention(features)
        channel_weights = self.channel_attention(features)
        
        # Apply attention
        attended_features = features * spatial_weights * channel_weights
        
        # Global average pooling
        output = torch.mean(attended_features, dim=(2, 3))
        
        return output
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _optimize_for_cpu(self):
        # Enable MKL optimizations if available
        torch.set_num_threads(torch.get_num_threads())
        # Use efficient memory format
        self.to(memory_format=torch.channels_last)
        # Enable inference mode optimizations
        torch._C._jit_set_profiling_executor(False)
