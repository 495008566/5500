import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class SEModule(nn.Module):
    """Squeeze-and-Excitation (SE) attention module"""
    
    def __init__(self,
                 channels: int,
                 reduction: int = 16):
        """
        Initialize SE module
        
        Args:
            channels: Number of input channels
            reduction: Channel reduction ratio
        """
        super(SEModule, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        b, c, _, _ = x.size()
        
        # Global average pooling
        y = self.avg_pool(x).view(b, c)
        
        # Channel attention
        y = self.fc(y).view(b, c, 1, 1)
        
        # Scale the input
        return x * y.expand_as(x)

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self,
                 channels: int,
                 reduction: int = 16,
                 kernel_size: int = 7):
        """
        Initialize CBAM
        
        Args:
            channels: Number of input channels
            reduction: Channel reduction ratio
            kernel_size: Kernel size for spatial attention
        """
        super(CBAM, self).__init__()
        
        # Channel attention
        self.channel_gate = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Channel attention
        channel_att = self.channel_gate(F.adaptive_avg_pool2d(x, 1))
        x = x * channel_att
        
        # Spatial attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_att = self.spatial_gate(torch.cat([max_pool, avg_pool], dim=1))
        
        return x * spatial_att

class ViewTransformationNetwork(nn.Module):
    """Network for transforming features across different views"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 512,
                 num_views: int = 11):
        """
        Initialize view transformation network
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_views: Number of view angles
        """
        super(ViewTransformationNetwork, self).__init__()
        
        self.num_views = num_views
        
        # View embedding
        self.view_embedding = nn.Embedding(num_views, hidden_dim)
        
        # Feature transformation
        self.transform = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,
                features: torch.Tensor,
                source_view: torch.Tensor,
                target_view: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Transform features from source view to target view
        
        Args:
            features: Input features
            source_view: Source view indices
            target_view: Target view indices (if None, transform to canonical view)
            
        Returns:
            Transformed features
        """
        batch_size = features.size(0)
        
        # Get source view embedding
        source_embed = self.view_embedding(source_view)
        
        # If target view not specified, use canonical view (middle view)
        if target_view is None:
            target_view = torch.ones_like(source_view) * (self.num_views // 2)
        
        # Get target view embedding
        target_embed = self.view_embedding(target_view)
        
        # Compute view difference embedding
        view_diff = target_embed - source_embed
        
        # Concatenate features with view difference
        combined = torch.cat([features, view_diff], dim=1)
        
        # Transform features
        transformed = self.transform(combined)
        
        return transformed

class DynamicWeightModule(nn.Module):
    """Dynamic weight module for body region feature fusion"""
    
    def __init__(self,
                 feature_dim: int,
                 num_regions: int = 3):
        """
        Initialize dynamic weight module
        
        Args:
            feature_dim: Feature dimension
            num_regions: Number of body regions
        """
        super(DynamicWeightModule, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * num_regions, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_regions),
            nn.Softmax(dim=1)
        )
    
    def forward(self,
                features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dynamic weights for feature fusion
        
        Args:
            features: Input features (batch_size, num_regions, feature_dim)
            
        Returns:
            Tuple of:
            - Fused features
            - Attention weights
        """
        batch_size, num_regions, feature_dim = features.size()
        
        # Flatten features
        flat_features = features.view(batch_size, -1)
        
        # Compute attention weights
        weights = self.attention(flat_features)
        
        # Apply weights
        weighted_features = features * weights.unsqueeze(-1)
        
        # Sum weighted features
        fused = weighted_features.sum(dim=1)
        
        return fused, weights

if __name__ == "__main__":
    # Test attention modules
    x = torch.randn(4, 64, 32, 16)
    
    # Test SE module
    se = SEModule(64)
    y_se = se(x)
    print(f"SE output shape: {y_se.shape}")
    
    # Test CBAM
    cbam = CBAM(64)
    y_cbam = cbam(x)
    print(f"CBAM output shape: {y_cbam.shape}")
    
    # Test view transformation
    features = torch.randn(4, 256)
    source_view = torch.randint(0, 11, (4,))
    vtn = ViewTransformationNetwork(256)
    y_vtn = vtn(features, source_view)
    print(f"VTN output shape: {y_vtn.shape}")
    
    # Test dynamic weights
    region_features = torch.randn(4, 3, 256)  # 3 regions
    weight_module = DynamicWeightModule(256)
    fused, weights = weight_module(region_features)
    print(f"Fused features shape: {fused.shape}")
    print(f"Attention weights shape: {weights.shape}")
