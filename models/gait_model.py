import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional
from .components.attention import CBAM, ViewTransformationNetwork, DynamicWeightModule
from .components.feature_pyramid import GaitFeatureExtractor
from .components.pose_estimator import PoseEstimator

class GaitModel(nn.Module):
    """Complete gait recognition model with attention and view transformation"""
    
    def __init__(self, config):
        super(GaitModel, self).__init__()
        self.config = config
        
        # Create backbone network
        self.backbone = self._create_backbone(config.backbone)
        backbone_channels = [256, 512, 1024, 2048]  # ResNet channels
        
        # Pose estimation
        self.pose_estimator = PoseEstimator(pretrained=True)
        
        # Feature extraction
        self.feature_extractor = GaitFeatureExtractor(backbone_channels)
        
        # View transformation
        self.view_transform = ViewTransformationNetwork(config.feature_dim, config.view_angles)
        
        # Attention modules
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)
        
        # Dynamic weight module
        self.weight_module = DynamicWeightModule(config.feature_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, config.num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Optimize for CPU if needed
        if torch.cuda.is_available() == False:
            self._optimize_for_cpu()
    
    def _create_backbone(self, backbone: str) -> nn.Module:
        """Create backbone network with skip connections"""
        if backbone == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif backbone == 'resnet101':
            model = models.resnet101(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove classification head
        return nn.Sequential(*list(model.children())[:-2])
    
    def forward(self,
                x: torch.Tensor,
                view_angles: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with skip connections and attention
        
        Args:
            x: Input tensor
            view_angles: View angle indices (optional)
            
        Returns:
            Dict containing model outputs
        """
        batch_size = x.size(0)
        
        # Extract backbone features with skip connections
        features = []
        x1 = self.backbone[0:5](x)  # conv1 + bn1 + relu + maxpool + layer1
        x1 = self.cbam1(x1)
        features.append(x1)
        
        x2 = self.backbone[5](x1)  # layer2
        x2 = self.cbam2(x2)
        features.append(x2)
        
        x3 = self.backbone[6](x2)  # layer3
        x3 = self.cbam3(x3)
        features.append(x3)
        
        x4 = self.backbone[7](x3)  # layer4
        x4 = self.cbam4(x4)
        features.append(x4)
        
        # Pose estimation
        pose_outputs = self.pose_estimator(x)
        keypoints, confidences = self.pose_estimator.extract_keypoints(
            pose_outputs['heatmaps']
        )
        
        # For simplified training, create dummy body regions
        # In a real implementation, we would extract keypoints and segment body regions
        body_regions = {
            'upper': None,  # These will be ignored in the simplified feature extractor
            'lower': None,
            'full': None
        }
        
        # Extract features
        gait_features = self.feature_extractor(features, body_regions)
        
        # Apply view transformation if angles provided
        if view_angles is not None:
            gait_features['fused'] = self.view_transform(
                gait_features['fused'],
                view_angles
            )
        
        # For simplified training, use the fused features directly
        # In a real implementation, we would apply dynamic weighting to different body regions
        weighted_features = gait_features['fused']
        attention_weights = torch.ones(batch_size, 3) / 3.0  # Equal weights for visualization
        
        # Classification
        logits = self.classifier(weighted_features)
        
        # For simplified training, return features and logits directly
        # In a real implementation, we would return a dictionary with all outputs
        return weighted_features, logits
    
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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _optimize_for_cpu(self):
        """Optimize model for CPU inference"""
        torch.set_num_threads(torch.get_num_threads())
        self.to(memory_format=torch.channels_last)
