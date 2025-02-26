import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class GaitLoss(nn.Module):
    """Combined loss function for gait recognition"""
    
    def __init__(self,
                 lambda_id: float = 1.0,
                 lambda_triplet: float = 1.0,
                 lambda_pose: float = 0.1,
                 margin: float = 0.3):
        """
        Initialize loss function
        
        Args:
            lambda_id: Weight for identity loss
            lambda_triplet: Weight for triplet loss
            lambda_pose: Weight for pose estimation loss
            margin: Margin for triplet loss
        """
        super(GaitLoss, self).__init__()
        
        self.lambda_id = lambda_id
        self.lambda_triplet = lambda_triplet
        self.lambda_pose = lambda_pose
        self.margin = margin
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            Dict containing loss components
        """
        # Identity classification loss
        id_loss = self.ce_loss(outputs['logits'], targets['identity'])
        
        # Triplet loss
        anchor = outputs['features']
        positive = targets['positive_features']
        negative = targets['negative_features']
        
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        triplet_loss = F.relu(pos_dist - neg_dist + self.margin).mean()
        
        # Pose estimation loss
        pose_loss = self.mse_loss(
            outputs['pose_keypoints'],
            targets['keypoints']
        )
        
        # Combine losses
        total_loss = (
            self.lambda_id * id_loss +
            self.lambda_triplet * triplet_loss +
            self.lambda_pose * pose_loss
        )
        
        return {
            'total_loss': total_loss,
            'id_loss': id_loss,
            'triplet_loss': triplet_loss,
            'pose_loss': pose_loss
        }

if __name__ == "__main__":
    # Test loss function
    criterion = GaitLoss()
    
    # Create dummy outputs
    outputs = {
        'logits': torch.randn(4, 124),  # 124 classes
        'features': torch.randn(4, 256),
        'pose_keypoints': torch.randn(4, 17, 2)  # 17 keypoints
    }
    
    # Create dummy targets
    targets = {
        'identity': torch.randint(0, 124, (4,)),
        'keypoints': torch.randn(4, 17, 2),
        'positive_features': torch.randn(4, 256),
        'negative_features': torch.randn(4, 256)
    }
    
    # Compute loss
    losses = criterion(outputs, targets)
    
    print("\nLoss Values:")
    for name, value in losses.items():
        print(f"{name}: {value.item():.4f}")
