import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import seaborn as sns
from sklearn.manifold import TSNE

class FeatureVisualizer:
    """Visualization tools for model features and attention"""
    
    def __init__(self, output_dir: str):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_attention_maps(self,
                          image: torch.Tensor,
                          attention_weights: Dict[str, torch.Tensor],
                          save_name: str):
        """
        Plot attention weight maps
        
        Args:
            image: Input image tensor (C, H, W)
            attention_weights: Dict of attention weights
            save_name: Name for saving plot
        """
        plt.figure(figsize=(15, 5))
        
        # Plot original image
        plt.subplot(1, len(attention_weights) + 1, 1)
        img = image.permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img)
        plt.title('Input Image')
        plt.axis('off')
        
        # Plot attention maps
        for idx, (name, weights) in enumerate(attention_weights.items(), 1):
            plt.subplot(1, len(attention_weights) + 1, idx + 1)
            
            # Average across channels if needed
            if weights.dim() > 2:
                weights = weights.mean(dim=1)
            
            # Normalize weights
            weights = weights.cpu().numpy()
            weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
            
            plt.imshow(weights, cmap='jet')
            plt.title(f'{name} Attention')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{save_name}_attention.png')
        plt.close()
    
    def plot_feature_maps(self,
                         features: torch.Tensor,
                         save_name: str,
                         num_channels: int = 16):
        """
        Plot feature activation maps
        
        Args:
            features: Feature tensor (C, H, W)
            num_channels: Number of channels to plot
            save_name: Name for saving plot
        """
        features = features.cpu().numpy()
        
        # Select channels
        if features.shape[0] > num_channels:
            channel_idx = np.linspace(0, features.shape[0]-1, num_channels, dtype=int)
            features = features[channel_idx]
        
        # Create grid
        rows = int(np.sqrt(num_channels))
        cols = int(np.ceil(num_channels / rows))
        
        plt.figure(figsize=(15, 15))
        for idx in range(min(num_channels, features.shape[0])):
            plt.subplot(rows, cols, idx + 1)
            
            # Normalize feature map
            feat_map = features[idx]
            feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
            
            plt.imshow(feat_map, cmap='viridis')
            plt.title(f'Channel {idx}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{save_name}_features.png')
        plt.close()
    
    def plot_embedding_space(self,
                           embeddings: torch.Tensor,
                           labels: torch.Tensor,
                           save_name: str):
        """
        Plot t-SNE visualization of embeddings
        
        Args:
            embeddings: Feature embeddings
            labels: Identity labels
            save_name: Name for saving plot
        """
        # Convert to numpy
        embeddings = embeddings.cpu().numpy()
        labels = labels.cpu().numpy()
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='tab20',
            alpha=0.6
        )
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of Feature Embeddings')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        plt.savefig(self.output_dir / f'{save_name}_embeddings.png')
        plt.close()
    
    def plot_cross_view_matrix(self,
                             accuracies: Dict[int, Dict[int, float]],
                             save_name: str):
        """
        Plot cross-view accuracy matrix
        
        Args:
            accuracies: Dict mapping source angles to dict of target angle accuracies
            save_name: Name for saving plot
        """
        # Convert to matrix
        angles = sorted(accuracies.keys())
        matrix = np.zeros((len(angles), len(angles)))
        
        for i, src_angle in enumerate(angles):
            for j, tgt_angle in enumerate(angles):
                matrix[i, j] = accuracies[src_angle][tgt_angle]
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=[str(a) for a in angles],
            yticklabels=[str(a) for a in angles]
        )
        plt.title('Cross-View Recognition Accuracy')
        plt.xlabel('Target View Angle')
        plt.ylabel('Source View Angle')
        
        plt.savefig(self.output_dir / f'{save_name}_cross_view.png')
        plt.close()
    
    def visualize_model_outputs(self,
                              image: torch.Tensor,
                              outputs: Dict[str, torch.Tensor],
                              save_name: str):
        """
        Visualize all model outputs
        
        Args:
            image: Input image
            outputs: Model outputs dictionary
            save_name: Name for saving visualizations
        """
        # Plot attention maps
        if 'attention_weights' in outputs:
            self.plot_attention_maps(
                image,
                outputs['attention_weights'],
                save_name
            )
        
        # Plot feature maps
        if 'features' in outputs:
            self.plot_feature_maps(
                outputs['features'],
                save_name=save_name
            )
        
        # Plot pose keypoints
        if 'pose_keypoints' in outputs and 'pose_confidences' in outputs:
            keypoints = outputs['pose_keypoints'][0].cpu().numpy()
            confidences = outputs['pose_confidences'][0].cpu().numpy()
            
            plt.figure(figsize=(5, 10))
            img = image.permute(1, 2, 0).cpu().numpy()
            plt.imshow(img)
            
            # Plot keypoints
            for kpt, conf in zip(keypoints, confidences):
                if conf > 0.5:
                    plt.plot(kpt[0], kpt[1], 'ro')
            
            plt.title('Pose Keypoints')
            plt.axis('off')
            plt.savefig(self.output_dir / f'{save_name}_pose.png')
            plt.close()

if __name__ == "__main__":
    # Test visualizer
    visualizer = FeatureVisualizer('visualization_test')
    
    # Create dummy data
    image = torch.randn(3, 256, 128)
    attention = {
        'spatial': torch.rand(32, 32),
        'channel': torch.rand(64, 16, 8)
    }
    features = torch.rand(64, 32, 16)
    embeddings = torch.randn(100, 256)
    labels = torch.randint(0, 10, (100,))
    
    # Test visualization functions
    visualizer.plot_attention_maps(image, attention, 'test')
    visualizer.plot_feature_maps(features, save_name='test')
    visualizer.plot_embedding_space(embeddings, labels, 'test')
    
    # Test cross-view matrix
    accuracies = {
        0: {0: 0.9, 45: 0.8, 90: 0.7},
        45: {0: 0.8, 45: 0.85, 90: 0.75},
        90: {0: 0.7, 45: 0.75, 90: 0.8}
    }
    visualizer.plot_cross_view_matrix(accuracies, 'test')
