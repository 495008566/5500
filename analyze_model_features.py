#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import logging

# Import custom modules
from models.gait_model import GaitModel
from preprocessing.datasets.dataset_factory import get_dataset
from utils.logging_utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze model features and dataset')
    parser.add_argument('--dataset', type=str, default='casia_b',
                        help='Dataset name (casia_b, oumvlp, gait3d)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--config_dir', type=str, default='config',
                        help='Directory containing configuration files')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    return parser.parse_args()

def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(model_config, checkpoint_path=None):
    """Load model from checkpoint or create a new one"""
    model = GaitModel(model_config)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume checkpoint is the state dict directly
                model.load_state_dict(checkpoint)
                
            logging.info("Checkpoint loaded successfully")
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
    else:
        logging.warning("No checkpoint provided or checkpoint not found. Using random initialization.")
    
    return model

def extract_features(model, data_loader, device):
    """Extract features from the model"""
    model.eval()
    all_features = []
    all_labels = []
    all_views = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            # Handle both tensor and dictionary data formats
            if isinstance(data, dict):
                images = data['image'].to(device)
                view_angles = data.get('view_angle')
                if view_angles is not None:
                    view_angles = view_angles.to(device)
                
                if isinstance(labels, dict):
                    target_ids = labels['identity'].to(device)
                else:
                    target_ids = labels.to(device)
            else:
                # For tensor data (simplified dataset)
                images = data.to(device)
                target_ids = labels.to(device)
                view_angles = None
            
            # Forward pass
            features, _ = model(images, view_angles)
            
            # Store features and labels
            all_features.append(features.cpu().numpy())
            all_labels.append(target_ids.cpu().numpy())
            
            # Store view angles if available
            if view_angles is not None:
                all_views.append(view_angles.cpu().numpy())
    
    # Concatenate all features and labels
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    if len(all_views) > 0:
        all_views = np.concatenate(all_views, axis=0)
    else:
        all_views = None
    
    return all_features, all_labels, all_views

def visualize_features(features, labels, views=None, output_dir=None):
    """Visualize features using t-SNE and PCA"""
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Reduce dimensionality with PCA first for efficiency
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features_pca)
    
    # Plot t-SNE visualization by class
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            features_tsne[mask, 0],
            features_tsne[mask, 1],
            c=[colors[i]],
            label=f'ID {label}',
            alpha=0.7,
            s=50
        )
    
    plt.title('t-SNE Visualization of Gait Features by Identity')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), ncol=2)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'tsne_by_identity.png'), dpi=300)
    else:
        plt.show()
    
    # If view angles are available, plot t-SNE by view angle
    if views is not None:
        plt.figure(figsize=(12, 10))
        unique_views = np.unique(views)
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_views)))
        
        for i, view in enumerate(unique_views):
            mask = views == view
            plt.scatter(
                features_tsne[mask, 0],
                features_tsne[mask, 1],
                c=[colors[i]],
                label=f'View {view}°',
                alpha=0.7,
                s=50
            )
        
        plt.title('t-SNE Visualization of Gait Features by View Angle')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(loc='best', bbox_to_anchor=(1.05, 1), ncol=1)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'tsne_by_view.png'), dpi=300)
        else:
            plt.show()
    
    # Compute feature statistics
    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0)
    feature_min = np.min(features, axis=0)
    feature_max = np.max(features, axis=0)
    
    # Plot feature statistics
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(feature_mean, bins=50)
    plt.title('Distribution of Feature Means')
    plt.xlabel('Mean Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(feature_std, bins=50)
    plt.title('Distribution of Feature Standard Deviations')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_statistics.png'), dpi=300)
    else:
        plt.show()
    
    # Save feature statistics
    if output_dir:
        np.save(os.path.join(output_dir, 'feature_mean.npy'), feature_mean)
        np.save(os.path.join(output_dir, 'feature_std.npy'), feature_std)
        np.save(os.path.join(output_dir, 'feature_min.npy'), feature_min)
        np.save(os.path.join(output_dir, 'feature_max.npy'), feature_max)
    
    # Compute intra-class and inter-class distances
    intra_class_distances = []
    inter_class_distances = []
    
    for label in unique_labels:
        # Get features for this class
        class_features = features[labels == label]
        
        # Compute pairwise distances within this class
        if len(class_features) > 1:
            for i in range(len(class_features)):
                for j in range(i+1, len(class_features)):
                    dist = np.linalg.norm(class_features[i] - class_features[j])
                    intra_class_distances.append(dist)
        
        # Get features for other classes
        other_features = features[labels != label]
        other_labels = labels[labels != label]
        
        # Sample some inter-class distances (to avoid computing all pairs)
        if len(class_features) > 0 and len(other_features) > 0:
            # Sample at most 100 features from this class
            sample_size = min(100, len(class_features))
            sampled_indices = np.random.choice(len(class_features), sample_size, replace=False)
            
            for i in sampled_indices:
                # Sample at most 100 features from other classes
                other_sample_size = min(100, len(other_features))
                other_sampled_indices = np.random.choice(len(other_features), other_sample_size, replace=False)
                
                for j in other_sampled_indices:
                    dist = np.linalg.norm(class_features[i] - other_features[j])
                    inter_class_distances.append(dist)
    
    # Plot distance distributions
    plt.figure(figsize=(12, 6))
    plt.hist(intra_class_distances, bins=50, alpha=0.5, label='Intra-class')
    plt.hist(inter_class_distances, bins=50, alpha=0.5, label='Inter-class')
    plt.title('Distribution of Intra-class and Inter-class Distances')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'distance_distributions.png'), dpi=300)
        
        # Save distance statistics
        with open(os.path.join(output_dir, 'distance_statistics.txt'), 'w') as f:
            f.write(f"Intra-class distances: mean={np.mean(intra_class_distances):.4f}, std={np.std(intra_class_distances):.4f}\n")
            f.write(f"Inter-class distances: mean={np.mean(inter_class_distances):.4f}, std={np.std(inter_class_distances):.4f}\n")
            f.write(f"Ratio (inter/intra): {np.mean(inter_class_distances)/np.mean(intra_class_distances):.4f}\n")
    else:
        plt.show()

def analyze_dataset(dataset, output_dir=None):
    """Analyze dataset statistics"""
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Count number of samples per class
    if hasattr(dataset, 'labels'):
        labels = dataset.labels
    elif hasattr(dataset, 'identities'):
        labels = dataset.identities
    else:
        logging.warning("Could not find labels in dataset. Skipping class distribution analysis.")
        return
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Plot class distribution
    plt.figure(figsize=(12, 6))
    plt.bar(unique_labels, counts)
    plt.title('Number of Samples per Identity')
    plt.xlabel('Identity ID')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300)
    else:
        plt.show()
    
    # If view angles are available, analyze view angle distribution
    if hasattr(dataset, 'view_angles'):
        try:
            view_angles = np.array(dataset.view_angles)
            unique_views, view_counts = np.unique(view_angles, return_counts=True)
            
            plt.figure(figsize=(12, 6))
            plt.bar(unique_views, view_counts)
            plt.title('Number of Samples per View Angle')
            plt.xlabel('View Angle (degrees)')
            plt.ylabel('Number of Samples')
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'view_distribution.png'), dpi=300)
            else:
                plt.show()
            
            # Skip class-view distribution analysis for now
            logging.info("Skipping class-view distribution analysis")
        except Exception as e:
            logging.warning(f"Error analyzing view angles: {str(e)}")
            logging.warning("Skipping view angle analysis")
        
        # Skip heatmap plotting for now
        logging.info("Skipping heatmap plotting")
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'class_view_distribution.png'), dpi=300)
        else:
            plt.show()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    
    # Load configurations
    model_config_path = os.path.join(args.config_dir, 'model_config.yaml')
    model_config = load_config(model_config_path)
    
    # Convert config to object for easier access
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key, value)
    
    model_config = Config(model_config)
    
    # Create dataset
    logging.info(f"Creating {args.dataset} dataset...")
    dataset = get_dataset(args.dataset, split='test')
    
    # Analyze dataset
    logging.info("Analyzing dataset...")
    analyze_dataset(dataset, os.path.join(output_dir, 'dataset'))
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_config, args.checkpoint_path)
    model = model.to(device)
    
    # Extract features
    logging.info("Extracting features...")
    features, labels, views = extract_features(model, data_loader, device)
    
    # Visualize features
    logging.info("Visualizing features...")
    visualize_features(features, labels, views, os.path.join(output_dir, 'features'))
    
    logging.info("Analysis completed!")

if __name__ == "__main__":
    main()
