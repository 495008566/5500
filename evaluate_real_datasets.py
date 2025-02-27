#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import time
import os
import numpy as np
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Import custom modules
from models.gait_model import GaitModel
from preprocessing.datasets.dataset_factory import get_dataset
from utils.logging_utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate gait recognition model on real datasets')
    parser.add_argument('--dataset', type=str, default='casia_b',
                        help='Dataset name (casia_b, oumvlp, gait3d)')
    parser.add_argument('--config_dir', type=str, default='config',
                        help='Directory containing configuration files')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                        help='Checkpoint file to load')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model on the test set"""
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, labels in tqdm(test_loader, desc="Evaluating"):
            # Handle sequence data if needed
            if len(frames.shape) == 5:
                batch_size, seq_len, channels, height, width = frames.shape
                # Use multiple frames for evaluation
                frame_indices = [0, seq_len//2, seq_len-1]  # Use first, middle, and last frame
                
                # Process each selected frame
                batch_logits = []
                
                for idx in frame_indices:
                    frame = frames[:, idx, :, :, :]
                    
                    if device == 'cuda':
                        frame = frame.cuda()
                        labels = labels.cuda()
                    
                    # Forward pass
                    features, logits = model(frame)
                    batch_logits.append(logits)
                
                # Average logits across frames
                logits = torch.mean(torch.stack(batch_logits), dim=0)
            else:
                # Single frame input
                if device == 'cuda':
                    frames = frames.cuda()
                    labels = labels.cuda()
                
                # Forward pass
                features, logits = model(frames)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Calculate accuracy
            _, preds = torch.max(logits, 1)
            
            # Update statistics
            test_loss += loss.item() * labels.size(0)
            test_acc += torch.sum(preds == labels).item()
            
            # Store predictions and labels for further analysis
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate final statistics
    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)
    
    return test_loss, test_acc, all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_accuracy_by_view(dataset, all_preds, all_labels, output_path):
    """Plot accuracy by view angle"""
    if not hasattr(dataset, 'view_angles'):
        return
    
    view_angles = np.array(dataset.view_angles)
    unique_views = np.unique(view_angles)
    view_acc = []
    
    for view in unique_views:
        view_indices = np.where(view_angles == view)[0]
        view_preds = [all_preds[i] for i in view_indices]
        view_labels = [all_labels[i] for i in view_indices]
        view_acc.append(np.mean(np.array(view_preds) == np.array(view_labels)))
    
    plt.figure(figsize=(10, 6))
    plt.bar(unique_views, view_acc)
    plt.title('Accuracy by View Angle')
    plt.xlabel('View Angle (degrees)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create directories
    results_dir = os.path.join(args.results_dir, args.dataset)
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(results_dir)
    
    # Load configurations
    model_config_path = os.path.join(args.config_dir, 'model_config.yaml')
    
    model_config = load_config(model_config_path)
    
    # Convert configs to objects for easier access
    model_config = Config(model_config)
    
    # Create dataset
    logging.info(f"Creating {args.dataset} test dataset...")
    test_dataset = get_dataset(args.dataset, split='test')
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = GaitModel(model_config)
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, args.dataset, args.checkpoint)
    logging.info(f"Loading checkpoint from {checkpoint_path}...")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logging.info("Checkpoint loaded successfully")
    else:
        logging.error(f"Checkpoint not found at {checkpoint_path}")
        return
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        model = model.cuda()
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate model
    logging.info("Evaluating model...")
    test_loss, test_acc, all_preds, all_labels = evaluate_model(model, test_loader, criterion, device)
    
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    
    # Generate classification report
    class_names = [str(i) for i in range(model_config.num_classes)]
    report = classification_report(all_labels, all_preds, target_names=class_names)
    logging.info(f"Classification Report:\n{report}")
    
    # Save classification report to file
    report_path = os.path.join(results_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(f"Classification Report:\n{report}")
    
    # Plot confusion matrix
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_preds, class_names, cm_path)
    
    # Plot accuracy by view angle
    view_acc_path = os.path.join(results_dir, 'accuracy_by_view.png')
    plot_accuracy_by_view(test_dataset, all_preds, all_labels, view_acc_path)
    
    logging.info(f"Evaluation results saved to {results_dir}")

if __name__ == "__main__":
    main()
