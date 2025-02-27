#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
import time
import os
import numpy as np
from pathlib import Path

class GaitTrainer:
    def __init__(self, model_config, train_config, checkpoint_dir, log_dir):
        self.model_config = model_config
        self.train_config = train_config
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Create directories
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Create model
        from models.gait_model import GaitModel
        self.model = GaitModel(model_config)
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model
        
        # Create optimizer
        if train_config.learning_rate:
            lr = train_config.learning_rate
        else:
            lr = 0.001
            
        if train_config.weight_decay:
            weight_decay = train_config.weight_decay
        else:
            weight_decay = 0.0005
            
        if hasattr(train_config, 'optimizer') and train_config.optimizer == 'sgd':
            momentum = train_config.momentum if hasattr(train_config, 'momentum') else 0.9
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:  # default to adam
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        
        # Create learning rate scheduler
        if hasattr(train_config, 'lr_scheduler') and train_config.lr_scheduler == 'step':
            step_size = train_config.lr_step_size if hasattr(train_config, 'lr_step_size') else 30
            gamma = train_config.lr_gamma if hasattr(train_config, 'lr_gamma') else 0.1
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        else:  # default to cosine
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=model_config.num_epochs
            )
        
        # Create loss function
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Create triplet loss
        margin = train_config.triplet_margin if hasattr(train_config, 'triplet_margin') else 0.3
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
        
        # Create data augmentation
        self.use_augmentation = train_config.use_augmentation if hasattr(train_config, 'use_augmentation') else True
        if self.use_augmentation:
            logging.info("Using data augmentation")
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ])
    
    def train(self, train_loader, val_loader):
        best_acc = 0.0
        
        for epoch in range(self.model_config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_acc = 0.0
            start_time = time.time()
            
            for i, (sequences, labels) in enumerate(train_loader):
                if torch.cuda.is_available():
                    sequences = sequences.cuda()
                    labels = labels.cuda()
                
                # Apply data augmentation if enabled
                if self.use_augmentation and torch.cuda.is_available():
                    # Only apply augmentation during training and when using GPU
                    # Apply augmentation to each image in the batch
                    augmented_sequences = []
                    for i in range(sequences.size(0)):
                        aug_seq = self.augmentation(sequences[i])
                        augmented_sequences.append(aug_seq)
                    sequences = torch.stack(augmented_sequences)
                
                # Forward pass
                self.optimizer.zero_grad()
                features, logits = self.model(sequences)
                
                # Calculate cross-entropy loss with label smoothing
                if hasattr(self.train_config, 'use_label_smoothing') and self.train_config.use_label_smoothing:
                    smooth_factor = getattr(self.train_config, 'label_smoothing_factor', 0.1)
                    # Create one-hot encoding
                    smooth_labels = torch.zeros_like(logits).scatter_(
                        1, labels.unsqueeze(1), 1.0
                    )
                    # Apply smoothing
                    smooth_labels = smooth_labels * (1 - smooth_factor) + smooth_factor / logits.size(1)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                    ce_loss = -(smooth_labels * log_probs).sum(dim=1).mean()
                else:
                    ce_loss = self.ce_loss(logits, labels)
                
                # Improved triplet loss with hard mining
                unique_labels = torch.unique(labels)
                triplet_loss = 0.0
                
                if len(unique_labels) > 1:  # Need at least 2 classes for triplet loss
                    # Compute pairwise distances between all features
                    dist_matrix = torch.cdist(features, features)
                    
                    # For each anchor, find hardest positive and negative
                    for anchor_idx in range(len(labels)):
                        anchor_label = labels[anchor_idx]
                        
                        # Find positive samples (same class, excluding anchor)
                        pos_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
                        pos_indices = pos_indices[pos_indices != anchor_idx]  # Exclude anchor itself
                        
                        if len(pos_indices) == 0:
                            continue  # Skip if no other samples of same class
                            
                        # Find hardest positive (same class, maximum distance)
                        pos_dists = dist_matrix[anchor_idx, pos_indices]
                        hardest_pos_idx = pos_indices[pos_dists.argmax()]
                        
                        # Find negative samples (different class)
                        neg_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]
                        
                        if len(neg_indices) == 0:
                            continue  # Skip if no samples of different class
                            
                        # Find hardest negative (different class, minimum distance)
                        neg_dists = dist_matrix[anchor_idx, neg_indices]
                        hardest_neg_idx = neg_indices[neg_dists.argmin()]
                        
                        # Calculate triplet loss for this anchor
                        anchor = features[anchor_idx]
                        positive = features[hardest_pos_idx]
                        negative = features[hardest_neg_idx]
                        
                        triplet_loss += self.triplet_loss(
                            anchor.unsqueeze(0),
                            positive.unsqueeze(0),
                            negative.unsqueeze(0)
                        )
                    
                    # Normalize by number of valid triplets
                    if triplet_loss > 0:
                        triplet_loss = triplet_loss / len(labels)
                
                # Add center loss to pull same-class features closer
                center_loss = 0.0
                if len(unique_labels) > 0:
                    for label in unique_labels:
                        indices = (labels == label).nonzero(as_tuple=True)[0]
                        if len(indices) > 1:
                            class_features = features[indices]
                            center = class_features.mean(dim=0)
                            center_loss += ((class_features - center.unsqueeze(0))**2).sum() / len(indices)
                    
                    center_loss = center_loss / len(unique_labels)
                
                # Combine losses with appropriate weights
                lambda_ce = getattr(self.train_config, 'lambda_ce', 1.0)
                lambda_triplet = getattr(self.train_config, 'lambda_triplet', 0.5)
                lambda_center = getattr(self.train_config, 'lambda_center', 0.1)
                
                if len(unique_labels) > 1:
                    loss = lambda_ce * ce_loss + lambda_triplet * triplet_loss + lambda_center * center_loss
                else:
                    loss = ce_loss
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Calculate accuracy
                _, preds = torch.max(logits, 1)
                acc = torch.sum(preds == labels).item() / labels.size(0)
                
                # Update statistics
                train_loss += loss.item()
                train_acc += acc
                
                if (i+1) % 10 == 0:
                    logging.info(f"Epoch [{epoch+1}/{self.model_config.num_epochs}], "
                                f"Step [{i+1}/{len(train_loader)}], "
                                f"Loss: {loss.item():.4f}, "
                                f"Accuracy: {acc:.4f}")
            
            # Calculate epoch statistics
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            epoch_time = time.time() - start_time
            
            logging.info(f"Epoch [{epoch+1}/{self.model_config.num_epochs}], "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Train Accuracy: {train_acc:.4f}, "
                        f"Time: {epoch_time:.2f}s")
            
            # Update learning rate
            self.scheduler.step()
            
            # Validation
            if hasattr(self.train_config, 'eval_freq') and (epoch+1) % self.train_config.eval_freq == 0:
                eval_freq = self.train_config.eval_freq
            else:
                eval_freq = 5
                
            if (epoch+1) % eval_freq == 0:
                val_loss, val_acc = self.validate(val_loader)
                
                logging.info(f"Epoch [{epoch+1}/{self.model_config.num_epochs}], "
                            f"Val Loss: {val_loss:.4f}, "
                            f"Val Accuracy: {val_acc:.4f}")
                
                # Save checkpoint if best accuracy
                if val_acc > best_acc:
                    best_acc = val_acc
                    self.save_checkpoint(epoch, val_acc, is_best=True)
            
            # Save regular checkpoint
            if hasattr(self.train_config, 'save_freq'):
                save_freq = self.train_config.save_freq
            else:
                save_freq = 10
                
            if (epoch+1) % save_freq == 0:
                self.save_checkpoint(epoch, val_acc if (epoch+1) % eval_freq == 0 else 0.0)
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                if torch.cuda.is_available():
                    sequences = sequences.cuda()
                    labels = labels.cuda()
                
                # Forward pass
                features, logits = self.model(sequences)
                
                # Calculate loss
                ce_loss = self.ce_loss(logits, labels)
                loss = ce_loss
                
                # Calculate accuracy
                _, preds = torch.max(logits, 1)
                acc = torch.sum(preds == labels).item() / labels.size(0)
                
                # Update statistics
                val_loss += loss.item()
                val_acc += acc
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch, accuracy, is_best=False):
        # Save only the model state dict for better compatibility
        state_dict = self.model.state_dict()
        
        if is_best:
            torch.save(state_dict, os.path.join(self.checkpoint_dir, 'best_model.pth'))
        
        # Also save full checkpoint for training resumption
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': accuracy
        }
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
