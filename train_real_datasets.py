import torch
from torch.utils.data import DataLoader
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
import argparse

from models.gait_model import GaitModel
from preprocessing.datasets.dataset_factory import get_dataset
from config import ModelConfig, TrainingConfig
from train_simple import GaitTrainer


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train gait recognition model with real datasets')
    parser.add_argument('--dataset', type=str, default='casia_b', choices=['casia_b', 'oumvlp', 'gait3d'],
                        help='Dataset to use for training')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of epochs for training')
    args = parser.parse_args()
    
    # Setup logging
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.log_dir}/training_{args.dataset}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Load configurations
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
    # Update configurations based on arguments
    model_config.batch_size = args.batch_size
    model_config.num_epochs = args.num_epochs
    
    # Create datasets
    logging.info(f"Creating {args.dataset} dataset...")
    train_dataset = get_dataset(args.dataset, split='train')
    val_dataset = get_dataset(args.dataset, split='test')
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=model_config.batch_size,
        shuffle=True,
        num_workers=model_config.num_workers
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=model_config.batch_size,
        shuffle=False,
        num_workers=model_config.num_workers
    )
    
    # Create trainer
    trainer = GaitTrainer(
        model_config=model_config,
        train_config=train_config,
        checkpoint_dir=f"{args.checkpoint_dir}/{args.dataset}",
        log_dir=f"{args.log_dir}/{args.dataset}"
    )
    
    # Train model
    logging.info(f"Starting training on {args.dataset}...")
    trainer.train(train_loader, val_loader)
    
    logging.info(f"Training completed on {args.dataset}!")


if __name__ == "__main__":
    main()
