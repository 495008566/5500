#!/bin/bash
# Create necessary directories
mkdir -p data/casia_b
mkdir -p checkpoints/casia_b
mkdir -p logs/casia_b
mkdir -p results/casia_b

# Run training
python train_real_datasets.py --dataset casia_b --checkpoint_dir checkpoints/casia_b --log_dir logs/casia_b
