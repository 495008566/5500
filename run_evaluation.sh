#!/bin/bash
# Create necessary directories
mkdir -p results/casia_b

# Run evaluation
python evaluate.py --dataset casia_b --checkpoint_path checkpoints/casia_b/best_model.pth --output_dir results/casia_b
