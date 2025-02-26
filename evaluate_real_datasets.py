import torch
from torch.utils.data import DataLoader
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from preprocessing.datasets.dataset_factory import create_dataset
from config import ModelConfig
from evaluate import GaitEvaluator


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate gait recognition model with real datasets')
    parser.add_argument('--dataset', type=str, default='casia_b', choices=['casia_b', 'oumvlp', 'gait3d'],
                        help='Dataset to use for evaluation')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save evaluation results')
    args = parser.parse_args()
    
    # Setup logging
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.output_dir}/evaluation_{args.dataset}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Load configurations
    model_config = ModelConfig()
    
    # Create dataset
    logging.info(f"Creating {args.dataset} test dataset...")
    test_dataset = create_dataset(args.dataset, split='test')
    
    # Create data loader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=model_config.batch_size,
        shuffle=False,
        num_workers=model_config.num_workers
    )
    
    # Create evaluator
    evaluator = GaitEvaluator(
        model_config=model_config,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir
    )
    
    # Evaluate model
    logging.info(f"Starting evaluation on {args.dataset}...")
    metrics = evaluator.evaluate(test_loader, cross_view=True)
    
    # Generate visualizations
    logging.info("Generating visualizations...")
    
    # Create directory for plots
    plots_dir = Path(args.output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot cross-view accuracy
    angles = list(metrics['cross_view_accuracy'].keys())
    accuracies = list(metrics['cross_view_accuracy'].values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(angles, accuracies, 'b-o')
    plt.xlabel('View Angle (degrees)')
    plt.ylabel('Rank-1 Accuracy (%)')
    plt.title('Cross-View Gait Recognition Accuracy')
    plt.grid(True)
    plt.savefig(str(plots_dir / "cross_view_accuracy.png"))
    
    # Save metrics to file
    with open(str(Path(args.output_dir) / "metrics.txt"), "w") as f:
        f.write(f"Overall Accuracy: {metrics['accuracy']:.2f}%\n")
        f.write(f"Processing Time: {metrics['processing_time']:.4f} seconds per sample\n")
        f.write(f"Memory Usage: {metrics['memory_usage']:.2f} MB\n")
        f.write("\nCross-View Accuracy:\n")
        for angle, acc in metrics['cross_view_accuracy'].items():
            f.write(f"  {angle}°: {acc:.2f}%\n")
    
    logging.info(f"Evaluation completed on {args.dataset}!")
    logging.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
