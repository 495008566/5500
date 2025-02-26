import torch
from torch.utils.data import DataLoader
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from preprocessing.datasets.dataset_factory import get_dataset
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
    test_dataset = get_dataset(args.dataset, split='test')
    
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
    
    # Plot cross-view accuracy if available
    # Check if we have any cross-view metrics
    cross_view_angles = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
    cross_view_metrics = {}
    
    for angle in cross_view_angles:
        key = f'accuracy_{angle}deg'
        if key in metrics:
            cross_view_metrics[angle] = metrics[key]
    
    if cross_view_metrics:
        angles = list(cross_view_metrics.keys())
        accuracies = list(cross_view_metrics.values())
        
        plt.figure(figsize=(10, 6))
        plt.plot(angles, accuracies, 'b-o')
        plt.xlabel('View Angle (degrees)')
        plt.ylabel('Rank-1 Accuracy (%)')
        plt.title('Cross-View Gait Recognition Accuracy')
        plt.grid(True)
        plt.savefig(str(plots_dir / "cross_view_accuracy.png"))
    else:
        logging.warning("Cross-view accuracy data not available. Skipping cross-view plot.")
    
    # Save metrics to file
    with open(str(Path(args.output_dir) / "metrics.txt"), "w") as f:
        f.write(f"Overall Accuracy: {metrics['accuracy']:.2f}%\n")
        f.write(f"Processing Time: {metrics['process_time']:.4f} seconds per sample\n")
        f.write(f"Memory Usage: {metrics['memory_usage']:.2f} MB\n")
        
        # Write cross-view accuracy if available
        if cross_view_metrics:
            f.write("\nCross-View Accuracy:\n")
            for angle, acc in cross_view_metrics.items():
                f.write(f"  {angle}°: {acc:.2f}%\n")
        else:
            f.write("\nCross-View Accuracy: Not available\n")
    
    logging.info(f"Evaluation completed on {args.dataset}!")
    logging.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
