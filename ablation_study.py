import torch
from torch.utils.data import DataLoader
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.datasets.dataset_factory import get_dataset
from config import ModelConfig
from evaluate import GaitEvaluator


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run ablation studies for gait recognition model')
    parser.add_argument('--dataset', type=str, default='casia_b', choices=['casia_b', 'oumvlp', 'gait3d'],
                        help='Dataset to use for evaluation')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/ablation',
                        help='Directory to save ablation study results')
    parser.add_argument('--components', type=str, nargs='+',
                        choices=['view_transform', 'attention', 'feature_pyramid', 'all'],
                        default=['all'],
                        help='Model components to ablate')
    args = parser.parse_args()
    
    # Setup logging
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.output_dir}/ablation_{args.dataset}.log"),
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
    
    # Run ablation studies
    results = {}
    
    # Baseline: Full model
    logging.info("Evaluating full model (baseline)...")
    model_config.use_view_transform = True
    model_config.use_attention = True
    model_config.use_feature_pyramid = True
    
    evaluator = GaitEvaluator(
        model_config=model_config,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir
    )
    
    baseline_metrics = evaluator.evaluate(test_loader, cross_view=True)
    results['Full Model'] = baseline_metrics
    
    # Ablation studies
    if 'all' in args.components or 'view_transform' in args.components:
        logging.info("Evaluating model without view transformation...")
        model_config.use_view_transform = False
        model_config.use_attention = True
        model_config.use_feature_pyramid = True
        
        evaluator = GaitEvaluator(
            model_config=model_config,
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir
        )
        
        metrics = evaluator.evaluate(test_loader, cross_view=True)
        results['w/o View Transform'] = metrics
    
    if 'all' in args.components or 'attention' in args.components:
        logging.info("Evaluating model without attention mechanism...")
        model_config.use_view_transform = True
        model_config.use_attention = False
        model_config.use_feature_pyramid = True
        
        evaluator = GaitEvaluator(
            model_config=model_config,
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir
        )
        
        metrics = evaluator.evaluate(test_loader, cross_view=True)
        results['w/o Attention'] = metrics
    
    if 'all' in args.components or 'feature_pyramid' in args.components:
        logging.info("Evaluating model without feature pyramid...")
        model_config.use_view_transform = True
        model_config.use_attention = True
        model_config.use_feature_pyramid = False
        
        evaluator = GaitEvaluator(
            model_config=model_config,
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir
        )
        
        metrics = evaluator.evaluate(test_loader, cross_view=True)
        results['w/o Feature Pyramid'] = metrics
    
    # Generate visualizations
    logging.info("Generating ablation study visualizations...")
    
    # Create directory for plots
    plots_dir = Path(args.output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Extract data for plotting
    components = list(results.keys())
    overall_acc = [results[c]['accuracy'] for c in components]
    
    # Handle cross-view metrics if available
    cross_45_acc = []
    cross_90_acc = []
    
    for c in components:
        # Check if we have any cross-view metrics
        cross_view_angles = [0, 18, 36, 45, 54, 72, 90, 108, 126, 144, 162, 180]
        cross_view_metrics = {}
        
        for angle in cross_view_angles:
            key = f'accuracy_{angle}deg'
            if key in results[c]:
                cross_view_metrics[angle] = results[c][key]
        
        # Calculate cross-45 and cross-90 accuracy
        if 45 in cross_view_metrics or 54 in cross_view_metrics:
            cross_45 = np.mean([cross_view_metrics.get(45, 0), cross_view_metrics.get(54, 0)])
        else:
            cross_45 = 0
        
        cross_90 = cross_view_metrics.get(90, 0)
        
        cross_45_acc.append(cross_45)
        cross_90_acc.append(cross_90)
    
    # Plot ablation study results
    x = np.arange(len(components))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width, overall_acc, width, label='Overall Accuracy')
    ax.bar(x, cross_45_acc, width, label='Cross-45° Accuracy')
    ax.bar(x + width, cross_90_acc, width, label='Cross-90° Accuracy')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Ablation Study Results')
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.legend()
    
    plt.savefig(str(plots_dir / "ablation_study.png"))
    
    # Save ablation results to file
    with open(str(Path(args.output_dir) / "ablation_results.txt"), "w") as f:
        f.write("Ablation Study Results\n")
        f.write("======================\n\n")
        
        # Calculate improvements relative to ablated versions
        baseline_acc = results['Full Model']['accuracy']
        
        # Get cross-view metrics for baseline if available
        cross_view_angles = [0, 18, 36, 45, 54, 72, 90, 108, 126, 144, 162, 180]
        baseline_cross_metrics = {}
        
        for angle in cross_view_angles:
            key = f'accuracy_{angle}deg'
            if key in results['Full Model']:
                baseline_cross_metrics[angle] = results['Full Model'][key]
        
        # Calculate baseline cross-45 and cross-90 accuracy
        if 45 in baseline_cross_metrics or 54 in baseline_cross_metrics:
            baseline_cross_45 = np.mean([baseline_cross_metrics.get(45, 0), baseline_cross_metrics.get(54, 0)])
        else:
            baseline_cross_45 = 0
        
        baseline_cross_90 = baseline_cross_metrics.get(90, 0) if 90 in baseline_cross_metrics else 0
        
        baseline_time = results['Full Model']['process_time']
        baseline_memory = results['Full Model']['memory_usage']
        
        # Write overall improvements
        f.write("Component Contributions:\n")
        for component in components[1:]:  # Skip the full model
            acc_diff = baseline_acc - results[component]['accuracy']
            
            # Get cross-view metrics for this component if available
            component_cross_metrics = {}
            
            for angle in cross_view_angles:
                key = f'accuracy_{angle}deg'
                if key in results[component]:
                    component_cross_metrics[angle] = results[component][key]
            
            # Calculate component cross-45 and cross-90 accuracy
            if 45 in component_cross_metrics or 54 in component_cross_metrics:
                component_cross_45 = np.mean([component_cross_metrics.get(45, 0), component_cross_metrics.get(54, 0)])
            else:
                component_cross_45 = 0
            
            component_cross_90 = component_cross_metrics.get(90, 0) if 90 in component_cross_metrics else 0
            
            cross_45_diff = baseline_cross_45 - component_cross_45
            cross_90_diff = baseline_cross_90 - component_cross_90
            
            time_diff = (results[component]['process_time'] - baseline_time) / results[component]['process_time'] * 100
            memory_diff = (results[component]['memory_usage'] - baseline_memory) / results[component]['memory_usage'] * 100
            
            f.write(f"\n{component}:\n")
            f.write(f"  Overall Accuracy: {'+' if acc_diff > 0 else ''}{acc_diff:.1f}%\n")
            f.write(f"  Cross-45° Accuracy: {'+' if cross_45_diff > 0 else ''}{cross_45_diff:.1f}%\n")
            f.write(f"  Cross-90° Accuracy: {'+' if cross_90_diff > 0 else ''}{cross_90_diff:.1f}%\n")
            f.write(f"  Processing Speed: {'+' if time_diff > 0 else ''}{time_diff:.1f}%\n")
            f.write(f"  Memory Efficiency: {'+' if memory_diff > 0 else ''}{memory_diff:.1f}%\n")
        
        # Write detailed results for each configuration
        f.write("\n\nDetailed Results:\n")
        for component, metrics in results.items():
            f.write(f"\n{component}:\n")
            f.write(f"  Overall Accuracy: {metrics['accuracy']:.2f}%\n")
            f.write(f"  Processing Time: {metrics['process_time']:.4f} seconds per sample\n")
            f.write(f"  Memory Usage: {metrics['memory_usage']:.2f} MB\n")
            
            # Write cross-view accuracy if available
            f.write("  Cross-View Accuracy:\n")
            for angle in cross_view_angles:
                key = f'accuracy_{angle}deg'
                if key in metrics:
                    f.write(f"    {angle}°: {metrics[key]:.2f}%\n")
    
    logging.info(f"Ablation study completed on {args.dataset}!")
    logging.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
