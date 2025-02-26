import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
from tqdm import tqdm
import time
import psutil
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

from models.gait_model import GaitModel
from config import ModelConfig
from utils import setup_logging

class GaitEvaluator:
    """Evaluator for gait recognition model"""
    
    def __init__(self,
                 model_config: ModelConfig,
                 checkpoint_path: str,
                 output_dir: str):
        """
        Initialize evaluator
        
        Args:
            model_config: Model configuration
            checkpoint_path: Path to model checkpoint
            output_dir: Directory to save evaluation results
        """
        self.config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Setup logging
        setup_logging(str(self.output_dir))
        
        # Initialize metrics
        self.metrics = {
            'accuracy': [],
            'cross_view_accuracy': {},
            'processing_time': [],
            'memory_usage': []
        }
    
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load model from checkpoint"""
        try:
            # Try loading with standard format
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model = GaitModel(self.config).to(self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume checkpoint is the state dict directly
                model.load_state_dict(checkpoint)
                
            model.eval()
            return model
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            # Create a new model for evaluation without loading weights
            logging.warning("Creating new model without loading weights")
            model = GaitModel(self.config).to(self.device)
            model.eval()
            return model
    
    def _measure_memory(self) -> float:
        """Measure current memory usage"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    @torch.no_grad()
    def evaluate_batch(self,
                      data: Dict[str, torch.Tensor],
                      targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate single batch"""
        batch_size = data['image'].size(0)
        
        # Measure processing time and memory
        start_time = time.time()
        mem_before = self._measure_memory()
        
        # Forward pass
        outputs = self.model(data['image'], data.get('view_angle'))
        
        # Measure processing time and memory
        process_time = (time.time() - start_time) / batch_size
        mem_used = self._measure_memory() - mem_before
        
        # Compute accuracy
        pred = outputs['logits'].max(1)[1]
        accuracy = (pred == targets['identity']).float().mean().item()
        
        return {
            'accuracy': accuracy,
            'process_time': process_time,
            'memory_used': mem_used
        }
    
    def evaluate(self,
                test_loader: DataLoader,
                cross_view: bool = False) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test data loader
            cross_view: Whether to evaluate cross-view performance
            
        Returns:
            Dict containing evaluation metrics
        """
        # Initialize metrics
        all_preds = []
        all_targets = []
        accuracies = []
        process_times = []
        memory_usages = []
        
        # Cross-view metrics
        if cross_view:
            cross_view_acc = {angle: [] for angle in range(0, 181, 45)}
            cross_view_preds = {angle: [] for angle in range(0, 181, 45)}
            cross_view_targets = {angle: [] for angle in range(0, 181, 45)}
        
        # Memory usage before evaluation
        mem_before = self._measure_memory()
        
        # Evaluation loop
        pbar = tqdm(test_loader, desc='Evaluating')
        for batch_idx, (data, targets) in enumerate(pbar):
            # Move to device
            # Handle both tensor and dictionary data formats
            if isinstance(data, dict):
                data = {k: v.to(self.device) for k, v in data.items()}
                targets = {k: v.to(self.device) for k, v in targets.items()}
                images = data['image']
                view_angles = data.get('view_angle')
            else:
                # For tensor data (simplified dataset)
                images = data.to(self.device)
                targets = targets.to(self.device)
                view_angles = None
            
            # Evaluate batch
            start_time = time.time()
            features, logits = self.model(images, view_angles)
            batch_time = (time.time() - start_time) / images.size(0)
            
            # Get predictions
            pred_logits = logits
            pred_probs = torch.softmax(pred_logits, dim=1)
            pred_labels = pred_logits.max(1)[1]
            
            # Update metrics
            all_preds.append(pred_probs.detach().cpu().numpy())
            
            # Handle both tensor and dictionary targets
            if isinstance(targets, dict):
                target_ids = targets['identity']
                all_targets.append(target_ids.cpu().numpy())
            else:
                target_ids = targets
                all_targets.append(target_ids.cpu().numpy())
            
            # Calculate accuracy
            accuracies.append(
                (pred_labels == target_ids).float().mean().item()
            )
            process_times.append(batch_time)
            
            # Update cross-view metrics
            if cross_view and isinstance(data, dict) and 'view_angle' in data:
                angles = data['view_angle'].cpu().numpy()
                batch_preds = pred_labels.cpu().numpy()
                batch_targets = target_ids.cpu().numpy()
                
                for angle in cross_view_acc.keys():
                    mask = angles == angle
                    if mask.any():
                        cross_view_acc[angle].append(
                            (batch_preds[mask] == batch_targets[mask]).mean()
                        )
                        cross_view_preds[angle].extend(pred_probs[mask].cpu().numpy())
                        cross_view_targets[angle].extend(batch_targets[mask])
            
            # Update progress bar
            pbar.set_postfix({
                'acc': f"{np.mean(accuracies):.4f}",
                'time': f"{batch_time:.4f}s"
            })
        
        # Concatenate predictions and targets
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # Handle NaN values
        valid_indices = ~np.isnan(all_preds).any(axis=1)
        if not np.all(valid_indices):
            logging.warning(f"Found {np.sum(~valid_indices)} samples with NaN values. Filtering them out.")
            all_preds = all_preds[valid_indices]
            all_targets = all_targets[valid_indices]
        
        # Compute memory usage
        mem_used = self._measure_memory() - mem_before
        
        # Compute overall metrics
        metrics: Dict[str, float] = {
            'accuracy': float(np.mean(accuracies)) if accuracies else 0.0,
            'process_time': float(np.mean(process_times)) if process_times else 0.0,
            'memory_usage': float(mem_used),
            'mAP': float(average_precision_score(
                all_targets,
                all_preds,
                average='macro'
            )) if len(all_preds) > 0 and len(all_targets) > 0 and not np.isnan(all_preds).any() else 0.0
        }
        
        # Compute cross-view metrics
        if cross_view:
            for angle in cross_view_acc.keys():
                if cross_view_acc[angle]:
                    metrics[f'accuracy_{angle}deg'] = float(np.mean(cross_view_acc[angle]))
                    if cross_view_preds[angle]:
                        try:
                            metrics[f'mAP_{angle}deg'] = float(average_precision_score(
                                cross_view_targets[angle],
                                cross_view_preds[angle],
                                average='macro'
                            ))
                        except Exception as e:
                            logging.warning(f"Error calculating mAP for {angle}deg: {str(e)}")
                            metrics[f'mAP_{angle}deg'] = 0.0
        
        return metrics
    
    def evaluate_ablation(self,
                         test_loader: DataLoader,
                         components: List[str],
                         cross_view: bool = True) -> Dict[str, float]:
        """
        Perform ablation study
        
        Args:
            test_loader: Test data loader
            components: List of components to ablate
            cross_view: Whether to evaluate cross-view performance
            
        Returns:
            Dict containing ablation results
        """
        results: Dict[str, float] = {}
        
        # Evaluate full model
        logging.info("Evaluating full model...")
        full_metrics = self.evaluate(test_loader, cross_view=cross_view)
        
        # Store full model metrics
        for metric, value in full_metrics.items():
            results[f'full_model_{metric}'] = float(value)
        
        # Evaluate with each component removed
        for component in components:
            logging.info(f"Evaluating without {component}...")
            
            # Temporarily disable component
            if hasattr(self.model, component):
                original_module = getattr(self.model, component)
                setattr(self.model, component, nn.Identity())
            
            # Evaluate
            metrics = self.evaluate(test_loader, cross_view=cross_view)
            
            # Store metrics
            for metric, value in metrics.items():
                results[f'without_{component}_{metric}'] = float(value)
            
            # Restore component
            if hasattr(self.model, component):
                setattr(self.model, component, original_module)
            
            # Calculate performance drop
            acc_drop = results['full_model_accuracy'] - metrics['accuracy']
            results[f'{component}_contribution'] = float(acc_drop)
            
            if cross_view:
                for angle in [45, 90]:
                    key = f'accuracy_{angle}deg'
                    if key in metrics and f'full_model_{key}' in results:
                        cross_view_drop = results[f'full_model_{key}'] - metrics[key]
                        results[f'{component}_contribution_{angle}deg'] = float(cross_view_drop)
        
        return results
    
    def plot_results(self, metrics: Dict[str, float]):
        """Plot evaluation results"""
        # Accuracy plot
        plt.figure(figsize=(10, 5))
        
        # Cross-view accuracy
        angles = []
        accuracies = []
        for key, value in metrics.items():
            if key.startswith('accuracy_') and key.endswith('deg'):
                angle = int(key.split('_')[1][:-3])
                angles.append(angle)
                accuracies.append(value)
        
        if angles:
            plt.subplot(1, 2, 1)
            plt.plot(angles, accuracies, 'b-o')
            plt.xlabel('View Angle (degrees)')
            plt.ylabel('Accuracy')
            plt.title('Cross-View Recognition Accuracy')
            plt.grid(True)
        
        # Processing time distribution
        if 'process_time' in metrics:
            plt.subplot(1, 2, 2)
            plt.hist(self.metrics['processing_time'], bins=20)
            plt.xlabel('Processing Time (s)')
            plt.ylabel('Frequency')
            plt.title('Processing Time Distribution')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'evaluation_results.png')
        plt.close()
    
    def save_results(self, metrics: Dict[str, float]):
        """Save evaluation results"""
        import json
        
        # Save metrics
        metrics_file = self.output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Log results
        logging.info("\nEvaluation Results:")
        logging.info(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"Average Processing Time: {metrics['process_time']:.4f}s")
        logging.info(f"Memory Usage: {metrics['memory_usage']:.2f}MB")
        
        if 'accuracy_90deg' in metrics:
            logging.info(f"90° Cross-View Accuracy: {metrics['accuracy_90deg']:.4f}")
        if 'accuracy_45deg' in metrics:
            logging.info(f"45° Cross-View Accuracy: {metrics['accuracy_45deg']:.4f}")

if __name__ == "__main__":
    # Load configurations
    model_config = ModelConfig()
    
    # Create evaluator
    evaluator = GaitEvaluator(
        model_config=model_config,
        checkpoint_path="checkpoints/model_best.pth",
        output_dir="evaluation_results"
    )
    
    # Create dummy test loader
    test_loader = DataLoader(
        dataset=None,  # Add your dataset here
        batch_size=model_config.batch_size,
        shuffle=False,
        num_workers=model_config.num_workers
    )
    
    # Evaluate model
    metrics = evaluator.evaluate(test_loader, cross_view=True)
    
    # Perform ablation study
    ablation_components = ['view_transform', 'attention_module', 'feature_pyramid']
    ablation_results = evaluator.evaluate_ablation(test_loader, ablation_components)
    
    # Plot and save results
    evaluator.plot_results(metrics)
    evaluator.save_results(metrics)
