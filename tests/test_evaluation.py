import unittest
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import GaitEvaluator
from src.config import ModelConfig
from src.baselines.gei_baseline import GEIBaseline
from src.visualization import FeatureVisualizer

class TestEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        cls.model_config = ModelConfig()
        cls.output_dir = Path("test_outputs")
        cls.output_dir.mkdir(exist_ok=True)
        
        # Create evaluator
        cls.evaluator = GaitEvaluator(
            model_config=cls.model_config,
            checkpoint_path=None,  # Will be mocked
            output_dir=str(cls.output_dir)
        )
        
        # Create GEI baseline
        cls.baseline = GEIBaseline(
            num_classes=cls.model_config.num_classes,
            device='cpu'
        )
        
        # Create visualizer
        cls.visualizer = FeatureVisualizer(str(cls.output_dir))
    
    def test_performance_metrics(self):
        """Test performance metrics meet requirements"""
        # Create dummy data
        batch_size = 4
        sequence_length = 30
        
        sequences = [
            [np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
             for _ in range(sequence_length)]
            for _ in range(batch_size)
        ]
        labels = [i % self.model_config.num_classes for i in range(batch_size)]
        view_angles = [0, 45, 90, 45]
        
        # Test baseline
        baseline_metrics = self.baseline.evaluate(
            sequences,
            labels,
            cross_view=True,
            view_angles=view_angles
        )
        
        # Verify metrics
        self.assertGreaterEqual(baseline_metrics['accuracy'], 0.70)
        if 'accuracy_45deg' in baseline_metrics:
            self.assertGreaterEqual(baseline_metrics['accuracy_45deg'], 0.80)
        if 'accuracy_90deg' in baseline_metrics:
            self.assertGreaterEqual(baseline_metrics['accuracy_90deg'], 0.70)
        self.assertLess(baseline_metrics['process_time'], 0.14)
    
    def test_lighting_conditions(self):
        """Test performance under different lighting conditions"""
        # Create dummy data with varying lighting
        sequence_length = 30
        sequences = {
            'normal': [np.random.randint(0, 255, (128, 64, 3)) for _ in range(sequence_length)],
            'dark': [np.random.randint(0, 128, (128, 64, 3)) for _ in range(sequence_length)],
            'bright': [np.random.randint(128, 255, (128, 64, 3)) for _ in range(sequence_length)]
        }
        labels = [0]  # Single sequence test
        
        # Test each condition
        for condition, sequence in sequences.items():
            metrics = self.baseline.evaluate([sequence], labels)
            self.assertGreater(metrics['accuracy'], 0.70, f"Failed under {condition} lighting")
    
    def test_occlusion_robustness(self):
        """Test performance with occlusions"""
        sequence_length = 30
        sequence = [np.random.randint(0, 255, (128, 64, 3)) for _ in range(sequence_length)]
        
        # Add 30% occlusion
        for frame in sequence:
            h, w = frame.shape[:2]
            occl_h = int(h * 0.3)
            occl_w = int(w * 0.3)
            x = np.random.randint(0, w - occl_w)
            y = np.random.randint(0, h - occl_h)
            frame[y:y+occl_h, x:x+occl_w] = 0
        
        metrics = self.baseline.evaluate([sequence], [0])
        self.assertGreater(metrics['accuracy'], 0.70, "Failed with 30% occlusion")
    
    def test_visualization(self):
        """Test visualization functions"""
        # Create dummy data
        image = torch.randn(3, 128, 64)
        attention = {
            'spatial': torch.rand(32, 32),
            'channel': torch.rand(64, 16, 8)
        }
        features = torch.rand(64, 32, 16)
        embeddings = torch.randn(100, 256)
        labels = torch.randint(0, 10, (100,))
        
        # Test all visualization functions
        self.visualizer.plot_attention_maps(image, attention, 'test_attention')
        self.visualizer.plot_feature_maps(features, 'test_features')
        self.visualizer.plot_embedding_space(embeddings, labels, 'test_embeddings')
        
        # Test cross-view matrix
        accuracies = {
            0: {0: 0.9, 45: 0.8, 90: 0.7},
            45: {0: 0.8, 45: 0.85, 90: 0.75},
            90: {0: 0.7, 45: 0.75, 90: 0.8}
        }
        self.visualizer.plot_cross_view_matrix(accuracies, 'test_cross_view')
        
        # Verify files were created
        self.assertTrue((self.output_dir / 'test_attention_attention.png').exists())
        self.assertTrue((self.output_dir / 'test_features_features.png').exists())
        self.assertTrue((self.output_dir / 'test_embeddings_embeddings.png').exists())
        self.assertTrue((self.output_dir / 'test_cross_view_cross_view.png').exists())

if __name__ == '__main__':
    unittest.main()
