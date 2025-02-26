import os
import time
import torch
import numpy as np
from models import GaitModel, FeatureExtractor, TemporalModel
from config import ModelConfig, TrainingConfig
from utils import optimize_cpu_performance, setup_logging

def verify_setup():
    print("=== Environment Verification ===")
    
    # Test CPU optimizations
    optimize_cpu_performance()
    print(f"Number of CPU threads: {torch.get_num_threads()}")
    
    # Initialize configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Test model instantiation
    print("\nTesting model initialization...")
    try:
        gait_model = GaitModel(model_config)
        feature_extractor = FeatureExtractor(model_config)
        temporal_model = TemporalModel(input_dim=256)
        print("✓ Model initialization successful")
    except Exception as e:
        print(f"✗ Model initialization failed: {str(e)}")
        return False
    
    # Test forward pass with dummy data
    print("\nTesting forward pass...")
    try:
        batch_size = 4
        seq_length = 30
        channels = 3
        height, width = model_config.img_size
        
        # Create dummy input
        x = torch.randn(batch_size, channels, height, width)
        
        # Test feature extraction
        start_time = time.time()
        features = feature_extractor(x)
        feature_time = time.time() - start_time
        print(f"Feature extraction time: {feature_time:.3f}s")
        
        # Test temporal processing
        seq_features = features.unsqueeze(1).repeat(1, seq_length, 1)
        start_time = time.time()
        temporal_features, _ = temporal_model(seq_features)
        temporal_time = time.time() - start_time
        print(f"Temporal processing time: {temporal_time:.3f}s")
        
        # Test full model
        start_time = time.time()
        logits, embeddings = gait_model(x)
        model_time = time.time() - start_time
        print(f"Full model forward pass time: {model_time:.3f}s")
        
        print("✓ Forward pass successful")
        print(f"\nModel output shapes:")
        print(f"Feature extractor output: {features.shape}")
        print(f"Temporal model output: {temporal_features.shape}")
        print(f"Final logits: {logits.shape}")
        print(f"Final embeddings: {embeddings.shape}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {str(e)}")
        return False
    
    print("\n=== Setup Verification Complete ===")
    return True

if __name__ == "__main__":
    verify_setup()
