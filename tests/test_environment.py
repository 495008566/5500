import torch
import torchvision
import numpy as np
import cv2
import sklearn
import matplotlib.pyplot as plt
from tqdm import tqdm

def test_environment():
    print("Testing environment setup...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    
    # Test CUDA tensor operations if available
    if torch.cuda.is_available():
        print("\nTesting CUDA tensor operations...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        z = torch.matmul(x, y)
        end_time.record()
        torch.cuda.synchronize()
        print(f"Matrix multiplication time: {start_time.elapsed_time(end_time):.2f} ms")

if __name__ == "__main__":
    test_environment()
