import torch
import numpy as np
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, List

def setup_logging(log_dir: str = "logs") -> None:
    """Setup logging configuration"""
    Path(log_dir).mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/training_{int(time.time())}.log"),
            logging.StreamHandler()
        ]
    )

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy"""
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    return correct / targets.size(0)

def save_checkpoint(
    state: dict,
    is_best: bool,
    checkpoint_dir: str,
    filename: str = "checkpoint.pth"
) -> None:
    """Save checkpoint to disk"""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(checkpoint_dir) / filename
    torch.save(state, filepath)
    if is_best:
        best_filepath = Path(checkpoint_dir) / "model_best.pth"
        torch.save(state, best_filepath)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 7, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

def optimize_cpu_performance():
    """Optimize PyTorch performance on CPU"""
    if not torch.cuda.is_available():
        torch.set_num_threads(torch.get_num_threads())
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
