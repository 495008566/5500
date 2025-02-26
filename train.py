import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

from models.gait_model import GaitModel
from models.loss import GaitLoss
from config import ModelConfig, TrainingConfig
from utils import setup_logging, AverageMeter, save_checkpoint, EarlyStopping

class GaitTrainer:
    """Trainer class for gait recognition model"""
    
    def __init__(self,
                 model_config: ModelConfig,
                 train_config: TrainingConfig,
                 checkpoint_dir: str,
                 log_dir: str):
        """
        Initialize trainer
        
        Args:
            model_config: Model configuration
            train_config: Training configuration
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.config = model_config
        self.train_config = train_config
        self.device = torch.device(model_config.device)
        
        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(log_dir)
        self.writer = SummaryWriter(log_dir)
        
        # Create model
        self.model = GaitModel(model_config).to(self.device)
        
        # Loss function
        self.criterion = GaitLoss(
            lambda_id=train_config.lambda_ce,
            lambda_triplet=train_config.lambda_triplet,
            lambda_pose=0.1
        )
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=7)
        
        # Initialize best metrics
        self.best_acc = 0.0
        self.best_loss = float('inf')
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer"""
        if self.train_config.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:  # SGD
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        if self.train_config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=1e-6
            )
        else:  # StepLR
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        # Metrics
        losses = AverageMeter()
        id_losses = AverageMeter()
        triplet_losses = AverageMeter()
        pose_losses = AverageMeter()
        accuracies = AverageMeter()
        
        # Training loop
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, targets) in enumerate(pbar):
            # Apply augmentation and move to device
            if isinstance(data, dict):
                data = {k: v.to(self.device) for k, v in data.items()}
            else:
                data = data.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Forward pass
            outputs = self.model(data['image'], data.get('view_angle'))
            
            # Compute loss
            loss_dict = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            self.optimizer.step()
            
            # Update metrics
            batch_size = data['image'].size(0)
            losses.update(loss_dict['total_loss'].item(), batch_size)
            id_losses.update(loss_dict['id_loss'].item(), batch_size)
            triplet_losses.update(loss_dict['triplet_loss'].item(), batch_size)
            pose_losses.update(loss_dict['pose_loss'].item(), batch_size)
            
            # Compute accuracy
            pred = outputs['logits'].max(1)[1]
            acc = (pred == targets['identity']).float().mean()
            accuracies.update(acc.item(), batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses.avg:.4f}",
                'acc': f"{accuracies.avg:.4f}"
            })
        
        return {
            'loss': losses.avg,
            'id_loss': id_losses.avg,
            'triplet_loss': triplet_losses.avg,
            'pose_loss': pose_losses.avg,
            'accuracy': accuracies.avg
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        # Metrics
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        # Validation loop
        pbar = tqdm(val_loader, desc='Validation')
        for batch_idx, (data, targets) in enumerate(pbar):
            # Move to device
            data = {k: v.to(self.device) for k, v in data.items()}
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Forward pass
            outputs = self.model(data['image'], data.get('view_angle'))
            
            # Compute loss
            loss_dict = self.criterion(outputs, targets)
            
            # Update metrics
            batch_size = data['image'].size(0)
            losses.update(loss_dict['total_loss'].item(), batch_size)
            
            # Compute accuracy
            pred = outputs['logits'].max(1)[1]
            acc = (pred == targets['identity']).float().mean()
            accuracies.update(acc.item(), batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses.avg:.4f}",
                'acc': f"{accuracies.avg:.4f}"
            })
        
        return {
            'loss': losses.avg,
            'accuracy': accuracies.avg
        }
    
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              num_epochs: Optional[int] = None):
        """
        Train model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (if None, use config)
        """
        num_epochs = num_epochs or self.config.num_epochs
        
        for epoch in range(num_epochs):
            logging.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Log training metrics
            for name, value in train_metrics.items():
                self.writer.add_scalar(f'train/{name}', value, epoch)
            
            # Validate
            if val_loader is not None and (epoch + 1) % self.train_config.val_frequency == 0:
                val_metrics = self.validate(val_loader)
                
                # Log validation metrics
                for name, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{name}', value, epoch)
                
                # Check for improvement
                val_loss = val_metrics['loss']
                val_acc = val_metrics['accuracy']
                
                is_best = val_acc > self.best_acc
                self.best_acc = max(val_acc, self.best_acc)
                
                # Save checkpoint
                if (epoch + 1) % self.train_config.save_frequency == 0 or is_best:
                    state = {
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        'best_acc': self.best_acc,
                        'config': self.config
                    }
                    
                    save_checkpoint(
                        state,
                        is_best,
                        str(self.checkpoint_dir),
                        f"checkpoint_{epoch+1}.pth"
                    )
                
                # Early stopping
                if self.early_stopping(val_loss):
                    logging.info("Early stopping triggered")
                    break
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            self.writer.add_scalar('train/lr', current_lr, epoch)
            
            # Log epoch summary
            logging.info(
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"Train Acc: {train_metrics['accuracy']:.4f}"
            )
            if val_loader is not None:
                logging.info(
                    f"Val Loss: {val_metrics['loss']:.4f} "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )

if __name__ == "__main__":
    # Load configurations
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
    # Create trainer
    trainer = GaitTrainer(
        model_config=model_config,
        train_config=train_config,
        checkpoint_dir="checkpoints",
        log_dir="logs"
    )
    
    # Create dummy data loaders for testing
    train_loader = DataLoader(
        dataset=None,  # Add your dataset here
        batch_size=model_config.batch_size,
        shuffle=True,
        num_workers=model_config.num_workers
    )
    
    val_loader = DataLoader(
        dataset=None,  # Add your dataset here
        batch_size=model_config.batch_size,
        shuffle=False,
        num_workers=model_config.num_workers
    )
    
    # Train model
    trainer.train(train_loader, val_loader)
