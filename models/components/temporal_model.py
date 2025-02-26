import torch
import torch.nn as nn
from typing import Tuple

class TemporalModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super(TemporalModel, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights for better convergence
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, sequence_length, input_dim)
        
        # GRU forward pass
        outputs, _ = self.gru(x)  # outputs: (batch_size, sequence_length, hidden_dim * 2)
        
        # Attention mechanism
        attention_weights = self.attention(outputs)  # (batch_size, sequence_length, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum of the outputs
        weighted_outputs = outputs * attention_weights
        temporal_features = torch.sum(weighted_outputs, dim=1)
        
        return temporal_features, attention_weights
    
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
