__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

from functools import partial
from typing import Optional, Tuple, Dict, List, Union
import torch
from torch import nn


class MLPRNN(nn.Module):
    """Multi-Layer Perceptron Recurrent Neural Network Block."""

    def __init__(self, input_dim, hidden_dim=512, dropout=0.25):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(input_dim),
                                 nn.Linear(input_dim, hidden_dim, bias=False),
                                 nn.SiLU(),
                                 nn.Dropout(dropout))
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.downsample = nn.Sequential(nn.LayerNorm(hidden_dim * 2),
                                        nn.Dropout(dropout),
                                        nn.Linear(int(hidden_dim * 2), hidden_dim, bias=False),
                                        nn.SiLU())

    def forward(self, x):
        self.rnn.flatten_parameters()
        x = self.mlp(x)
        with torch.autocast(device_type='cuda', enabled=False):
            x = x.float().contiguous()
            x, _ = self.rnn(x)
        return self.downsample(x)


class RMREncoder(nn.Module):
    """Residual Multi-Layer Perceptron Recurrent Neural Network Encoder."""

    def __init__(self, input_dim=256, hidden_dim=512, num_layers=2, dropout=0.25):
        super().__init__()
        self.first_layer = MLPRNN(input_dim, hidden_dim, dropout)
        self.layers = nn.ModuleList([MLPRNN(hidden_dim, hidden_dim, dropout) for _ in range(num_layers - 1)])

    def forward(self, x):
        x = self.first_layer(x)
        for l in self.layers:
            x = l(x) + x
        return x


class RMRSyllableClassifier(nn.Module):
    """Residual Multi-Layer Perceptron Recurrent Neural Network Syllable Classifier."""

    def __init__(self,
                 num_classes: int = 0,
                 input_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 encoder_dropout: float = 0.25,
                 head_dropout: float = 0.25,
                 head_norm: bool = False):

        super().__init__()
        self.encoder = RMREncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                                  dropout=encoder_dropout)
        if head_norm:
            self.head = nn.Sequential(nn.LayerNorm(hidden_dim),
                                      nn.Dropout(head_dropout),
                                      nn.Linear(hidden_dim, num_classes))
        else:
            self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)
