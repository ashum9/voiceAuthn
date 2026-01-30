"""
Custom module for ECAPA-TDNN speaker embeddings.
This provides the model architecture for SpeechBrain inference.
"""

import torch
import torch.nn as nn
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from speechbrain.lobes.features import Fbank


class Classifier(nn.Module):
    """Simple classifier for speaker identification."""
    
    def __init__(self, input_size, out_neurons, lin_blocks=1, lin_neurons=192, dropout=0.5):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        for i in range(lin_blocks):
            self.blocks.append(nn.Linear(input_size if i == 0 else lin_neurons, lin_neurons))
            self.blocks.append(nn.BatchNorm1d(lin_neurons))
            self.blocks.append(nn.ReLU())
            
        self.output = nn.Linear(lin_neurons, out_neurons)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        x = self.output(x)
        return self.softmax(x)
