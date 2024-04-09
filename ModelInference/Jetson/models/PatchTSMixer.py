#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import PatchTSMixerConfig, PatchTSMixerForTimeSeriesClassification

class PatchTSMixer(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, context_length=30):
        super(PatchTSMixer, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.config = PatchTSMixerConfig(num_input_channels=input_dim, 
                                         d_model=hidden_dim,
                                         patch_len=8,
                                         #prediction_length=1,
                                         num_targets=output_dim,
                                         num_layers=layer_dim,
                                         dropout=dropout_prob,
                                         context_length=context_length,
                                         return_dict=True)
        
        self.PatchTSMixerClassifier = PatchTSMixerForTimeSeriesClassification(self.config)

        # Fully connected layer
        #self.fc1 = nn.Linear(hidden_dim, int(hidden_dim/2))
        #self.fc2 = nn.Linear(int(hidden_dim/2), output_dim)


    def forward(self, x):
        out = self.PatchTSMixerClassifier(x)
        return out["prediction_outputs"]