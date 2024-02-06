#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

class CNN2DGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(CNN2DGRUModel, self).__init__()

        self.dropInput = nn.Dropout(p=0.1)
        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # in_channels is the number of features (modalities/channels)
        # out_channels is the number of kernels
        # expected input size (N_batchsize，C_in_channel, Height, Width) 
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=input_dim, \
                               kernel_size=(3, input_dim), stride=1, padding=(1,0), bias=False)
        # expected output size (N_batchsize，C_out_channel, Height_out, Width_out)
        ## Notes
        # 1. We could construct the input to be 
        #    (N, C=1, Height=lagWindowSize, Width=features=input_dim) 
        # 2. As we set:
        #       kernel_size=(2, input_dim), stride=2, 
        #       padding=(2,0),out_channels=input_dim*2
        #    the output dimension = (N, C=input_dim*2, Height=lagWindowSize/2, Width=1)
        
        self.bn0 = nn.BatchNorm2d(input_dim)
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=input_dim, \
                               kernel_size=(3, input_dim), stride=1, padding=(1,0), bias=False)
        self.bn1 = nn.BatchNorm2d(input_dim)
        
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=input_dim, \
                               kernel_size=(3, input_dim), stride=1, padding=(1,0), bias=False)
        self.bn2 = nn.BatchNorm2d(input_dim)

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, 
            batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        # Convert input from (N,Length,Channel) to (N,1,Lenght, Channel)
        #print('='*10)
        #print(x.shape)
         
        x = torch.unsqueeze(x, dim=1)
        x = self.dropInput(x)
        #print(x.shape)
        x = F.relu(self.bn0(self.conv0(x)))
        #print(x.shape)
        x = x.permute(0,3,2,1)
        #print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        #print(x.shape)
        x = x.permute(0,3,2,1)
        #print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        #print(x.shape)
        # Convert output dimension from
        #   (N, C=input_dim*2, Height=lagWindowSize, Width=1)
        # back to 
        #   (N,Length=lagWindowSize,Channel=input_dim*2)
        x = x.squeeze(dim=3).permute(0,2,1)
        #print(x.shape)

        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return self.sigmoid(out)
