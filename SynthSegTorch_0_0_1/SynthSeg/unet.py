"""PyTorch implementation of the UNet model for SynthSeg.

This module contains the PyTorch implementation of the UNet model
used in the SynthSeg framework for medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UNet(nn.Module):
    """PyTorch implementation of a UNet model for medical image segmentation.
    
    This class implements a UNet architecture with configurable depth, feature counts,
    and other parameters. It supports 2D and 3D inputs and includes options for
    residual connections, batch normalization, and different activation functions.
    
    Args:
        input_shape (tuple): Shape of the input tensor (excluding batch dimension).
        nb_labels (int): Number of output channels/labels.
        nb_levels (int): Number of levels in the UNet (depth of the network).
        nb_conv_per_level (int): Number of convolutions per level.
        conv_size (int): Size of the convolution kernels.
        nb_features (int): Number of features in the first level.
        feat_mult (int): Multiplier for the number of features at each level.
        activation (str): Activation function to use ('relu', 'elu', etc.).
        use_residuals (bool): Whether to use residual connections.
        batch_norm (bool): Whether to use batch normalization.
        conv_dropout (float): Dropout rate for convolutional layers.
    """
    
    def __init__(self,
                 input_shape,
                 nb_labels,
                 nb_levels=5,
                 nb_conv_per_level=2,
                 conv_size=3,
                 nb_features=16,
                 feat_mult=2,
                 activation='elu',
                 use_residuals=False,
                 batch_norm=False,
                 conv_dropout=0.0):
        super(UNet, self).__init__()
        
        # Store parameters
        self.input_shape = input_shape
        self.nb_labels = nb_labels
        self.nb_levels = nb_levels
        self.nb_conv_per_level = nb_conv_per_level
        self.conv_size = conv_size
        self.nb_features = nb_features
        self.feat_mult = feat_mult
        self.use_residuals = use_residuals
        self.batch_norm = batch_norm
        self.conv_dropout = conv_dropout
        
        # Determine dimensionality (2D or 3D)
        self.ndims = len(input_shape) - 1  # Subtract 1 for channel dimension
        
        # Set activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Create encoder and decoder layers
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.encoder_residual_layers = nn.ModuleList() if use_residuals else None
        self.decoder_residual_layers = nn.ModuleList() if use_residuals else None
        self.encoder_bn_layers = nn.ModuleList() if batch_norm else None
        self.decoder_bn_layers = nn.ModuleList() if batch_norm else None
        self.pool_layers = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()
        
        # Create encoder path
        for level in range(nb_levels):
            level_layers = nn.ModuleList()
            nb_lvl_feats = nb_features * (feat_mult ** level)
            
            # Create convolutional layers for this level
            for conv in range(nb_conv_per_level):
                if conv == 0 and level == 0:
                    # First convolution at first level takes input channels
                    in_channels = input_shape[-1]
                elif conv == 0:
                    # First convolution at other levels takes features from previous level
                    in_channels = nb_features * (feat_mult ** (level - 1))
                else:
                    # Subsequent convolutions take features from previous convolution
                    in_channels = nb_lvl_feats
                
                # Create convolutional layer
                if self.ndims == 2:
                    layer = nn.Conv2d(in_channels, nb_lvl_feats, conv_size, padding=conv_size//2)
                else:  # 3D
                    layer = nn.Conv3d(in_channels, nb_lvl_feats, conv_size, padding=conv_size//2)
                
                level_layers.append(layer)
            
            self.encoder_layers.append(level_layers)
            
            # Create residual connection layers if needed
            if use_residuals:
                if self.ndims == 2:
                    res_layer = nn.Conv2d(input_shape[-1] if level == 0 else nb_features * (feat_mult ** (level - 1)),
                                         nb_lvl_feats, 1)
                else:  # 3D
                    res_layer = nn.Conv3d(input_shape[-1] if level == 0 else nb_features * (feat_mult ** (level - 1)),
                                         nb_lvl_feats, 1)
                self.encoder_residual_layers.append(res_layer)
            
            # Create batch normalization layers if needed
            if batch_norm:
                if self.ndims == 2:
                    bn_layer = nn.BatchNorm2d(nb_lvl_feats)
                else:  # 3D
                    bn_layer = nn.BatchNorm3d(nb_lvl_feats)
                self.encoder_bn_layers.append(bn_layer)
            
            # Create pooling layer (except for last level)
            if level < nb_levels - 1:
                if self.ndims == 2:
                    pool_layer = nn.MaxPool2d(2)
                else:  # 3D
                    pool_layer = nn.MaxPool3d(2)
                self.pool_layers.append(pool_layer)
        
        # Create decoder path
        for level in range(nb_levels - 2, -1, -1):
            level_layers = nn.ModuleList()
            nb_lvl_feats = nb_features * (feat_mult ** level)
            
            # Create upconvolution layer
            if self.ndims == 2:
                upconv_layer = nn.ConvTranspose2d(nb_features * (feat_mult ** (level + 1)),
                                                nb_lvl_feats, 2, stride=2)
            else:  # 3D
                upconv_layer = nn.ConvTranspose3d(nb_features * (feat_mult ** (level + 1)),
                                                nb_lvl_feats, 2, stride=2)
            self.upconv_layers.append(upconv_layer)
            
            # Create convolutional layers for this level
            for conv in range(nb_conv_per_level):
                if conv == 0:
                    # First convolution takes concatenated features
                    in_channels = nb_lvl_feats * 2  # Features from upconv + skip connection
                else:
                    # Subsequent convolutions take features from previous convolution
                    in_channels = nb_lvl_feats
                
                # Create convolutional layer
                if self.ndims == 2:
                    layer = nn.Conv2d(in_channels, nb_lvl_feats, conv_size, padding=conv_size//2)
                else:  # 3D
                    layer = nn.Conv3d(in_channels, nb_lvl_feats, conv_size, padding=conv_size//2)
                
                level_layers.append(layer)
            
            self.decoder_layers.append(level_layers)
            
            # Create residual connection layers if needed
            if use_residuals:
                if self.ndims == 2:
                    res_layer = nn.Conv2d(nb_lvl_feats * 2, nb_lvl_feats, 1)
                else:  # 3D
                    res_layer = nn.Conv3d(nb_lvl_feats * 2, nb_lvl_feats, 1)
                self.decoder_residual_layers.append(res_layer)
            
            # Create batch normalization layers if needed
            if batch_norm:
                if self.ndims == 2:
                    bn_layer = nn.BatchNorm2d(nb_lvl_feats)
                else:  # 3D
                    bn_layer = nn.BatchNorm3d(nb_lvl_feats)
                self.decoder_bn_layers.append(bn_layer)
        
        # Create final prediction layer
        if self.ndims == 2:
            self.final_conv = nn.Conv2d(nb_features, nb_labels, 1)
        else:  # 3D
            self.final_conv = nn.Conv3d(nb_features, nb_labels, 1)
    
    def forward(self, x):
        """Forward pass through the UNet.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, *spatial_dims, channels]
                Note: PyTorch expects [batch_size, channels, *spatial_dims], so we'll transpose.
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, *spatial_dims, nb_labels]
        """
        # Transpose input to PyTorch channel-first format
        # From [batch_size, *spatial_dims, channels] to [batch_size, channels, *spatial_dims]
        if self.ndims == 2:
            x = x.permute(0, 3, 1, 2)
        else:  # 3D
            x = x.permute(0, 4, 1, 2, 3)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for level in range(self.nb_levels):
            # Store input to this level for residual connection
            level_input = x
            
            # Apply convolutions
            for conv in range(self.nb_conv_per_level):
                x = self.encoder_layers[level][conv](x)
                
                # Apply dropout if specified
                if self.conv_dropout > 0:
                    x = F.dropout(x, p=self.conv_dropout, training=self.training)
                
                # Apply activation for all but the last convolution if using residuals
                if not (conv == self.nb_conv_per_level - 1 and self.use_residuals):
                    x = self.activation(x)
            
            # Apply residual connection if specified
            if self.use_residuals:
                x = x + self.encoder_residual_layers[level](level_input)
                x = self.activation(x)
            
            # Apply batch normalization if specified
            if self.batch_norm:
                x = self.encoder_bn_layers[level](x)
            
            # Store result for skip connection
            if level < self.nb_levels - 1:
                skip_connections.append(x)
                
                # Apply pooling for all but the last level
                x = self.pool_layers[level](x)
        
        # Decoder path
        for level in range(self.nb_levels - 2, -1, -1):
            # Apply upconvolution
            x = self.upconv_layers[self.nb_levels - 2 - level](x)
            
            # Concatenate with skip connection
            x = torch.cat([x, skip_connections[level]], dim=1)
            
            # Store input to this level for residual connection
            level_input = x
            
            # Apply convolutions
            for conv in range(self.nb_conv_per_level):
                x = self.decoder_layers[self.nb_levels - 2 - level][conv](x)
                
                # Apply dropout if specified
                if self.conv_dropout > 0:
                    x = F.dropout(x, p=self.conv_dropout, training=self.training)
                
                # Apply activation for all but the last convolution if using residuals
                if not (conv == self.nb_conv_per_level - 1 and self.use_residuals):
                    x = self.activation(x)
            
            # Apply residual connection if specified
            if self.use_residuals:
                x = x + self.decoder_residual_layers[self.nb_levels - 2 - level](level_input)
                x = self.activation(x)
            
            # Apply batch normalization if specified
            if self.batch_norm:
                x = self.decoder_bn_layers[self.nb_levels - 2 - level](x)
        
        # Final convolution to get predictions
        x = self.final_conv(x)
        
        # Apply softmax along the channel dimension
        x = F.softmax(x, dim=1)
        
        # Transpose output back to channel-last format
        # From [batch_size, channels, *spatial_dims] to [batch_size, *spatial_dims, channels]
        if self.ndims == 2:
            x = x.permute(0, 2, 3, 1)
        else:  # 3D
            x = x.permute(0, 2, 3, 4, 1)
        
        return x