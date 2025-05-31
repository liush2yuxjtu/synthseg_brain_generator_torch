"""PyTorch implementation of custom neural network layers for the neuron module.

This module contains PyTorch implementations of custom neural network layers
that are used in the neuron module, including spatial transformers and other
specialized layers for medical image processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialTransformer(nn.Module):
    """PyTorch implementation of a spatial transformer layer.
    
    This layer applies a spatial transformation to an input tensor using a
    deformation field. The transformation can be performed using linear or
    nearest neighbor interpolation.
    
    Args:
        interp_method (str): Interpolation method. Options are 'linear' or 'nearest'.
        indexing (str): Indexing convention. Options are 'ij' (matrix) or 'xy' (cartesian).
        single_transform (bool): Whether to use a single transform for all images in a batch.
        fill_value (float): Value to use for points outside the input tensor.
    """
    
    def __init__(self, interp_method='linear', indexing='ij', single_transform=False, fill_value=0):
        super(SpatialTransformer, self).__init__()
        
        self.interp_method = interp_method
        self.indexing = indexing
        self.single_transform = single_transform
        self.fill_value = fill_value
        
        # Check interpolation method
        assert interp_method in ['linear', 'nearest'], \
            f"Interpolation method must be 'linear' or 'nearest', got {interp_method}"
        
        # Check indexing convention
        assert indexing in ['ij', 'xy'], \
            f"Indexing must be 'ij' or 'xy', got {indexing}"
    
    def forward(self, input_tensor, deformation_field):
        """Apply spatial transformation to input tensor.
        
        Args:
            input_tensor (torch.Tensor): Input tensor to transform.
                Shape: [batch_size, *spatial_dims, channels]
            deformation_field (torch.Tensor): Deformation field.
                Shape: [batch_size, *spatial_dims, ndims]
                
        Returns:
            torch.Tensor: Transformed tensor with the same shape as input_tensor.
        """
        # Get tensor shapes
        input_shape = input_tensor.shape
        batch_size = input_shape[0]
        spatial_dims = input_shape[1:-1]
        n_dims = len(spatial_dims)
        n_channels = input_shape[-1]
        
        # Check that deformation field has the correct shape
        expected_def_shape = (batch_size, *spatial_dims, n_dims)
        assert deformation_field.shape == expected_def_shape, \
            f"Deformation field shape {deformation_field.shape} doesn't match expected shape {expected_def_shape}"
        
        # Handle single transform case
        if self.single_transform and batch_size > 1:
            deformation_field = deformation_field[0:1].expand(batch_size, *spatial_dims, n_dims)
        
        # Reshape tensors for grid_sample
        # PyTorch expects [batch_size, channels, *spatial_dims]
        input_tensor = input_tensor.permute(0, -1, *range(1, n_dims+1))
        
        # Create sampling grid
        grid = self._create_sampling_grid(deformation_field, spatial_dims)
        
        # Apply interpolation
        if self.interp_method == 'linear':
            mode = 'bilinear' if n_dims == 2 else 'trilinear'
        else:  # 'nearest'
            mode = 'nearest'
        
        # Apply grid_sample
        # PyTorch's grid_sample expects normalized coordinates in [-1, 1]
        output = F.grid_sample(
            input_tensor, 
            grid, 
            mode=mode, 
            padding_mode='zeros', 
            align_corners=True
        )
        
        # Reshape output back to original format [batch_size, *spatial_dims, channels]
        output = output.permute(0, *range(2, n_dims+2), 1)
        
        return output
    
    def _create_sampling_grid(self, deformation_field, spatial_dims):
        """Create a sampling grid from a deformation field.
        
        Args:
            deformation_field (torch.Tensor): Deformation field.
                Shape: [batch_size, *spatial_dims, ndims]
            spatial_dims (tuple): Spatial dimensions of the input tensor.
                
        Returns:
            torch.Tensor: Sampling grid for grid_sample.
                Shape: [batch_size, *spatial_dims, ndims]
        """
        batch_size = deformation_field.shape[0]
        n_dims = len(spatial_dims)
        
        # Create normalized coordinate grid
        # PyTorch's grid_sample expects normalized coordinates in [-1, 1]
        grid_tensors = []
        for i, size in enumerate(spatial_dims):
            # Create normalized coordinates from -1 to 1
            if self.indexing == 'ij':
                coords = torch.linspace(-1, 1, size, device=deformation_field.device)
            else:  # 'xy'
                coords = torch.linspace(1, -1, size, device=deformation_field.device) if i == 0 else \
                         torch.linspace(-1, 1, size, device=deformation_field.device)
            
            grid_tensors.append(coords)
        
        # Create meshgrid
        if self.indexing == 'ij':
            mesh_tensors = torch.meshgrid(*grid_tensors, indexing='ij')
        else:  # 'xy'
            mesh_tensors = torch.meshgrid(*grid_tensors, indexing='xy')
        
        # Stack to create the grid
        grid = torch.stack(mesh_tensors, dim=-1)
        
        # Expand to batch size
        grid = grid.unsqueeze(0).expand(batch_size, *spatial_dims, n_dims)
        
        # Add deformation field to grid
        # Convert deformation field from voxel units to normalized coordinates
        norm_factors = torch.tensor([2.0 / (size - 1) for size in spatial_dims], 
                                   device=deformation_field.device)
        normalized_deformation = deformation_field * norm_factors.view(1, 1, 1, -1)
        
        # Apply deformation
        grid = grid + normalized_deformation
        
        # Reshape grid for grid_sample
        # PyTorch's grid_sample expects [batch_size, *spatial_dims, ndims]
        return grid


class VecInt(nn.Module):
    """Vector integration layer.
    
    Integrates a vector field via scaling and squaring.
    
    Args:
        inshape (tuple): Input tensor shape (not including batch dimension).
        nsteps (int): Number of scaling and squaring steps.
    """
    
    def __init__(self, inshape, nsteps=7):
        super(VecInt, self).__init__()
        
        self.inshape = inshape
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(interp_method='linear')
    
    def forward(self, vec):
        """Integrate vector field.
        
        Args:
            vec (torch.Tensor): Vector field to integrate.
                Shape: [batch_size, *spatial_dims, ndims]
                
        Returns:
            torch.Tensor: Integrated vector field (displacement field).
                Shape: [batch_size, *spatial_dims, ndims]
        """
        # Scale vector field for stability
        vec = vec * self.scale
        
        # Recursively apply scaling and squaring
        for _ in range(self.nsteps):
            # Create identity grid
            identity = self._get_identity_grid(vec)
            
            # Apply current vector field to identity grid
            warped_identity = identity + vec
            
            # Compose vector field with itself
            vec = vec + self.transformer(vec, warped_identity)
        
        return vec
    
    def _get_identity_grid(self, vec):
        """Create identity grid with the same shape as the vector field.
        
        Args:
            vec (torch.Tensor): Vector field tensor.
                Shape: [batch_size, *spatial_dims, ndims]
                
        Returns:
            torch.Tensor: Identity grid.
                Shape: [batch_size, *spatial_dims, ndims]
        """
        batch_size = vec.shape[0]
        spatial_dims = vec.shape[1:-1]
        n_dims = vec.shape[-1]
        
        # Create coordinate grid
        grid_tensors = []
        for i, size in enumerate(spatial_dims):
            coords = torch.arange(size, dtype=vec.dtype, device=vec.device)
            grid_tensors.append(coords)
        
        # Create meshgrid
        mesh_tensors = torch.meshgrid(*grid_tensors, indexing='ij')
        
        # Stack to create the grid
        grid = torch.stack(mesh_tensors, dim=-1)
        
        # Expand to batch size
        grid = grid.unsqueeze(0).expand(batch_size, *spatial_dims, n_dims)
        
        return grid