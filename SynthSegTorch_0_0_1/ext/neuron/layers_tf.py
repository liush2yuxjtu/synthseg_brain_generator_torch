"""TensorFlow implementation of custom neural network layers for the neuron module.

This module contains TensorFlow implementations of custom neural network layers
that are used in the neuron module, including spatial transformers and other
specialized layers for medical image processing.
"""

import tensorflow as tf
import keras.layers as KL
import keras.initializers as KI
import numpy as np


class SpatialTransformer(KL.Layer):
    """TensorFlow implementation of a spatial transformer layer.
    
    This layer applies a spatial transformation to an input tensor using a
    deformation field. The transformation can be performed using linear or
    nearest neighbor interpolation.
    
    Args:
        interp_method (str): Interpolation method. Options are 'linear' or 'nearest'.
        indexing (str): Indexing convention. Options are 'ij' (matrix) or 'xy' (cartesian).
        single_transform (bool): Whether to use a single transform for all images in a batch.
        fill_value (float): Value to use for points outside the input tensor.
    """
    
    def __init__(self, interp_method='linear', indexing='ij', single_transform=False, fill_value=0, **kwargs):
        self.interp_method = interp_method
        self.indexing = indexing
        self.single_transform = single_transform
        self.fill_value = fill_value
        super(SpatialTransformer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Check interpolation method
        assert self.interp_method in ['linear', 'nearest'], \
            f"Interpolation method must be 'linear' or 'nearest', got {self.interp_method}"
        
        # Check indexing convention
        assert self.indexing in ['ij', 'xy'], \
            f"Indexing must be 'ij' or 'xy', got {self.indexing}"
        
        super(SpatialTransformer, self).build(input_shape)
    
    def call(self, inputs):
        """Apply spatial transformation to input tensor.
        
        Args:
            inputs (list): List containing [input_tensor, deformation_field].
                input_tensor: Tensor to transform. Shape: [batch_size, *spatial_dims, channels]
                deformation_field: Deformation field. Shape: [batch_size, *spatial_dims, ndims]
                
        Returns:
            tf.Tensor: Transformed tensor with the same shape as input_tensor.
        """
        # Unpack inputs
        input_tensor, deformation_field = inputs
        
        # Get tensor shapes
        input_shape = tf.shape(input_tensor)
        batch_size = input_shape[0]
        spatial_dims = input_shape[1:-1]
        n_dims = len(spatial_dims)
        n_channels = input_shape[-1]
        
        # Handle single transform case
        if self.single_transform and batch_size > 1:
            deformation_field = tf.tile(deformation_field[0:1], [batch_size] + [1] * (n_dims + 1))
        
        # Create sampling grid
        grid = self._create_sampling_grid(deformation_field, spatial_dims)
        
        # Apply interpolation
        if self.interp_method == 'linear':
            output = self._linear_interpolate(input_tensor, grid)
        else:  # 'nearest'
            output = self._nearest_interpolate(input_tensor, grid)
        
        return output
    
    def _create_sampling_grid(self, deformation_field, spatial_dims):
        """Create a sampling grid from a deformation field.
        
        Args:
            deformation_field (tf.Tensor): Deformation field.
                Shape: [batch_size, *spatial_dims, ndims]
            spatial_dims (tuple): Spatial dimensions of the input tensor.
                
        Returns:
            tf.Tensor: Sampling grid for interpolation.
                Shape: [batch_size, *spatial_dims, ndims]
        """
        batch_size = tf.shape(deformation_field)[0]
        n_dims = len(spatial_dims)
        
        # Create coordinate grid
        grid_tensors = []
        for i, size in enumerate(spatial_dims):
            coords = tf.range(size, dtype=deformation_field.dtype)
            grid_tensors.append(coords)
        
        # Create meshgrid
        if self.indexing == 'ij':
            mesh_tensors = tf.meshgrid(*grid_tensors, indexing='ij')
        else:  # 'xy'
            mesh_tensors = tf.meshgrid(*grid_tensors, indexing='xy')
        
        # Stack to create the grid
        grid = tf.stack(mesh_tensors, axis=-1)
        
        # Expand to batch size
        grid = tf.tile(tf.expand_dims(grid, axis=0), [batch_size] + [1] * (n_dims + 1))
        
        # Add deformation field to grid
        grid = grid + deformation_field
        
        return grid
    
    def _linear_interpolate(self, input_tensor, grid):
        """Apply linear interpolation using the sampling grid.
        
        Args:
            input_tensor (tf.Tensor): Input tensor to transform.
                Shape: [batch_size, *spatial_dims, channels]
            grid (tf.Tensor): Sampling grid.
                Shape: [batch_size, *spatial_dims, ndims]
                
        Returns:
            tf.Tensor: Transformed tensor with the same shape as input_tensor.
        """
        # Get tensor shapes
        input_shape = tf.shape(input_tensor)
        batch_size = input_shape[0]
        spatial_dims = input_shape[1:-1]
        n_dims = len(spatial_dims)
        n_channels = input_shape[-1]
        
        # Clip grid to valid range
        grid = tf.clip_by_value(grid, 0, tf.cast(spatial_dims, grid.dtype) - 1)
        
        # Get integer and fractional parts of coordinates
        grid_floor = tf.floor(grid)
        grid_ceil = tf.ceil(grid)
        grid_frac = grid - grid_floor
        
        # Convert to integer for indexing
        grid_floor = tf.cast(grid_floor, tf.int32)
        grid_ceil = tf.cast(grid_ceil, tf.int32)
        
        # Initialize output tensor
        output = tf.zeros_like(input_tensor)
        
        # Apply linear interpolation for each dimension
        for corner in self._get_corners(n_dims):
            # Compute corner coordinates
            corner_coords = []
            corner_weight = tf.ones_like(grid[..., 0:1])
            
            for i in range(n_dims):
                if corner[i] == 0:
                    corner_coords.append(grid_floor[..., i])
                    corner_weight *= (1 - grid_frac[..., i:i+1])
                else:
                    corner_coords.append(grid_ceil[..., i])
                    corner_weight *= grid_frac[..., i:i+1]
            
            # Gather values at corner coordinates
            corner_tensor = self._gather_nd_batch(input_tensor, corner_coords)
            
            # Apply weight and add to output
            output += corner_tensor * corner_weight
        
        return output
    
    def _nearest_interpolate(self, input_tensor, grid):
        """Apply nearest neighbor interpolation using the sampling grid.
        
        Args:
            input_tensor (tf.Tensor): Input tensor to transform.
                Shape: [batch_size, *spatial_dims, channels]
            grid (tf.Tensor): Sampling grid.
                Shape: [batch_size, *spatial_dims, ndims]
                
        Returns:
            tf.Tensor: Transformed tensor with the same shape as input_tensor.
        """
        # Get tensor shapes
        input_shape = tf.shape(input_tensor)
        batch_size = input_shape[0]
        spatial_dims = input_shape[1:-1]
        n_dims = len(spatial_dims)
        
        # Round grid to nearest integer
        grid = tf.round(grid)
        
        # Clip grid to valid range
        grid = tf.clip_by_value(grid, 0, tf.cast(spatial_dims, grid.dtype) - 1)
        
        # Convert to integer for indexing
        grid = tf.cast(grid, tf.int32)
        
        # Gather values at grid coordinates
        output = self._gather_nd_batch(input_tensor, [grid[..., i] for i in range(n_dims)])
        
        return output
    
    def _get_corners(self, n_dims):
        """Get all corner combinations for n_dims.
        
        Args:
            n_dims (int): Number of dimensions.
                
        Returns:
            list: List of corner combinations (0 or 1 for each dimension).
        """
        corners = []
        for i in range(2 ** n_dims):
            corner = []
            for j in range(n_dims):
                corner.append((i >> j) & 1)
            corners.append(corner)
        return corners
    
    def _gather_nd_batch(self, tensor, indices):
        """Gather values from tensor at specified indices for each batch.
        
        Args:
            tensor (tf.Tensor): Input tensor.
                Shape: [batch_size, *spatial_dims, channels]
            indices (list): List of index tensors for each dimension.
                Each tensor has shape [batch_size, *spatial_dims]
                
        Returns:
            tf.Tensor: Gathered values.
                Shape: [batch_size, *spatial_dims, channels]
        """
        # Get tensor shapes
        batch_size = tf.shape(tensor)[0]
        spatial_shape = tf.shape(tensor)[1:-1]
        n_dims = len(indices)
        
        # Create batch indices
        batch_idx = tf.range(batch_size)
        batch_idx = tf.reshape(batch_idx, [batch_size] + [1] * n_dims)
        batch_idx = tf.tile(batch_idx, [1] + list(spatial_shape))
        
        # Stack indices
        indices = [batch_idx] + indices
        stacked_indices = tf.stack(indices, axis=-1)
        
        # Gather values
        return tf.gather_nd(tensor, stacked_indices)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]


class VecInt(KL.Layer):
    """Vector integration layer.
    
    Integrates a vector field via scaling and squaring.
    
    Args:
        inshape (tuple): Input tensor shape (not including batch dimension).
        nsteps (int): Number of scaling and squaring steps.
    """
    
    def __init__(self, inshape, nsteps=7, **kwargs):
        self.inshape = inshape
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        super(VecInt, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Create spatial transformer for vector composition
        self.transformer = SpatialTransformer(interp_method='linear')
        super(VecInt, self).build(input_shape)
    
    def call(self, inputs):
        """Integrate vector field.
        
        Args:
            inputs (tf.Tensor): Vector field to integrate.
                Shape: [batch_size, *spatial_dims, ndims]
                
        Returns:
            tf.Tensor: Integrated vector field (displacement field).
                Shape: [batch_size, *spatial_dims, ndims]
        """
        # Scale vector field for stability
        vec = inputs * self.scale
        
        # Recursively apply scaling and squaring
        for _ in range(self.nsteps):
            # Create identity grid
            identity = self._get_identity_grid(vec)
            
            # Apply current vector field to identity grid
            warped_identity = identity + vec
            
            # Compose vector field with itself
            vec = vec + self.transformer([vec, warped_identity])
        
        return vec
    
    def _get_identity_grid(self, vec):
        """Create identity grid with the same shape as the vector field.
        
        Args:
            vec (tf.Tensor): Vector field tensor.
                Shape: [batch_size, *spatial_dims, ndims]
                
        Returns:
            tf.Tensor: Identity grid.
                Shape: [batch_size, *spatial_dims, ndims]
        """
        batch_size = tf.shape(vec)[0]
        spatial_dims = tf.shape(vec)[1:-1]
        n_dims = tf.shape(vec)[-1]
        
        # Create coordinate grid
        grid_tensors = []
        for i in range(n_dims):
            size = spatial_dims[i]
            coords = tf.range(size, dtype=vec.dtype)
            grid_tensors.append(coords)
        
        # Create meshgrid
        mesh_tensors = tf.meshgrid(*grid_tensors, indexing='ij')
        
        # Stack to create the grid
        grid = tf.stack(mesh_tensors, axis=-1)
        
        # Expand to batch size
        grid = tf.tile(tf.expand_dims(grid, axis=0), [batch_size] + [1] * (n_dims + 1))
        
        return grid
    
    def compute_output_shape(self, input_shape):
        return input_shape