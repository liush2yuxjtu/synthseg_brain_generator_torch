"""TensorFlow utilities for editing tensors in lab2im.

This file contains TensorFlow implementations of utility functions for editing tensors in lab2im.

If you use this code, please cite the first SynthSeg paper:
https://github.com/BBillot/lab2im/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


# python imports
import tensorflow as tf
import numpy as np

# project imports
from ext.lab2im import utils


def get_padding_shape(shape, stride=1, kernel_size=3, dilation=1):
    """Get the padding shape for a convolution operation.
    
    Args:
        shape: Shape of the input tensor.
        stride: Stride of the convolution.
        kernel_size: Size of the convolution kernel.
        dilation: Dilation of the convolution.
        
    Returns:
        list: Padding shape for the convolution.
    """
    # Ensure kernel_size and stride are lists
    kernel_size = utils.reformat_to_list(kernel_size, len(shape))
    stride = utils.reformat_to_list(stride, len(shape))
    dilation = utils.reformat_to_list(dilation, len(shape))
    
    # Calculate padding for each dimension
    padding = []
    for i in range(len(shape)):
        # Calculate effective kernel size with dilation
        effective_kernel_size = kernel_size[i] + (kernel_size[i] - 1) * (dilation[i] - 1)
        # Calculate padding
        padding.append(int(np.ceil((effective_kernel_size - stride[i]) / 2)))
    
    return padding


def pad_tensor(tensor, padding_shape, mode='CONSTANT', constant_values=0):
    """Pad a tensor with the given padding shape.
    
    Args:
        tensor: Input tensor to pad.
        padding_shape: Padding shape for each dimension.
        mode: Padding mode ('CONSTANT', 'REFLECT', 'SYMMETRIC').
        constant_values: Value to pad with if mode is 'CONSTANT'.
        
    Returns:
        tf.Tensor: Padded tensor.
    """
    # Convert padding_shape to TensorFlow format
    # TensorFlow expects [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], ...]
    paddings = [[0, 0]]  # No padding for batch dimension
    for pad in padding_shape:
        paddings.append([pad, pad])
    paddings.append([0, 0])  # No padding for channel dimension
    
    # Apply padding
    return tf.pad(tensor, paddings, mode=mode, constant_values=constant_values)


def gaussian_kernel(shape, sigma, dtype=tf.float32):
    """Create a Gaussian kernel.
    
    Args:
        shape: Shape of the kernel.
        sigma: Standard deviation of the Gaussian.
        dtype: Data type of the kernel.
        
    Returns:
        tf.Tensor: Gaussian kernel.
    """
    # Ensure shape and sigma are lists
    shape = utils.reformat_to_list(shape)
    sigma = utils.reformat_to_list(sigma, len(shape))
    
    # Create meshgrid for each dimension
    ranges = [tf.range(-(s // 2), -(s // 2) + s, dtype=dtype) for s in shape]
    grids = tf.meshgrid(*ranges, indexing='ij')
    
    # Create Gaussian kernel
    kernel = tf.zeros(shape, dtype=dtype)
    for i, grid in enumerate(grids):
        kernel = kernel + (grid / sigma[i]) ** 2
    
    kernel = tf.exp(-0.5 * kernel)
    
    # Normalize kernel
    kernel = kernel / tf.reduce_sum(kernel)
    
    return kernel


def apply_affine_transform(tensor, affine_matrix, interpolation='linear'):
    """Apply an affine transformation to a tensor.
    
    Args:
        tensor: Input tensor to transform.
        affine_matrix: Affine transformation matrix.
        interpolation: Interpolation method ('linear' or 'nearest').
        
    Returns:
        tf.Tensor: Transformed tensor.
    """
    # Get tensor shape
    batch_size = tf.shape(tensor)[0]
    spatial_dims = tensor.shape[1:-1]
    n_dims = len(spatial_dims)
    n_channels = tf.shape(tensor)[-1]
    
    # Create sampling grid
    grid = create_affine_sampling_grid(spatial_dims, affine_matrix)
    
    # Apply transformation
    order = 1 if interpolation == 'linear' else 0
    transformed = tf.raw_ops.ImageProjectiveTransformV3(
        images=tensor,
        transforms=affine_matrix,
        output_shape=tf.shape(tensor)[1:-1],
        interpolation=order,
        fill_mode='REFLECT'
    )
    
    return transformed


def create_affine_sampling_grid(spatial_dims, affine_matrix):
    """Create a sampling grid for affine transformation.
    
    Args:
        spatial_dims: Spatial dimensions of the tensor.
        affine_matrix: Affine transformation matrix.
        
    Returns:
        tf.Tensor: Sampling grid for grid_sample.
    """
    # Get batch size and number of dimensions
    batch_size = tf.shape(affine_matrix)[0]
    n_dims = len(spatial_dims)
    
    # Create normalized coordinate grid
    ranges = [tf.linspace(-1.0, 1.0, s) for s in spatial_dims]
    grids = tf.meshgrid(*ranges, indexing='ij')
    
    # Flatten grid and add homogeneous coordinate
    grid = tf.stack([tf.reshape(grid, [-1]) for grid in grids], axis=0)
    grid = tf.concat([grid, tf.ones([1, tf.shape(grid)[1]])], axis=0)
    
    # Apply affine transformation
    grid = tf.matmul(affine_matrix, grid)
    
    # Reshape grid to match TensorFlow format
    grid = tf.transpose(grid[:, :n_dims])
    grid = tf.reshape(grid, [*spatial_dims, n_dims])
    
    return grid


def resample_tensor(tensor, new_shape, interpolation='linear'):
    """Resample a tensor to a new shape.
    
    Args:
        tensor: Input tensor to resample.
        new_shape: New shape for the tensor.
        interpolation: Interpolation method ('linear' or 'nearest').
        
    Returns:
        tf.Tensor: Resampled tensor.
    """
    # Get tensor shape
    batch_size = tf.shape(tensor)[0]
    spatial_dims = tensor.shape[1:-1]
    n_dims = len(spatial_dims)
    n_channels = tf.shape(tensor)[-1]
    
    # Resample tensor
    method = tf.image.ResizeMethod.BILINEAR if interpolation == 'linear' else tf.image.ResizeMethod.NEAREST_NEIGHBOR
    
    # Reshape tensor for tf.image.resize
    if n_dims == 3:
        # For 3D, we need to process each slice separately
        tensor_reshaped = tf.reshape(tensor, [batch_size * spatial_dims[0], spatial_dims[1], spatial_dims[2], n_channels])
        resampled = tf.image.resize(tensor_reshaped, [new_shape[1], new_shape[2]], method=method)
        resampled = tf.reshape(resampled, [batch_size, spatial_dims[0], new_shape[1], new_shape[2], n_channels])
        
        # Now resize along the first spatial dimension
        resampled = tf.transpose(resampled, [0, 2, 3, 1, 4])
        resampled = tf.reshape(resampled, [batch_size * new_shape[1] * new_shape[2], spatial_dims[0], 1, n_channels])
        resampled = tf.image.resize(resampled, [new_shape[0], 1], method=method)
        resampled = tf.reshape(resampled, [batch_size, new_shape[1], new_shape[2], new_shape[0], n_channels])
        resampled = tf.transpose(resampled, [0, 3, 1, 2, 4])
    else:
        # For 2D, we can use tf.image.resize directly
        resampled = tf.image.resize(tensor, new_shape, method=method)
    
    return resampled


def integrate_svf(svf, n_steps=8):
    """Integrate a stationary velocity field (SVF) to obtain a diffeomorphic transformation.
    
    Args:
        svf: Stationary velocity field tensor.
        n_steps: Number of integration steps.
        
    Returns:
        tf.Tensor: Diffeomorphic transformation.
    """
    # Get tensor shape
    batch_size = tf.shape(svf)[0]
    spatial_dims = svf.shape[1:-1]
    n_dims = len(spatial_dims)
    
    # Initialize displacement field with SVF
    disp = svf / (2 ** n_steps)
    
    # Perform scaling and squaring
    for _ in range(n_steps):
        # Create sampling grid for current displacement
        grid = create_displacement_sampling_grid(spatial_dims, disp)
        
        # Compose displacement with itself
        disp_composed = tf.raw_ops.ImageProjectiveTransformV3(
            images=disp,
            transforms=grid,
            output_shape=tf.shape(disp)[1:-1],
            interpolation=1,  # bilinear
            fill_mode='REFLECT'
        )
        
        # Add displacement
        disp = disp + disp_composed
    
    return disp


def create_displacement_sampling_grid(spatial_dims, displacement):
    """Create a sampling grid for displacement field composition.
    
    Args:
        spatial_dims: Spatial dimensions of the tensor.
        displacement: Displacement field tensor.
        
    Returns:
        tf.Tensor: Sampling grid for grid_sample.
    """
    # Get batch size and number of dimensions
    batch_size = tf.shape(displacement)[0]
    n_dims = len(spatial_dims)
    
    # Create normalized coordinate grid
    ranges = [tf.linspace(-1.0, 1.0, s) for s in spatial_dims]
    grids = tf.meshgrid(*ranges, indexing='ij')
    
    # Create identity grid
    identity_grid = tf.stack(grids, axis=-1)
    identity_grid = tf.expand_dims(identity_grid, axis=0)
    identity_grid = tf.tile(identity_grid, [batch_size, 1, 1, 1, 1])
    
    # Add displacement to identity grid
    grid = identity_grid + displacement
    
    return grid


def smooth_tensor(tensor, sigma):
    """Smooth a tensor with a Gaussian kernel.
    
    Args:
        tensor: Input tensor to smooth.
        sigma: Standard deviation of the Gaussian kernel.
        
    Returns:
        tf.Tensor: Smoothed tensor.
    """
    # Get tensor shape
    batch_size = tf.shape(tensor)[0]
    spatial_dims = tensor.shape[1:-1]
    n_dims = len(spatial_dims)
    n_channels = tf.shape(tensor)[-1]
    
    # Ensure sigma is a list
    sigma = utils.reformat_to_list(sigma, n_dims)
    
    # Create Gaussian kernel
    kernel_size = [int(np.ceil(3 * s)) * 2 + 1 for s in sigma]  # Ensure odd kernel size
    kernel = gaussian_kernel(kernel_size, sigma, dtype=tensor.dtype)
    
    # Apply Gaussian smoothing
    # For simplicity, we'll apply 1D convolutions sequentially for each dimension
    smoothed = tensor
    for dim in range(n_dims):
        # Create 1D kernel for this dimension
        kernel_shape = [1] * n_dims
        kernel_shape[dim] = kernel_size[dim]
        kernel_1d = gaussian_kernel(kernel_shape, [sigma[dim]], dtype=tensor.dtype)
        kernel_1d = tf.reshape(kernel_1d, [*kernel_shape, 1, 1])
        
        # Pad tensor for this dimension
        padding = [[0, 0]] * (n_dims + 2)  # +2 for batch and channel dimensions
        padding[dim + 1] = [kernel_size[dim] // 2, kernel_size[dim] // 2]
        padded = tf.pad(smoothed, padding, mode='REFLECT')
        
        # Apply 1D convolution
        smoothed = tf.nn.conv3d(padded, kernel_1d, strides=[1, 1, 1, 1, 1], padding='VALID')
    
    return smoothed


def crop_tensor(tensor, crop_shape):
    """Crop a tensor to the given shape.
    
    Args:
        tensor: Input tensor to crop.
        crop_shape: Shape to crop to.
        
    Returns:
        tf.Tensor: Cropped tensor.
    """
    # Get tensor shape
    batch_size = tf.shape(tensor)[0]
    spatial_dims = tensor.shape[1:-1]
    n_dims = len(spatial_dims)
    n_channels = tf.shape(tensor)[-1]
    
    # Ensure crop_shape is a list
    crop_shape = utils.reformat_to_list(crop_shape, n_dims)
    
    # Calculate start and end indices for cropping
    start_idx = [(s - c) // 2 for s, c in zip(spatial_dims, crop_shape)]
    end_idx = [s + c for s, c in zip(start_idx, crop_shape)]
    
    # Create slices for cropping
    slices = [slice(None)]  # For batch dimension
    slices.extend([slice(s, e) for s, e in zip(start_idx, end_idx)])
    slices.append(slice(None))  # For channel dimension
    
    # Apply cropping
    cropped = tensor[slices]
    
    return cropped


def rescale_tensor_values(tensor, new_min=0, new_max=1):
    """Rescale tensor values to a new range.
    
    Args:
        tensor: Input tensor to rescale.
        new_min: New minimum value.
        new_max: New maximum value.
        
    Returns:
        tf.Tensor: Rescaled tensor.
    """
    # Get tensor shape
    batch_size = tf.shape(tensor)[0]
    n_channels = tf.shape(tensor)[-1]
    
    # Reshape tensor for easier processing
    tensor_flat = tf.reshape(tensor, [batch_size, -1, n_channels])
    
    # Get min and max values per batch and channel
    tensor_min = tf.reduce_min(tensor_flat, axis=1, keepdims=True)
    tensor_max = tf.reduce_max(tensor_flat, axis=1, keepdims=True)
    
    # Rescale tensor
    tensor_rescaled = (tensor - tf.reshape(tensor_min, [batch_size, 1, 1, 1, n_channels])) / \
                      (tf.reshape(tensor_max - tensor_min, [batch_size, 1, 1, 1, n_channels]) + tf.keras.backend.epsilon())
    tensor_rescaled = tensor_rescaled * (new_max - new_min) + new_min
    
    return tensor_rescaled


def add_noise(tensor, sigma=0.1):
    """Add Gaussian noise to a tensor.
    
    Args:
        tensor: Input tensor.
        sigma: Standard deviation of the noise.
        
    Returns:
        tf.Tensor: Tensor with added noise.
    """
    # Generate Gaussian noise
    noise = tf.random.normal(tf.shape(tensor), mean=0, stddev=sigma, dtype=tensor.dtype)
    
    # Add noise to tensor
    return tensor + noise


def apply_bias_field(tensor, bias_field):
    """Apply a bias field to a tensor.
    
    Args:
        tensor: Input tensor.
        bias_field: Bias field tensor.
        
    Returns:
        tf.Tensor: Tensor with bias field applied.
    """
    # Apply bias field (multiplicative)
    return tensor * bias_field


def generate_bias_field(tensor_shape, bias_field_std=0.3, bias_scale=0.025):
    """Generate a random bias field.
    
    Args:
        tensor_shape: Shape of the tensor to generate bias field for.
        bias_field_std: Standard deviation of the bias field.
        bias_scale: Scale of the bias field.
        
    Returns:
        tf.Tensor: Generated bias field.
    """
    # Get tensor shape
    batch_size = tensor_shape[0]
    spatial_dims = tensor_shape[1:-1]
    n_dims = len(spatial_dims)
    n_channels = tensor_shape[-1]
    
    # Calculate bias field shape (smaller than tensor shape)
    bias_shape = [max(3, int(s * bias_scale)) for s in spatial_dims]
    
    # Generate random bias field
    bias_field = tf.random.normal([batch_size, *bias_shape, n_channels], mean=0, stddev=bias_field_std)
    
    # Smooth bias field
    bias_field = smooth_tensor(bias_field, sigma=1.0)
    
    # Resize bias field to tensor shape
    bias_field = resample_tensor(bias_field, spatial_dims, interpolation='linear')
    
    # Exponentiate bias field
    bias_field = tf.exp(bias_field)
    
    return bias_field


def convert_labels(tensor, source_labels, target_labels):
    """Convert label values in a tensor.
    
    Args:
        tensor: Input tensor containing label indices.
        source_labels: List of source label values.
        target_labels: List of target label values.
        
    Returns:
        tf.Tensor: Tensor with converted label values.
    """
    # Create output tensor with same shape as input
    output = tf.zeros_like(tensor)
    
    # Convert each label
    for source, target in zip(source_labels, target_labels):
        output = tf.where(tf.equal(tensor, source), tf.ones_like(tensor) * target, output)
    
    return output


def one_hot_encoding(tensor, n_classes):
    """Convert a tensor of label indices to one-hot encoding.
    
    Args:
        tensor: Input tensor containing label indices.
        n_classes: Number of classes.
        
    Returns:
        tf.Tensor: One-hot encoded tensor.
    """
    # Get tensor shape
    batch_size = tf.shape(tensor)[0]
    spatial_dims = tensor.shape[1:-1]
    n_dims = len(spatial_dims)
    
    # Reshape tensor to [batch_size, -1]
    tensor_flat = tf.reshape(tensor, [batch_size, -1])
    tensor_flat = tf.cast(tensor_flat, tf.int32)
    
    # Create one-hot encoding
    one_hot = tf.one_hot(tensor_flat, n_classes, axis=-1)
    
    # Reshape back to original spatial dimensions
    one_hot = tf.reshape(one_hot, [batch_size, *spatial_dims, n_classes])
    
    return one_hot


def mask_tensor(tensor, mask, value=0):
    """Apply a mask to a tensor.
    
    Args:
        tensor: Input tensor.
        mask: Binary mask tensor.
        value: Value to fill masked regions with.
        
    Returns:
        tf.Tensor: Masked tensor.
    """
    # Apply mask
    return tf.where(mask > 0, tensor, tf.ones_like(tensor) * value)