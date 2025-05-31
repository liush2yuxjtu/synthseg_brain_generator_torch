"""PyTorch utilities for editing tensors in lab2im.

This file contains PyTorch implementations of utility functions for editing tensors in lab2im.

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
import torch
import torch.nn.functional as F
import numpy as np

# project imports
from . import utils


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


def pad_tensor(tensor, padding_shape, mode='constant', value=0):
    """Pad a tensor with the given padding shape.
    
    Args:
        tensor: Input tensor to pad.
        padding_shape: Padding shape for each dimension.
        mode: Padding mode ('constant', 'reflect', 'replicate', or 'circular').
        value: Value to pad with if mode is 'constant'.
        
    Returns:
        torch.Tensor: Padded tensor.
    """
    # Convert padding_shape to PyTorch format (last dim first)
    # PyTorch expects (padding_left, padding_right, padding_top, padding_bottom, ...)
    padding = []
    for pad in reversed(padding_shape):
        padding.extend([pad, pad])
    
    # Apply padding
    return F.pad(tensor, padding, mode=mode, value=value)


def gaussian_kernel(shape, sigma, dtype=torch.float32, device='cpu'):
    """Create a Gaussian kernel.
    
    Args:
        shape: Shape of the kernel.
        sigma: Standard deviation of the Gaussian.
        dtype: Data type of the kernel.
        device: Device to create the kernel on.
        
    Returns:
        torch.Tensor: Gaussian kernel.
    """
    # Ensure shape and sigma are lists
    shape = utils.reformat_to_list(shape)
    sigma = utils.reformat_to_list(sigma, len(shape))
    
    # Create meshgrid for each dimension
    ranges = [torch.arange(-(s // 2), -(s // 2) + s, dtype=dtype, device=device) for s in shape]
    grids = torch.meshgrid(*ranges, indexing='ij')
    
    # Create Gaussian kernel
    kernel = torch.zeros(shape, dtype=dtype, device=device)
    for i, grid in enumerate(grids):
        kernel = kernel + (grid / sigma[i]) ** 2
    
    kernel = torch.exp(-0.5 * kernel)
    
    # Normalize kernel
    kernel = kernel / torch.sum(kernel)
    
    return kernel


def apply_affine_transform(tensor, affine_matrix, interpolation='linear'):
    """Apply an affine transformation to a tensor.
    
    Args:
        tensor: Input tensor to transform.
        affine_matrix: Affine transformation matrix.
        interpolation: Interpolation method ('linear' or 'nearest').
        
    Returns:
        torch.Tensor: Transformed tensor.
    """
    # Get tensor shape
    batch_size = tensor.shape[0]
    spatial_dims = tensor.shape[1:-1]
    n_dims = len(spatial_dims)
    n_channels = tensor.shape[-1]
    
    # Reshape tensor to [batch_size, n_channels, *spatial_dims]
    tensor = tensor.permute(0, n_dims + 1, *range(1, n_dims + 1))
    
    # Create sampling grid
    grid = create_affine_sampling_grid(spatial_dims, affine_matrix)
    
    # Apply transformation
    mode = 'bilinear' if interpolation == 'linear' else 'nearest'
    transformed = F.grid_sample(tensor, grid, mode=mode, align_corners=True)
    
    # Reshape back to [batch_size, *spatial_dims, n_channels]
    transformed = transformed.permute(0, *range(2, n_dims + 2), 1)
    
    return transformed


def create_affine_sampling_grid(spatial_dims, affine_matrix):
    """Create a sampling grid for affine transformation.
    
    Args:
        spatial_dims: Spatial dimensions of the tensor.
        affine_matrix: Affine transformation matrix.
        
    Returns:
        torch.Tensor: Sampling grid for grid_sample.
    """
    # Get batch size and number of dimensions
    batch_size = affine_matrix.shape[0]
    n_dims = len(spatial_dims)
    
    # Create normalized coordinate grid
    ranges = [torch.linspace(-1, 1, s, device=affine_matrix.device) for s in spatial_dims]
    grids = torch.meshgrid(*ranges, indexing='ij')
    
    # Flatten grid and add homogeneous coordinate
    grid = torch.stack([grid.flatten() for grid in grids], dim=0)
    grid = torch.cat([grid, torch.ones(1, grid.shape[1], device=grid.device)], dim=0)
    
    # Apply affine transformation
    grid = affine_matrix @ grid
    
    # Reshape grid to match grid_sample input format
    grid = grid[:, :n_dims].view(batch_size, *spatial_dims, n_dims)
    
    return grid


def resample_tensor(tensor, new_shape, interpolation='linear'):
    """Resample a tensor to a new shape.
    
    Args:
        tensor: Input tensor to resample.
        new_shape: New shape for the tensor.
        interpolation: Interpolation method ('linear' or 'nearest').
        
    Returns:
        torch.Tensor: Resampled tensor.
    """
    # Get tensor shape
    batch_size = tensor.shape[0]
    spatial_dims = tensor.shape[1:-1]
    n_dims = len(spatial_dims)
    n_channels = tensor.shape[-1]
    
    # Reshape tensor to [batch_size, n_channels, *spatial_dims]
    tensor = tensor.permute(0, n_dims + 1, *range(1, n_dims + 1))
    
    # Resample tensor
    mode = 'bilinear' if interpolation == 'linear' else 'nearest'
    resampled = F.interpolate(tensor, size=new_shape, mode=mode, align_corners=True if mode == 'bilinear' else None)
    
    # Reshape back to [batch_size, *new_shape, n_channels]
    resampled = resampled.permute(0, *range(2, n_dims + 2), 1)
    
    return resampled


def integrate_svf(svf, n_steps=8):
    """Integrate a stationary velocity field (SVF) to obtain a diffeomorphic transformation.
    
    Args:
        svf: Stationary velocity field tensor.
        n_steps: Number of integration steps.
        
    Returns:
        torch.Tensor: Diffeomorphic transformation.
    """
    # Get tensor shape
    batch_size = svf.shape[0]
    spatial_dims = svf.shape[1:-1]
    n_dims = len(spatial_dims)
    
    # Initialize displacement field with SVF
    disp = svf / (2 ** n_steps)
    
    # Perform scaling and squaring
    for _ in range(n_steps):
        # Create sampling grid for current displacement
        grid = create_displacement_sampling_grid(spatial_dims, disp)
        
        # Compose displacement with itself
        disp_composed = F.grid_sample(disp.permute(0, n_dims + 1, *range(1, n_dims + 1)),
                                     grid,
                                     mode='bilinear',
                                     align_corners=True)
        
        # Add displacement
        disp = disp + disp_composed.permute(0, *range(2, n_dims + 2), 1)
    
    return disp


def create_displacement_sampling_grid(spatial_dims, displacement):
    """Create a sampling grid for displacement field composition.
    
    Args:
        spatial_dims: Spatial dimensions of the tensor.
        displacement: Displacement field tensor.
        
    Returns:
        torch.Tensor: Sampling grid for grid_sample.
    """
    # Get batch size and number of dimensions
    batch_size = displacement.shape[0]
    n_dims = len(spatial_dims)
    
    # Create normalized coordinate grid
    ranges = [torch.linspace(-1, 1, s, device=displacement.device) for s in spatial_dims]
    grids = torch.meshgrid(*ranges, indexing='ij')
    
    # Create identity grid
    identity_grid = torch.stack(grids, dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    
    # Add displacement to identity grid
    grid = identity_grid + displacement
    
    return grid


def smooth_tensor(tensor, sigma):
    """Smooth a tensor with a Gaussian kernel.
    
    Args:
        tensor: Input tensor to smooth.
        sigma: Standard deviation of the Gaussian kernel.
        
    Returns:
        torch.Tensor: Smoothed tensor.
    """
    # Get tensor shape
    batch_size = tensor.shape[0]
    spatial_dims = tensor.shape[1:-1]
    n_dims = len(spatial_dims)
    n_channels = tensor.shape[-1]
    
    # Ensure sigma is a list
    sigma = utils.reformat_to_list(sigma, n_dims)
    
    # Create Gaussian kernel
    kernel_size = [int(np.ceil(3 * s)) * 2 + 1 for s in sigma]  # Ensure odd kernel size
    kernel = gaussian_kernel(kernel_size, sigma, dtype=tensor.dtype, device=tensor.device)
    
    # Reshape tensor to [batch_size, n_channels, *spatial_dims]
    tensor = tensor.permute(0, n_dims + 1, *range(1, n_dims + 1))
    
    # Apply Gaussian smoothing
    # For simplicity, we'll apply 1D convolutions sequentially for each dimension
    smoothed = tensor
    for dim in range(n_dims):
        # Create 1D kernel for this dimension
        kernel_shape = [1] * n_dims
        kernel_shape[dim] = kernel_size[dim]
        kernel_1d = gaussian_kernel(kernel_shape, [sigma[dim]], dtype=tensor.dtype, device=tensor.device)
        kernel_1d = kernel_1d.view(1, 1, *kernel_shape)
        
        # Pad tensor for this dimension
        padding = [0] * (2 * n_dims)
        padding[2 * dim] = kernel_size[dim] // 2
        padding[2 * dim + 1] = kernel_size[dim] // 2
        padded = F.pad(smoothed, tuple(padding), mode='reflect')
        
        # Apply 1D convolution
        smoothed = F.conv3d(padded, kernel_1d, groups=n_channels)
    
    # Reshape back to [batch_size, *spatial_dims, n_channels]
    smoothed = smoothed.permute(0, *range(2, n_dims + 2), 1)
    
    return smoothed


def crop_tensor(tensor, crop_shape):
    """Crop a tensor to the given shape.
    
    Args:
        tensor: Input tensor to crop.
        crop_shape: Shape to crop to.
        
    Returns:
        torch.Tensor: Cropped tensor.
    """
    # Get tensor shape
    batch_size = tensor.shape[0]
    spatial_dims = tensor.shape[1:-1]
    n_dims = len(spatial_dims)
    n_channels = tensor.shape[-1]
    
    # Ensure crop_shape is a list
    crop_shape = utils.reformat_to_list(crop_shape, n_dims)
    
    # Calculate start and end indices for cropping
    start_idx = [(s - c) // 2 for s, c in zip(spatial_dims, crop_shape)]
    end_idx = [s + c for s, c in zip(start_idx, crop_shape)]
    
    # Create slices for cropping
    slices = [slice(s, e) for s, e in zip(start_idx, end_idx)]
    
    # Apply cropping
    cropped = tensor[:, slices[0], slices[1], slices[2], :] if n_dims == 3 else \
              tensor[:, slices[0], slices[1], :] if n_dims == 2 else \
              tensor[:, slices[0], :]
    
    return cropped


def rescale_tensor_values(tensor, new_min=0, new_max=1):
    """Rescale tensor values to a new range.
    
    Args:
        tensor: Input tensor to rescale.
        new_min: New minimum value.
        new_max: New maximum value.
        
    Returns:
        torch.Tensor: Rescaled tensor.
    """
    # Get tensor shape
    batch_size = tensor.shape[0]
    n_channels = tensor.shape[-1]
    
    # Reshape tensor for easier processing
    tensor_flat = tensor.reshape(batch_size, -1, n_channels)
    
    # Get min and max values per batch and channel
    tensor_min = tensor_flat.min(dim=1, keepdim=True)[0]
    tensor_max = tensor_flat.max(dim=1, keepdim=True)[0]
    
    # Rescale tensor
    tensor_rescaled = (tensor - tensor_min.reshape(batch_size, 1, 1, 1, n_channels)) / \
                      (tensor_max - tensor_min).reshape(batch_size, 1, 1, 1, n_channels)
    tensor_rescaled = tensor_rescaled * (new_max - new_min) + new_min
    
    return tensor_rescaled


def add_noise(tensor, sigma=0.1):
    """Add Gaussian noise to a tensor.
    
    Args:
        tensor: Input tensor.
        sigma: Standard deviation of the noise.
        
    Returns:
        torch.Tensor: Tensor with added noise.
    """
    # Generate Gaussian noise
    noise = torch.randn_like(tensor) * sigma
    
    # Add noise to tensor
    return tensor + noise


def apply_bias_field(tensor, bias_field):
    """Apply a bias field to a tensor.
    
    Args:
        tensor: Input tensor.
        bias_field: Bias field tensor.
        
    Returns:
        torch.Tensor: Tensor with bias field applied.
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
        torch.Tensor: Generated bias field.
    """
    # Get tensor shape
    batch_size = tensor_shape[0]
    spatial_dims = tensor_shape[1:-1]
    n_dims = len(spatial_dims)
    n_channels = tensor_shape[-1]
    
    # Calculate bias field shape (smaller than tensor shape)
    bias_shape = [max(3, int(s * bias_scale)) for s in spatial_dims]
    
    # Generate random bias field
    bias_field = torch.randn([batch_size, *bias_shape, n_channels], device=tensor_shape.device) * bias_field_std
    
    # Smooth bias field
    bias_field = smooth_tensor(bias_field, sigma=1.0)
    
    # Resize bias field to tensor shape
    bias_field = resample_tensor(bias_field, spatial_dims, interpolation='linear')
    
    # Exponentiate bias field
    bias_field = torch.exp(bias_field)
    
    return bias_field


def convert_labels(tensor, source_labels, target_labels):
    """Convert label values in a tensor.
    
    Args:
        tensor: Input tensor containing label indices.
        source_labels: List of source label values.
        target_labels: List of target label values.
        
    Returns:
        torch.Tensor: Tensor with converted label values.
    """
    # Create output tensor with same shape as input
    output = torch.zeros_like(tensor)
    
    # Convert each label
    for source, target in zip(source_labels, target_labels):
        output[tensor == source] = target
    
    return output


def one_hot_encoding(tensor, n_classes):
    """Convert a tensor of label indices to one-hot encoding.
    
    Args:
        tensor: Input tensor containing label indices.
        n_classes: Number of classes.
        
    Returns:
        torch.Tensor: One-hot encoded tensor.
    """
    # Get tensor shape
    batch_size = tensor.shape[0]
    spatial_dims = tensor.shape[1:-1]
    n_dims = len(spatial_dims)
    
    # Reshape tensor to [batch_size, -1]
    tensor_flat = tensor.reshape(batch_size, -1).long()
    
    # Create one-hot encoding
    one_hot = torch.zeros([batch_size, tensor_flat.shape[1], n_classes], device=tensor.device)
    one_hot.scatter_(2, tensor_flat.unsqueeze(-1), 1)
    
    # Reshape back to original spatial dimensions
    one_hot = one_hot.reshape(batch_size, *spatial_dims, n_classes)
    
    return one_hot


def mask_tensor(tensor, mask, value=0):
    """Apply a mask to a tensor.
    
    Args:
        tensor: Input tensor.
        mask: Binary mask tensor.
        value: Value to fill masked regions with.
        
    Returns:
        torch.Tensor: Masked tensor.
    """
    # Apply mask
    return torch.where(mask > 0, tensor, torch.ones_like(tensor) * value)


def get_dims(shape, max_channels=10):
    """Get the number of dimensions and channels from the shape of a tensor.
    
    Args:
        shape: Shape of a tensor.
        max_channels: Maximum possible number of channels.
        
    Returns:
        tuple: The number of dimensions and channels associated with the provided shape.
    """
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels


def get_ras_axes(aff, n_dims=3):
    """Find the RAS axes corresponding to each dimension of a volume, based on its affine matrix.
    
    Args:
        aff: Affine matrix. Can be a 2D numpy array.
        n_dims: Number of dimensions (excluding channels) of the volume.
        
    Returns:
        numpy.ndarray: Axes corresponding to RAS orientations.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    for i in range(n_dims):
        if i not in img_ras_axes:
            unique, counts = np.unique(img_ras_axes, return_counts=True)
            incorrect_value = unique[np.argmax(counts)]
            img_ras_axes[np.where(img_ras_axes == incorrect_value)[0][-1]] = i
    
    return img_ras_axes


def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=None, return_copy=True):
    """Align a volume to a reference orientation specified by an affine matrix.
    
    Args:
        volume: Input tensor or numpy array.
        aff: Affine matrix of the floating volume.
        aff_ref: Affine matrix of the target orientation. Default is identity matrix.
        return_aff: Whether to return the affine matrix of the aligned volume.
        n_dims: Number of dimensions (excluding channels) of the volume.
        return_copy: Whether to return the original volume or a copy.
        
    Returns:
        torch.Tensor or numpy.ndarray: Aligned volume, with corresponding affine matrix if return_aff is True.
    """
    # Convert torch tensor to numpy if needed
    is_torch = isinstance(volume, torch.Tensor)
    if is_torch:
        device = volume.device
        dtype = volume.dtype
        volume_np = volume.detach().cpu().numpy()
    else:
        volume_np = volume
    
    # Work on copy
    new_volume = volume_np.copy() if return_copy else volume_np
    aff_flo = aff.copy()
    
    # Default value for aff_ref
    if aff_ref is None:
        aff_ref = np.eye(4)
    
    # Extract RAS axes
    if n_dims is None:
        n_dims, _ = get_dims(new_volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)
    
    # Align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            new_volume = np.swapaxes(new_volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])[0][0]
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = ras_axes_flo[i], ras_axes_flo[swapped_axis_idx]
    
    # Align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            new_volume = np.flip(new_volume, axis=i)
            aff_flo[:, i] = -aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (new_volume.shape[i] - 1)
    
    # Convert back to torch tensor if needed
    if is_torch:
        new_volume = torch.tensor(new_volume, dtype=dtype, device=device)
    
    if return_aff:
        return new_volume, aff_flo
    else:
        return new_volume