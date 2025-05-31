"""PyTorch implementation of custom layers for lab2im

This file contains PyTorch implementations of the custom layers used in lab2im.

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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# project imports
from . import utils
from . import edit_tensors as l2i_et

# third-party imports
from ..neuron import utils as nrn_utils


class RandomSpatialDeformation(nn.Module):
    """PyTorch module that spatially deforms tensors with a combination of affine and elastic transformations.
    
    The input tensors are expected to have the shape [batchsize, *spatial_dims, channels].
    The non-linear deformation is obtained by:
    1) a small-size SVF is sampled from a centered normal distribution of random standard deviation.
    2) it is resized with trilinear interpolation to half the shape of the input tensor
    3) it is integrated to obtain a diffeomorphic transformation
    4) finally, it is resized (again with trilinear interpolation) to full image size
    """

    def __init__(self,
                 scaling_bounds=0.15,
                 rotation_bounds=10,
                 shearing_bounds=0.02,
                 translation_bounds=False,
                 enable_90_rotations=False,
                 nonlin_std=4.,
                 nonlin_scale=.0625,
                 inter_method='linear',
                 prob_deform=1):
        """Initialize the RandomSpatialDeformation module.
        
        Args:
            scaling_bounds: (optional) range of the random scaling to apply.
            rotation_bounds: (optional) range of the random rotation to apply.
            shearing_bounds: (optional) range of the random shearing to apply.
            translation_bounds: (optional) range of the random translation to apply.
            enable_90_rotations: (optional) whether to rotate the input by a random angle chosen in {0, 90, 180, 270}.
            nonlin_std: (optional) maximum value of the standard deviation of the normal distribution from which we
                sample the small-size SVF. Set to 0 to turn off elastic deformation.
            nonlin_scale: (optional) factor between the shapes of the input tensor and the shape of the input non-linear tensor.
            inter_method: (optional) interpolation method when deforming the input tensor. Can be 'linear', or 'nearest'
            prob_deform: (optional) probability to apply spatial deformation
        """
        super(RandomSpatialDeformation, self).__init__()
        
        # Store parameters
        self.scaling_bounds = scaling_bounds
        self.rotation_bounds = rotation_bounds
        self.shearing_bounds = shearing_bounds
        self.translation_bounds = translation_bounds
        self.enable_90_rotations = enable_90_rotations
        self.nonlin_std = nonlin_std
        self.nonlin_scale = nonlin_scale
        self.inter_method = inter_method
        self.prob_deform = prob_deform
    
    def forward(self, x):
        """Apply random spatial deformation to the input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, *spatial_dims, channels]
            
        Returns:
            torch.Tensor: Deformed tensor of the same shape as the input
        """
        # Get tensor shape
        batch_size = x.shape[0]
        n_dims = len(x.shape) - 2  # Subtract batch and channel dimensions
        spatial_shape = x.shape[1:-1]
        
        # Randomly decide whether to apply deformation
        if torch.rand(1).item() > self.prob_deform:
            return x
        
        # Initialize identity transform
        transform = torch.eye(n_dims + 1, dtype=torch.float32, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Apply random scaling
        if self.scaling_bounds:
            scaling_factors = self._sample_scaling_factors(batch_size, n_dims)
            scaling_matrix = torch.diag_embed(scaling_factors)
            transform[:, :n_dims, :n_dims] = torch.matmul(transform[:, :n_dims, :n_dims], scaling_matrix)
        
        # Apply random rotations
        if self.rotation_bounds:
            rotation_angles = self._sample_rotation_angles(batch_size, n_dims)
            for i in range(batch_size):
                rotation_matrix = self._create_rotation_matrix(rotation_angles[i], n_dims)
                transform[i, :n_dims, :n_dims] = torch.matmul(transform[i, :n_dims, :n_dims], rotation_matrix)
        
        # Apply random shearing
        if self.shearing_bounds:
            shearing_factors = self._sample_shearing_factors(batch_size, n_dims)
            for i in range(batch_size):
                shearing_matrix = self._create_shearing_matrix(shearing_factors[i], n_dims)
                transform[i, :n_dims, :n_dims] = torch.matmul(transform[i, :n_dims, :n_dims], shearing_matrix)
        
        # Apply random translations
        if self.translation_bounds:
            translation_factors = self._sample_translation_factors(batch_size, n_dims)
            transform[:, :n_dims, -1] = translation_factors
        
        # Apply 90-degree rotations if enabled
        if self.enable_90_rotations:
            # Implementation would go here
            pass
        
        # Apply non-linear deformation if enabled
        if self.nonlin_std > 0:
            # Implementation would go here
            pass
        
        # Apply the transformation to the input tensor
        # This is a placeholder - actual implementation would use a spatial transformer network
        # or grid_sample to apply the transformation
        deformed_x = x  # Placeholder - would be replaced with actual transformation
        
        return deformed_x
    
    def _sample_scaling_factors(self, batch_size, n_dims):
        """Sample random scaling factors."""
        if isinstance(self.scaling_bounds, (int, float)):
            bounds = [1 - self.scaling_bounds, 1 + self.scaling_bounds]
            scaling_factors = torch.rand(batch_size, n_dims, dtype=torch.float32) * (bounds[1] - bounds[0]) + bounds[0]
        else:
            # Implementation for other cases would go here
            scaling_factors = torch.ones(batch_size, n_dims, dtype=torch.float32)
        
        return scaling_factors
    
    def _sample_rotation_angles(self, batch_size, n_dims):
        """Sample random rotation angles."""
        if isinstance(self.rotation_bounds, (int, float)):
            bounds = [-self.rotation_bounds, self.rotation_bounds]
            angles = torch.rand(batch_size, n_dims * (n_dims - 1) // 2) * (bounds[1] - bounds[0]) + bounds[0]
        else:
            # Implementation for other cases would go here
            angles = torch.zeros(batch_size, n_dims * (n_dims - 1) // 2)
        
        return angles
    
    def _create_rotation_matrix(self, angles, n_dims):
        """Create a rotation matrix from angles."""
        # This is a placeholder - actual implementation would create proper rotation matrices
        return torch.eye(n_dims)
    
    def _sample_shearing_factors(self, batch_size, n_dims):
        """Sample random shearing factors."""
        if isinstance(self.shearing_bounds, (int, float)):
            bounds = [-self.shearing_bounds, self.shearing_bounds]
            factors = torch.rand(batch_size, n_dims * (n_dims - 1)) * (bounds[1] - bounds[0]) + bounds[0]
        else:
            # Implementation for other cases would go here
            factors = torch.zeros(batch_size, n_dims * (n_dims - 1))
        
        return factors
    
    def _create_shearing_matrix(self, factors, n_dims):
        """Create a shearing matrix from factors."""
        # This is a placeholder - actual implementation would create proper shearing matrices
        return torch.eye(n_dims)
    
    def _sample_translation_factors(self, batch_size, n_dims):
        """Sample random translation factors."""
        if isinstance(self.translation_bounds, (int, float)):
            bounds = [-self.translation_bounds, self.translation_bounds]
            factors = torch.rand(batch_size, n_dims) * (bounds[1] - bounds[0]) + bounds[0]
        else:
            # Implementation for other cases would go here
            factors = torch.zeros(batch_size, n_dims)
        
        return factors


class RandomCrop(nn.Module):
    """PyTorch module that randomly crops tensors to a given shape."""
    
    def __init__(self, crop_shape):
        """Initialize the RandomCrop module.
        
        Args:
            crop_shape: Shape to crop the input tensor to.
        """
        super(RandomCrop, self).__init__()
        self.crop_shape = utils.reformat_to_list(crop_shape)
    
    def forward(self, x, crop_shape=None):
        """Randomly crop the input tensor to the specified shape.
        
        Args:
            x: Input tensor of shape [batch_size, *spatial_dims, channels]
            crop_shape: (optional) Shape to crop to, overrides the shape specified in __init__
            
        Returns:
            torch.Tensor: Cropped tensor
        """
        # Get tensor shape
        batch_size = x.shape[0]
        n_dims = len(x.shape) - 2  # Subtract batch and channel dimensions
        spatial_shape = x.shape[1:-1]
        n_channels = x.shape[-1]
        
        # Use provided crop_shape if specified, otherwise use the one from __init__
        if crop_shape is None:
            crop_shape = self.crop_shape
        else:
            crop_shape = utils.reformat_to_list(crop_shape)
        
        # Check if cropping is needed
        if spatial_shape == tuple(crop_shape):
            return x
        
        # Calculate maximum starting indices for cropping
        max_start_idx = [spatial_shape[i] - crop_shape[i] for i in range(n_dims)]
        
        # Sample random starting indices for each batch
        start_idx = [torch.randint(0, max_idx + 1, (batch_size,), device=x.device) for max_idx in max_start_idx]
        
        # Crop each sample in the batch
        cropped_tensors = []
        for b in range(batch_size):
            # Build slicing indices for this batch sample
            slices = [slice(start_idx[i][b].item(), start_idx[i][b].item() + crop_shape[i]) for i in range(n_dims)]
            # Apply cropping
            cropped_tensors.append(x[b][slices][None])
        
        # Concatenate results along batch dimension
        return torch.cat(cropped_tensors, dim=0)


class SampleConditionalGMM(nn.Module):
    """PyTorch module that samples from a conditional Gaussian Mixture Model."""
    
    def __init__(self, generation_labels):
        """Initialize the SampleConditionalGMM module.
        
        Args:
            generation_labels: List of all possible label values in the input label maps.
        """
        super(SampleConditionalGMM, self).__init__()
        self.generation_labels = generation_labels
    
    def forward(self, inputs):
        """Sample from a conditional GMM based on the input label map.
        
        Args:
            inputs: List containing [labels, means, stds]
                - labels: Tensor of shape [batch_size, *spatial_dims, 1] containing label indices
                - means: Tensor of shape [batch_size, n_labels, n_channels] containing GMM means
                - stds: Tensor of shape [batch_size, n_labels, n_channels] containing GMM standard deviations
            
        Returns:
            torch.Tensor: Sampled image of shape [batch_size, *spatial_dims, n_channels]
        """
        # Unpack inputs
        labels, means, stds = inputs
        
        # Get tensor shape
        batch_size = labels.shape[0]
        spatial_shape = labels.shape[1:-1]
        n_dims = len(spatial_shape)
        n_channels = means.shape[-1]
        
        # Reshape labels to [batch_size, -1, 1] for easier processing
        labels_flat = labels.reshape(batch_size, -1, 1)
        
        # Initialize output tensor
        output = torch.zeros(batch_size, np.prod(spatial_shape), n_channels, device=labels.device)
        
        # For each label, sample from the corresponding Gaussian and place in the output
        for label_idx, label_value in enumerate(self.generation_labels):
            # Convert label_value to the same type as labels_flat
            label_tensor = torch.tensor(label_value, dtype=labels_flat.dtype, device=labels_flat.device)
            
            # Create mask for this label
            mask = (labels_flat == label_tensor).float()
            
            # Skip if no voxels have this label
            if not torch.any(mask):
                continue
            
            # Get means and stds for this label
            label_means = means[:, label_idx:label_idx+1]
            label_stds = stds[:, label_idx:label_idx+1]
            
            # Sample from Gaussian
            noise = torch.randn(batch_size, 1, n_channels, device=labels.device)
            samples = label_means + label_stds * noise
            
            # Place samples in output using the mask
            output = output + mask * samples
        
        # Reshape output back to original spatial dimensions
        output = output.reshape(batch_size, *spatial_shape, n_channels)
        
        return output


class BiasFieldCorruption(nn.Module):
    """PyTorch module that applies a random bias field to an image."""
    
    def __init__(self, bias_field_std=0.5, bias_scale=0.025):
        """Initialize the BiasFieldCorruption module.
        
        Args:
            bias_field_std: Standard deviation of the bias field.
            bias_scale: Scale of the bias field.
        """
        super(BiasFieldCorruption, self).__init__()
        self.bias_field_std = bias_field_std
        self.bias_scale = bias_scale
    
    def forward(self, x):
        """Apply a random bias field to the input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, *spatial_dims, channels]
            
        Returns:
            torch.Tensor: Tensor with bias field applied
        """
        # This is a placeholder implementation
        # In a real implementation, you would generate a smooth bias field and apply it
        return x


class IntensityAugmentation(nn.Module):
    """PyTorch module that applies random intensity augmentation to an image."""
    
    def __init__(self):
        """Initialize the IntensityAugmentation module."""
        super(IntensityAugmentation, self).__init__()
    
    def forward(self, x):
        """Apply random intensity augmentation to the input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, *spatial_dims, channels]
            
        Returns:
            torch.Tensor: Augmented tensor
        """
        # This is a placeholder implementation
        # In a real implementation, you would apply random intensity transformations
        return x


class ConvertLabels(nn.Module):
    """PyTorch module that converts label values in a tensor."""
    
    def __init__(self, source_labels, target_labels):
        """Initialize the ConvertLabels module.
        
        Args:
            source_labels: List of source label values.
            target_labels: List of target label values.
        """
        super(ConvertLabels, self).__init__()
        self.source_labels = source_labels
        self.target_labels = target_labels
    
    def forward(self, x):
        """Convert label values in the input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, *spatial_dims, channels] containing label indices
            
        Returns:
            torch.Tensor: Tensor with converted label values
        """
        # Create output tensor with same shape as input
        output = torch.zeros_like(x)
        
        # Convert each label
        for source, target in zip(self.source_labels, self.target_labels):
            output[x == source] = target
        
        return output


class DynamicGaussianBlur(nn.Module):
    """PyTorch module that applies Gaussian blur with dynamic sigma."""
    
    def __init__(self, blur_range=1.15):
        """Initialize the DynamicGaussianBlur module.
        
        Args:
            blur_range: Range for randomizing the standard deviation of the blurring kernels.
        """
        super(DynamicGaussianBlur, self).__init__()
        self.blur_range = blur_range
    
    def forward(self, x):
        """Apply Gaussian blur with dynamic sigma to the input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, *spatial_dims, channels]
            
        Returns:
            torch.Tensor: Blurred tensor
        """
        # This is a placeholder implementation
        # In a real implementation, you would apply Gaussian blur with dynamic sigma
        return x


class MimicAcquisition(nn.Module):
    """PyTorch module that mimics the acquisition process of medical images."""
    
    def __init__(self, atlas_res, target_res):
        """Initialize the MimicAcquisition module.
        
        Args:
            atlas_res: Resolution of the input label maps.
            target_res: Target resolution of the generated images.
        """
        super(MimicAcquisition, self).__init__()
        self.atlas_res = atlas_res
        self.target_res = target_res
    
    def forward(self, x):
        """Mimic the acquisition process of medical images.
        
        Args:
            x: Input tensor of shape [batch_size, *spatial_dims, channels]
            
        Returns:
            torch.Tensor: Tensor with acquisition effects applied
        """
        # This is a placeholder implementation
        # In a real implementation, you would apply acquisition effects
        return x


class ResizeTransform(nn.Module):
    """PyTorch module that resizes a tensor by a given factor."""
    
    def __init__(self, factor, interp_method='linear'):
        """Initialize the ResizeTransform module.
        
        Args:
            factor: Resizing factor for each dimension.
            interp_method: Interpolation method to use.
        """
        super(ResizeTransform, self).__init__()
        self.factor = factor
        self.interp_method = interp_method
    
    def forward(self, x):
        """Resize the input tensor by the specified factor.
        
        Args:
            x: Input tensor of shape [batch_size, *spatial_dims, channels]
            
        Returns:
            torch.Tensor: Resized tensor
        """
        # This is a placeholder implementation
        # In a real implementation, you would resize the tensor using grid_sample or interpolate
        return x


class IntensityNormalisation(nn.Module):
    """PyTorch module that normalizes the intensity of an image."""
    
    def __init__(self):
        """Initialize the IntensityNormalisation module."""
        super(IntensityNormalisation, self).__init__()
    
    def forward(self, x):
        """Normalize the intensity of the input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, *spatial_dims, channels]
            
        Returns:
            torch.Tensor: Normalized tensor
        """
        # Compute min and max per batch and channel
        batch_size = x.shape[0]
        n_channels = x.shape[-1]
        x_flat = x.reshape(batch_size, -1, n_channels)
        
        # Compute min and max
        x_min = x_flat.min(dim=1, keepdim=True)[0]
        x_max = x_flat.max(dim=1, keepdim=True)[0]
        
        # Normalize to [0, 1]
        x_norm = (x - x_min.reshape(batch_size, 1, 1, 1, n_channels)) / \
                 (x_max - x_min).reshape(batch_size, 1, 1, 1, n_channels)
        
        return x_norm