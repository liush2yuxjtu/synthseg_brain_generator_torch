"""TensorFlow/Keras implementation of custom layers for lab2im

This file contains TensorFlow/Keras implementations of the custom layers used in lab2im.

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
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K

# project imports
from ext.lab2im import utils
from ext.lab2im import edit_tensors_tf as l2i_et

# third-party imports
from ext.neuron import utils as nrn_utils


class RandomSpatialDeformation(KL.Layer):
    """Keras layer that spatially deforms tensors with a combination of affine and elastic transformations.
    
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
                 prob_deform=1,
                 **kwargs):
        """Initialize the RandomSpatialDeformation layer.
        
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
        super(RandomSpatialDeformation, self).__init__(**kwargs)
        
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
    
    def build(self, input_shape):
        """Build the layer."""
        # Get tensor shape
        self.n_dims = len(input_shape) - 2  # Subtract batch and channel dimensions
        self.spatial_shape = input_shape[1:-1]
        
        super(RandomSpatialDeformation, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        """Apply random spatial deformation to the input tensor.
        
        Args:
            inputs: Input tensor of shape [batch_size, *spatial_dims, channels]
            
        Returns:
            tf.Tensor: Deformed tensor of the same shape as the input
        """
        # Randomly decide whether to apply deformation
        rand = tf.random.uniform([], 0, 1)
        if rand > self.prob_deform:
            return inputs
        
        # Get tensor shape
        batch_size = tf.shape(inputs)[0]
        
        # Initialize identity transform
        transform = tf.eye(self.n_dims + 1, batch_shape=[batch_size])
        
        # Apply random scaling
        if self.scaling_bounds:
            scaling_factors = self._sample_scaling_factors(batch_size)
            scaling_matrix = tf.linalg.diag(scaling_factors)
            transform = tf.matmul(transform, scaling_matrix)
        
        # Apply random rotations
        if self.rotation_bounds:
            rotation_angles = self._sample_rotation_angles(batch_size)
            rotation_matrix = self._create_rotation_matrix(rotation_angles)
            transform = tf.matmul(transform, rotation_matrix)
        
        # Apply random shearing
        if self.shearing_bounds:
            shearing_factors = self._sample_shearing_factors(batch_size)
            shearing_matrix = self._create_shearing_matrix(shearing_factors)
            transform = tf.matmul(transform, shearing_matrix)
        
        # Apply random translations
        if self.translation_bounds:
            translation_factors = self._sample_translation_factors(batch_size)
            transform = tf.tensor_scatter_nd_update(
                transform,
                indices=tf.constant([[i, j, self.n_dims] for i in range(batch_size) for j in range(self.n_dims)]),
                updates=tf.reshape(translation_factors, [-1])
            )
        
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
        deformed_inputs = inputs  # Placeholder - would be replaced with actual transformation
        
        return deformed_inputs
    
    def _sample_scaling_factors(self, batch_size):
        """Sample random scaling factors."""
        if isinstance(self.scaling_bounds, (int, float)):
            bounds = [1 - self.scaling_bounds, 1 + self.scaling_bounds]
            scaling_factors = tf.random.uniform([batch_size, self.n_dims], bounds[0], bounds[1])
        else:
            # Implementation for other cases would go here
            scaling_factors = tf.ones([batch_size, self.n_dims])
        
        return scaling_factors
    
    def _sample_rotation_angles(self, batch_size):
        """Sample random rotation angles."""
        if isinstance(self.rotation_bounds, (int, float)):
            bounds = [-self.rotation_bounds, self.rotation_bounds]
            angles = tf.random.uniform([batch_size, self.n_dims * (self.n_dims - 1) // 2], bounds[0], bounds[1])
        else:
            # Implementation for other cases would go here
            angles = tf.zeros([batch_size, self.n_dims * (self.n_dims - 1) // 2])
        
        return angles
    
    def _create_rotation_matrix(self, angles):
        """Create a rotation matrix from angles."""
        # This is a placeholder - actual implementation would create proper rotation matrices
        batch_size = tf.shape(angles)[0]
        return tf.eye(self.n_dims, batch_shape=[batch_size])
    
    def _sample_shearing_factors(self, batch_size):
        """Sample random shearing factors."""
        if isinstance(self.shearing_bounds, (int, float)):
            bounds = [-self.shearing_bounds, self.shearing_bounds]
            factors = tf.random.uniform([batch_size, self.n_dims * (self.n_dims - 1)], bounds[0], bounds[1])
        else:
            # Implementation for other cases would go here
            factors = tf.zeros([batch_size, self.n_dims * (self.n_dims - 1)])
        
        return factors
    
    def _create_shearing_matrix(self, factors):
        """Create a shearing matrix from factors."""
        # This is a placeholder - actual implementation would create proper shearing matrices
        batch_size = tf.shape(factors)[0]
        return tf.eye(self.n_dims, batch_shape=[batch_size])
    
    def _sample_translation_factors(self, batch_size):
        """Sample random translation factors."""
        if isinstance(self.translation_bounds, (int, float)):
            bounds = [-self.translation_bounds, self.translation_bounds]
            factors = tf.random.uniform([batch_size, self.n_dims], bounds[0], bounds[1])
        else:
            # Implementation for other cases would go here
            factors = tf.zeros([batch_size, self.n_dims])
        
        return factors
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        return input_shape


class RandomCrop(KL.Layer):
    """Keras layer that randomly crops tensors to a given shape."""
    
    def __init__(self, crop_shape, **kwargs):
        """Initialize the RandomCrop layer.
        
        Args:
            crop_shape: Shape to crop the input tensor to.
        """
        super(RandomCrop, self).__init__(**kwargs)
        self.crop_shape = utils.reformat_to_list(crop_shape)
    
    def build(self, input_shape):
        """Build the layer."""
        # Get tensor shape
        self.n_dims = len(input_shape) - 2  # Subtract batch and channel dimensions
        self.spatial_shape = input_shape[1:-1]
        
        super(RandomCrop, self).build(input_shape)
    
    def call(self, inputs, crop_shape=None, **kwargs):
        """Randomly crop the input tensor to the specified shape.
        
        Args:
            inputs: Input tensor of shape [batch_size, *spatial_dims, channels]
            crop_shape: (optional) Shape to crop to, overrides the shape specified in __init__
            
        Returns:
            tf.Tensor: Cropped tensor
        """
        # Get tensor shape
        batch_size = tf.shape(inputs)[0]
        n_channels = tf.shape(inputs)[-1]
        
        # Use provided crop_shape if specified, otherwise use the one from __init__
        if crop_shape is None:
            crop_shape = self.crop_shape
        else:
            crop_shape = utils.reformat_to_list(crop_shape)
        
        # Check if cropping is needed
        if self.spatial_shape == tuple(crop_shape):
            return inputs
        
        # Calculate maximum starting indices for cropping
        max_start_idx = [self.spatial_shape[i] - crop_shape[i] for i in range(self.n_dims)]
        
        # Sample random starting indices for each batch
        start_idx = [tf.random.uniform([batch_size], 0, max_idx + 1, dtype=tf.int32) for max_idx in max_start_idx]
        
        # Crop each sample in the batch
        cropped_tensors = []
        for b in range(batch_size):
            # Build slicing indices for this batch sample
            slices = [slice(start_idx[i][b], start_idx[i][b] + crop_shape[i]) for i in range(self.n_dims)]
            # Apply cropping
            cropped_tensors.append(inputs[b][slices][None])
        
        # Concatenate results along batch dimension
        return tf.concat(cropped_tensors, axis=0)
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        output_shape = list(input_shape)
        for i in range(self.n_dims):
            output_shape[i+1] = self.crop_shape[i]
        return tuple(output_shape)


class SampleConditionalGMM(KL.Layer):
    """Keras layer that samples from a conditional Gaussian Mixture Model."""
    
    def __init__(self, generation_labels, **kwargs):
        """Initialize the SampleConditionalGMM layer.
        
        Args:
            generation_labels: List of all possible label values in the input label maps.
        """
        super(SampleConditionalGMM, self).__init__(**kwargs)
        self.generation_labels = generation_labels
    
    def call(self, inputs, **kwargs):
        """Sample from a conditional GMM based on the input label map.
        
        Args:
            inputs: List containing [labels, means, stds]
                - labels: Tensor of shape [batch_size, *spatial_dims, 1] containing label indices
                - means: Tensor of shape [batch_size, n_labels, n_channels] containing GMM means
                - stds: Tensor of shape [batch_size, n_labels, n_channels] containing GMM standard deviations
            
        Returns:
            tf.Tensor: Sampled image of shape [batch_size, *spatial_dims, n_channels]
        """
        # Unpack inputs
        labels, means, stds = inputs
        
        # Get tensor shape
        batch_size = tf.shape(labels)[0]
        spatial_shape = tf.shape(labels)[1:-1]
        n_dims = len(labels.shape) - 2
        n_channels = tf.shape(means)[-1]
        
        # Reshape labels to [batch_size, -1, 1] for easier processing
        labels_flat = tf.reshape(labels, [batch_size, -1, 1])
        
        # Initialize output tensor
        output = tf.zeros([batch_size, tf.reduce_prod(spatial_shape), n_channels])
        
        # For each label, sample from the corresponding Gaussian and place in the output
        for label_idx, label_value in enumerate(self.generation_labels):
            # Create mask for this label
            mask = tf.cast(tf.equal(labels_flat, label_value), tf.float32)
            
            # Skip if no voxels have this label
            if tf.reduce_sum(mask) == 0:
                continue
            
            # Get means and stds for this label
            label_means = means[:, label_idx:label_idx+1]
            label_stds = stds[:, label_idx:label_idx+1]
            
            # Sample from Gaussian
            noise = tf.random.normal([batch_size, 1, n_channels])
            samples = label_means + label_stds * noise
            
            # Place samples in output using the mask
            output = output + mask * samples
        
        # Reshape output back to original spatial dimensions
        output = tf.reshape(output, tf.concat([[batch_size], spatial_shape, [n_channels]], axis=0))
        
        return output
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        # input_shape is a list of shapes for [labels, means, stds]
        labels_shape = input_shape[0]
        means_shape = input_shape[1]
        
        # Output shape is same as labels but with n_channels from means
        output_shape = list(labels_shape)
        output_shape[-1] = means_shape[-1]
        
        return tuple(output_shape)


class BiasFieldCorruption(KL.Layer):
    """Keras layer that applies a random bias field to an image."""
    
    def __init__(self, bias_field_std=0.5, bias_scale=0.025, **kwargs):
        """Initialize the BiasFieldCorruption layer.
        
        Args:
            bias_field_std: Standard deviation of the bias field.
            bias_scale: Scale of the bias field.
        """
        super(BiasFieldCorruption, self).__init__(**kwargs)
        self.bias_field_std = bias_field_std
        self.bias_scale = bias_scale
    
    def call(self, inputs, **kwargs):
        """Apply a random bias field to the input tensor.
        
        Args:
            inputs: Input tensor of shape [batch_size, *spatial_dims, channels]
            
        Returns:
            tf.Tensor: Tensor with bias field applied
        """
        # This is a placeholder implementation
        # In a real implementation, you would generate a smooth bias field and apply it
        return inputs
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        return input_shape


class IntensityAugmentation(KL.Layer):
    """Keras layer that applies random intensity augmentation to an image."""
    
    def __init__(self, **kwargs):
        """Initialize the IntensityAugmentation layer."""
        super(IntensityAugmentation, self).__init__(**kwargs)
    
    def call(self, inputs, **kwargs):
        """Apply random intensity augmentation to the input tensor.
        
        Args:
            inputs: Input tensor of shape [batch_size, *spatial_dims, channels]
            
        Returns:
            tf.Tensor: Augmented tensor
        """
        # This is a placeholder implementation
        # In a real implementation, you would apply random intensity transformations
        return inputs
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        return input_shape


class ConvertLabels(KL.Layer):
    """Keras layer that converts label values in a tensor."""
    
    def __init__(self, source_labels, target_labels, **kwargs):
        """Initialize the ConvertLabels layer.
        
        Args:
            source_labels: List of source label values.
            target_labels: List of target label values.
        """
        super(ConvertLabels, self).__init__(**kwargs)
        self.source_labels = source_labels
        self.target_labels = target_labels
    
    def call(self, inputs, **kwargs):
        """Convert label values in the input tensor.
        
        Args:
            inputs: Input tensor of shape [batch_size, *spatial_dims, channels] containing label indices
            
        Returns:
            tf.Tensor: Tensor with converted label values
        """
        # Create output tensor with same shape as input
        output = tf.zeros_like(inputs)
        
        # Convert each label
        for source, target in zip(self.source_labels, self.target_labels):
            output = tf.where(tf.equal(inputs, source), tf.ones_like(inputs) * target, output)
        
        return output
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        return input_shape


class DynamicGaussianBlur(KL.Layer):
    """Keras layer that applies Gaussian blur with dynamic sigma."""
    
    def __init__(self, blur_range=1.15, **kwargs):
        """Initialize the DynamicGaussianBlur layer.
        
        Args:
            blur_range: Range for randomizing the standard deviation of the blurring kernels.
        """
        super(DynamicGaussianBlur, self).__init__(**kwargs)
        self.blur_range = blur_range
    
    def call(self, inputs, **kwargs):
        """Apply Gaussian blur with dynamic sigma to the input tensor.
        
        Args:
            inputs: Input tensor of shape [batch_size, *spatial_dims, channels]
            
        Returns:
            tf.Tensor: Blurred tensor
        """
        # This is a placeholder implementation
        # In a real implementation, you would apply Gaussian blur with dynamic sigma
        return inputs
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        return input_shape


class MimicAcquisition(KL.Layer):
    """Keras layer that mimics the acquisition process of medical images."""
    
    def __init__(self, atlas_res, target_res, **kwargs):
        """Initialize the MimicAcquisition layer.
        
        Args:
            atlas_res: Resolution of the input label maps.
            target_res: Target resolution of the generated images.
        """
        super(MimicAcquisition, self).__init__(**kwargs)
        self.atlas_res = atlas_res
        self.target_res = target_res
    
    def call(self, inputs, **kwargs):
        """Mimic the acquisition process of medical images.
        
        Args:
            inputs: Input tensor of shape [batch_size, *spatial_dims, channels]
            
        Returns:
            tf.Tensor: Tensor with acquisition effects applied
        """
        # This is a placeholder implementation
        # In a real implementation, you would apply acquisition effects
        return inputs
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        return input_shape


class ResizeTransform(KL.Layer):
    """Keras layer that resizes a tensor by a given factor."""
    
    def __init__(self, factor, interp_method='linear', **kwargs):
        """Initialize the ResizeTransform layer.
        
        Args:
            factor: Resizing factor for each dimension.
            interp_method: Interpolation method to use.
        """
        super(ResizeTransform, self).__init__(**kwargs)
        self.factor = factor
        self.interp_method = interp_method
    
    def call(self, inputs, **kwargs):
        """Resize the input tensor by the specified factor.
        
        Args:
            inputs: Input tensor of shape [batch_size, *spatial_dims, channels]
            
        Returns:
            tf.Tensor: Resized tensor
        """
        # This is a placeholder implementation
        # In a real implementation, you would resize the tensor using tf.image.resize or similar
        return inputs
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        # Calculate new spatial dimensions based on factor
        output_shape = list(input_shape)
        for i in range(1, len(input_shape) - 1):
            output_shape[i] = int(input_shape[i] * self.factor)
        
        return tuple(output_shape)


class IntensityNormalisation(KL.Layer):
    """Keras layer that normalizes the intensity of an image."""
    
    def __init__(self, **kwargs):
        """Initialize the IntensityNormalisation layer."""
        super(IntensityNormalisation, self).__init__(**kwargs)
    
    def call(self, inputs, **kwargs):
        """Normalize the intensity of the input tensor.
        
        Args:
            inputs: Input tensor of shape [batch_size, *spatial_dims, channels]
            
        Returns:
            tf.Tensor: Normalized tensor
        """
        # Compute min and max per batch and channel
        batch_size = tf.shape(inputs)[0]
        n_channels = tf.shape(inputs)[-1]
        inputs_flat = tf.reshape(inputs, [batch_size, -1, n_channels])
        
        # Compute min and max
        inputs_min = tf.reduce_min(inputs_flat, axis=1, keepdims=True)
        inputs_max = tf.reduce_max(inputs_flat, axis=1, keepdims=True)
        
        # Normalize to [0, 1]
        inputs_norm = (inputs - tf.reshape(inputs_min, [batch_size, 1, 1, 1, n_channels])) / \
                      (tf.reshape(inputs_max - inputs_min, [batch_size, 1, 1, 1, n_channels]) + K.epsilon())
        
        return inputs_norm
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        return input_shape