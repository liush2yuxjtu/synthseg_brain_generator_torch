"""PyTorch implementation of the labels to image model for SynthSeg.

This module contains the PyTorch implementation of the model that generates
synthetic MRI images from label maps using a Gaussian Mixture Model (GMM).
"""

import torch
import torch.nn as nn
import numpy as np

from ..ext.lab2im import layers as l2i_layers


class LabelsToImageModel(nn.Module):
    """PyTorch model for generating images from label maps using a GMM.
    
    This model takes label maps as input and generates synthetic MRI images
    by sampling from a Gaussian Mixture Model (GMM) for each label. It also
    applies various spatial and intensity augmentations to the generated images.
    
    Args:
        labels_shape (tuple): Shape of the input label maps (excluding batch dimension).
        n_channels (int): Number of channels in the output image.
        generation_labels (list): List of labels for which to generate image intensities.
        n_neutral_labels (int): Number of labels with neutral values (i.e., not generated from the GMM).
        atlas_res (float): Resolution of the atlas in mm.
        target_res (float or list): Target resolution(s) in mm.
        output_shape (tuple): Shape of the output images (excluding batch and channel dimensions).
        output_div_by_n (int): Ensure output shape is divisible by this value.
        blur_range (float): Range of standard deviation for Gaussian blurring.
        bias_field_std (float): Standard deviation of the bias field.
        bias_shape_factor (float): Shape factor of the bias field.
        gamma_std (float): Standard deviation of the gamma augmentation.
        apply_affine (bool): Whether to apply random affine transformations.
        scaling_bounds (tuple): Bounds for random scaling.
        rotation_bounds (tuple): Bounds for random rotation.
        shearing_bounds (tuple): Bounds for random shearing.
        translation_bounds (tuple): Bounds for random translation.
        nonlin_std (float): Standard deviation of the random nonlinear deformation.
        nonlin_shape_factor (float): Shape factor of the nonlinear deformation.
        simulate_registration (bool): Whether to simulate registration.
        flipping (bool): Whether to apply random flipping.
        apply_bias_field (bool): Whether to apply bias field augmentation.
        apply_intensity_augmentation (bool): Whether to apply intensity augmentation.
        apply_gamma_augmentation (bool): Whether to apply gamma augmentation.
    """
    
    def __init__(self,
                 labels_shape,
                 n_channels=1,
                 generation_labels=None,
                 n_neutral_labels=None,
                 atlas_res=1.0,
                 target_res=1.0,
                 output_shape=None,
                 output_div_by_n=None,
                 blur_range=1.15,
                 bias_field_std=0.3,
                 bias_shape_factor=0.025,
                 gamma_std=0.1,
                 apply_affine=True,
                 scaling_bounds=0.15,
                 rotation_bounds=15,
                 shearing_bounds=0.012,
                 translation_bounds=False,
                 nonlin_std=3.,
                 nonlin_shape_factor=0.04,
                 simulate_registration=False,
                 flipping=True,
                 apply_bias_field=True,
                 apply_intensity_augmentation=True,
                 apply_gamma_augmentation=True):
        super(LabelsToImageModel, self).__init__()
        
        # Store parameters
        self.labels_shape = labels_shape
        self.n_channels = n_channels
        self.generation_labels = generation_labels
        self.n_neutral_labels = n_neutral_labels
        self.atlas_res = atlas_res
        self.target_res = target_res
        self.output_shape = output_shape
        self.output_div_by_n = output_div_by_n
        self.blur_range = blur_range
        self.bias_field_std = bias_field_std
        self.bias_shape_factor = bias_shape_factor
        self.gamma_std = gamma_std
        self.apply_affine = apply_affine
        self.scaling_bounds = scaling_bounds
        self.rotation_bounds = rotation_bounds
        self.shearing_bounds = shearing_bounds
        self.translation_bounds = translation_bounds
        self.nonlin_std = nonlin_std
        self.nonlin_shape_factor = nonlin_shape_factor
        self.simulate_registration = simulate_registration
        self.flipping = flipping
        self.apply_bias_field = apply_bias_field
        self.apply_intensity_augmentation = apply_intensity_augmentation
        self.apply_gamma_augmentation = apply_gamma_augmentation
        
        # Initialize layers
        self._build_model()
    
    def _build_model(self):
        """Build the model by initializing all the layers."""
        # Spatial augmentation layers
        if self.apply_affine or self.nonlin_std > 0:
            self.spatial_deformation = l2i_layers.RandomSpatialDeformation(
                scaling_bounds=self.scaling_bounds,
                rotation_bounds=self.rotation_bounds,
                shearing_bounds=self.shearing_bounds,
                translation_bounds=self.translation_bounds,
                nonlin_std=self.nonlin_std if self.apply_affine else 0,
                nonlin_scale=self.nonlin_shape_factor
            )
        
        # Cropping layer if output shape is specified
        if self.output_shape is not None:
            self.random_crop = l2i_layers.RandomCrop(self.output_shape)
        
        # GMM sampling layer
        self.gmm_sampler = l2i_layers.SampleConditionalGMM(self.generation_labels)
        
        # Bias field augmentation
        if self.apply_bias_field:
            self.bias_field = l2i_layers.BiasFieldCorruption(
                self.bias_field_std,
                bias_scale=self.bias_shape_factor
            )
        
        # Intensity augmentation
        if self.apply_intensity_augmentation:
            self.intensity_augmentation = l2i_layers.IntensityAugmentation()
        
        # Gaussian blur
        if self.blur_range > 0:
            self.blur = l2i_layers.DynamicGaussianBlur(self.blur_range)
    
    def forward(self, labels, means, stds, prior_means=None, prior_stds=None):
        """Generate synthetic images from label maps.
        
        Args:
            labels (torch.Tensor): Input label maps.
                Shape: [batch_size, *spatial_dims, 1]
            means (torch.Tensor): Means for the GMM.
                Shape: [batch_size, n_labels, n_channels]
            stds (torch.Tensor): Standard deviations for the GMM.
                Shape: [batch_size, n_labels, n_channels]
            prior_means (torch.Tensor, optional): Prior means for the GMM.
                Shape: [batch_size, n_labels, n_channels]
            prior_stds (torch.Tensor, optional): Prior standard deviations for the GMM.
                Shape: [batch_size, n_labels, n_channels]
                
        Returns:
            tuple: Tuple containing:
                - image (torch.Tensor): Generated image.
                    Shape: [batch_size, *output_spatial_dims, n_channels]
                - labels (torch.Tensor): Transformed label map.
                    Shape: [batch_size, *output_spatial_dims, 1]
        """
        # Apply spatial deformation if enabled
        if hasattr(self, 'spatial_deformation'):
            labels = self.spatial_deformation(labels)
        
        # Apply random cropping if output shape is specified
        if hasattr(self, 'random_crop'):
            labels = self.random_crop(labels)
        
        # Sample from GMM
        image = self.gmm_sampler([labels, means, stds])
        
        # Apply bias field if enabled
        if hasattr(self, 'bias_field'):
            image = self.bias_field(image)
        
        # Apply intensity augmentation if enabled
        if hasattr(self, 'intensity_augmentation'):
            image = self.intensity_augmentation(image)
        
        # Apply Gaussian blur if enabled
        if hasattr(self, 'blur'):
            image = self.blur(image)
        
        return image, labels


def labels_to_image_model(labels_shape,
                          n_channels=1,
                          generation_labels=None,
                          n_neutral_labels=None,
                          output_labels=None,  # 添加output_labels参数，但不使用它
                          atlas_res=1.0,
                          target_res=1.0,
                          output_shape=None,
                          output_div_by_n=None,
                          blur_range=1.15,
                          bias_field_std=0.3,
                          bias_shape_factor=0.025,
                          gamma_std=0.1,
                          apply_affine=True,
                          scaling_bounds=0.15,
                          rotation_bounds=15,
                          shearing_bounds=0.012,
                          translation_bounds=False,
                          nonlin_std=3.,
                          nonlin_shape_factor=0.04,
                          simulate_registration=False,
                          flipping=True,
                          apply_bias_field=True,
                          apply_intensity_augmentation=True,
                          apply_gamma_augmentation=True,
                          aff=None,  # 忽略此参数
                          nonlin_scale=None,  # 忽略此参数
                          randomise_res=False,  # 忽略此参数
                          max_res_iso=None,  # 忽略此参数
                          max_res_aniso=None,  # 忽略此参数
                          data_res=None,  # 忽略此参数
                          thickness=None,  # 忽略此参数
                          bias_scale=None,  # 忽略此参数
                          return_gradients=False,  # 忽略此参数
                          device=None):  # 忽略此参数
    """Create a PyTorch model for generating images from label maps using a GMM.
    
    This is a convenience function that creates an instance of the LabelsToImageModel class.
    
    Args:
        labels_shape (tuple): Shape of the input label maps (excluding batch dimension).
        n_channels (int): Number of channels in the output image.
        generation_labels (list): List of labels for which to generate image intensities.
        n_neutral_labels (int): Number of labels with neutral values (i.e., not generated from the GMM).
        output_labels (list): List of labels to output (ignored in this implementation).
        atlas_res (float): Resolution of the atlas in mm.
        target_res (float or list): Target resolution(s) in mm.
        output_shape (tuple): Shape of the output images (excluding batch and channel dimensions).
        output_div_by_n (int): Ensure output shape is divisible by this value.
        blur_range (float): Range of standard deviation for Gaussian blurring.
        bias_field_std (float): Standard deviation of the bias field.
        bias_shape_factor (float): Shape factor of the bias field.
        gamma_std (float): Standard deviation of the gamma augmentation.
        apply_affine (bool): Whether to apply random affine transformations.
        scaling_bounds (tuple): Bounds for random scaling.
        rotation_bounds (tuple): Bounds for random rotation.
        shearing_bounds (tuple): Bounds for random shearing.
        translation_bounds (tuple): Bounds for random translation.
        nonlin_std (float): Standard deviation of the random nonlinear deformation.
        nonlin_shape_factor (float): Shape factor of the nonlinear deformation.
        simulate_registration (bool): Whether to simulate registration.
        flipping (bool): Whether to apply random flipping.
        apply_bias_field (bool): Whether to apply bias field augmentation.
        apply_intensity_augmentation (bool): Whether to apply intensity augmentation.
        apply_gamma_augmentation (bool): Whether to apply gamma augmentation.
        aff (numpy.ndarray): Affine matrix for spatial transformations (ignored).
        nonlin_scale (float): Scale factor for nonlinear deformations (ignored).
        randomise_res (bool): Whether to randomize resolution (ignored).
        max_res_iso (float): Maximum isotropic resolution (ignored).
        max_res_aniso (float): Maximum anisotropic resolution (ignored).
        data_res (float): Data resolution (ignored).
        thickness (float): Slice thickness (ignored).
        bias_scale (float): Scale factor for bias field (ignored).
        return_gradients (bool): Whether to return gradients (ignored).
        device (str): Device to use for computation (ignored).
        
    Returns:
        LabelsToImageModel: A PyTorch model for generating images from label maps.
    """
    # 使用nonlin_scale替代nonlin_shape_factor，如果提供了nonlin_scale
    if nonlin_scale is not None:
        nonlin_shape_factor = nonlin_scale
        
    # 使用bias_scale替代bias_shape_factor，如果提供了bias_scale
    if bias_scale is not None:
        bias_shape_factor = bias_scale
        
    return LabelsToImageModel(
        labels_shape=labels_shape,
        n_channels=n_channels,
        generation_labels=generation_labels,
        n_neutral_labels=n_neutral_labels,
        atlas_res=atlas_res,
        target_res=target_res,
        output_shape=output_shape,
        output_div_by_n=output_div_by_n,
        blur_range=blur_range,
        bias_field_std=bias_field_std,
        bias_shape_factor=bias_shape_factor,
        gamma_std=gamma_std,
        apply_affine=apply_affine,
        scaling_bounds=scaling_bounds,
        rotation_bounds=rotation_bounds,
        shearing_bounds=shearing_bounds,
        translation_bounds=translation_bounds,
        nonlin_std=nonlin_std,
        nonlin_shape_factor=nonlin_shape_factor,
        simulate_registration=simulate_registration,
        flipping=flipping,
        apply_bias_field=apply_bias_field,
        apply_intensity_augmentation=apply_intensity_augmentation,
        apply_gamma_augmentation=apply_gamma_augmentation
    )