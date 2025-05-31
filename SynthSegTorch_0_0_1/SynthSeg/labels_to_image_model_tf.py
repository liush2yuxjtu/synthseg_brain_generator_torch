"""TensorFlow implementation of the labels to image model for SynthSeg.

This module contains the TensorFlow implementation of the model that generates
synthetic MRI images from label maps using a Gaussian Mixture Model (GMM).
"""

import tensorflow as tf
import keras.layers as KL
import keras.models as KM
import numpy as np

from ext.lab2im import layers_tf as l2i_layers


def labels_to_image_model_tf(labels_shape,
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
    """Create a TensorFlow model for generating images from label maps using a GMM.
    
    This function builds a Keras/TensorFlow model that takes label maps as input and
    generates synthetic MRI images by sampling from a Gaussian Mixture Model (GMM) for
    each label. It also applies various spatial and intensity augmentations to the
    generated images.
    
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
        
    Returns:
        keras.Model: A Keras/TensorFlow model for generating images from label maps.
    """
    # Define inputs
    labels_input = KL.Input(shape=labels_shape, name='labels_input')
    means_input = KL.Input(shape=[len(generation_labels), n_channels], name='means_input')
    stds_input = KL.Input(shape=[len(generation_labels), n_channels], name='stds_input')
    
    # Optional inputs for prior distributions
    prior_means_input = KL.Input(shape=[len(generation_labels), n_channels], name='prior_means_input')
    prior_stds_input = KL.Input(shape=[len(generation_labels), n_channels], name='prior_stds_input')
    
    # Apply spatial deformation if enabled
    labels = labels_input
    if apply_affine or nonlin_std > 0:
        spatial_deformation = l2i_layers.RandomSpatialDeformation(
            labels_shape,
            apply_affine=apply_affine,
            scaling_bounds=scaling_bounds,
            rotation_bounds=rotation_bounds,
            shearing_bounds=shearing_bounds,
            translation_bounds=translation_bounds,
            nonlin_std=nonlin_std,
            nonlin_shape_factor=nonlin_shape_factor,
            simulate_registration=simulate_registration,
            flipping=flipping
        )
        labels = spatial_deformation(labels)
    
    # Apply random cropping if output shape is specified
    if output_shape is not None:
        random_crop = l2i_layers.RandomCrop(output_shape, output_div_by_n)
        labels = random_crop(labels)
    
    # Sample from GMM
    gmm_sampler = l2i_layers.SampleConditionalGMM(generation_labels, n_neutral_labels)
    image = gmm_sampler([labels, means_input, stds_input, prior_means_input, prior_stds_input])
    
    # Apply bias field if enabled
    if apply_bias_field:
        bias_field = l2i_layers.BiasFieldCorruption(
            labels_shape[:-1],
            bias_field_std,
            bias_shape_factor
        )
        image = bias_field(image)
    
    # Apply intensity augmentation if enabled
    if apply_intensity_augmentation:
        intensity_augmentation = l2i_layers.IntensityAugmentation(gamma_std)
        image = intensity_augmentation(image)
    
    # Apply Gaussian blur if enabled
    if blur_range > 0:
        blur = l2i_layers.DynamicGaussianBlur(blur_range)
        image = blur(image)
    
    # Create model with all inputs and outputs
    return KM.Model(
        inputs=[labels_input, means_input, stds_input, prior_means_input, prior_stds_input],
        outputs=[image, labels],
        name='labels_to_image_model'
    )