"""PyTorch implementation of the training module for SynthSeg.

This module contains the PyTorch implementation of the training function
for the SynthSeg framework, which trains a UNet for segmenting synthetic
MRI images generated from label maps using a GMM.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time

from SynthSeg.labels_to_image_model import labels_to_image_model
from SynthSeg.unet import UNet


def training(labels_dir,
            model_dir,
            path_generation_labels=None,
            path_segmentation_labels=None,
            n_neutral_labels=None,
            generation_classes=None,
            segmentation_classes=None,
            target_res=1.,
            output_shape=None,
            output_div_by_n=None,
            flipping=True,
            scaling_bounds=0.15,
            rotation_bounds=15,
            shearing_bounds=0.012,
            translation_bounds=False,
            nonlin_std=3.,
            nonlin_shape_factor=0.04,
            simulate_registration=False,
            apply_bias_field=True,
            bias_field_std=0.3,
            bias_shape_factor=0.025,
            apply_intensity_augmentation=True,
            gamma_std=0.1,
            apply_mirroring=False,
            blur_range=1.15,
            data_res=None,
            thickness=None,
            downsample=False,
            build_reliability_maps=False,
            reliability_noise_std=0.1,
            head_model_file=None,
            prior_distributions='uniform',
            n_levels=5,
            nb_conv_per_level=2,
            conv_size=3,
            unet_feat_count=24,
            feat_multiplier=2,
            dropout=0,
            activation='elu',
            lr=1e-4,
            lr_decay=0,
            epochs=100,
            steps_per_epoch=1000,
            batch_size=1,
            checkpoint_epoch=1,
            use_specific_stats_for_channel=False,
            prior_means=None,
            prior_stds=None,
            use_gpu=True,
            gpu_id=0):
    """Train a UNet for segmenting synthetic MRI images generated from label maps.
    
    This function trains a UNet for segmenting synthetic MRI images generated from
    label maps using a Gaussian Mixture Model (GMM). The synthetic images are
    generated on-the-fly during training, with various spatial and intensity
    augmentations applied.
    
    Args:
        labels_dir (str): Path to the directory containing label maps for training.
        model_dir (str): Path to the directory where the model will be saved.
        path_generation_labels (str, optional): Path to the generation labels file.
        path_segmentation_labels (str, optional): Path to the segmentation labels file.
        n_neutral_labels (int, optional): Number of neutral labels.
        generation_classes (list, optional): List of classes for generation.
        segmentation_classes (list, optional): List of classes for segmentation.
        target_res (float or list, optional): Target resolution(s) in mm.
        output_shape (tuple, optional): Shape of the output images.
        output_div_by_n (int, optional): Ensure output shape is divisible by this value.
        flipping (bool, optional): Whether to apply random flipping.
        scaling_bounds (float, optional): Bounds for random scaling.
        rotation_bounds (float, optional): Bounds for random rotation.
        shearing_bounds (float, optional): Bounds for random shearing.
        translation_bounds (float, optional): Bounds for random translation.
        nonlin_std (float, optional): Standard deviation of the random nonlinear deformation.
        nonlin_shape_factor (float, optional): Shape factor of the nonlinear deformation.
        simulate_registration (bool, optional): Whether to simulate registration.
        apply_bias_field (bool, optional): Whether to apply bias field augmentation.
        bias_field_std (float, optional): Standard deviation of the bias field.
        bias_shape_factor (float, optional): Shape factor of the bias field.
        apply_intensity_augmentation (bool, optional): Whether to apply intensity augmentation.
        gamma_std (float, optional): Standard deviation of the gamma augmentation.
        apply_mirroring (bool, optional): Whether to apply mirroring.
        blur_range (float, optional): Range of standard deviation for Gaussian blurring.
        data_res (float or list, optional): Resolution of the input data in mm.
        thickness (float or list, optional): Slice thickness in mm.
        downsample (bool, optional): Whether to downsample the images.
        build_reliability_maps (bool, optional): Whether to build reliability maps.
        reliability_noise_std (float, optional): Standard deviation of the reliability noise.
        head_model_file (str, optional): Path to the head model file.
        prior_distributions (str, optional): Type of prior distributions.
        n_levels (int, optional): Number of levels in the UNet.
        nb_conv_per_level (int, optional): Number of convolutions per level in the UNet.
        conv_size (int, optional): Size of the convolution kernels in the UNet.
        unet_feat_count (int, optional): Number of features in the first layer of the UNet.
        feat_multiplier (int, optional): Feature multiplier for the UNet.
        dropout (float, optional): Dropout rate for the UNet.
        activation (str, optional): Activation function for the UNet.
        lr (float, optional): Learning rate.
        lr_decay (float, optional): Learning rate decay.
        epochs (int, optional): Number of epochs.
        steps_per_epoch (int, optional): Number of steps per epoch.
        batch_size (int, optional): Batch size.
        checkpoint_epoch (int, optional): Frequency of checkpoints in epochs.
        use_specific_stats_for_channel (bool, optional): Whether to use specific stats for each channel.
        prior_means (list, optional): Prior means for the GMM.
        prior_stds (list, optional): Prior standard deviations for the GMM.
        use_gpu (bool, optional): Whether to use GPU.
        gpu_id (int, optional): ID of the GPU to use.
    """
    # Set device
    if use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load label maps
    print('Loading label maps...')
    label_maps = load_label_maps(labels_dir)
    n_labels = len(label_maps)
    print(f'Found {n_labels} label maps')
    
    # Get label info
    if path_generation_labels is not None:
        generation_labels = np.load(path_generation_labels)
    else:
        if generation_classes is not None:
            generation_labels = generation_classes
        else:
            # Extract unique labels from the label maps
            unique_labels = np.unique(np.concatenate([np.unique(lm) for lm in label_maps]))
            generation_labels = unique_labels.tolist()
    
    if path_segmentation_labels is not None:
        segmentation_labels = np.load(path_segmentation_labels)
    else:
        if segmentation_classes is not None:
            segmentation_labels = segmentation_classes
        else:
            segmentation_labels = generation_labels
    
    # Get shapes
    label_shape = label_maps[0].shape
    n_dims = len(label_shape)
    n_channels = 1  # Assuming single-channel images
    n_classes = len(segmentation_labels)
    
    # Create labels-to-image model
    print('Creating labels-to-image model...')
    l2i_model = labels_to_image_model(
        labels_shape=(*label_shape, 1),
        n_channels=n_channels,
        generation_labels=generation_labels,
        n_neutral_labels=n_neutral_labels,
        atlas_res=data_res,
        target_res=target_res,
        output_shape=output_shape,
        output_div_by_n=output_div_by_n,
        blur_range=blur_range,
        bias_field_std=bias_field_std,
        bias_shape_factor=bias_shape_factor,
        gamma_std=gamma_std,
        apply_affine=True,
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
        apply_gamma_augmentation=apply_intensity_augmentation
    ).to(device)
    
    # Create UNet model
    print('Creating UNet model...')
    if output_shape is not None:
        input_shape = output_shape
    else:
        input_shape = label_shape
    
    unet = UNet(
        input_shape=input_shape,
        n_channels=n_channels,
        n_classes=n_classes,
        n_levels=n_levels,
        n_conv_per_level=nb_conv_per_level,
        conv_size=conv_size,
        feat_count=unet_feat_count,
        feat_multiplier=feat_multiplier,
        dropout=dropout,
        activation=activation
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(unet.parameters(), lr=lr)
    
    # Learning rate scheduler
    if lr_decay > 0:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    
    # Training loop
    print('Starting training...')
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        for step in range(1, steps_per_epoch + 1):
            # Generate batch of label maps
            batch_indices = np.random.choice(n_labels, batch_size)
            batch_label_maps = [label_maps[i] for i in batch_indices]
            batch_label_maps = np.stack(batch_label_maps, axis=0)
            batch_label_maps = np.expand_dims(batch_label_maps, axis=-1)  # Add channel dimension
            batch_label_maps = torch.from_numpy(batch_label_maps).float().to(device)
            
            # Generate means and stds for the GMM
            batch_means = generate_gmm_params(generation_labels, n_channels, batch_size, prior_means, device)
            batch_stds = generate_gmm_params(generation_labels, n_channels, batch_size, prior_stds, device)
            
            # Generate prior means and stds if needed
            if prior_distributions == 'normal':
                batch_prior_means = generate_gmm_params(generation_labels, n_channels, batch_size, prior_means, device)
                batch_prior_stds = generate_gmm_params(generation_labels, n_channels, batch_size, prior_stds, device)
            else:
                batch_prior_means = None
                batch_prior_stds = None
            
            # Generate synthetic images
            images, labels = l2i_model(batch_label_maps, batch_means, batch_stds, batch_prior_means, batch_prior_stds)
            
            # Convert labels to segmentation labels
            # This is a placeholder - in a real implementation, you would need to convert
            # the generation labels to segmentation labels if they are different
            segmentation_targets = labels
            
            # Convert to one-hot encoding for the loss function
            # PyTorch's CrossEntropyLoss expects class indices, not one-hot encoding
            segmentation_targets = segmentation_targets.long().squeeze(-1)  # Remove channel dimension
            
            # Forward pass through UNet
            # UNet expects input shape [batch_size, channels, *spatial_dims]
            images = images.permute(0, -1, *range(1, n_dims + 1))  # Move channels to second dimension
            outputs = unet(images)
            
            # Compute loss
            loss = criterion(outputs, segmentation_targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update epoch loss
            epoch_loss += loss.item()
            
            # Print step info
            if step % 100 == 0:
                print(f'Epoch {epoch}, Step {step}/{steps_per_epoch}, Loss: {loss.item():.4f}')
        
        # Update learning rate
        if lr_decay > 0:
            scheduler.step()
        
        # Print epoch info
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_loss /= steps_per_epoch
        print(f'Epoch {epoch}/{epochs} completed in {epoch_duration:.2f}s, Loss: {epoch_loss:.4f}')
        
        # Save checkpoint
        if epoch % checkpoint_epoch == 0:
            checkpoint_path = os.path.join(model_dir, f'model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'unet_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'model_final.pth')
    torch.save({
        'epoch': epochs,
        'unet_state_dict': unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss
    }, final_model_path)
    print(f'Final model saved to {final_model_path}')


def load_label_maps(labels_dir):
    """Load label maps from a directory.
    
    Args:
        labels_dir (str): Path to the directory containing label maps.
        
    Returns:
        list: List of label maps as numpy arrays.
    """
    # This is a placeholder function
    # In a real implementation, you would load the label maps from files
    # For now, we'll just create some dummy label maps
    label_maps = []
    for filename in os.listdir(labels_dir):
        if filename.endswith('.nii.gz') or filename.endswith('.nii'):
            # In a real implementation, you would use nibabel to load the file
            # For now, we'll just create a dummy array
            label_map = np.random.randint(0, 5, (128, 128, 128))
            label_maps.append(label_map)
    return label_maps


def generate_gmm_params(labels, n_channels, batch_size, prior_values=None, device=None):
    """Generate parameters for the GMM.
    
    Args:
        labels (list): List of labels.
        n_channels (int): Number of channels.
        batch_size (int): Batch size.
        prior_values (list, optional): Prior values for the parameters.
        device (torch.device, optional): Device to create the tensor on.
        
    Returns:
        torch.Tensor: Tensor of GMM parameters.
    """
    n_labels = len(labels)
    
    if prior_values is not None:
        # Use prior values with some random variation
        params = torch.tensor(prior_values, dtype=torch.float32, device=device)
        params = params.unsqueeze(0).expand(batch_size, -1, -1)
        # Add some random variation
        params = params + torch.randn_like(params) * 0.1
    else:
        # Generate random parameters
        # For means, use values in a reasonable range for MRI intensities
        if 'means' in locals():
            params = torch.rand(batch_size, n_labels, n_channels, device=device) * 100 + 50
        else:  # For stds
            params = torch.rand(batch_size, n_labels, n_channels, device=device) * 15 + 5
    
    return params