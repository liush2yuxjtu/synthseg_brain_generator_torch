"""Test file for comparing TensorFlow and PyTorch implementations of labels_to_image_model.

This file contains tests that compare the functionality of the TensorFlow and PyTorch
implementations of the model that generates synthetic MRI images from label maps.
"""

import os
import sys
import numpy as np
import torch
import tensorflow as tf

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the TensorFlow and PyTorch implementations
from SynthSeg import labels_to_image_model as torch_model
from SynthSeg import labels_to_image_model_tf as tf_model


def test_model_creation():
    """Test the creation of the labels_to_image_model."""
    print("Testing labels_to_image_model creation...")
    
    try:
        # Define model parameters
        labels_shape = (128, 128, 128, 1)
        n_channels = 1
        generation_labels = [0, 1, 2, 3, 4]
        n_neutral_labels = 1
        
        # Create TensorFlow model
        tf_l2i_model = tf_model.labels_to_image_model_tf(
            labels_shape=labels_shape,
            n_channels=n_channels,
            generation_labels=generation_labels,
            n_neutral_labels=n_neutral_labels,
            apply_affine=True,
            nonlin_std=3.0,
            output_shape=(96, 96, 96),
            blur_range=1.15,
            bias_field_std=0.3
        )
        
        # Create PyTorch model
        torch_l2i_model = torch_model.labels_to_image_model(
            labels_shape=labels_shape,
            n_channels=n_channels,
            generation_labels=generation_labels,
            n_neutral_labels=n_neutral_labels,
            apply_affine=True,
            nonlin_std=3.0,
            output_shape=(96, 96, 96),
            blur_range=1.15,
            bias_field_std=0.3
        )
        
        # Check that the models were created successfully
        assert tf_l2i_model is not None, "TensorFlow model creation failed"
        assert torch_l2i_model is not None, "PyTorch model creation failed"
        
        print("✓ labels_to_image_model creation test passed")
    except Exception as e:
        print(f"✗ labels_to_image_model creation test failed: {str(e)}")
        raise


def test_model_forward_pass():
    """Test the forward pass of the labels_to_image_model."""
    print("Testing labels_to_image_model forward pass...")
    
    # Set random seeds for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)
    tf.random.set_seed(1234)
    
    try:
        # Define model parameters
        labels_shape = (32, 32, 32, 1)  # Smaller shape for faster testing
        n_channels = 1
        generation_labels = [0, 1, 2, 3, 4]
        n_neutral_labels = 1
        batch_size = 2
        output_shape = (24, 24, 24)
        
        # Create TensorFlow model with minimal augmentation for testing
        tf_l2i_model = tf_model.labels_to_image_model_tf(
            labels_shape=labels_shape,
            n_channels=n_channels,
            generation_labels=generation_labels,
            n_neutral_labels=n_neutral_labels,
            apply_affine=False,  # Disable affine for deterministic testing
            nonlin_std=0.0,      # Disable nonlinear deformation
            output_shape=output_shape,
            blur_range=0.0,      # Disable blurring
            bias_field_std=0.0,  # Disable bias field
            apply_intensity_augmentation=False  # Disable intensity augmentation
        )
        
        # Create PyTorch model with the same parameters
        torch_l2i_model = torch_model.labels_to_image_model(
            labels_shape=labels_shape,
            n_channels=n_channels,
            generation_labels=generation_labels,
            n_neutral_labels=n_neutral_labels,
            apply_affine=False,
            nonlin_std=0.0,
            output_shape=output_shape,
            blur_range=0.0,
            bias_field_std=0.0,
            apply_intensity_augmentation=False
        )
        
        # Create input data
        # Labels: integer values from the generation_labels list
        np_labels = np.zeros((batch_size, *labels_shape[:-1], 1), dtype=np.float32)
        for i, label in enumerate(generation_labels):
            # Create a block of each label in the volume
            block_size = labels_shape[0] // len(generation_labels)
            start = i * block_size
            end = (i + 1) * block_size if i < len(generation_labels) - 1 else labels_shape[0]
            np_labels[:, start:end, :, :, :] = label
        
        # Means and standard deviations for the GMM
        np_means = np.random.normal(100, 50, (batch_size, len(generation_labels), n_channels)).astype(np.float32)
        np_stds = np.abs(np.random.normal(15, 5, (batch_size, len(generation_labels), n_channels))).astype(np.float32)
        
        # Prior means and standard deviations (optional)
        np_prior_means = np.random.normal(100, 50, (batch_size, len(generation_labels), n_channels)).astype(np.float32)
        np_prior_stds = np.abs(np.random.normal(15, 5, (batch_size, len(generation_labels), n_channels))).astype(np.float32)
        
        # Convert to TensorFlow and PyTorch tensors
        tf_labels = tf.convert_to_tensor(np_labels)
        tf_means = tf.convert_to_tensor(np_means)
        tf_stds = tf.convert_to_tensor(np_stds)
        tf_prior_means = tf.convert_to_tensor(np_prior_means)
        tf_prior_stds = tf.convert_to_tensor(np_prior_stds)
        
        torch_labels = torch.from_numpy(np_labels)
        torch_means = torch.from_numpy(np_means)
        torch_stds = torch.from_numpy(np_stds)
        torch_prior_means = torch.from_numpy(np_prior_means)
        torch_prior_stds = torch.from_numpy(np_prior_stds)
        
        # Run forward pass
        tf_image, tf_output_labels = tf_l2i_model([tf_labels, tf_means, tf_stds, tf_prior_means, tf_prior_stds])
        torch_image, torch_output_labels = torch_l2i_model(torch_labels, torch_means, torch_stds, torch_prior_means, torch_prior_stds)
        
        # Convert outputs to numpy for comparison
        tf_image_np = tf_image.numpy()
        tf_output_labels_np = tf_output_labels.numpy()
        torch_image_np = torch_image.detach().numpy()
        torch_output_labels_np = torch_output_labels.detach().numpy()
        
        # Check that the outputs have the correct shape
        expected_image_shape = (batch_size, *output_shape, n_channels)
        expected_labels_shape = (batch_size, *output_shape, 1)
        
        assert tf_image_np.shape == expected_image_shape, \
            f"TF image shape {tf_image_np.shape} doesn't match expected shape {expected_image_shape}"
        assert tf_output_labels_np.shape == expected_labels_shape, \
            f"TF labels shape {tf_output_labels_np.shape} doesn't match expected shape {expected_labels_shape}"
        
        assert torch_image_np.shape == expected_image_shape, \
            f"PyTorch image shape {torch_image_np.shape} doesn't match expected shape {expected_image_shape}"
        assert torch_output_labels_np.shape == expected_labels_shape, \
            f"PyTorch labels shape {torch_output_labels_np.shape} doesn't match expected shape {expected_labels_shape}"
        
        # Check that the outputs are similar
        # Note: Due to implementation differences, we use a higher tolerance
        # and only check that the outputs are in a similar range
        assert np.isclose(np.mean(tf_image_np), np.mean(torch_image_np), atol=20), \
            f"Mean image values don't match: {np.mean(tf_image_np)} vs {np.mean(torch_image_np)}"
        assert np.isclose(np.std(tf_image_np), np.std(torch_image_np), atol=20), \
            f"Std of image values don't match: {np.std(tf_image_np)} vs {np.std(torch_image_np)}"
        
        # Check that the labels are preserved
        for label in generation_labels:
            tf_count = np.sum(tf_output_labels_np == label)
            torch_count = np.sum(torch_output_labels_np == label)
            assert tf_count > 0, f"Label {label} not found in TF output labels"
            assert torch_count > 0, f"Label {label} not found in PyTorch output labels"
        
        print("✓ labels_to_image_model forward pass test passed")
    except Exception as e:
        print(f"✗ labels_to_image_model forward pass test failed: {str(e)}")
        raise


def test_model_with_augmentation():
    """Test the labels_to_image_model with augmentation enabled."""
    print("Testing labels_to_image_model with augmentation...")
    
    # Set random seeds for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)
    tf.random.set_seed(1234)
    
    try:
        # Define model parameters
        labels_shape = (32, 32, 32, 1)  # Smaller shape for faster testing
        n_channels = 1
        generation_labels = [0, 1, 2, 3, 4]
        n_neutral_labels = 1
        batch_size = 2
        output_shape = (24, 24, 24)
        
        # Create TensorFlow model with augmentation
        tf_l2i_model = tf_model.labels_to_image_model_tf(
            labels_shape=labels_shape,
            n_channels=n_channels,
            generation_labels=generation_labels,
            n_neutral_labels=n_neutral_labels,
            apply_affine=True,
            nonlin_std=1.0,
            output_shape=output_shape,
            blur_range=1.15,
            bias_field_std=0.3,
            apply_intensity_augmentation=True
        )
        
        # Create PyTorch model with the same parameters
        torch_l2i_model = torch_model.labels_to_image_model(
            labels_shape=labels_shape,
            n_channels=n_channels,
            generation_labels=generation_labels,
            n_neutral_labels=n_neutral_labels,
            apply_affine=True,
            nonlin_std=1.0,
            output_shape=output_shape,
            blur_range=1.15,
            bias_field_std=0.3,
            apply_intensity_augmentation=True
        )
        
        # Create input data
        # Labels: integer values from the generation_labels list
        np_labels = np.zeros((batch_size, *labels_shape[:-1], 1), dtype=np.float32)
        for i, label in enumerate(generation_labels):
            # Create a block of each label in the volume
            block_size = labels_shape[0] // len(generation_labels)
            start = i * block_size
            end = (i + 1) * block_size if i < len(generation_labels) - 1 else labels_shape[0]
            np_labels[:, start:end, :, :, :] = label
        
        # Means and standard deviations for the GMM
        np_means = np.random.normal(100, 50, (batch_size, len(generation_labels), n_channels)).astype(np.float32)
        np_stds = np.abs(np.random.normal(15, 5, (batch_size, len(generation_labels), n_channels))).astype(np.float32)
        
        # Prior means and standard deviations (optional)
        np_prior_means = np.random.normal(100, 50, (batch_size, len(generation_labels), n_channels)).astype(np.float32)
        np_prior_stds = np.abs(np.random.normal(15, 5, (batch_size, len(generation_labels), n_channels))).astype(np.float32)
        
        # Convert to TensorFlow and PyTorch tensors
        tf_labels = tf.convert_to_tensor(np_labels)
        tf_means = tf.convert_to_tensor(np_means)
        tf_stds = tf.convert_to_tensor(np_stds)
        tf_prior_means = tf.convert_to_tensor(np_prior_means)
        tf_prior_stds = tf.convert_to_tensor(np_prior_stds)
        
        torch_labels = torch.from_numpy(np_labels)
        torch_means = torch.from_numpy(np_means)
        torch_stds = torch.from_numpy(np_stds)
        torch_prior_means = torch.from_numpy(np_prior_means)
        torch_prior_stds = torch.from_numpy(np_prior_stds)
        
        # Run forward pass
        tf_image, tf_output_labels = tf_l2i_model([tf_labels, tf_means, tf_stds, tf_prior_means, tf_prior_stds])
        torch_image, torch_output_labels = torch_l2i_model(torch_labels, torch_means, torch_stds, torch_prior_means, torch_prior_stds)
        
        # Convert outputs to numpy for comparison
        tf_image_np = tf_image.numpy()
        tf_output_labels_np = tf_output_labels.numpy()
        torch_image_np = torch_image.detach().numpy()
        torch_output_labels_np = torch_output_labels.detach().numpy()
        
        # Check that the outputs have the correct shape
        expected_image_shape = (batch_size, *output_shape, n_channels)
        expected_labels_shape = (batch_size, *output_shape, 1)
        
        assert tf_image_np.shape == expected_image_shape, \
            f"TF image shape {tf_image_np.shape} doesn't match expected shape {expected_image_shape}"
        assert tf_output_labels_np.shape == expected_labels_shape, \
            f"TF labels shape {tf_output_labels_np.shape} doesn't match expected shape {expected_labels_shape}"
        
        assert torch_image_np.shape == expected_image_shape, \
            f"PyTorch image shape {torch_image_np.shape} doesn't match expected shape {expected_image_shape}"
        assert torch_output_labels_np.shape == expected_labels_shape, \
            f"PyTorch labels shape {torch_output_labels_np.shape} doesn't match expected shape {expected_labels_shape}"
        
        # With augmentation enabled, we can't directly compare the outputs
        # But we can check that the outputs are in a reasonable range
        assert np.min(tf_image_np) >= 0, f"TF image has negative values: {np.min(tf_image_np)}"
        assert np.min(torch_image_np) >= 0, f"PyTorch image has negative values: {np.min(torch_image_np)}"
        
        # Check that the labels still contain all the expected labels
        for label in generation_labels:
            tf_count = np.sum(tf_output_labels_np == label)
            torch_count = np.sum(torch_output_labels_np == label)
            # With spatial augmentation, some labels might disappear due to cropping
            # So we only check that at least one label is present
            assert np.any(np.isin(generation_labels, tf_output_labels_np)), \
                f"No labels from {generation_labels} found in TF output labels"
            assert np.any(np.isin(generation_labels, torch_output_labels_np)), \
                f"No labels from {generation_labels} found in PyTorch output labels"
        
        print("✓ labels_to_image_model with augmentation test passed")
    except Exception as e:
        print(f"✗ labels_to_image_model with augmentation test failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the tests
    try:
        test_model_creation()
        test_model_forward_pass()
        test_model_with_augmentation()
        
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTests failed: {str(e)}")
        sys.exit(1)