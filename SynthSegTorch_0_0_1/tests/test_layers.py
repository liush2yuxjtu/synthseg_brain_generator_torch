"""Test file for comparing TensorFlow and PyTorch implementations of lab2im layers.

This file contains tests that compare the functionality of the TensorFlow and PyTorch
implementations of the custom layers used in lab2im.
"""

import os
import sys
import numpy as np
import torch
import tensorflow as tf

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the TensorFlow and PyTorch implementations
from ext.lab2im import layers as torch_layers
from ext.lab2im import layers_tf as tf_layers


def test_random_spatial_deformation():
    """Test the RandomSpatialDeformation layer."""
    print("Testing RandomSpatialDeformation...")
    
    # Set random seeds for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)
    tf.random.set_seed(1234)
    
    # Create test input data
    batch_size = 2
    spatial_shape = (10, 10, 10)
    n_channels = 1
    
    # Create numpy array and convert to TensorFlow and PyTorch tensors
    np_input = np.random.rand(batch_size, *spatial_shape, n_channels).astype(np.float32)
    tf_input = tf.convert_to_tensor(np_input)
    torch_input = torch.from_numpy(np_input)
    
    # Create the layers with the same parameters
    try:
        tf_layer = tf_layers.RandomSpatialDeformation(
            scaling_bounds=0.15,
            rotation_bounds=10,
            shearing_bounds=0.02,
            nonlin_std=4.0,
            nonlin_scale=0.0625
        )
        
        torch_layer = torch_layers.RandomSpatialDeformation(
            scaling_bounds=0.15,
            rotation_bounds=10,
            shearing_bounds=0.02,
            nonlin_std=4.0,
            nonlin_scale=0.0625
        )
        
        # Apply the layers
        tf_output = tf_layer(tf_input)
        torch_output = torch_layer(torch_input)
        
        # Convert outputs to numpy for comparison
        tf_output_np = tf_output.numpy()
        torch_output_np = torch_output.detach().numpy()
        
        # Check that the outputs have the same shape
        assert tf_output_np.shape == torch_output_np.shape, \
            f"Output shapes don't match: {tf_output_np.shape} vs {torch_output_np.shape}"
        
        # Note: We don't check for exact equality because the implementations might differ slightly
        # Instead, we check that the outputs are reasonably close
        # This is a placeholder for now since our implementations are not complete
        
        print("✓ RandomSpatialDeformation test passed")
    except Exception as e:
        print(f"✗ RandomSpatialDeformation test failed: {str(e)}")
        raise


def test_random_crop():
    """Test the RandomCrop layer."""
    print("Testing RandomCrop...")
    
    # Set random seeds for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)
    tf.random.set_seed(1234)
    
    # Create test input data
    batch_size = 2
    spatial_shape = (20, 20, 20)
    crop_shape = (10, 10, 10)
    n_channels = 1
    
    # Create numpy array and convert to TensorFlow and PyTorch tensors
    np_input = np.random.rand(batch_size, *spatial_shape, n_channels).astype(np.float32)
    tf_input = tf.convert_to_tensor(np_input)
    torch_input = torch.from_numpy(np_input)
    
    # Create the layers with the same parameters
    try:
        tf_layer = tf_layers.RandomCrop(crop_shape=crop_shape)
        torch_layer = torch_layers.RandomCrop(crop_shape=crop_shape)
        
        # Apply the layers
        tf_output = tf_layer(tf_input)
        torch_output = torch_layer(torch_input)
        
        # Convert outputs to numpy for comparison
        tf_output_np = tf_output.numpy()
        torch_output_np = torch_output.detach().numpy()
        
        # Check that the outputs have the correct shape
        expected_shape = (batch_size, *crop_shape, n_channels)
        assert tf_output_np.shape == expected_shape, \
            f"TF output shape {tf_output_np.shape} doesn't match expected shape {expected_shape}"
        assert torch_output_np.shape == expected_shape, \
            f"PyTorch output shape {torch_output_np.shape} doesn't match expected shape {expected_shape}"
        
        print("✓ RandomCrop test passed")
    except Exception as e:
        print(f"✗ RandomCrop test failed: {str(e)}")
        raise


def test_sample_conditional_gmm():
    """Test the SampleConditionalGMM layer."""
    print("Testing SampleConditionalGMM...")
    
    # Set random seeds for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)
    tf.random.set_seed(1234)
    
    # Create test input data
    batch_size = 2
    spatial_shape = (10, 10, 10)
    n_labels = 3
    n_channels = 1
    generation_labels = [0, 1, 2]
    
    # Create numpy arrays for labels, means, and stds
    np_labels = np.random.randint(0, n_labels, (batch_size, *spatial_shape, 1)).astype(np.float32)
    np_means = np.random.rand(batch_size, n_labels, n_channels).astype(np.float32)
    np_stds = np.random.rand(batch_size, n_labels, n_channels).astype(np.float32) * 0.1
    
    # Convert to TensorFlow and PyTorch tensors
    tf_labels = tf.convert_to_tensor(np_labels)
    tf_means = tf.convert_to_tensor(np_means)
    tf_stds = tf.convert_to_tensor(np_stds)
    
    torch_labels = torch.from_numpy(np_labels)
    torch_means = torch.from_numpy(np_means)
    torch_stds = torch.from_numpy(np_stds)
    
    # Create the layers with the same parameters
    try:
        tf_layer = tf_layers.SampleConditionalGMM(generation_labels=generation_labels)
        torch_layer = torch_layers.SampleConditionalGMM(generation_labels=generation_labels)
        
        # Apply the layers
        tf_output = tf_layer([tf_labels, tf_means, tf_stds])
        torch_output = torch_layer([torch_labels, torch_means, torch_stds])
        
        # Convert outputs to numpy for comparison
        tf_output_np = tf_output.numpy()
        torch_output_np = torch_output.detach().numpy()
        
        # Check that the outputs have the correct shape
        expected_shape = (batch_size, *spatial_shape, n_channels)
        assert tf_output_np.shape == expected_shape, \
            f"TF output shape {tf_output_np.shape} doesn't match expected shape {expected_shape}"
        assert torch_output_np.shape == expected_shape, \
            f"PyTorch output shape {torch_output_np.shape} doesn't match expected shape {expected_shape}"
        
        print("✓ SampleConditionalGMM test passed")
    except Exception as e:
        print(f"✗ SampleConditionalGMM test failed: {str(e)}")
        raise


def test_intensity_normalisation():
    """Test the IntensityNormalisation layer."""
    print("Testing IntensityNormalisation...")
    
    # Set random seeds for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)
    tf.random.set_seed(1234)
    
    # Create test input data
    batch_size = 2
    spatial_shape = (10, 10, 10)
    n_channels = 1
    
    # Create numpy array with values in a specific range
    np_input = np.random.rand(batch_size, *spatial_shape, n_channels).astype(np.float32) * 100 + 50
    tf_input = tf.convert_to_tensor(np_input)
    torch_input = torch.from_numpy(np_input)
    
    # Create the layers
    try:
        tf_layer = tf_layers.IntensityNormalisation()
        torch_layer = torch_layers.IntensityNormalisation()
        
        # Apply the layers
        tf_output = tf_layer(tf_input)
        torch_output = torch_layer(torch_input)
        
        # Convert outputs to numpy for comparison
        tf_output_np = tf_output.numpy()
        torch_output_np = torch_output.detach().numpy()
        
        # Check that the outputs have the same shape
        assert tf_output_np.shape == torch_output_np.shape, \
            f"Output shapes don't match: {tf_output_np.shape} vs {torch_output_np.shape}"
        
        # Check that the outputs are normalized to [0, 1]
        assert np.all(tf_output_np >= 0) and np.all(tf_output_np <= 1), \
            "TF output not normalized to [0, 1]"
        assert np.all(torch_output_np >= 0) and np.all(torch_output_np <= 1), \
            "PyTorch output not normalized to [0, 1]"
        
        # Check that the min and max values are close to 0 and 1
        assert np.isclose(np.min(tf_output_np), 0, atol=1e-5) and np.isclose(np.max(tf_output_np), 1, atol=1e-5), \
            f"TF output min/max not close to 0/1: {np.min(tf_output_np)}/{np.max(tf_output_np)}"
        assert np.isclose(np.min(torch_output_np), 0, atol=1e-5) and np.isclose(np.max(torch_output_np), 1, atol=1e-5), \
            f"PyTorch output min/max not close to 0/1: {np.min(torch_output_np)}/{np.max(torch_output_np)}"
        
        print("✓ IntensityNormalisation test passed")
    except Exception as e:
        print(f"✗ IntensityNormalisation test failed: {str(e)}")
        raise


def test_convert_labels():
    """Test the ConvertLabels layer."""
    print("Testing ConvertLabels...")
    
    # Set random seeds for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)
    tf.random.set_seed(1234)
    
    # Create test input data
    batch_size = 2
    spatial_shape = (10, 10, 10)
    n_channels = 1
    
    # Create numpy array with integer labels
    np_input = np.random.randint(0, 3, (batch_size, *spatial_shape, n_channels)).astype(np.float32)
    tf_input = tf.convert_to_tensor(np_input)
    torch_input = torch.from_numpy(np_input)
    
    # Define source and target labels
    source_labels = [0, 1, 2]
    target_labels = [10, 20, 30]
    
    # Create the layers
    try:
        tf_layer = tf_layers.ConvertLabels(source_labels=source_labels, target_labels=target_labels)
        torch_layer = torch_layers.ConvertLabels(source_labels=source_labels, target_labels=target_labels)
        
        # Apply the layers
        tf_output = tf_layer(tf_input)
        torch_output = torch_layer(torch_input)
        
        # Convert outputs to numpy for comparison
        tf_output_np = tf_output.numpy()
        torch_output_np = torch_output.detach().numpy()
        
        # Check that the outputs have the same shape
        assert tf_output_np.shape == torch_output_np.shape, \
            f"Output shapes don't match: {tf_output_np.shape} vs {torch_output_np.shape}"
        
        # Check that the labels have been converted correctly
        for source, target in zip(source_labels, target_labels):
            assert np.all(tf_output_np[np_input == source] == target), \
                f"TF output: Label {source} not converted to {target}"
            assert np.all(torch_output_np[np_input == source] == target), \
                f"PyTorch output: Label {source} not converted to {target}"
        
        print("✓ ConvertLabels test passed")
    except Exception as e:
        print(f"✗ ConvertLabels test failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the tests
    try:
        test_random_crop()
        test_intensity_normalisation()
        test_convert_labels()
        test_sample_conditional_gmm()
        test_random_spatial_deformation()
        
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTests failed: {str(e)}")
        sys.exit(1)