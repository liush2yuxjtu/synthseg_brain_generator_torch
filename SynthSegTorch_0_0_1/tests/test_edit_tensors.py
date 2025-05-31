"""Test file for comparing TensorFlow and PyTorch implementations of edit_tensors.

This file contains tests that compare the functionality of the TensorFlow and PyTorch
implementations of the utility functions for editing tensors in lab2im.
"""

import os
import sys
import numpy as np
import torch
import tensorflow as tf

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the TensorFlow and PyTorch implementations
from ext.lab2im import edit_tensors as torch_et
from ext.lab2im import edit_tensors_tf as tf_et


def test_get_padding_shape():
    """Test the get_padding_shape function."""
    print("Testing get_padding_shape...")
    
    try:
        # Test with different shapes and parameters
        shape_3d = (10, 10, 10)
        shape_2d = (10, 10)
        
        # Test with default parameters
        torch_padding_3d = torch_et.get_padding_shape(shape_3d)
        tf_padding_3d = tf_et.get_padding_shape(shape_3d)
        
        assert torch_padding_3d == tf_padding_3d, \
            f"Padding shapes don't match for 3D: {torch_padding_3d} vs {tf_padding_3d}"
        
        # Test with custom parameters
        torch_padding_2d = torch_et.get_padding_shape(shape_2d, stride=2, kernel_size=5, dilation=2)
        tf_padding_2d = tf_et.get_padding_shape(shape_2d, stride=2, kernel_size=5, dilation=2)
        
        assert torch_padding_2d == tf_padding_2d, \
            f"Padding shapes don't match for 2D with custom params: {torch_padding_2d} vs {tf_padding_2d}"
        
        print("✓ get_padding_shape test passed")
    except Exception as e:
        print(f"✗ get_padding_shape test failed: {str(e)}")
        raise


def test_gaussian_kernel():
    """Test the gaussian_kernel function."""
    print("Testing gaussian_kernel...")
    
    try:
        # Test with different shapes and sigmas
        shape_1d = [5]
        shape_2d = [5, 5]
        shape_3d = [5, 5, 5]
        sigma_1d = [1.0]
        sigma_2d = [1.0, 1.0]
        sigma_3d = [1.0, 1.0, 1.0]
        
        # Test 1D kernel
        torch_kernel_1d = torch_et.gaussian_kernel(shape_1d, sigma_1d).cpu().numpy()
        tf_kernel_1d = tf_et.gaussian_kernel(shape_1d, sigma_1d).numpy()
        
        assert np.allclose(torch_kernel_1d, tf_kernel_1d, atol=1e-5), \
            f"1D Gaussian kernels don't match"
        
        # Test 2D kernel
        torch_kernel_2d = torch_et.gaussian_kernel(shape_2d, sigma_2d).cpu().numpy()
        tf_kernel_2d = tf_et.gaussian_kernel(shape_2d, sigma_2d).numpy()
        
        assert np.allclose(torch_kernel_2d, tf_kernel_2d, atol=1e-5), \
            f"2D Gaussian kernels don't match"
        
        # Test 3D kernel
        torch_kernel_3d = torch_et.gaussian_kernel(shape_3d, sigma_3d).cpu().numpy()
        tf_kernel_3d = tf_et.gaussian_kernel(shape_3d, sigma_3d).numpy()
        
        assert np.allclose(torch_kernel_3d, tf_kernel_3d, atol=1e-5), \
            f"3D Gaussian kernels don't match"
        
        print("✓ gaussian_kernel test passed")
    except Exception as e:
        print(f"✗ gaussian_kernel test failed: {str(e)}")
        raise


def test_crop_tensor():
    """Test the crop_tensor function."""
    print("Testing crop_tensor...")
    
    # Set random seeds for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)
    tf.random.set_seed(1234)
    
    try:
        # Create test input data
        batch_size = 2
        spatial_shape = (20, 20, 20)
        crop_shape = (10, 10, 10)
        n_channels = 1
        
        # Create numpy array and convert to TensorFlow and PyTorch tensors
        np_input = np.random.rand(batch_size, *spatial_shape, n_channels).astype(np.float32)
        tf_input = tf.convert_to_tensor(np_input)
        torch_input = torch.from_numpy(np_input)
        
        # Apply cropping
        tf_output = tf_et.crop_tensor(tf_input, crop_shape)
        torch_output = torch_et.crop_tensor(torch_input, crop_shape)
        
        # Convert outputs to numpy for comparison
        tf_output_np = tf_output.numpy()
        torch_output_np = torch_output.detach().numpy()
        
        # Check that the outputs have the correct shape
        expected_shape = (batch_size, *crop_shape, n_channels)
        assert tf_output_np.shape == expected_shape, \
            f"TF output shape {tf_output_np.shape} doesn't match expected shape {expected_shape}"
        assert torch_output_np.shape == expected_shape, \
            f"PyTorch output shape {torch_output_np.shape} doesn't match expected shape {expected_shape}"
        
        # Check that the outputs are the same
        assert np.allclose(tf_output_np, torch_output_np, atol=1e-5), \
            f"Cropped tensors don't match"
        
        print("✓ crop_tensor test passed")
    except Exception as e:
        print(f"✗ crop_tensor test failed: {str(e)}")
        raise


def test_rescale_tensor_values():
    """Test the rescale_tensor_values function."""
    print("Testing rescale_tensor_values...")
    
    # Set random seeds for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)
    tf.random.set_seed(1234)
    
    try:
        # Create test input data
        batch_size = 2
        spatial_shape = (10, 10, 10)
        n_channels = 1
        
        # Create numpy array with values in a specific range
        np_input = np.random.rand(batch_size, *spatial_shape, n_channels).astype(np.float32) * 100 + 50
        tf_input = tf.convert_to_tensor(np_input)
        torch_input = torch.from_numpy(np_input)
        
        # Apply rescaling
        new_min, new_max = -1, 1
        tf_output = tf_et.rescale_tensor_values(tf_input, new_min, new_max)
        torch_output = torch_et.rescale_tensor_values(torch_input, new_min, new_max)
        
        # Convert outputs to numpy for comparison
        tf_output_np = tf_output.numpy()
        torch_output_np = torch_output.detach().numpy()
        
        # Check that the outputs have the same shape
        assert tf_output_np.shape == torch_output_np.shape, \
            f"Output shapes don't match: {tf_output_np.shape} vs {torch_output_np.shape}"
        
        # Check that the outputs are rescaled to [new_min, new_max]
        assert np.all(tf_output_np >= new_min) and np.all(tf_output_np <= new_max), \
            f"TF output not rescaled to [{new_min}, {new_max}]"
        assert np.all(torch_output_np >= new_min) and np.all(torch_output_np <= new_max), \
            f"PyTorch output not rescaled to [{new_min}, {new_max}]"
        
        # Check that the min and max values are close to new_min and new_max
        assert np.isclose(np.min(tf_output_np), new_min, atol=1e-5) and np.isclose(np.max(tf_output_np), new_max, atol=1e-5), \
            f"TF output min/max not close to {new_min}/{new_max}: {np.min(tf_output_np)}/{np.max(tf_output_np)}"
        assert np.isclose(np.min(torch_output_np), new_min, atol=1e-5) and np.isclose(np.max(torch_output_np), new_max, atol=1e-5), \
            f"PyTorch output min/max not close to {new_min}/{new_max}: {np.min(torch_output_np)}/{np.max(torch_output_np)}"
        
        # Check that the outputs are similar
        assert np.allclose(tf_output_np, torch_output_np, atol=1e-5), \
            f"Rescaled tensors don't match"
        
        print("✓ rescale_tensor_values test passed")
    except Exception as e:
        print(f"✗ rescale_tensor_values test failed: {str(e)}")
        raise


def test_add_noise():
    """Test the add_noise function."""
    print("Testing add_noise...")
    
    # Set random seeds for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)
    tf.random.set_seed(1234)
    
    try:
        # Create test input data
        batch_size = 2
        spatial_shape = (10, 10, 10)
        n_channels = 1
        
        # Create numpy array
        np_input = np.random.rand(batch_size, *spatial_shape, n_channels).astype(np.float32)
        tf_input = tf.convert_to_tensor(np_input)
        torch_input = torch.from_numpy(np_input)
        
        # Apply noise
        sigma = 0.1
        tf_output = tf_et.add_noise(tf_input, sigma)
        torch_output = torch_et.add_noise(torch_input, sigma)
        
        # Convert outputs to numpy for comparison
        tf_output_np = tf_output.numpy()
        torch_output_np = torch_output.detach().numpy()
        
        # Check that the outputs have the same shape
        assert tf_output_np.shape == torch_output_np.shape, \
            f"Output shapes don't match: {tf_output_np.shape} vs {torch_output_np.shape}"
        
        # Check that noise was added (outputs should be different from inputs)
        assert not np.allclose(np_input, tf_output_np, atol=1e-5), \
            f"TF output is the same as input (no noise added)"
        assert not np.allclose(np_input, torch_output_np, atol=1e-5), \
            f"PyTorch output is the same as input (no noise added)"
        
        # Note: We can't directly compare the TF and PyTorch outputs because the random noise will be different
        # But we can check that the noise has the expected standard deviation
        tf_noise = tf_output_np - np_input
        torch_noise = torch_output_np - np_input
        
        assert np.isclose(np.std(tf_noise), sigma, atol=1e-2), \
            f"TF noise std {np.std(tf_noise)} not close to expected {sigma}"
        assert np.isclose(np.std(torch_noise), sigma, atol=1e-2), \
            f"PyTorch noise std {np.std(torch_noise)} not close to expected {sigma}"
        
        print("✓ add_noise test passed")
    except Exception as e:
        print(f"✗ add_noise test failed: {str(e)}")
        raise


def test_convert_labels():
    """Test the convert_labels function."""
    print("Testing convert_labels...")
    
    # Set random seeds for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)
    tf.random.set_seed(1234)
    
    try:
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
        
        # Apply label conversion
        tf_output = tf_et.convert_labels(tf_input, source_labels, target_labels)
        torch_output = torch_et.convert_labels(torch_input, source_labels, target_labels)
        
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
        
        # Check that the outputs are the same
        assert np.allclose(tf_output_np, torch_output_np, atol=1e-5), \
            f"Converted label tensors don't match"
        
        print("✓ convert_labels test passed")
    except Exception as e:
        print(f"✗ convert_labels test failed: {str(e)}")
        raise


def test_one_hot_encoding():
    """Test the one_hot_encoding function."""
    print("Testing one_hot_encoding...")
    
    # Set random seeds for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)
    tf.random.set_seed(1234)
    
    try:
        # Create test input data
        batch_size = 2
        spatial_shape = (5, 5, 5)
        n_channels = 1
        n_classes = 4
        
        # Create numpy array with integer labels
        np_input = np.random.randint(0, n_classes, (batch_size, *spatial_shape, n_channels)).astype(np.float32)
        tf_input = tf.convert_to_tensor(np_input)
        torch_input = torch.from_numpy(np_input)
        
        # Apply one-hot encoding
        tf_output = tf_et.one_hot_encoding(tf_input, n_classes)
        torch_output = torch_et.one_hot_encoding(torch_input, n_classes)
        
        # Convert outputs to numpy for comparison
        tf_output_np = tf_output.numpy()
        torch_output_np = torch_output.detach().numpy()
        
        # Check that the outputs have the correct shape
        expected_shape = (batch_size, *spatial_shape, n_classes)
        assert tf_output_np.shape == expected_shape, \
            f"TF output shape {tf_output_np.shape} doesn't match expected shape {expected_shape}"
        assert torch_output_np.shape == expected_shape, \
            f"PyTorch output shape {torch_output_np.shape} doesn't match expected shape {expected_shape}"
        
        # Check that the outputs are one-hot encoded
        assert np.all(np.sum(tf_output_np, axis=-1) == 1), \
            f"TF output is not one-hot encoded"
        assert np.all(np.sum(torch_output_np, axis=-1) == 1), \
            f"PyTorch output is not one-hot encoded"
        
        # Check that the outputs are the same
        assert np.allclose(tf_output_np, torch_output_np, atol=1e-5), \
            f"One-hot encoded tensors don't match"
        
        print("✓ one_hot_encoding test passed")
    except Exception as e:
        print(f"✗ one_hot_encoding test failed: {str(e)}")
        raise


def test_mask_tensor():
    """Test the mask_tensor function."""
    print("Testing mask_tensor...")
    
    # Set random seeds for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)
    tf.random.set_seed(1234)
    
    try:
        # Create test input data
        batch_size = 2
        spatial_shape = (10, 10, 10)
        n_channels = 1
        
        # Create numpy arrays for input and mask
        np_input = np.random.rand(batch_size, *spatial_shape, n_channels).astype(np.float32)
        np_mask = np.random.randint(0, 2, (batch_size, *spatial_shape, n_channels)).astype(np.float32)
        
        # Convert to TensorFlow and PyTorch tensors
        tf_input = tf.convert_to_tensor(np_input)
        tf_mask = tf.convert_to_tensor(np_mask)
        torch_input = torch.from_numpy(np_input)
        torch_mask = torch.from_numpy(np_mask)
        
        # Apply masking
        mask_value = -1
        tf_output = tf_et.mask_tensor(tf_input, tf_mask, mask_value)
        torch_output = torch_et.mask_tensor(torch_input, torch_mask, mask_value)
        
        # Convert outputs to numpy for comparison
        tf_output_np = tf_output.numpy()
        torch_output_np = torch_output.detach().numpy()
        
        # Check that the outputs have the same shape
        assert tf_output_np.shape == torch_output_np.shape, \
            f"Output shapes don't match: {tf_output_np.shape} vs {torch_output_np.shape}"
        
        # Check that the masking was applied correctly
        assert np.all(tf_output_np[np_mask > 0] == np_input[np_mask > 0]), \
            f"TF output: Mask not applied correctly (masked values changed)"
        assert np.all(tf_output_np[np_mask == 0] == mask_value), \
            f"TF output: Mask not applied correctly (unmasked values not set to {mask_value})"
        
        assert np.all(torch_output_np[np_mask > 0] == np_input[np_mask > 0]), \
            f"PyTorch output: Mask not applied correctly (masked values changed)"
        assert np.all(torch_output_np[np_mask == 0] == mask_value), \
            f"PyTorch output: Mask not applied correctly (unmasked values not set to {mask_value})"
        
        # Check that the outputs are the same
        assert np.allclose(tf_output_np, torch_output_np, atol=1e-5), \
            f"Masked tensors don't match"
        
        print("✓ mask_tensor test passed")
    except Exception as e:
        print(f"✗ mask_tensor test failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the tests
    try:
        test_get_padding_shape()
        test_gaussian_kernel()
        test_crop_tensor()
        test_rescale_tensor_values()
        test_add_noise()
        test_convert_labels()
        test_one_hot_encoding()
        test_mask_tensor()
        
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTests failed: {str(e)}")
        sys.exit(1)