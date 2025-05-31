"""Test file for comparing TensorFlow and PyTorch implementations of neuron layers.

This file contains tests that compare the functionality of the TensorFlow and PyTorch
implementations of the custom neural network layers in the neuron module.
"""

import os
import sys
import numpy as np
import torch
import tensorflow as tf

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the TensorFlow and PyTorch implementations
from ext.neuron import layers as torch_layers
from ext.neuron import layers_tf as tf_layers


def test_spatial_transformer_linear():
    """Test the SpatialTransformer layer with linear interpolation."""
    print("Testing SpatialTransformer with linear interpolation...")
    
    # Set random seeds for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)
    tf.random.set_seed(1234)
    
    try:
        # Create test input data
        batch_size = 2
        spatial_shape = (10, 10, 10)
        n_channels = 1
        n_dims = len(spatial_shape)
        
        # Create input tensor
        np_input = np.random.rand(batch_size, *spatial_shape, n_channels).astype(np.float32)
        tf_input = tf.convert_to_tensor(np_input)
        torch_input = torch.from_numpy(np_input)
        
        # Create deformation field (small random displacements)
        np_deformation = np.random.normal(0, 0.5, (batch_size, *spatial_shape, n_dims)).astype(np.float32)
        tf_deformation = tf.convert_to_tensor(np_deformation)
        torch_deformation = torch.from_numpy(np_deformation)
        
        # Create transformer layers
        tf_transformer = tf_layers.SpatialTransformer(interp_method='linear')
        torch_transformer = torch_layers.SpatialTransformer(interp_method='linear')
        
        # Apply transformations
        tf_output = tf_transformer([tf_input, tf_deformation])
        torch_output = torch_transformer(torch_input, torch_deformation)
        
        # Convert outputs to numpy for comparison
        tf_output_np = tf_output.numpy()
        torch_output_np = torch_output.detach().numpy()
        
        # Check that the outputs have the same shape
        assert tf_output_np.shape == torch_output_np.shape, \
            f"Output shapes don't match: {tf_output_np.shape} vs {torch_output_np.shape}"
        
        # Check that the outputs are similar
        # Note: Due to implementation differences, we use a higher tolerance
        assert np.allclose(tf_output_np, torch_output_np, atol=1e-2), \
            f"Transformed tensors don't match"
        
        print("✓ SpatialTransformer (linear) test passed")
    except Exception as e:
        print(f"✗ SpatialTransformer (linear) test failed: {str(e)}")
        raise


def test_spatial_transformer_nearest():
    """Test the SpatialTransformer layer with nearest interpolation."""
    print("Testing SpatialTransformer with nearest interpolation...")
    
    # Set random seeds for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)
    tf.random.set_seed(1234)
    
    try:
        # Create test input data
        batch_size = 2
        spatial_shape = (10, 10, 10)
        n_channels = 1
        n_dims = len(spatial_shape)
        
        # Create input tensor with discrete values
        np_input = np.random.randint(0, 5, (batch_size, *spatial_shape, n_channels)).astype(np.float32)
        tf_input = tf.convert_to_tensor(np_input)
        torch_input = torch.from_numpy(np_input)
        
        # Create deformation field (small random displacements)
        np_deformation = np.random.normal(0, 0.5, (batch_size, *spatial_shape, n_dims)).astype(np.float32)
        tf_deformation = tf.convert_to_tensor(np_deformation)
        torch_deformation = torch.from_numpy(np_deformation)
        
        # Create transformer layers
        tf_transformer = tf_layers.SpatialTransformer(interp_method='nearest')
        torch_transformer = torch_layers.SpatialTransformer(interp_method='nearest')
        
        # Apply transformations
        tf_output = tf_transformer([tf_input, tf_deformation])
        torch_output = torch_transformer(torch_input, torch_deformation)
        
        # Convert outputs to numpy for comparison
        tf_output_np = tf_output.numpy()
        torch_output_np = torch_output.detach().numpy()
        
        # Check that the outputs have the same shape
        assert tf_output_np.shape == torch_output_np.shape, \
            f"Output shapes don't match: {tf_output_np.shape} vs {torch_output_np.shape}"
        
        # Check that the outputs are similar
        # Note: Due to implementation differences, we use a higher tolerance
        assert np.allclose(tf_output_np, torch_output_np, atol=1e-2), \
            f"Transformed tensors don't match"
        
        print("✓ SpatialTransformer (nearest) test passed")
    except Exception as e:
        print(f"✗ SpatialTransformer (nearest) test failed: {str(e)}")
        raise


def test_spatial_transformer_single_transform():
    """Test the SpatialTransformer layer with single_transform=True."""
    print("Testing SpatialTransformer with single_transform=True...")
    
    # Set random seeds for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)
    tf.random.set_seed(1234)
    
    try:
        # Create test input data
        batch_size = 3
        spatial_shape = (8, 8, 8)
        n_channels = 1
        n_dims = len(spatial_shape)
        
        # Create input tensor
        np_input = np.random.rand(batch_size, *spatial_shape, n_channels).astype(np.float32)
        tf_input = tf.convert_to_tensor(np_input)
        torch_input = torch.from_numpy(np_input)
        
        # Create deformation field (only one transform that will be applied to all inputs)
        np_deformation = np.random.normal(0, 0.5, (1, *spatial_shape, n_dims)).astype(np.float32)
        tf_deformation = tf.convert_to_tensor(np_deformation)
        torch_deformation = torch.from_numpy(np_deformation)
        
        # Create transformer layers with single_transform=True
        tf_transformer = tf_layers.SpatialTransformer(interp_method='linear', single_transform=True)
        torch_transformer = torch_layers.SpatialTransformer(interp_method='linear', single_transform=True)
        
        # Apply transformations
        tf_output = tf_transformer([tf_input, tf_deformation])
        torch_output = torch_transformer(torch_input, torch_deformation)
        
        # Convert outputs to numpy for comparison
        tf_output_np = tf_output.numpy()
        torch_output_np = torch_output.detach().numpy()
        
        # Check that the outputs have the same shape
        assert tf_output_np.shape == torch_output_np.shape, \
            f"Output shapes don't match: {tf_output_np.shape} vs {torch_output_np.shape}"
        
        # Check that the outputs are similar
        assert np.allclose(tf_output_np, torch_output_np, atol=1e-2), \
            f"Transformed tensors don't match"
        
        print("✓ SpatialTransformer (single_transform) test passed")
    except Exception as e:
        print(f"✗ SpatialTransformer (single_transform) test failed: {str(e)}")
        raise


def test_vec_int():
    """Test the VecInt layer."""
    print("Testing VecInt...")
    
    # Set random seeds for reproducibility
    np.random.seed(1234)
    torch.manual_seed(1234)
    tf.random.set_seed(1234)
    
    try:
        # Create test input data
        batch_size = 2
        spatial_shape = (8, 8, 8)
        n_dims = len(spatial_shape)
        
        # Create velocity field (small random velocities)
        np_velocity = np.random.normal(0, 0.1, (batch_size, *spatial_shape, n_dims)).astype(np.float32)
        tf_velocity = tf.convert_to_tensor(np_velocity)
        torch_velocity = torch.from_numpy(np_velocity)
        
        # Create VecInt layers
        tf_vec_int = tf_layers.VecInt(inshape=(*spatial_shape, n_dims), nsteps=3)
        torch_vec_int = torch_layers.VecInt(inshape=(*spatial_shape, n_dims), nsteps=3)
        
        # Apply vector integration
        tf_output = tf_vec_int(tf_velocity)
        torch_output = torch_vec_int(torch_velocity)
        
        # Convert outputs to numpy for comparison
        tf_output_np = tf_output.numpy()
        torch_output_np = torch_output.detach().numpy()
        
        # Check that the outputs have the same shape
        assert tf_output_np.shape == torch_output_np.shape, \
            f"Output shapes don't match: {tf_output_np.shape} vs {torch_output_np.shape}"
        
        # Check that the outputs are similar
        # Note: Due to implementation differences, we use a higher tolerance
        assert np.allclose(tf_output_np, torch_output_np, atol=1e-1), \
            f"Integrated vector fields don't match"
        
        print("✓ VecInt test passed")
    except Exception as e:
        print(f"✗ VecInt test failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the tests
    try:
        test_spatial_transformer_linear()
        test_spatial_transformer_nearest()
        test_spatial_transformer_single_transform()
        test_vec_int()
        
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTests failed: {str(e)}")
        sys.exit(1)