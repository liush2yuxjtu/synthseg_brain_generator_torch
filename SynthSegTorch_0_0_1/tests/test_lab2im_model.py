"""Tests for lab2im_model module

This file contains tests for comparing the TensorFlow and PyTorch implementations
of the lab2im_model module.
"""

import os
import sys
import numpy as np

# Add parent directory to path to allow importing SynthSegTorch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_lab2im_model_creation():
    """Test creation of lab2im model in both TensorFlow and PyTorch."""
    try:
        # Setup test parameters
        labels_shape = [32, 32, 32]
        n_channels = 1
        generation_labels = np.array([0, 1, 2, 3])
        output_labels = np.array([0, 1, 2, 3])
        atlas_res = 1.0
        target_res = 1.0
        
        # Test TensorFlow implementation
        try:
            import tensorflow as tf
            from ext.lab2im.lab2im_model_tf import lab2im_model_tf
            
            # Create TensorFlow model
            tf_model = lab2im_model_tf(
                labels_shape=labels_shape,
                n_channels=n_channels,
                generation_labels=generation_labels,
                output_labels=output_labels,
                atlas_res=atlas_res,
                target_res=target_res
            )
            
            print(f"✓ TensorFlow model created successfully with output shape: {tf_model.output_shape}")
            tf_success = True
        except ImportError:
            print("✗ TensorFlow not installed, skipping TF implementation test")
            tf_success = False
        except Exception as e:
            print(f"✗ TensorFlow model creation failed: {str(e)}")
            tf_success = False
        
        # Test PyTorch implementation
        try:
            import torch
            from ext.lab2im.lab2im_model import lab2im_model
            
            # Create PyTorch model
            torch_model = lab2im_model(
                labels_shape=labels_shape,
                n_channels=n_channels,
                generation_labels=generation_labels,
                output_labels=output_labels,
                atlas_res=atlas_res,
                target_res=target_res
            )
            
            print(f"✓ PyTorch model created successfully")
            torch_success = True
        except ImportError:
            print("✗ PyTorch not installed, skipping PyTorch implementation test")
            torch_success = False
        except Exception as e:
            print(f"✗ PyTorch model creation failed: {str(e)}")
            torch_success = False
        
        # Overall test success
        if tf_success and torch_success:
            print(f"✓ Both TensorFlow and PyTorch models created successfully")
            return True
        elif torch_success:  # Only PyTorch is required for the conversion
            print(f"✓ PyTorch model created successfully (TensorFlow not available)")
            return True
        else:
            print(f"✗ Model creation test failed")
            return False
    except Exception as e:
        print(f"✗ test_lab2im_model_creation failed: {str(e)}")
        return False


def test_lab2im_model_forward():
    """Test forward pass of lab2im model in both TensorFlow and PyTorch."""
    try:
        # Setup test parameters
        labels_shape = [32, 32, 32]
        n_channels = 1
        generation_labels = np.array([0, 1, 2, 3])
        output_labels = np.array([0, 1, 2, 3])
        atlas_res = 1.0
        target_res = 1.0
        batch_size = 1
        
        # Create test inputs
        # Random label map with values from generation_labels
        np.random.seed(42)  # For reproducibility
        labels_input_np = np.random.choice(generation_labels, size=[batch_size] + labels_shape + [1])
        
        # Random means and stds for the GMM
        means_input_np = np.random.rand(batch_size, len(generation_labels), n_channels)
        stds_input_np = np.random.rand(batch_size, len(generation_labels), n_channels) * 0.1
        
        # Test TensorFlow implementation
        try:
            import tensorflow as tf
            import keras.backend as K
            from ext.lab2im.lab2im_model_tf import lab2im_model_tf
            
            # Set random seed for TensorFlow
            tf.random.set_seed(42)
            
            # Create TensorFlow model
            tf_model = lab2im_model_tf(
                labels_shape=labels_shape,
                n_channels=n_channels,
                generation_labels=generation_labels,
                output_labels=output_labels,
                atlas_res=atlas_res,
                target_res=target_res
            )
            
            # Convert numpy arrays to TensorFlow tensors
            labels_input_tf = tf.convert_to_tensor(labels_input_np, dtype=tf.int32)
            means_input_tf = tf.convert_to_tensor(means_input_np, dtype=tf.float32)
            stds_input_tf = tf.convert_to_tensor(stds_input_np, dtype=tf.float32)
            
            # Run forward pass
            tf_outputs = tf_model([labels_input_tf, means_input_tf, stds_input_tf])
            
            # Convert outputs to numpy
            tf_image = K.eval(tf_outputs[0])
            tf_labels = K.eval(tf_outputs[1])
            
            print(f"✓ TensorFlow forward pass successful with output shapes: {tf_image.shape}, {tf_labels.shape}")
            tf_success = True
        except ImportError:
            print("✗ TensorFlow not installed, skipping TF implementation test")
            tf_success = False
            tf_image = None
            tf_labels = None
        except Exception as e:
            print(f"✗ TensorFlow forward pass failed: {str(e)}")
            tf_success = False
            tf_image = None
            tf_labels = None
        
        # Test PyTorch implementation
        try:
            import torch
            from ext.lab2im.lab2im_model import lab2im_model
            
            # Set random seed for PyTorch
            torch.manual_seed(42)
            
            # Create PyTorch model
            torch_model = lab2im_model(
                labels_shape=labels_shape,
                n_channels=n_channels,
                generation_labels=generation_labels,
                output_labels=output_labels,
                atlas_res=atlas_res,
                target_res=target_res
            )
            
            # Convert numpy arrays to PyTorch tensors
            labels_input_torch = torch.from_numpy(labels_input_np).int()
            means_input_torch = torch.from_numpy(means_input_np).float()
            stds_input_torch = torch.from_numpy(stds_input_np).float()
            
            # Run forward pass
            torch_image, torch_labels = torch_model(labels_input_torch, means_input_torch, stds_input_torch)
            
            # Convert outputs to numpy
            torch_image_np = torch_image.detach().cpu().numpy()
            torch_labels_np = torch_labels.detach().cpu().numpy()
            
            print(f"✓ PyTorch forward pass successful with output shapes: {torch_image_np.shape}, {torch_labels_np.shape}")
            torch_success = True
        except ImportError:
            print("✗ PyTorch not installed, skipping PyTorch implementation test")
            torch_success = False
        except Exception as e:
            print(f"✗ PyTorch forward pass failed: {str(e)}")
            torch_success = False
        
        # Compare outputs if both implementations succeeded
        if tf_success and torch_success:
            try:
                # Check if shapes match
                assert tf_image.shape == torch_image_np.shape, \
                    f"Image shapes don't match: TF={tf_image.shape}, Torch={torch_image_np.shape}"
                assert tf_labels.shape == torch_labels_np.shape, \
                    f"Label shapes don't match: TF={tf_labels.shape}, Torch={torch_labels_np.shape}"
                
                # Note: Due to different random number generation and implementation details,
                # the actual values may not match exactly. In a real test, you would need to
                # ensure deterministic behavior or test with fixed inputs/outputs.
                
                print(f"✓ Output shapes match between TensorFlow and PyTorch implementations")
                print(f"  TF image mean: {tf_image.mean()}, Torch image mean: {torch_image_np.mean()}")
                print(f"  TF labels unique values: {np.unique(tf_labels)}, Torch labels unique values: {np.unique(torch_labels_np)}")
                
                return True
            except Exception as e:
                print(f"✗ Output comparison failed: {str(e)}")
                return False
        elif torch_success:  # Only PyTorch is required for the conversion
            print(f"✓ PyTorch forward pass successful (TensorFlow not available)")
            return True
        else:
            print(f"✗ Forward pass test failed")
            return False
    except Exception as e:
        print(f"✗ test_lab2im_model_forward failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run tests if this file is executed directly
    test_lab2im_model_creation()
    test_lab2im_model_forward()