import os
import sys
import unittest
import numpy as np
import torch
import tensorflow as tf

# Add the parent directory to the path so we can import the SynthSeg modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PyTorch implementations
from SynthSeg.labels_to_image_model import labels_to_image_model

# Import TensorFlow implementations
from SynthSeg.labels_to_image_model_tf import labels_to_image_model_tf


class TestLabelsToImageModelFull(unittest.TestCase):
    
    def setUp(self):
        # Set random seeds for reproducibility
        np.random.seed(1234)
        torch.manual_seed(1234)
        tf.random.set_seed(1234)
        
        # Create sample data for testing
        self.batch_size = 2
        self.height = 32
        self.width = 32
        self.depth = 32
        self.n_channels = 1
        self.n_labels = 5
        
        # Define model parameters
        self.labels_shape = (self.height, self.width, self.depth, 1)
        self.generation_labels = np.array([0, 1, 2, 3, 4])
        self.n_neutral_labels = 1
        self.output_shape = (24, 24, 24)
        
        # Create random label maps
        self.labels_np = np.random.randint(
            0, self.n_labels, 
            (self.batch_size, self.height, self.width, self.depth, 1)
        ).astype(np.int32)
        
        # Create random means and standard deviations for the GMM
        self.means_np = np.random.rand(
            self.batch_size, self.n_labels, self.n_channels
        ).astype(np.float32)
        
        self.stds_np = np.random.rand(
            self.batch_size, self.n_labels, self.n_channels
        ).astype(np.float32) * 0.1 + 0.05  # Ensure positive std values
        
        # Create random prior means and standard deviations
        self.prior_means_np = np.random.rand(
            self.batch_size, self.n_labels, self.n_channels
        ).astype(np.float32)
        
        self.prior_stds_np = np.random.rand(
            self.batch_size, self.n_labels, self.n_channels
        ).astype(np.float32) * 0.1 + 0.05  # Ensure positive std values
        
        # Convert to PyTorch tensors
        self.labels_torch = torch.from_numpy(self.labels_np)
        self.means_torch = torch.from_numpy(self.means_np)
        self.stds_torch = torch.from_numpy(self.stds_np)
        self.prior_means_torch = torch.from_numpy(self.prior_means_np)
        self.prior_stds_torch = torch.from_numpy(self.prior_stds_np)
        
        # Convert to TensorFlow tensors
        self.labels_tf = tf.convert_to_tensor(self.labels_np)
        self.means_tf = tf.convert_to_tensor(self.means_np)
        self.stds_tf = tf.convert_to_tensor(self.stds_np)
        self.prior_means_tf = tf.convert_to_tensor(self.prior_means_np)
        self.prior_stds_tf = tf.convert_to_tensor(self.prior_stds_np)
    
    def test_model_creation_minimal(self):
        """Test the creation of the labels_to_image_model with minimal parameters."""
        print("Testing labels_to_image_model creation with minimal parameters...")
        
        try:
            # Create TensorFlow model with minimal parameters
            tf_l2i_model = labels_to_image_model_tf(
                labels_shape=self.labels_shape,
                n_channels=self.n_channels,
                generation_labels=self.generation_labels,
                n_neutral_labels=self.n_neutral_labels,
                apply_affine=False,
                nonlin_std=0.0,
                blur_range=0.0,
                bias_field_std=0.0,
                apply_bias_field=False,
                apply_intensity_augmentation=False,
                apply_gamma_augmentation=False
            )
            
            # Create PyTorch model with minimal parameters
            torch_l2i_model = labels_to_image_model(
                labels_shape=self.labels_shape,
                n_channels=self.n_channels,
                generation_labels=self.generation_labels,
                n_neutral_labels=self.n_neutral_labels,
                apply_affine=False,
                nonlin_std=0.0,
                blur_range=0.0,
                bias_field_std=0.0,
                apply_bias_field=False,
                apply_intensity_augmentation=False,
                apply_gamma_augmentation=False
            )
            
            # Check that models were created successfully
            self.assertIsNotNone(tf_l2i_model)
            self.assertIsNotNone(torch_l2i_model)
            
            print("Model creation with minimal parameters successful.")
            
        except Exception as e:
            self.fail(f"Error in test_model_creation_minimal: {str(e)}")
    
    def test_model_creation_full(self):
        """Test the creation of the labels_to_image_model with all parameters."""
        print("Testing labels_to_image_model creation with all parameters...")
        
        try:
            # Create TensorFlow model with all parameters
            tf_l2i_model = labels_to_image_model_tf(
                labels_shape=self.labels_shape,
                n_channels=self.n_channels,
                generation_labels=self.generation_labels,
                n_neutral_labels=self.n_neutral_labels,
                atlas_res=1.0,
                target_res=1.0,
                output_shape=self.output_shape,
                output_div_by_n=8,
                blur_range=1.15,
                bias_field_std=0.3,
                bias_shape_factor=0.025,
                gamma_std=0.1,
                apply_affine=True,
                scaling_bounds=0.15,
                rotation_bounds=15,
                shearing_bounds=0.012,
                translation_bounds=False,
                nonlin_std=3.0,
                nonlin_shape_factor=0.04,
                simulate_registration=False,
                flipping=True,
                apply_bias_field=True,
                apply_intensity_augmentation=True,
                apply_gamma_augmentation=True
            )
            
            # Create PyTorch model with all parameters
            torch_l2i_model = labels_to_image_model(
                labels_shape=self.labels_shape,
                n_channels=self.n_channels,
                generation_labels=self.generation_labels,
                n_neutral_labels=self.n_neutral_labels,
                atlas_res=1.0,
                target_res=1.0,
                output_shape=self.output_shape,
                output_div_by_n=8,
                blur_range=1.15,
                bias_field_std=0.3,
                bias_shape_factor=0.025,
                gamma_std=0.1,
                apply_affine=True,
                scaling_bounds=0.15,
                rotation_bounds=15,
                shearing_bounds=0.012,
                translation_bounds=False,
                nonlin_std=3.0,
                nonlin_shape_factor=0.04,
                simulate_registration=False,
                flipping=True,
                apply_bias_field=True,
                apply_intensity_augmentation=True,
                apply_gamma_augmentation=True
            )
            
            # Check that models were created successfully
            self.assertIsNotNone(tf_l2i_model)
            self.assertIsNotNone(torch_l2i_model)
            
            print("Model creation with all parameters successful.")
            
        except Exception as e:
            self.fail(f"Error in test_model_creation_full: {str(e)}")
    
    def test_forward_pass_minimal(self):
        """Test the forward pass of the labels_to_image_model with minimal augmentation."""
        print("Testing labels_to_image_model forward pass with minimal augmentation...")
        
        try:
            # Create TensorFlow model with minimal augmentation
            tf_l2i_model = labels_to_image_model_tf(
                labels_shape=self.labels_shape,
                n_channels=self.n_channels,
                generation_labels=self.generation_labels,
                n_neutral_labels=self.n_neutral_labels,
                apply_affine=False,
                nonlin_std=0.0,
                blur_range=0.0,
                bias_field_std=0.0,
                apply_bias_field=False,
                apply_intensity_augmentation=False,
                apply_gamma_augmentation=False
            )
            
            # Create PyTorch model with minimal augmentation
            torch_l2i_model = labels_to_image_model(
                labels_shape=self.labels_shape,
                n_channels=self.n_channels,
                generation_labels=self.generation_labels,
                n_neutral_labels=self.n_neutral_labels,
                apply_affine=False,
                nonlin_std=0.0,
                blur_range=0.0,
                bias_field_std=0.0,
                apply_bias_field=False,
                apply_intensity_augmentation=False,
                apply_gamma_augmentation=False
            )
            
            # Forward pass through TensorFlow model
            tf_image, tf_labels = tf_l2i_model([
                self.labels_tf, 
                self.means_tf, 
                self.stds_tf, 
                self.prior_means_tf, 
                self.prior_stds_tf
            ])
            
            # Forward pass through PyTorch model
            torch_image, torch_labels = torch_l2i_model(
                self.labels_torch, 
                self.means_torch, 
                self.stds_torch, 
                self.prior_means_torch, 
                self.prior_stds_torch
            )
            
            # Convert PyTorch outputs to numpy for comparison
            torch_image_np = torch_image.detach().numpy()
            torch_labels_np = torch_labels.detach().numpy()
            
            # Convert TensorFlow outputs to numpy for comparison
            tf_image_np = tf_image.numpy()
            tf_labels_np = tf_labels.numpy()
            
            # Check that outputs have the same shape
            self.assertEqual(torch_image_np.shape, tf_image_np.shape)
            self.assertEqual(torch_labels_np.shape, tf_labels_np.shape)
            
            # Check that outputs are similar (allowing for some numerical differences)
            # We use a high tolerance because the implementations might have some differences
            image_diff = np.mean(np.abs(torch_image_np - tf_image_np))
            print(f"Mean absolute difference in image outputs: {image_diff}")
            self.assertLess(image_diff, 0.5, "Image outputs differ significantly")
            
            # For labels, we expect them to be identical since there's no randomness
            # in the minimal configuration
            labels_equal = np.array_equal(torch_labels_np, tf_labels_np)
            if not labels_equal:
                print(f"Labels differ: TF unique: {np.unique(tf_labels_np)}, PyTorch unique: {np.unique(torch_labels_np)}")
            self.assertTrue(labels_equal, "Label outputs should be identical")
            
            print("Forward pass with minimal augmentation successful.")
            
        except Exception as e:
            self.fail(f"Error in test_forward_pass_minimal: {str(e)}")
    
    def test_output_shape(self):
        """Test that the output shape is correct when specifying output_shape."""
        print("Testing output shape specification...")
        
        try:
            # Create TensorFlow model with output shape
            tf_l2i_model = labels_to_image_model_tf(
                labels_shape=self.labels_shape,
                n_channels=self.n_channels,
                generation_labels=self.generation_labels,
                n_neutral_labels=self.n_neutral_labels,
                output_shape=self.output_shape,
                apply_affine=False,
                nonlin_std=0.0,
                blur_range=0.0,
                bias_field_std=0.0,
                apply_bias_field=False,
                apply_intensity_augmentation=False,
                apply_gamma_augmentation=False
            )
            
            # Create PyTorch model with output shape
            torch_l2i_model = labels_to_image_model(
                labels_shape=self.labels_shape,
                n_channels=self.n_channels,
                generation_labels=self.generation_labels,
                n_neutral_labels=self.n_neutral_labels,
                output_shape=self.output_shape,
                apply_affine=False,
                nonlin_std=0.0,
                blur_range=0.0,
                bias_field_std=0.0,
                apply_bias_field=False,
                apply_intensity_augmentation=False,
                apply_gamma_augmentation=False
            )
            
            # Forward pass through TensorFlow model
            tf_image, tf_labels = tf_l2i_model([
                self.labels_tf, 
                self.means_tf, 
                self.stds_tf, 
                self.prior_means_tf, 
                self.prior_stds_tf
            ])
            
            # Forward pass through PyTorch model
            torch_image, torch_labels = torch_l2i_model(
                self.labels_torch, 
                self.means_torch, 
                self.stds_torch, 
                self.prior_means_torch, 
                self.prior_stds_torch
            )
            
            # Check that outputs have the correct shape
            expected_shape = (self.batch_size,) + self.output_shape + (self.n_channels,)
            self.assertEqual(tf_image.shape, expected_shape)
            self.assertEqual(torch_image.shape, expected_shape)
            
            expected_labels_shape = (self.batch_size,) + self.output_shape + (1,)
            self.assertEqual(tf_labels.shape, expected_labels_shape)
            self.assertEqual(torch_labels.shape, expected_labels_shape)
            
            print("Output shape test successful.")
            
        except Exception as e:
            self.fail(f"Error in test_output_shape: {str(e)}")


if __name__ == '__main__':
    unittest.main()