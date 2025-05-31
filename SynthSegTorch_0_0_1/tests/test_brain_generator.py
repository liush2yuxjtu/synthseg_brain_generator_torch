import os
import sys
import unittest
import numpy as np
import torch
import tensorflow as tf

# Add the parent directory to the path so we can import the SynthSeg modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SynthSeg.brain_generator import BrainGenerator
from SynthSeg.brain_generator_tf import BrainGenerator_tf


class TestBrainGenerator(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for test label maps
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a simple label map for testing
        self.label_map_shape = (32, 32, 32)
        self.label_map = np.zeros(self.label_map_shape, dtype=np.int32)
        
        # Add some simple structures (e.g., a cube in the center)
        center = np.array(self.label_map_shape) // 2
        size = 10
        x_min, x_max = center[0] - size // 2, center[0] + size // 2
        y_min, y_max = center[1] - size // 2, center[1] + size // 2
        z_min, z_max = center[2] - size // 2, center[2] + size // 2
        
        # Label 1 for the cube
        self.label_map[x_min:x_max, y_min:y_max, z_min:z_max] = 1
        
        # Label 2 for a smaller cube inside
        inner_size = 4
        x_inner_min = center[0] - inner_size // 2
        x_inner_max = center[0] + inner_size // 2
        y_inner_min = center[1] - inner_size // 2
        y_inner_max = center[1] + inner_size // 2
        z_inner_min = center[2] - inner_size // 2
        z_inner_max = center[2] + inner_size // 2
        
        self.label_map[x_inner_min:x_inner_max, y_inner_min:y_inner_max, z_inner_min:z_inner_max] = 2
        
        # Save the label map
        self.label_map_path = os.path.join(self.test_dir, 'test_label_map.npy')
        np.save(self.label_map_path, self.label_map)
        
        # Common parameters for both generators
        self.path_label_maps = [self.label_map_path]
        self.n_neutral_labels = 3  # Background (0) + 2 labels
        self.generation_labels = np.arange(self.n_neutral_labels)
        self.output_labels = np.arange(self.n_neutral_labels)
        
    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.label_map_path):
            os.remove(self.label_map_path)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
    
    def test_brain_generator_creation(self):
        """Test that both PyTorch and TensorFlow generators can be created with the same parameters."""
        try:
            # Create PyTorch generator
            torch_generator = BrainGenerator(
                path_label_maps=self.path_label_maps,
                generation_labels=self.generation_labels,
                output_labels=self.output_labels,
                batch_size=1,
                n_channels=1,
                target_res=1.0,
                output_shape=None,
                output_div_by_n=None,
                generation_classes=None,
                prior_means=None,
                prior_stds=None,
                prior_distributions='uniform',
                use_specific_stats_for_channel=False,
                mix_prior_and_random=False
            )
            
            # Create TensorFlow generator
            tf_generator = BrainGenerator_tf(
                path_label_maps=self.path_label_maps,
                generation_labels=self.generation_labels,
                output_labels=self.output_labels,
                batch_size=1,
                n_channels=1,
                target_res=1.0,
                output_shape=None,
                output_div_by_n=None,
                generation_classes=None,
                prior_means=None,
                prior_stds=None,
                prior_distributions='uniform',
                use_specific_stats_for_channel=False,
                mix_prior_and_random=False
            )
            
            self.assertIsNotNone(torch_generator)
            self.assertIsNotNone(tf_generator)
            
            print("Both PyTorch and TensorFlow brain generators created successfully.")
            
        except Exception as e:
            self.fail(f"Failed to create brain generators: {str(e)}")
    
    def test_brain_generation_minimal(self):
        """Test brain generation with minimal augmentation."""
        try:
            # Create generators with minimal augmentation
            torch_generator = BrainGenerator(
                path_label_maps=self.path_label_maps,
                generation_labels=self.generation_labels,
                output_labels=self.output_labels,
                batch_size=1,
                n_channels=1,
                # Disable augmentations
                flipping=False,
                scaling_bounds=[1., 1.],
                rotation_bounds=[0., 0.],
                shearing_bounds=[0., 0.],
                translation_bounds=[0., 0.],
                nonlin_std=0.0,
                bias_field_std=0.0,
                blur_background=False
            )
            
            tf_generator = BrainGenerator_tf(
                path_label_maps=self.path_label_maps,
                generation_labels=self.generation_labels,
                output_labels=self.output_labels,
                batch_size=1,
                n_channels=1,
                # Disable augmentations
                flipping=False,
                scaling_bounds=[1., 1.],
                rotation_bounds=[0., 0.],
                shearing_bounds=[0., 0.],
                translation_bounds=[0., 0.],
                nonlin_std=0.0,
                bias_field_std=0.0,
                blur_background=False
            )
            
            # Generate images and segmentations
            torch_image, torch_labels = torch_generator.generate_brain()
            tf_image, tf_labels = tf_generator.generate_brain()
            
            # Convert PyTorch tensors to numpy if needed
            if isinstance(torch_image, torch.Tensor):
                torch_image = torch_image.cpu().numpy()
            if isinstance(torch_labels, torch.Tensor):
                torch_labels = torch_labels.cpu().numpy()
                
            # Check shapes
            self.assertEqual(torch_image.shape, tf_image.shape)
            self.assertEqual(torch_labels.shape, tf_labels.shape)
            
            # Check that the outputs are similar (not exactly equal due to implementation differences)
            # For labels, they should be identical since no augmentation is applied
            np.testing.assert_array_equal(torch_labels, tf_labels)
            
            # For images, allow some tolerance due to different implementations
            # The overall structure should be similar, but values might differ slightly
            # We'll check correlation instead of exact equality
            correlation = np.corrcoef(torch_image.flatten(), tf_image.flatten())[0, 1]
            self.assertGreater(correlation, 0.9, "Image correlation should be high")
            
            print("Brain generation with minimal augmentation passed.")
            
        except Exception as e:
            self.fail(f"Failed in brain generation with minimal augmentation: {str(e)}")
    
    def test_brain_generation_with_augmentation(self):
        """Test brain generation with augmentations enabled."""
        try:
            # Set random seeds for reproducibility
            np.random.seed(1234)
            torch.manual_seed(1234)
            tf.random.set_seed(1234)
            
            # Create generators with augmentations
            torch_generator = BrainGenerator(
                path_label_maps=self.path_label_maps,
                generation_labels=self.generation_labels,
                output_labels=self.output_labels,
                batch_size=1,
                n_channels=1,
                # Enable some augmentations
                flipping=True,
                scaling_bounds=[0.9, 1.1],
                rotation_bounds=[-10, 10],
                shearing_bounds=[-0.1, 0.1],
                translation_bounds=[-5, 5],
                nonlin_std=1.0,
                bias_field_std=0.3
            )
            
            tf_generator = BrainGenerator_tf(
                path_label_maps=self.path_label_maps,
                generation_labels=self.generation_labels,
                output_labels=self.output_labels,
                batch_size=1,
                n_channels=1,
                # Enable same augmentations
                flipping=True,
                scaling_bounds=[0.9, 1.1],
                rotation_bounds=[-10, 10],
                shearing_bounds=[-0.1, 0.1],
                translation_bounds=[-5, 5],
                nonlin_std=1.0,
                bias_field_std=0.3
            )
            
            # Generate images and segmentations
            torch_image, torch_labels = torch_generator.generate_brain()
            tf_image, tf_labels = tf_generator.generate_brain()
            
            # Convert PyTorch tensors to numpy if needed
            if isinstance(torch_image, torch.Tensor):
                torch_image = torch_image.cpu().numpy()
            if isinstance(torch_labels, torch.Tensor):
                torch_labels = torch_labels.cpu().numpy()
            
            # Check shapes
            self.assertEqual(torch_image.shape, tf_image.shape)
            self.assertEqual(torch_labels.shape, tf_labels.shape)
            
            # With augmentations, outputs will be different, but should have similar statistical properties
            # Check that both outputs have reasonable values
            self.assertTrue(np.min(torch_image) >= 0, "PyTorch image values should be non-negative")
            self.assertTrue(np.min(tf_image) >= 0, "TensorFlow image values should be non-negative")
            
            # Check that labels are still valid (integers within the expected range)
            self.assertTrue(np.all(np.unique(torch_labels) <= self.n_neutral_labels), 
                           "PyTorch labels should be within expected range")
            self.assertTrue(np.all(np.unique(tf_labels) <= self.n_neutral_labels), 
                           "TensorFlow labels should be within expected range")
            
            print("Brain generation with augmentation passed.")
            
        except Exception as e:
            self.fail(f"Failed in brain generation with augmentation: {str(e)}")


if __name__ == '__main__':
    unittest.main()