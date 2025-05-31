import os
import sys
import unittest
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the SynthSeg modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SynthSeg.brain_generator import BrainGenerator


class TestTorchBrainGenerator(unittest.TestCase):
    
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
        
        # Common parameters for the generator
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
        """Test that PyTorch generator can be created with the parameters."""
        try:
            # Create PyTorch generator
            torch_generator = BrainGenerator(
                labels_dir=self.test_dir,
                generation_labels=self.generation_labels,
                output_labels=self.output_labels,
                batchsize=1,
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
            print("PyTorch brain generator created successfully.")
            
        except Exception as e:
            self.fail(f"Failed to create brain generator: {str(e)}")
    
    def test_brain_generation_minimal(self):
        """Test brain generation with minimal augmentation."""
        try:
            # Create generator with minimal augmentation
            torch_generator = BrainGenerator(
                labels_dir=self.test_dir,
                generation_labels=self.generation_labels,
                output_labels=self.output_labels,
                batchsize=1,
                n_channels=1,
                # Disable augmentations
                flipping=False,
                scaling_bounds=0.0,
                rotation_bounds=0.0,
                shearing_bounds=0.0,
                translation_bounds=False,
                nonlin_std=0.0,
                bias_field_std=0.0
            )
            
            # Generate images and segmentations
            torch_image, torch_labels = torch_generator.generate_brain()
            
            # Convert PyTorch tensors to numpy if needed
            if isinstance(torch_image, torch.Tensor):
                torch_image = torch_image.cpu().numpy()
            if isinstance(torch_labels, torch.Tensor):
                torch_labels = torch_labels.cpu().numpy()
                
            # Check shapes
            self.assertEqual(torch_image.shape, torch_labels.shape)
            
            # Save visualization
            self._save_visualization(torch_image, torch_labels, "minimal")
            
            print("Brain generation with minimal augmentation passed.")
            
        except Exception as e:
            self.fail(f"Failed in brain generation with minimal augmentation: {str(e)}")
    
    def test_brain_generation_with_augmentation(self):
        """Test brain generation with augmentations enabled."""
        try:
            # Set random seeds for reproducibility
            np.random.seed(1234)
            torch.manual_seed(1234)
            
            # Create generator with augmentations
            torch_generator = BrainGenerator(
                labels_dir=self.test_dir,
                generation_labels=self.generation_labels,
                output_labels=self.output_labels,
                batchsize=1,
                n_channels=1,
                # Enable some augmentations
                flipping=True,
                scaling_bounds=0.1,
                rotation_bounds=10,
                shearing_bounds=0.01,
                translation_bounds=5,
                nonlin_std=1.0,
                bias_field_std=0.3
            )
            
            # Generate images and segmentations
            torch_image, torch_labels = torch_generator.generate_brain()
            
            # Convert PyTorch tensors to numpy if needed
            if isinstance(torch_image, torch.Tensor):
                torch_image = torch_image.cpu().numpy()
            if isinstance(torch_labels, torch.Tensor):
                torch_labels = torch_labels.cpu().numpy()
            
            # Check shapes
            self.assertEqual(torch_image.shape, torch_labels.shape)
            
            # Save visualization
            self._save_visualization(torch_image, torch_labels, "augmented")
            
            # Check that labels are still valid (integers within the expected range)
            self.assertTrue(np.all(np.unique(torch_labels) <= self.n_neutral_labels), 
                           "PyTorch labels should be within expected range")
            
            print("Brain generation with augmentation passed.")
            
        except Exception as e:
            self.fail(f"Failed in brain generation with augmentation: {str(e)}")
    
    def _save_visualization(self, image, labels, name):
        """Save visualization of the generated image and labels."""
        # Create output directory
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving visualization to {output_dir}")
        
        # Get middle slices
        if len(image.shape) == 3:  # 3D single channel
            slice_idx = image.shape[0] // 2
            image_slice = image[slice_idx, :, :]
            labels_slice = labels[slice_idx, :, :]
            print(f"Image shape: {image.shape}, using middle slice {slice_idx}")
        elif len(image.shape) == 4:  # 3D with channel
            slice_idx = image.shape[0] // 2
            image_slice = image[slice_idx, :, :, 0]
            labels_slice = labels[slice_idx, :, :, 0]
            print(f"Image shape: {image.shape}, using middle slice {slice_idx}")
        else:
            print(f"Unexpected image shape: {image.shape}")
            # Try to handle other cases
            if len(image.shape) > 0:
                # Just take the first slice of whatever dimensions we have
                image_slice = image.reshape(image.size // np.prod(image.shape[-2:]), *image.shape[-2:])[0]
                labels_slice = labels.reshape(labels.size // np.prod(labels.shape[-2:]), *labels.shape[-2:])[0]
            else:
                print("Cannot create visualization for this shape")
                return
        
        print(f"Image slice shape: {image_slice.shape}, min: {np.min(image_slice)}, max: {np.max(image_slice)}")
        print(f"Labels slice shape: {labels_slice.shape}, unique values: {np.unique(labels_slice)}")
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot image
        im1 = axes[0].imshow(image_slice, cmap='gray')
        axes[0].set_title('Generated Image')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Plot labels
        im2 = axes[1].imshow(labels_slice, cmap='viridis')
        axes[1].set_title('Labels')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Save figure
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'brain_generation_{name}.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"Visualization saved to {output_path}")


if __name__ == '__main__':
    unittest.main()