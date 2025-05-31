import os
import sys
import unittest
import numpy as np
import torch
import tensorflow as tf

# Add the parent directory to the path so we can import the SynthSeg modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PyTorch implementations
from SynthSeg.metrics_model import DiceLoss, WeightedL2Loss

# Import TensorFlow implementations
from ext.lab2im.layers import DiceLoss as DiceLoss_tf
from ext.lab2im.layers import WeightedL2Loss as WeightedL2Loss_tf


class TestMetricsModel(unittest.TestCase):
    
    def setUp(self):
        # Set random seeds for reproducibility
        np.random.seed(1234)
        torch.manual_seed(1234)
        tf.random.set_seed(1234)
        
        # Create sample data for testing
        self.batch_size = 2
        self.height = 16
        self.width = 16
        self.depth = 16
        self.n_labels = 4
        
        # Create 3D data
        self.shape_3d = (self.batch_size, self.height, self.width, self.depth, self.n_labels)
        
        # Create random ground truth and predictions (3D)
        self.gt_np_3d = np.random.rand(*self.shape_3d).astype(np.float32)
        self.pred_np_3d = np.random.rand(*self.shape_3d).astype(np.float32)
        
        # Normalize to make them probabilistic
        self.gt_np_3d = self.gt_np_3d / np.sum(self.gt_np_3d, axis=-1, keepdims=True)
        self.pred_np_3d = self.pred_np_3d / np.sum(self.pred_np_3d, axis=-1, keepdims=True)
        
        # Convert to PyTorch tensors
        self.gt_torch_3d = torch.from_numpy(self.gt_np_3d)
        self.pred_torch_3d = torch.from_numpy(self.pred_np_3d)
        
        # Convert to TensorFlow tensors
        self.gt_tf_3d = tf.convert_to_tensor(self.gt_np_3d)
        self.pred_tf_3d = tf.convert_to_tensor(self.pred_np_3d)
        
        # Create 2D data
        self.shape_2d = (self.batch_size, self.height, self.width, self.n_labels)
        
        # Create random ground truth and predictions (2D)
        self.gt_np_2d = np.random.rand(*self.shape_2d).astype(np.float32)
        self.pred_np_2d = np.random.rand(*self.shape_2d).astype(np.float32)
        
        # Normalize to make them probabilistic
        self.gt_np_2d = self.gt_np_2d / np.sum(self.gt_np_2d, axis=-1, keepdims=True)
        self.pred_np_2d = self.pred_np_2d / np.sum(self.pred_np_2d, axis=-1, keepdims=True)
        
        # Convert to PyTorch tensors
        self.gt_torch_2d = torch.from_numpy(self.gt_np_2d)
        self.pred_torch_2d = torch.from_numpy(self.pred_np_2d)
        
        # Convert to TensorFlow tensors
        self.gt_tf_2d = tf.convert_to_tensor(self.gt_np_2d)
        self.pred_tf_2d = tf.convert_to_tensor(self.pred_np_2d)
    
    def test_dice_loss_2d(self):
        """Test that PyTorch and TensorFlow implementations of DiceLoss produce similar results for 2D data."""
        try:
            # Create loss functions
            dice_loss_torch = DiceLoss()
            dice_loss_tf = DiceLoss_tf()
            
            # Compute losses
            loss_torch = dice_loss_torch(self.gt_torch_2d, self.pred_torch_2d).numpy()
            loss_tf = dice_loss_tf([self.gt_tf_2d, self.pred_tf_2d]).numpy()
            
            # Check that losses are similar
            self.assertAlmostEqual(loss_torch, loss_tf, places=4, 
                                  msg=f"PyTorch loss {loss_torch} and TensorFlow loss {loss_tf} differ significantly")
            
            print(f"2D DiceLoss - PyTorch: {loss_torch}, TensorFlow: {loss_tf}")
            
        except Exception as e:
            self.fail(f"Error in test_dice_loss_2d: {str(e)}")
    
    def test_dice_loss_3d(self):
        """Test that PyTorch and TensorFlow implementations of DiceLoss produce similar results for 3D data."""
        try:
            # Create loss functions
            dice_loss_torch = DiceLoss()
            dice_loss_tf = DiceLoss_tf()
            
            # Compute losses
            loss_torch = dice_loss_torch(self.gt_torch_3d, self.pred_torch_3d).numpy()
            loss_tf = dice_loss_tf([self.gt_tf_3d, self.pred_tf_3d]).numpy()
            
            # Check that losses are similar
            self.assertAlmostEqual(loss_torch, loss_tf, places=4, 
                                  msg=f"PyTorch loss {loss_torch} and TensorFlow loss {loss_tf} differ significantly")
            
            print(f"3D DiceLoss - PyTorch: {loss_torch}, TensorFlow: {loss_tf}")
            
        except Exception as e:
            self.fail(f"Error in test_dice_loss_3d: {str(e)}")
    
    def test_dice_loss_with_class_weights(self):
        """Test DiceLoss with class weights."""
        try:
            # Create class weights
            class_weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
            
            # Create loss functions with class weights
            dice_loss_torch = DiceLoss(class_weights=class_weights)
            dice_loss_tf = DiceLoss_tf(class_weights=class_weights)
            
            # Compute losses
            loss_torch = dice_loss_torch(self.gt_torch_2d, self.pred_torch_2d).numpy()
            loss_tf = dice_loss_tf([self.gt_tf_2d, self.pred_tf_2d]).numpy()
            
            # Check that losses are similar
            self.assertAlmostEqual(loss_torch, loss_tf, places=4, 
                                  msg=f"PyTorch loss {loss_torch} and TensorFlow loss {loss_tf} differ significantly")
            
            print(f"DiceLoss with class weights - PyTorch: {loss_torch}, TensorFlow: {loss_tf}")
            
        except Exception as e:
            self.fail(f"Error in test_dice_loss_with_class_weights: {str(e)}")
    
    def test_weighted_l2_loss_2d(self):
        """Test that PyTorch and TensorFlow implementations of WeightedL2Loss produce similar results for 2D data."""
        try:
            # Create loss functions
            wl2_loss_torch = WeightedL2Loss(target_value=5)
            wl2_loss_tf = WeightedL2Loss_tf(target_value=5)
            
            # Compute losses
            loss_torch = wl2_loss_torch(self.gt_torch_2d, self.pred_torch_2d).numpy()
            loss_tf = wl2_loss_tf([self.gt_tf_2d, self.pred_tf_2d]).numpy()
            
            # Check that losses are similar
            self.assertAlmostEqual(loss_torch, loss_tf, places=4, 
                                  msg=f"PyTorch loss {loss_torch} and TensorFlow loss {loss_tf} differ significantly")
            
            print(f"2D WeightedL2Loss - PyTorch: {loss_torch}, TensorFlow: {loss_tf}")
            
        except Exception as e:
            self.fail(f"Error in test_weighted_l2_loss_2d: {str(e)}")
    
    def test_weighted_l2_loss_3d(self):
        """Test that PyTorch and TensorFlow implementations of WeightedL2Loss produce similar results for 3D data."""
        try:
            # Create loss functions
            wl2_loss_torch = WeightedL2Loss(target_value=5)
            wl2_loss_tf = WeightedL2Loss_tf(target_value=5)
            
            # Compute losses
            loss_torch = wl2_loss_torch(self.gt_torch_3d, self.pred_torch_3d).numpy()
            loss_tf = wl2_loss_tf([self.gt_tf_3d, self.pred_tf_3d]).numpy()
            
            # Check that losses are similar
            self.assertAlmostEqual(loss_torch, loss_tf, places=4, 
                                  msg=f"PyTorch loss {loss_torch} and TensorFlow loss {loss_tf} differ significantly")
            
            print(f"3D WeightedL2Loss - PyTorch: {loss_torch}, TensorFlow: {loss_tf}")
            
        except Exception as e:
            self.fail(f"Error in test_weighted_l2_loss_3d: {str(e)}")


if __name__ == '__main__':
    unittest.main()