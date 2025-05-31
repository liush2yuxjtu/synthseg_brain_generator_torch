"""
If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


# python imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# project imports
from ext.lab2im import edit_tensors


def metrics_model(input_model, label_list, metrics='dice'):
    """
    Create a PyTorch model that computes metrics between predictions and ground truth.
    
    :param input_model: PyTorch model that outputs predictions
    :param label_list: list of labels to consider for metrics computation
    :param metrics: type of metric to compute, either 'dice' or 'wl2'
    :return: PyTorch model that computes the specified metric
    """
    # Create a wrapper model that computes the metric
    class MetricsModel(nn.Module):
        def __init__(self, base_model, label_list, metrics_type):
            super(MetricsModel, self).__init__()
            self.base_model = base_model
            self.label_list = np.unique(label_list)
            self.metrics_type = metrics_type
            
            # Check that the number of labels matches the output channels
            n_labels = base_model.output_shape[-1] if hasattr(base_model, 'output_shape') else None
            if n_labels is not None:
                assert n_labels == len(self.label_list), 'label_list should be as long as the posteriors channels'
            
            # Initialize loss functions
            if metrics_type == 'dice':
                self.loss_fn = DiceLoss()
            elif metrics_type == 'wl2':
                self.loss_fn = WeightedL2Loss(target_value=5)
            else:
                raise ValueError(f'metrics should either be "dice" or "wl2", got {metrics_type}')
        
        def forward(self, x, labels_gt=None):
            # Get prediction from base model
            pred = self.base_model(x)
            
            # If no ground truth is provided, just return the prediction
            if labels_gt is None:
                return pred
            
            # Get ground truth and convert to one-hot encoding
            # Convert labels to match the label list
            labels_gt = edit_tensors.convert_labels(labels_gt, self.label_list)
            
            # Convert to one-hot encoding
            labels_gt_one_hot = edit_tensors.one_hot_encoding(labels_gt, len(self.label_list))
            
            # Compute the loss
            loss = self.loss_fn(labels_gt_one_hot, pred)
            
            return loss
    
    # Create and return the metrics model
    return MetricsModel(input_model, label_list, metrics)


class IdentityLoss(nn.Module):
    """
    Very simple loss, as the computation of the loss has been directly implemented in the model.
    """
    def __init__(self, keepdims=True):
        super(IdentityLoss, self).__init__()
        self.keepdims = keepdims
    
    def forward(self, y_true, y_predicted):
        """
        Because the metrics is already calculated in the model, we simply return y_predicted.
        We still need to put y_true in the inputs, as it's expected by PyTorch.
        """
        # Simply return the predicted value as the loss
        return y_predicted


class DiceLoss(nn.Module):
    """
    This layer computes the Dice loss between two tensors.
    These tensors are expected to have the same shape [batch, size_dim1, ..., size_dimN, n_labels].
    The first input tensor is the GT and the second is the prediction.
    
    :param class_weights: (optional) if given, the loss is obtained by a weighted average of the Dice across labels.
    Must be a sequence or 1d numpy array of length n_labels. Can also be -1, where the weights are dynamically set to
    the inverse of the volume of each label in the ground truth.
    :param boundary_weights: (optional) bonus weight that we apply to the voxels close to boundaries between structures
    when computing the loss. Default is 0 where no boundary weighting is applied.
    :param boundary_dist: (optional) if boundary_weight is not 0, the extra boundary weighting is applied to all voxels
    within this distance to a region boundary. Default is 3.
    :param skip_background: (optional) whether to skip boundary weighting for the background class, as this may be
    redundant when we have several labels. This is only used if boundary_weight is not 0.
    :param enable_checks: (optional) whether to make sure that the 2 input tensors are probabilistic (i.e. the label
    probabilities sum to 1 at each voxel location). Default is True.
    """
    def __init__(self, 
                 class_weights=None, 
                 boundary_weights=0, 
                 boundary_dist=3, 
                 skip_background=True, 
                 enable_checks=True):
        super(DiceLoss, self).__init__()
        self.class_weights = class_weights
        self.dynamic_weighting = False if class_weights != -1 else True
        self.class_weights_tensor = None
        self.boundary_weights = boundary_weights
        self.boundary_dist = boundary_dist
        self.skip_background = skip_background
        self.enable_checks = enable_checks
    
    def forward(self, gt, pred):
        # Make sure tensors are probabilistic
        if self.enable_checks:
            # Normalize to ensure probabilities sum to 1
            gt = gt / (torch.sum(gt, dim=-1, keepdim=True) + torch.finfo(gt.dtype).eps)
            pred = pred / (torch.sum(pred, dim=-1, keepdim=True) + torch.finfo(pred.dtype).eps)
        
        # Get spatial dimensions
        spatial_dims = list(range(1, len(gt.shape) - 1))
        
        # Compute dice loss for each label
        top = 2 * gt * pred
        bottom = torch.square(gt) + torch.square(pred)
        
        # Apply boundary weighting if needed
        if self.boundary_weights > 0:
            # Create a pooling layer to detect boundaries
            n_dims = len(gt.shape) - 2
            if n_dims == 2:
                avg_pool = nn.AvgPool2d(kernel_size=2*self.boundary_dist+1, stride=1, padding=self.boundary_dist)
            elif n_dims == 3:
                avg_pool = nn.AvgPool3d(kernel_size=2*self.boundary_dist+1, stride=1, padding=self.boundary_dist)
            else:
                raise ValueError(f"Unsupported number of dimensions: {n_dims}")
            
            # Apply pooling to detect boundaries
            avg = avg_pool(gt.permute(0, -1, *spatial_dims))  # Move channels to position expected by nn.AvgPool
            avg = avg.permute(0, *range(2, 2+n_dims), 1)  # Move channels back to last position
            
            # Identify boundary regions
            boundaries = (avg > 0).float() * (avg < (1 / len(spatial_dims) - 1e-4)).float()
            
            # Skip background if requested
            if self.skip_background and gt.shape[-1] > 1:
                # Zero out the background channel boundaries
                boundaries_channels = torch.unbind(boundaries, dim=-1)
                zeros = torch.zeros_like(boundaries_channels[0])
                boundaries_list = [zeros] + list(boundaries_channels[1:])
                boundaries = torch.stack(boundaries_list, dim=-1)
            
            # Apply boundary weights
            boundary_weights_tensor = 1 + self.boundary_weights * boundaries
            top *= boundary_weights_tensor
            bottom *= boundary_weights_tensor
        
        # Compute loss
        top = torch.sum(top, dim=spatial_dims)
        bottom = torch.sum(bottom, dim=spatial_dims)
        dice = (top + torch.finfo(gt.dtype).eps) / (bottom + torch.finfo(gt.dtype).eps)
        loss = 1 - dice
        
        # Apply class weighting if needed
        if self.dynamic_weighting:
            # The weight of a class is the inverse of its volume in the ground truth
            if self.boundary_weights > 0:
                # Account for boundary weighting when computing volume
                boundary_weights_tensor = 1 + self.boundary_weights * boundaries
                self.class_weights_tensor = 1 / torch.sum(gt * boundary_weights_tensor, dim=spatial_dims)
            else:
                self.class_weights_tensor = 1 / torch.sum(gt, dim=spatial_dims)
        
        if self.class_weights_tensor is not None or self.class_weights is not None:
            if self.class_weights_tensor is None:
                # Convert class weights to tensor
                if isinstance(self.class_weights, (list, np.ndarray)):
                    weights = torch.tensor(self.class_weights, dtype=gt.dtype, device=gt.device)
                else:
                    weights = torch.ones(gt.shape[-1], dtype=gt.dtype, device=gt.device) * self.class_weights
                self.class_weights_tensor = weights.unsqueeze(0)  # Add batch dimension
            
            # Normalize weights
            self.class_weights_tensor = self.class_weights_tensor / torch.sum(self.class_weights_tensor, dim=-1, keepdim=True)
            
            # Apply weights
            loss = torch.sum(loss * self.class_weights_tensor, dim=-1)
        
        # Return mean loss across batch
        return torch.mean(loss)


class WeightedL2Loss(nn.Module):
    """
    This layer computes a L2 loss weighted by a specified factor (target_value) between two tensors.
    This is designed to be used on the layer before the softmax.
    The tensors are expected to have the same shape [batchsize, size_dim1, ..., size_dimN, n_labels].
    The first input tensor is the GT and the second is the prediction.
    
    :param target_value: target value for the layer before softmax: target_value when gt = 1, -target_value when gt = 0.
    """
    def __init__(self, target_value=5):
        super(WeightedL2Loss, self).__init__()
        self.target_value = target_value
    
    def forward(self, gt, pred):
        # Extract number of labels
        n_labels = gt.shape[-1]
        
        # Compute weights based on background
        weights = torch.unsqueeze(1 - gt[..., 0] + 1e-8, -1)
        
        # Compute weighted L2 loss
        loss = torch.sum(weights * torch.square(pred - self.target_value * (2 * gt - 1))) / (torch.sum(weights) * n_labels)
        
        return loss