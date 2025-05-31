"""PyTorch implementation of the prediction module for SynthSeg.

This module contains the PyTorch implementation of the prediction function
for the SynthSeg framework, which uses a trained UNet to segment MRI images.
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import csv

from SynthSeg.unet import UNet
from ext.lab2im import utils
from ext.lab2im import edit_tensors


def dice(x, y):
    """Implementation of dice scores for 0/1 numpy array"""
    return 2 * np.sum(x * y) / (np.sum(x) + np.sum(y) + 1e-5)


def fast_dice(x, y, labels):
    """Fast implementation of Dice scores.
    :param x: input label map
    :param y: input label map of the same size as x
    :param labels: numpy array of labels to evaluate on
    :return: numpy array with Dice scores in the same order as labels.
    """

    assert x.shape == y.shape, 'both inputs should have same size, had {} and {}'.format(x.shape, y.shape)

    if len(labels) > 1:
        # sort labels
        labels_sorted = np.sort(labels)

        # build bins for histograms
        label_edges = np.sort(np.concatenate([labels_sorted - 0.1, labels_sorted + 0.1]))
        label_edges = np.insert(label_edges, [0, len(label_edges)], [labels_sorted[0] - 0.1, labels_sorted[-1] + 0.1])

        # compute Dice and re-arrange scores in initial order
        hst = np.histogram2d(x.flatten(), y.flatten(), bins=label_edges)[0]
        idx = np.arange(start=1, stop=2 * len(labels_sorted), step=2)
        dice_score = 2 * np.diag(hst)[idx] / (np.sum(hst, 0)[idx] + np.sum(hst, 1)[idx] + 1e-5)
        dice_score = dice_score[np.searchsorted(labels_sorted, labels)]

    else:
        dice_score = dice(x == labels[0], y == labels[0])

    return dice_score


def surface_distances(x, y, hausdorff_percentile=None, return_coordinate_max_distance=False):
    """Computes the maximum boundary distance (Hausdorff distance), and the average boundary distance of two masks.
    :param x: numpy array (boolean or 0/1)
    :param y: numpy array (boolean or 0/1)
    :param hausdorff_percentile: (optional) percentile (from 0 to 100) for which to compute the Hausdorff distance.
    Set this to 100 to compute the real Hausdorff distance (default). Can also be a list, where HD will be computed for
    the provided values.
    :param return_coordinate_max_distance: (optional) when set to true, the function will return the coordinates of the
    voxel with the highest distance (only if hausdorff_percentile=100).
    :return: max_dist, mean_dist(, coordinate_max_distance)
    max_dist: scalar with HD computed for the given percentile (or list if hausdorff_percentile was given as a list).
    mean_dist: scalar with average surface distance
    coordinate_max_distance: only returned return_coordinate_max_distance is True.
    """

    assert x.shape == y.shape, 'both inputs should have same size, had {} and {}'.format(x.shape, y.shape)
    n_dims = len(x.shape)

    hausdorff_percentile = 100 if hausdorff_percentile is None else hausdorff_percentile
    hausdorff_percentile = utils.reformat_to_list(hausdorff_percentile)

    # crop x and y around ROI
    # 这里我们需要实现crop_volume_around_region函数
    # 由于这个函数比较复杂，我们可以简化实现或者直接使用原始图像
    # 在实际应用中，应该实现完整的crop_volume_around_region函数
    # 简化实现：找到非零区域的边界
    def crop_volume_around_region(volume, margin=0):
        # 找到非零区域的索引
        indices = np.nonzero(volume)
        if len(indices[0]) == 0:  # 如果没有非零区域
            return volume, None
            
        # 找到边界
        min_idx = [np.min(idx) for idx in indices]
        max_idx = [np.max(idx) for idx in indices]
        
        # 添加边距
        min_idx = [max(0, idx - margin) for idx in min_idx]
        max_idx = [min(volume.shape[i] + 1, idx + margin + 1) for i, idx in enumerate(max_idx)]
        
        # 创建裁剪索引
        if n_dims == 3:
            crop_idx = [min_idx[0], min_idx[1], min_idx[2], max_idx[0], max_idx[1], max_idx[2]]
        elif n_dims == 2:
            crop_idx = [min_idx[0], min_idx[1], max_idx[0], max_idx[1]]
        else:
            raise ValueError(f"Unsupported number of dimensions: {n_dims}")
            
        return volume, crop_idx
    
    def crop_volume_with_idx(volume, crop_idx):
        if n_dims == 3:
            return volume[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]]
        elif n_dims == 2:
            return volume[crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]]
        else:
            raise ValueError(f"Unsupported number of dimensions: {n_dims}")
    
    _, crop_x = crop_volume_around_region(x)
    _, crop_y = crop_volume_around_region(y)

    # set distances to maximum volume shape if they are not defined
    if (crop_x is None) | (crop_y is None):
        return max(x.shape), max(x.shape)

    crop = np.concatenate([np.minimum(crop_x, crop_y)[:n_dims], np.maximum(crop_x, crop_y)[n_dims:]])
    x = crop_volume_with_idx(x, crop)
    y = crop_volume_with_idx(y, crop)

    # detect edge
    x_dist_int = distance_transform_edt(x * 1)
    x_edge = (x_dist_int == 1) * 1
    y_dist_int = distance_transform_edt(y * 1)
    y_edge = (y_dist_int == 1) * 1

    # calculate distance from edge
    x_dist = distance_transform_edt(np.logical_not(x_edge))
    y_dist = distance_transform_edt(np.logical_not(y_edge))

    # find distances from the 2 surfaces
    x_dists_to_y = y_dist[x_edge == 1]
    y_dists_to_x = x_dist[y_edge == 1]

    max_dist = list()
    coordinate_max_distance = None
    for hd_percentile in hausdorff_percentile:

        # find max distance from the 2 surfaces
        if hd_percentile == 100:
            max_dist.append(np.max(np.concatenate([x_dists_to_y, y_dists_to_x])))

            if return_coordinate_max_distance:
                indices_x_surface = np.where(x_edge == 1)
                idx_max_distance_x = np.where(x_dists_to_y == max_dist)[0]
                if idx_max_distance_x.size != 0:
                    coordinate_max_distance = np.stack(indices_x_surface).transpose()[idx_max_distance_x]
                else:
                    indices_y_surface = np.where(y_edge == 1)
                    idx_max_distance_y = np.where(y_dists_to_x == max_dist)[0]
                    coordinate_max_distance = np.stack(indices_y_surface).transpose()[idx_max_distance_y]

        # find percentile of max distance
        else:
            max_dist.append(np.percentile(np.concatenate([x_dists_to_y, y_dists_to_x]), hd_percentile))

    # find average distance between 2 surfaces
    if x_dists_to_y.shape[0] > 0:
        x_mean_dist_to_y = np.mean(x_dists_to_y)
    else:
        x_mean_dist_to_y = max(x.shape)
    if y_dists_to_x.shape[0] > 0:
        y_mean_dist_to_x = np.mean(y_dists_to_x)
    else:
        y_mean_dist_to_x = max(x.shape)
    mean_dist = (x_mean_dist_to_y + y_mean_dist_to_x) / 2

    # convert max dist back to scalar if HD only computed for 1 percentile
    if len(max_dist) == 1:
        max_dist = max_dist[0]

    # return coordinate of max distance if necessary
    if coordinate_max_distance is not None:
        return max_dist, mean_dist, coordinate_max_distance
    else:
        return max_dist, mean_dist


def predict(path_images,
           path_segmentations,
           path_model,
           labels_segmentation,
           n_neutral_labels=None,
           names_segmentation=None,
           path_posteriors=None,
           path_resampled=None,
           path_volumes=None,
           min_pad=None,
           cropping=None,
           target_res=1.,
           gradients=False,
           flip=True,
           topology_classes=None,
           sigma_smoothing=0.5,
           keep_biggest_component=True,
           n_levels=5,
           nb_conv_per_level=2,
           conv_size=3,
           unet_feat_count=24,
           feat_multiplier=2,
           activation='elu',
           gt_folder=None,
           compute_distances=False,
           recompute=True,
           verbose=True):
    """Segment images with a trained model.

    Args:
        path_images: path of images to segment. Can be the path to a directory or to a single image.
            If it is a directory, all images with extension nii.gz, nii, mgz, or npz inside this folder
            are segmented with the trained model.
        path_segmentations: path where segmentations will be saved. Must be the same type as path_images:
            (path to a folder if path_images is a folder, or path to a file if path_images is a file)
        path_model: path of the trained model.
        labels_segmentation: list of labels for which to compute Dice scores. Must be the same as the
            training label values.
        n_neutral_labels: (optional) list of neutral labels (i.e. non-sided), to help label mapping in
            flip test-time augmentation. Default is None, where no label is considered neutral.
        names_segmentation: (optional) list of names corresponding to the labels in labels_segmentation.
            Only used when path_volumes is provided. Must be the same length as labels_segmentation.
        path_posteriors: (optional) path where posteriors will be saved. Must be the same type as path_images.
            Default is None, where posteriors are not saved.
        path_resampled: (optional) path where images resampled to the target resolution will be saved.
            Must be the same type as path_images. Default is None, where resampled images are not saved.
        path_volumes: (optional) path where segmentation volumes will be saved. Must be the same type as
            path_images. The volumes of all segmentation labels will be saved in a single csv file.
            Default is None, where segmentation volumes are not computed.
        min_pad: (optional) minimum size of the padding for the input to the network. Default is None,
            where the padding size is automatically computed for each image.
        cropping: (optional) crop the images to the specified size. Can be an int, a sequence, or a
            1d numpy array. Default is None, where no cropping is performed.
        target_res: (optional) target resolution at which the images will be segmented. This will
            automatically resample the images if they are not already at the target resolution.
            Default is 1 (mm).
        gradients: (optional) whether to add image gradients as additional input channels. Default is False.
        flip: (optional) whether to use test-time augmentation by flipping the inputs. Default is True.
        topology_classes: (optional) list of classes for which to enforce topology constraints.
            Default is None, where no topology constraints are enforced.
        sigma_smoothing: (optional) standard deviation of the Gaussian kernel used to smooth the
            posteriors. Default is 0.5.
        keep_biggest_component: (optional) whether to keep only the biggest connected component in
            the segmentation. Default is True.
        n_levels: (optional) number of levels in the UNet. Default is 5.
        nb_conv_per_level: (optional) number of convolutions per level in the UNet. Default is 2.
        conv_size: (optional) size of the convolution kernels in the UNet. Default is 3.
        unet_feat_count: (optional) number of features in the first level of the UNet. Default is 24.
        feat_multiplier: (optional) multiplier for the number of features in each level of the UNet.
            Default is 2.
        activation: (optional) activation function in the UNet. Default is 'elu'.
        gt_folder: (optional) path of the ground truth segmentations for evaluation. Must be the same
            type as path_images. Default is None, where no evaluation is performed.
        compute_distances: (optional) whether to compute the Hausdorff distances between the
            segmentations and the ground truth. Default is False.
        recompute: (optional) whether to recompute segmentations that already exist. Default is True.
        verbose: (optional) whether to print progress. Default is True.

    Returns:
        tuple: (path_images, path_segmentations)
    """

    # Prepare input/output paths
    path_images, path_segmentations, path_posteriors, path_volumes, path_resampled, gt_folder = \
        prepare_output_files(path_images, path_segmentations, path_posteriors, path_volumes, path_resampled, gt_folder)

    # Get label list
    labels_segmentation = utils.get_list_labels(labels_segmentation)
    n_neutral_labels = utils.get_list_labels(n_neutral_labels) if n_neutral_labels is not None else []

    # Prepare names for volume file if needed
    if path_volumes is not None:
        if names_segmentation is None:
            names_segmentation = [str(lab) for lab in labels_segmentation]
        elif len(names_segmentation) != len(labels_segmentation):
            raise ValueError("names_segmentation and labels_segmentation must have the same length")

        # Create volume file and write header
        volume_header = ['subject']
        for name in names_segmentation:
            volume_header.append(name)
        if gt_folder is not None:
            for name in names_segmentation:
                volume_header.append(name + '_dice')
            if compute_distances:
                for name in names_segmentation:
                    volume_header.append(name + '_dist')
        utils.mkdir(os.path.dirname(path_volumes))
        with open(path_volumes, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(volume_header)

    # Get flip indices if needed
    flip_indices = None
    if flip:
        flip_indices = get_flip_indices(labels_segmentation, n_neutral_labels)

    # Build model
    if verbose:
        print('building model...')
    model = build_model(path_model, None, labels_segmentation, n_levels, nb_conv_per_level, conv_size,
                       unet_feat_count, feat_multiplier, activation, sigma_smoothing, flip_indices, gradients)

    # Loop over images
    for idx, (path_image, path_segmentation) in enumerate(zip(path_images, path_segmentations)):
        if verbose:
            print('processing', path_image, '({}/{})'.format(idx + 1, len(path_images)))

        # Check if result already exists
        if os.path.exists(path_segmentation) and not recompute:
            if verbose:
                print('segmentation already computed, skipping...')
            continue

        # Read image and corresponding info
        start = time.time()
        im, aff, h = utils.load_volume(path_image, im_only=False, squeeze=False)
        if verbose:
            print('reading time: {:.2f}s'.format(time.time() - start))

        # Preprocess image
        start = time.time()
        im, aff_aligned, h_aligned, im_shape, pad_shape, crop_idx = preprocess(im, aff, h, n_levels, target_res,
                                                                              cropping, min_pad, gradients)
        if verbose:
            print('preprocessing time: {:.2f}s'.format(time.time() - start))

        # Run model
        start = time.time()
        input_shape = list(im.shape[:-1]) + [im.shape[-1] + 3 * gradients]
        model.input_shape = input_shape
        posteriors = model(torch.from_numpy(im).float())
        if isinstance(posteriors, torch.Tensor):
            posteriors = posteriors.detach().numpy()
        if verbose:
            print('inference time: {:.2f}s'.format(time.time() - start))

        # Postprocess segmentation: align post-processed segmentation back to image space
        start = time.time()
        seg, posteriors = postprocess(posteriors, im_shape, pad_shape, crop_idx, aff_aligned, aff, h_aligned, h,
                                     topology_classes, keep_biggest_component, labels_segmentation)
        if verbose:
            print('postprocessing time: {:.2f}s'.format(time.time() - start))

        # Save outputs
        start = time.time()
        utils.save_volume(seg, aff, h, path_segmentation)
        if path_posteriors is not None:
            utils.save_volume(posteriors, aff, h, path_posteriors[idx])
        if path_resampled is not None:
            utils.save_volume(im, aff_aligned, h_aligned, path_resampled[idx])
        if verbose:
            print('saving time: {:.2f}s'.format(time.time() - start))

        # Compute volumes and Dice scores
        if path_volumes is not None:
            start = time.time()
            subject_name = os.path.basename(path_image).split('.')[0]
            volumes = np.zeros(len(labels_segmentation))
            for idx, label in enumerate(labels_segmentation):
                if label != 0:  # Skip background
                    volumes[idx] = np.sum(seg == label) * np.prod(utils.get_volume_info(h)[0])

            # Write volumes to CSV
            row = [subject_name] + volumes.tolist()

            # Compute Dice scores if ground truth is provided
            if gt_folder is not None:
                # Find corresponding ground truth file
                gt_file = os.path.join(gt_folder, os.path.basename(path_image))
                if os.path.exists(gt_file):
                    # Load ground truth segmentation
                    gt, aff_gt, h_gt = utils.load_volume(gt_file, im_only=False)

                    # Compute Dice scores
                    dice_scores = np.zeros(len(labels_segmentation))
                    for idx, label in enumerate(labels_segmentation):
                        if label != 0:  # Skip background
                            dice_scores[idx] = dice(seg == label, gt == label)
                    row += dice_scores.tolist()

                    # Compute Hausdorff distances if requested
                    if compute_distances:
                        distances = np.zeros(len(labels_segmentation))
                        for idx, label in enumerate(labels_segmentation):
                            if label != 0:  # Skip background
                                # Compute Hausdorff distance
                                max_dist, _ = surface_distances(seg == label, gt == label, [100])
                                distances[idx] = max_dist
                        row += distances.tolist()

            # Write row to CSV
            with open(path_volumes, 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)

            if verbose:
                print('volume computation time: {:.2f}s'.format(time.time() - start))

    return path_images, path_segmentations


def prepare_output_files(path_images, path_segmentations, path_posteriors=None, path_volumes=None,
                        path_resampled=None, gt_folder=None):
    """Prepare input and output paths for segmentation.

    Args:
        path_images: path of images to segment. Can be the path to a directory or to a single image.
        path_segmentations: path where segmentations will be saved.
        path_posteriors: (optional) path where posteriors will be saved.
        path_volumes: (optional) path where segmentation volumes will be saved.
        path_resampled: (optional) path where images resampled to the target resolution will be saved.
        gt_folder: (optional) path of the ground truth segmentations for evaluation.

    Returns:
        tuple: (path_images, path_segmentations, path_posteriors, path_volumes, path_resampled, gt_folder)
    """

    # Check inputs
    if not path_images:
        raise ValueError("path_images must be provided")
    if not path_segmentations:
        raise ValueError("path_segmentations must be provided")

    # Process image and segmentation paths
    if os.path.isdir(path_images):
        # Input is a directory, find all compatible images
        path_images = utils.list_images_in_folder(path_images)
        if not path_images:
            raise ValueError("Could not find any compatible images in the provided directory")

        # Create output directory if it doesn't exist
        utils.mkdir(path_segmentations)

        # Create output paths for each image
        basename = [os.path.basename(p) for p in path_images]
        path_segmentations = [os.path.join(path_segmentations, p) for p in basename]

    else:
        # Input is a single image
        if not os.path.exists(path_images):
            raise ValueError("Input image does not exist: {}".format(path_images))

        # Create output directory if needed
        utils.mkdir(os.path.dirname(path_segmentations))

        # Convert to lists for consistent processing
        path_images = [path_images]
        path_segmentations = [path_segmentations]

    # Process posterior paths if provided
    if path_posteriors is not None:
        if os.path.isdir(path_posteriors) or len(path_images) == 1:
            # Create output directory if needed
            utils.mkdir(path_posteriors if os.path.isdir(path_posteriors) else os.path.dirname(path_posteriors))

            # Create output paths for each image
            if os.path.isdir(path_posteriors):
                basename = [os.path.basename(p) for p in path_images]
                path_posteriors = [os.path.join(path_posteriors, p) for p in basename]
            else:
                path_posteriors = [path_posteriors]
        else:
            raise ValueError("path_posteriors must be a directory when processing multiple images")

    # Process resampled paths if provided
    if path_resampled is not None:
        if os.path.isdir(path_resampled) or len(path_images) == 1:
            # Create output directory if needed
            utils.mkdir(path_resampled if os.path.isdir(path_resampled) else os.path.dirname(path_resampled))

            # Create output paths for each image
            if os.path.isdir(path_resampled):
                basename = [os.path.basename(p) for p in path_images]
                path_resampled = [os.path.join(path_resampled, p) for p in basename]
            else:
                path_resampled = [path_resampled]
        else:
            raise ValueError("path_resampled must be a directory when processing multiple images")

    # Process ground truth folder if provided
    if gt_folder is not None:
        if not os.path.isdir(gt_folder):
            raise ValueError("gt_folder must be a directory")

    return path_images, path_segmentations, path_posteriors, path_volumes, path_resampled, gt_folder


def preprocess(im, aff, h, n_levels, target_res=1., cropping=None, min_pad=None, gradients=False):
    """Preprocess an image for segmentation.

    Args:
        im: input image, numpy array of shape [height, width, depth, channels]
        aff: affine matrix, numpy array of shape [4, 4]
        h: header
        n_levels: number of levels in the UNet
        target_res: target resolution in mm
        cropping: crop the images to the specified size
        min_pad: minimum size of the padding for the input to the network
        gradients: whether to add image gradients as additional input channels

    Returns:
        tuple: (preprocessed_image, aligned_affine, aligned_header, original_shape, pad_shape, crop_indices)
    """

    # Reorient image if needed
    im_shape = im.shape
    im, aff, h_aligned = utils.align_volume_to_ref(im, aff, h, aff_ref=np.eye(4), return_aff_map=True)

    # Resample image to target resolution
    if target_res:
        im, aff = utils.resample_volume(im, aff, h_aligned, new_vox_size=target_res)

    # Crop image if specified
    crop_idx = None
    if cropping is not None:
        im, crop_idx = utils.crop_volume(im, cropping, return_crop_idx=True)

    # Add batch and channel dimensions if needed
    if len(im.shape) == 3:
        im = im[..., np.newaxis]

    # Add gradient channels if specified
    if gradients:
        # Compute gradients
        grad_x = np.zeros_like(im)
        grad_y = np.zeros_like(im)
        grad_z = np.zeros_like(im)

        # Compute gradients along each axis
        grad_x[1:-1, :, :, :] = (im[2:, :, :, :] - im[:-2, :, :, :]) / 2
        grad_y[:, 1:-1, :, :] = (im[:, 2:, :, :] - im[:, :-2, :, :]) / 2
        grad_z[:, :, 1:-1, :] = (im[:, :, 2:, :] - im[:, :, :-2, :]) / 2

        # Concatenate gradients with original image
        im = np.concatenate([im, grad_x, grad_y, grad_z], axis=-1)

    # Normalize image to [0, 1]
    im = utils.rescale_volume(im, new_min=0., new_max=1., min_percentile=0.5, max_percentile=99.5)

    # Pad image to ensure it's divisible by 2^n_levels
    pad_shape, im = utils.pad_volume(im, min_pad=min_pad, return_pad_shape=True, div_by_n=2**n_levels)

    return im, aff, h_aligned, im_shape, pad_shape, crop_idx


def build_model(path_model,
               input_shape,
               labels_segmentation,
               n_levels,
               nb_conv_per_level,
               conv_size,
               unet_feat_count,
               feat_multiplier,
               activation,
               sigma_smoothing,
               flip_indices,
               gradients):
    """Build a segmentation model from a trained model file.

    Args:
        path_model: path to the trained model file
        input_shape: shape of the input tensor
        labels_segmentation: list of segmentation labels
        n_levels: number of levels in the UNet
        nb_conv_per_level: number of convolutions per level in the UNet
        conv_size: size of the convolution kernels in the UNet
        unet_feat_count: number of features in the first level of the UNet
        feat_multiplier: multiplier for the number of features in each level of the UNet
        activation: activation function in the UNet
        sigma_smoothing: standard deviation of the Gaussian kernel used to smooth the posteriors
        flip_indices: indices for flipping the segmentation labels
        gradients: whether to add image gradients as additional input channels

    Returns:
        model: PyTorch model for segmentation
    """

    # Check if model file exists
    if not os.path.isfile(path_model):
        raise ValueError("Model file does not exist: {}".format(path_model))

    # Create UNet model
    if input_shape is None:
        # Will be set later when we know the input shape
        n_channels = 1 + 3 * gradients
        input_shape = [None, None, None, n_channels]

    # Create UNet model
    model = UNet(input_shape=input_shape,
                nb_labels=len(labels_segmentation),
                nb_levels=n_levels,
                nb_conv_per_level=nb_conv_per_level,
                conv_size=conv_size,
                nb_features=unet_feat_count,
                feat_mult=feat_multiplier,
                activation=activation)

    # Load weights from trained model
    model.load_state_dict(torch.load(path_model))
    model.eval()  # Set to evaluation mode

    # Create a wrapper model that handles test-time augmentation and smoothing
    class SegmentationModel:
        def __init__(self, base_model, sigma_smoothing, flip_indices):
            self.base_model = base_model
            self.sigma_smoothing = sigma_smoothing
            self.flip_indices = flip_indices
            self.input_shape = None

        def __call__(self, x):
            # Convert numpy array to PyTorch tensor if needed
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()

            # Make sure input is on the correct device
            device = next(self.base_model.parameters()).device
            x = x.to(device)

            # Get predictions
            with torch.no_grad():
                # Forward pass
                posteriors = self.base_model(x)

                # Test-time augmentation with flipping if specified
                if self.flip_indices is not None:
                    # Flip input along the first spatial dimension
                    x_flipped = torch.flip(x, dims=[1])
                    posteriors_flipped = self.base_model(x_flipped)

                    # Flip the output back and reorder the labels
                    posteriors_flipped = torch.flip(posteriors_flipped, dims=[1])

                    # Reorder the labels according to flip_indices
                    posteriors_flipped_reordered = posteriors_flipped.clone()
                    for i, j in enumerate(self.flip_indices):
                        if j != i:
                            posteriors_flipped_reordered[..., j] = posteriors_flipped[..., i]

                    # Average the predictions
                    posteriors = (posteriors + posteriors_flipped_reordered) / 2

                # Apply Gaussian smoothing if specified
                if self.sigma_smoothing > 0:
                    # Convert to channel-first format for convolution
                    ndims = len(posteriors.shape) - 2  # Exclude batch and channel dims
                    if ndims == 2:
                        posteriors = posteriors.permute(0, 3, 1, 2)
                        kernel_size = int(6 * self.sigma_smoothing + 1)
                        if kernel_size % 2 == 0:
                            kernel_size += 1  # Ensure odd kernel size
                        padding = kernel_size // 2

                        # Create Gaussian kernel
                        x = torch.arange(-padding, padding + 1, dtype=torch.float32, device=device)
                        kernel = torch.exp(-(x**2) / (2 * self.sigma_smoothing**2))
                        kernel = kernel / kernel.sum()

                        # Apply separable Gaussian convolution
                        for b in range(posteriors.shape[0]):
                            for c in range(posteriors.shape[1]):
                                # Apply along height
                                temp = F.conv1d(posteriors[b, c].unsqueeze(0).unsqueeze(0),
                                               kernel.view(1, 1, -1),
                                               padding=(padding, 0))
                                # Apply along width
                                posteriors[b, c] = F.conv1d(temp.transpose(2, 3),
                                                          kernel.view(1, 1, -1),
                                                          padding=(padding, 0)).transpose(2, 3).squeeze()

                        # Convert back to channel-last format
                        posteriors = posteriors.permute(0, 2, 3, 1)
                    elif ndims == 3:
                        posteriors = posteriors.permute(0, 4, 1, 2, 3)
                        kernel_size = int(6 * self.sigma_smoothing + 1)
                        if kernel_size % 2 == 0:
                            kernel_size += 1  # Ensure odd kernel size
                        padding = kernel_size // 2

                        # Create Gaussian kernel
                        x = torch.arange(-padding, padding + 1, dtype=torch.float32, device=device)
                        kernel = torch.exp(-(x**2) / (2 * self.sigma_smoothing**2))
                        kernel = kernel / kernel.sum()

                        # Apply separable Gaussian convolution
                        for b in range(posteriors.shape[0]):
                            for c in range(posteriors.shape[1]):
                                # Apply along height
                                temp = F.conv1d(posteriors[b, c].reshape(1, 1, -1, posteriors.shape[3] * posteriors.shape[4]),
                                               kernel.view(1, 1, -1),
                                               padding=(padding, 0))
                                temp = temp.reshape(1, 1, posteriors.shape[2], posteriors.shape[3], posteriors.shape[4])

                                # Apply along width
                                temp = temp.permute(0, 1, 3, 2, 4)
                                temp = F.conv1d(temp.reshape(1, 1, -1, temp.shape[4]),
                                               kernel.view(1, 1, -1),
                                               padding=(padding, 0))
                                temp = temp.reshape(1, 1, posteriors.shape[3], posteriors.shape[2], posteriors.shape[4])
                                temp = temp.permute(0, 1, 3, 2, 4)

                                # Apply along depth
                                temp = temp.permute(0, 1, 4, 2, 3)
                                temp = F.conv1d(temp.reshape(1, 1, -1, temp.shape[4]),
                                               kernel.view(1, 1, -1),
                                               padding=(padding, 0))
                                temp = temp.reshape(1, 1, posteriors.shape[4], posteriors.shape[2], posteriors.shape[3])
                                temp = temp.permute(0, 1, 3, 4, 2)

                                posteriors[b, c] = temp.squeeze()

                        # Convert back to channel-last format
                        posteriors = posteriors.permute(0, 2, 3, 4, 1)

            # Convert back to numpy if input was numpy
            if isinstance(x, np.ndarray):
                posteriors = posteriors.cpu().numpy()

            return posteriors

    # Create and return the wrapper model
    return SegmentationModel(model, sigma_smoothing, flip_indices)


def postprocess(posteriors, im_shape, pad_shape, crop_idx, aff_aligned, aff, h_aligned, h,
               topology_classes=None, keep_biggest_component=True, labels_segmentation=None):
    """Postprocess the segmentation output.

    Args:
        posteriors: output of the network, numpy array of shape [batch, height, width, depth, channels]
        im_shape: original shape of the image
        pad_shape: shape of the padding
        crop_idx: indices used for cropping
        aff_aligned: affine matrix of the aligned image
        aff: original affine matrix
        h_aligned: header of the aligned image
        h: original header
        topology_classes: list of classes for which to enforce topology constraints
        keep_biggest_component: whether to keep only the biggest connected component
        labels_segmentation: list of segmentation labels

    Returns:
        tuple: (segmentation, posteriors)
    """

    # Get hard segmentation
    seg = np.argmax(posteriors, axis=-1)

    # Keep posteriors for later use
    posteriors_init = posteriors.copy()

    # Keep only the biggest connected component for each label if specified
    if keep_biggest_component:
        for label in np.unique(seg):
            if label != 0:  # Skip background
                mask = seg == label
                components, n_components = edit_tensors.label_connected_components(mask)
                if n_components > 1:
                    # Find the largest component
                    component_sizes = [np.sum(components == i) for i in range(1, n_components + 1)]
                    largest_component = np.argmax(component_sizes) + 1
                    # Keep only the largest component
                    seg[components != largest_component] = 0

    # Apply topology constraints if specified
    if topology_classes is not None and labels_segmentation is not None:
        # Convert topology_classes to indices in labels_segmentation
        topology_indices = [labels_segmentation.index(label) for label in topology_classes if label in labels_segmentation]

        # Reset posteriors for topology classes
        for idx in topology_indices:
            posteriors[..., idx] = 0

        # Set posteriors to 1 where the segmentation has the corresponding label
        for idx in topology_indices:
            label = labels_segmentation[idx]
            posteriors[seg == label, idx] = 1

    # Uncrop segmentation if needed
    if crop_idx is not None:
        # Create full-sized segmentation
        full_seg = np.zeros(im_shape[:3], dtype=seg.dtype)
        full_posteriors = np.zeros((*im_shape[:3], posteriors.shape[-1]), dtype=posteriors.dtype)

        # Get crop indices
        i_start, j_start, k_start, i_end, j_end, k_end = crop_idx

        # Insert cropped segmentation into full-sized segmentation
        full_seg[i_start:i_end, j_start:j_end, k_start:k_end] = seg
        full_posteriors[i_start:i_end, j_start:j_end, k_start:k_end, :] = posteriors

        seg = full_seg
        posteriors = full_posteriors

    # Unpad segmentation if needed
    if pad_shape is not None:
        # Get padding amounts
        pad_amounts = [(pad_shape[i] - im_shape[i]) // 2 for i in range(3)]

        # Crop out the padding
        seg = seg[pad_amounts[0]:pad_amounts[0] + im_shape[0],
                 pad_amounts[1]:pad_amounts[1] + im_shape[1],
                 pad_amounts[2]:pad_amounts[2] + im_shape[2]]

        posteriors = posteriors[pad_amounts[0]:pad_amounts[0] + im_shape[0],
                              pad_amounts[1]:pad_amounts[1] + im_shape[1],
                              pad_amounts[2]:pad_amounts[2] + im_shape[2], :]

    # Align segmentation back to original space
    seg, aff_final = utils.align_volume_to_ref(seg, aff_aligned, h_aligned, aff_ref=aff, return_aff_map=True)
    posteriors = utils.align_volume_to_ref(posteriors, aff_aligned, h_aligned, aff_ref=aff, n_dims=4)

    return seg, posteriors


def get_flip_indices(labels_segmentation, n_neutral_labels=None):
    """Get indices for flipping segmentation labels.

    Args:
        labels_segmentation: list of segmentation labels
        n_neutral_labels: list of neutral labels (i.e. non-sided)

    Returns:
        list: indices for flipping segmentation labels
    """

    # Initialize flip indices as identity mapping
    flip_indices = list(range(len(labels_segmentation)))

    # Handle neutral labels
    if n_neutral_labels is None:
        n_neutral_labels = []

    # Create mapping for left-right label pairs
    # This is a placeholder - you would need to implement the actual mapping
    # based on your specific label set
    # For example, if label 1 is left hippocampus and label 2 is right hippocampus,
    # you would set flip_indices[1] = 2 and flip_indices[2] = 1

    return flip_indices


def write_csv(path_csv, data, header=None):
    """Write data to a CSV file.

    Args:
        path_csv: path to the CSV file
        data: data to write to the CSV file
        header: header for the CSV file
    """

    # Create directory if it doesn't exist
    utils.mkdir(os.path.dirname(path_csv))

    # Write data to CSV file
    with open(path_csv, 'w') as csvFile:
        writer = csv.writer(csvFile)
        if header is not None:
            writer.writerow(header)
        writer.writerows(data)