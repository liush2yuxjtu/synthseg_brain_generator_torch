"""Utility functions for lab2im module.

This module contains utility functions for the lab2im module.
"""

# python imports
import os
import numpy as np
import nibabel as nib


def load_array_if_path(var):
    """Load a numpy array from a path if var is a string.
    
    :param var: variable to check and load if it's a path
    :return: loaded array if var is a path, var otherwise
    """
    if isinstance(var, str):
        if os.path.isfile(var):
            var = np.load(var)
    return var


def load_volume(path_volume, dtype=None, aff_ref=None):
    """Load a volume from a file.
    
    :param path_volume: path of the volume to load
    :param dtype: (optional) data type of the loaded volume. Default is None.
    :param aff_ref: (optional) reference affine matrix. Default is None.
    :return: loaded volume
    """
    # Check file extension
    if path_volume.endswith(('.npy', '.npz')):
        # For numpy files, load directly
        try:
            if path_volume.endswith('.npy'):
                volume = np.load(path_volume)
            else:  # .npz file
                volume = np.load(path_volume)['vol_data'] if 'vol_data' in np.load(path_volume) else \
                         np.load(path_volume)[np.load(path_volume).files[0]]
        except Exception as e:
            raise ValueError(f'Error loading numpy volume {path_volume}: {str(e)}')
    else:
        # For medical image formats (nii, nii.gz, mgz)
        try:
            volume_file = nib.load(path_volume)
            volume = volume_file.get_fdata()
        except Exception as e:
            raise ValueError(f'Error loading volume {path_volume}: {str(e)}')
    
    # Convert to specified data type if provided
    if dtype is not None:
        if dtype == 'int':
            volume = volume.astype(np.int32)
        else:
            volume = volume.astype(dtype)
    
    return volume


def add_axis(x, axis=0):
    """Add axis to a numpy array.
    
    :param x: input array
    :param axis: (optional) axis along which to add the new axis. Can be a single integer or a list of integers.
    :return: array with added axis
    """
    if isinstance(axis, (list, tuple)):
        for ax in axis:
            x = np.expand_dims(x, axis=ax)
    else:
        x = np.expand_dims(x, axis=axis)
    return x


def draw_value_from_distribution(hyperparameters, size, distribution='uniform', default_range_min=0, default_range_max=1, positive_only=False):
    """Draw values from a distribution.
    
    :param hyperparameters: parameters of the distribution. Can be:
        - None: will use default values
        - a sequence of length 2: [min, max] for uniform, [mean, std] for normal
        - a numpy array of shape (2, size)
    :param size: number of values to draw
    :param distribution: (optional) type of distribution, 'uniform' or 'normal'. Default is 'uniform'.
    :param default_range_min: (optional) default minimum value if hyperparameters is None. Default is 0.
    :param default_range_max: (optional) default maximum value if hyperparameters is None. Default is 1.
    :param positive_only: (optional) whether to ensure all values are positive. Default is False.
    :return: drawn values
    """
    # Set default hyperparameters if None
    if hyperparameters is None:
        if distribution == 'uniform':
            hyperparameters = [default_range_min, default_range_max]
        else:
            hyperparameters = [0.5 * (default_range_min + default_range_max), 0.5 * (default_range_max - default_range_min)]
    
    # Load from file if string
    hyperparameters = load_array_if_path(hyperparameters)
    
    # Draw from distribution
    if isinstance(hyperparameters, (list, tuple)):
        if distribution == 'uniform':
            values = np.random.uniform(low=hyperparameters[0], high=hyperparameters[1], size=size)
        elif distribution == 'normal':
            values = np.random.normal(loc=hyperparameters[0], scale=hyperparameters[1], size=size)
        else:
            raise ValueError(f'Distribution {distribution} not supported')
    elif isinstance(hyperparameters, np.ndarray) and hyperparameters.shape[0] == 2:
        if hyperparameters.shape[1] != size:
            raise ValueError(f'Hyperparameters shape {hyperparameters.shape} does not match size {size}')
        if distribution == 'uniform':
            values = np.random.uniform(low=hyperparameters[0, :], high=hyperparameters[1, :], size=size)
        elif distribution == 'normal':
            values = np.random.normal(loc=hyperparameters[0, :], scale=hyperparameters[1, :], size=size)
        else:
            raise ValueError(f'Distribution {distribution} not supported')
    else:
        raise ValueError(f'Hyperparameters shape {hyperparameters.shape} not supported')
    
    # Ensure positive values if requested
    if positive_only:
        values = np.maximum(values, 0)
    
    return values


def get_volume_info(path_volume, aff_ref=None, return_volume=False, max_channels=10):
    """Get information about a volume file.
    
    :param path_volume: path of the volume file
    :param aff_ref: (optional) reference affine matrix. If not None, the loaded volume is aligned to this affine matrix.
    All info relative to the volume is then given in this new space. Must be a numpy array of dimension 4x4.
    :param return_volume: (optional) whether to return the volume along with the information.
    :param max_channels: maximum possible number of channels for the input volume.
    :return: tuple containing volume information. If return_volume is False, returns (volume_shape, affine, n_dims, n_channels, header, voxel_size).
    If return_volume is True, returns (volume, volume_shape, affine, n_dims, n_channels, header, voxel_size).
    """
    # Check file extension
    if path_volume.endswith(('.npy', '.npz')):
        # For numpy files, load directly
        try:
            if path_volume.endswith('.npy'):
                volume = np.load(path_volume)
            else:  # .npz file
                volume = np.load(path_volume)['vol_data'] if 'vol_data' in np.load(path_volume) else \
                         np.load(path_volume)[np.load(path_volume).files[0]]
            # Create dummy header and affine
            header = None
            affine = np.eye(4) if aff_ref is None else aff_ref
        except Exception as e:
            raise ValueError(f'Error loading numpy volume {path_volume}: {str(e)}')
    else:
        # For medical image formats (nii, nii.gz, mgz)
        try:
            volume_file = nib.load(path_volume)
            volume = volume_file.get_fdata()
            header = volume_file.header
            affine = volume_file.affine
        except Exception as e:
            raise ValueError(f'Error loading volume {path_volume}: {str(e)}')
    
    # Understand if image is multichannel
    volume_shape = list(volume.shape)
    n_dims = len(volume_shape)
    n_channels = 1
    
    # Try to detect if last dimension is for channels
    if n_dims > 3 and volume_shape[-1] <= max_channels:
        n_dims -= 1
        n_channels = volume_shape[-1]
    
    # Get actual volume shape (excluding channels)
    volume_shape = volume_shape[:n_dims]
    
    # Get voxel size
    if header is not None and ('.nii' in path_volume or '.nii.gz' in path_volume):
        voxel_size = np.array(header['pixdim'][1:n_dims+1])
    elif header is not None and '.mgz' in path_volume:
        voxel_size = np.array(header['delta'])  # mgz image
    else:
        voxel_size = np.array([1.0] * n_dims)
    
    # Align to given affine matrix if provided
    if aff_ref is not None and not np.array_equal(affine, aff_ref):
        # Import here to avoid circular imports
        from ext.lab2im import edit_volumes
        
        # Get RAS axes
        ras_axes = edit_volumes.get_ras_axes(affine, n_dims=n_dims)
        ras_axes_ref = edit_volumes.get_ras_axes(aff_ref, n_dims=n_dims)
        
        # Align volume if return_volume is True
        if return_volume:
            volume = edit_volumes.align_volume_to_ref(volume, affine, aff_ref=aff_ref, n_dims=n_dims)
        
        # Update shape and voxel size
        volume_shape = np.array(volume_shape)
        voxel_size = np.array(voxel_size)
        volume_shape[ras_axes_ref] = volume_shape[ras_axes]
        voxel_size[ras_axes_ref] = voxel_size[ras_axes]
        volume_shape = volume_shape.tolist()
    
    # Return results
    if return_volume:
        return volume, volume_shape, affine, n_dims, n_channels, header, voxel_size
    else:
        return volume_shape, affine, n_dims, n_channels, header, voxel_size


def reformat_to_list(var, length=None):
    """Reformat a variable to a list of specified length.
    
    :param var: variable to reformat
    :param length: (optional) length of the output list. Default is None.
    :return: reformatted list
    """
    if var is None:
        return [None] * length if length is not None else None
    var = [var] if not isinstance(var, (list, tuple)) else list(var)
    if length is not None:
        if len(var) < length:
            var = var + [var[-1]] * (length - len(var))
        elif len(var) > length:
            var = var[:length]
    return var


def list_images_in_folder(path_dir, include_subject_id=False, include_extension=True):
    """List all image files in a directory.
    
    :param path_dir: path of the directory
    :param include_subject_id: (optional) whether to include subject id in the returned list. Default is False.
    :param include_extension: (optional) whether to include file extension in the returned list. Default is True.
    :return: list of image files
    """
    # List all files in directory
    files_list = sorted(os.listdir(path_dir))
    
    # Filter for image files (assuming common image extensions)
    image_extensions = ['.nii', '.nii.gz', '.mgz', '.npz', '.npy']
    images_list = [f for f in files_list if any(f.endswith(ext) for ext in image_extensions)]
    
    # Process filenames if needed
    if not include_extension:
        images_list = [os.path.splitext(f)[0] for f in images_list]
        # Handle double extensions like .nii.gz
        images_list = [os.path.splitext(f)[0] if f.endswith('.nii') else f for f in images_list]
    
    # Add subject id if requested
    if include_subject_id:
        subject_ids = [f.split('.')[0] for f in files_list]
        return [os.path.join(path_dir, f) for f in images_list]
    else:
        return [os.path.join(path_dir, f) for f in images_list]