"""lab2im module for SynthSegTorch

This module contains tools for generating images from label maps.
It is a PyTorch reimplementation of the original lab2im module.
"""

# Import utils module to make it available when importing ext.lab2im
from . import utils

# Import edit_tensors as edit_volumes for backward compatibility
from . import edit_tensors as edit_volumes