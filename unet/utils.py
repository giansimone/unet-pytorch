"""
Module containing utility functions for image processing.
"""
import torch
from skimage import io, exposure, util
from torchvision import transforms


def im_to_tensor(im_path: str) -> torch.Tensor:
    """ Read a TIF image and return a tensor
    
    Args:
        im_path: Path to load the image.
    
    Returns:
        Tensor: Torch tensor containing the image as uint8.
    """
    im = io.imread(im_path)
    im = util.img_as_ubyte(exposure.rescale_intensity(im))
    im = transforms.ToTensor()(im)
    return im
