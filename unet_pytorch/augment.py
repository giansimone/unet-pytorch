"""
Module to augment the number of images by rotating them.
"""
import os

from skimage import io, transform


def augment(im_path: str) -> None:
    """ Read and rotate an image (+90, +180, +270 degrees).
    
    Args:
        im_path: Path to load the image.
    """
    im = io.imread(im_path)
    filename, extension = os.path.splitext(im_path)
    for i in (90, 180, 270):
        new_im = transform.rotate(image=im, angle=i, order=0)
        new_im_path = filename + 'R' + str(i) + extension
        io.imsave(new_im_path, new_im)
