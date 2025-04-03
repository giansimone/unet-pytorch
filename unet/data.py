"""
Module to create the dataset for training the U-Net.
"""
import os
from glob import glob

import torch
from torch.utils.data import Dataset
from skimage import io

from unet.utils import im_to_tensor


class SegmentationDataset(Dataset):
    """Dataset to train the U-Net."""

    def __init__(self, x: list, y: list, device: str='cuda') -> None:
        """ Initialize the dataset with input and target file paths.

        Args:
            x (list): List of input file paths.
            y (list): List of target file paths.
            device (str): Device to use for tensor conversion.
        """
        self.x, self.y = x, y
        self.device = device

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple:
        # Select the sample
        input_path = self.x[idx]
        target_path = self.y[idx]

        # Load input and target
        x, y = (
            im_to_tensor(input_path).type(torch.float32).to(self.device),
            torch.from_numpy(io.imread(target_path)).type(torch.long).to(self.device),
        )

        return x, y


def data_set(data_path: str, device: str) -> SegmentationDataset:
    """Create a dataset to train the U-Net.
    
    Args:
        data_path: Path that contains the data.
        device: Device to use for tensor conversion.
    Returns:
        Object from class `SegmentationDataset`.
    """
    inputs = get_files_list(os.path.join(data_path, 'inputs'), 'tif')
    targets = get_files_list(os.path.join(data_path, 'targets'), 'png')

    return SegmentationDataset(inputs, targets, device)


def get_files_list(base_path: str, ext: str) -> list:
    """Get a list of files with a specific extension.
    
    Args:
        base_path: Path that contains the files.
        ext: Desired extension for the files.
    Returns:
        List of filenames with the desired extension.
    """
    filenames = glob(os.path.join(base_path, '*.' + ext))
    filenames.sort()
    return filenames
