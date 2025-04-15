"""
Module to create the dataset for training the U-Net.
"""
import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from skimage import io

from unet.utils import im_to_tensor


class SegmentationDataset(Dataset):
    """Dataset to train the U-Net."""

    def __init__(self, input_files: list, target_files: list) -> None:
        """ Initialize the dataset with input and target files.

        Args:
            input_files: List of input files.
            target_files: List of target files.
        """
        self.x = input_files
        self.y = target_files

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a sample from the dataset.
        Args:
            idx: Index of the sample to return.
        Returns:
            Tuple of input and target tensors.
        """
        input_file = self.x[idx]
        target_file = self.y[idx]

        x, y = (
            im_to_tensor(input_file).type(torch.float32),
            torch.from_numpy(io.imread(target_file)).type(torch.long),
        )

        return x, y


class UNetDataset():
    """Dataset class for the U-Net model."""

    def __init__(self, in_path: str, out_path: str) -> None:
        """Initialise the U-Net dataset.

        Args:
            in_path: Path to the input images.
            out_path: Path to the output masks.
        """
        self.in_path = in_path
        self.out_path = out_path

        self.inputs = self._get_filenames(in_path, 'tif')
        self.targets = self._get_filenames(out_path, 'png')

        assert len(self.inputs) == len(self.targets)

        train_inputs, test_inputs, train_targets, test_targets = train_test_split(
            self.inputs, self.targets, test_size=0.2, random_state=42
        )
        self.train_dataset = SegmentationDataset(train_inputs, train_targets)
        self.val_dataset = SegmentationDataset(test_inputs, test_targets)
        self.train_loader = None
        self.val_loader = None

    def _get_filenames(self, base_path: str, ext: str) -> list:
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

    def set_train_loader(self, batch_size: int) -> None:
        """Get the training data loader.
        
        Args:
            batch_size: Batch size for the data loader.
        Returns:
            DataLoader for the training dataset.
        """
        self.train_loader =  DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )

    def set_val_loader(self, batch_size: int) -> None:
        """Get the validation data loader.
        
        Args:
            batch_size: Batch size for the data loader.
        """
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

    @property
    def get_train_loader(self) -> DataLoader:
        """Get the training data loader."""
        return self.train_loader

    @property
    def get_val_loader(self) -> DataLoader:
        """Get the validation data loader."""
        return self.val_loader
