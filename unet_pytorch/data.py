"""
Module to create the dataset for training the U-Net.
"""
import os
import glob
import dataclasses

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from skimage import io

from unet_pytorch.utils import im_to_tensor


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


@dataclasses.dataclass(init=False)
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

    def _get_filenames(self, base_path: str, ext: str) -> list:
        """Get a list of files with a specific extension.
        
        Args:
            base_path: Path that contains the files.
            ext: Desired extension for the files.
        Returns:
            List of filenames with the desired extension.
        """
        filenames = glob.glob(os.path.join(base_path, '*.' + ext))
        filenames.sort()
        return filenames

    @property
    def get_train_dataset(self) -> SegmentationDataset:
        """Get the training dataset."""
        return self.train_dataset

    @property
    def get_val_dataset(self) -> SegmentationDataset:
        """Get the validation dataset."""
        return self.val_dataset


class UNetDataLoader:
    """Data loader for the U-Net model."""

    def __init__(self, dataset: UNetDataset, batch_size: int, num_workers: int = 0):
        """Initialise the U-Net data loader.

        Args:
            dataset: Dataset object containing the training and validation datasets.
            batch_size: Batch size for the data loader.
            num_workers: Number of workers for the data loader.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_loader = DataLoader(
            dataset.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            dataset.val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
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
