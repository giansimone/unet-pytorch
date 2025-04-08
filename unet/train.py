"""
Module to train the U-Net model.
"""
import os

import torch
from torch import nn

from unet.model import UNet
from unet.data import UNetDataset


class UNetTrainer:
    """Trainer class for the U-Net model."""

    def __init__(
            self,
            paths: dict[str, str],
            batch_size: int=4
        ) -> None:
        """Initialise the U-Net trainer.

        Args:
            paths: Dictionary containing paths for input, target images, and model checkpoints.
                - 'inputs': Path to input images.
                - 'targets': Path to target images.
                - 'model': Path to save model checkpoints.
            batch_size: Batch size for training.
        """
        self.paths = paths
        self.batch_size = batch_size

        self.unet_dataset = UNetDataset(self.paths['inputs'], self.paths['targets'])
        self.unet_dataset.set_train_loader(self.batch_size)
        self.unet_dataset.set_val_loader(self.batch_size)

        if torch.backends.mps.is_available():
            self.device = torch.device('mps') # Apple silicon
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = UNet().to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimiser = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, mode='min', patience=5, factor=0.5
        )

        self.history = {'train_loss': [], 'val_loss': []}

    def _train_one_epoch(self) -> float:
        """Train the model for one epoch.
        
        Returns:
            Average loss for the epoch.
        """
        train_loader = self.unet_dataset.get_train_loader
        self.model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)
            loss = self.loss_fn(pred, y.squeeze(1).long())

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            total_loss += loss.item() * x.size(0)

        return total_loss / len(train_loader.dataset)

    def _evaluate(self) -> float:
        """Evaluate the model on the validation set.
        
        Returns:
            Average loss for the validation set.
        """
        val_loader = self.unet_dataset.get_val_loader
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)
                loss = self.loss_fn(pred, y.squeeze(1).long())

                total_loss += loss.item() * x.size(0)

        return total_loss / len(val_loader.dataset)

    def train(self, epochs: int=50, save_interval: int=5) -> None:
        """Train the U-Net model.

        Args:
            epochs: Number of epochs to train for.
            save_interval: Interval for saving checkpoints.
        """
        best_val_loss = float('inf')
        for epoch in range(epochs):
            train_loss = self._train_one_epoch()
            self.history['train_loss'].append(train_loss)

            val_loss = self._evaluate()
            self.history['val_loss'].append(val_loss)

            self.scheduler.step(val_loss)

            print(
                f'Epoch {epoch + 1}/{epochs}, '
                f'Train loss: {train_loss:.4f}, '
                f'Validation loss: {val_loss:.4f}'
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, 'best_model.pth')

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch, val_loss, f'checkpoint_epoch_{epoch + 1}.pth')

        self.save_checkpoint(epochs, val_loss, 'final_model.pth')
        print('Training complete!')

    def save_checkpoint(self, epoch: int, val_loss: float, filename: str) -> None:
        """Save the model checkpoint.
        
        Args:
            epoch: Current epoch number.
            val_loss: Validation loss for the current epoch.
            filename: Filename for the checkpoint.
        """
        torch.save({
            'epoch': epoch + 1,
            'model_weights': self.model.state_dict(),
            'optimiser_weights': self.optimiser.state_dict(),
            'loss': val_loss,
            'history': self.history,
        }, os.path.join(self.paths['model'], filename)
        )

    def load_checkpoint(self, filename: str) -> tuple[int, float]:
        """Load the model checkpoint.
        
        Args:
            filename: Filename for the checkpoint.
        
        Returns:
            start_epoch: Starting epoch number.
            best_val_loss: Best validation loss.
        """
        checkpoint_path = os.path.join(self.paths['model'], filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        else:
            print(f'Checkpoint {checkpoint_path} not found.')
            return 0, float('inf')
        self.model.load_state_dict(checkpoint['model_weights'])
        self.optimiser.load_state_dict(checkpoint['optimiser_weights'])
        start_epoch = checkpoint['epoch'] - 1
        best_val_loss = checkpoint['loss']
        self.history = checkpoint['history']
        print(
            f'Loaded checkpoint from epoch {start_epoch + 1} '
            f'with validation loss {best_val_loss:.4f}'
        )
        return start_epoch, best_val_loss
