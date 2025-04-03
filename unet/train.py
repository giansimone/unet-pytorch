"""
Module to train the U-Net model.
"""
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from unet.model import UNet
from unet import data


def train(
        data_path: str,
        epochs: int=10,
        batch_size: int=4,
        learning_rate: float=1e-4
    ) -> None:
    """Train function for the U-Net.

    Args:
        data_path: Path to the dataset.
        epochs: Number of epochs to train the model.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimiser.
    """
    torch.manual_seed(42)

    if torch.backends.mps.is_available():
        device = torch.device('mps') # Apple silicon
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = data.data_set(data_path, device)
    data_loader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)

    model = UNet().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        model.train()
        for x, y in data_loader:

            pred = model(x)
            loss = loss_fn(pred, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        torch.save(model.state_dict(), os.path.join(data_path, 'weights.pth'))

    print('Training complete!')
