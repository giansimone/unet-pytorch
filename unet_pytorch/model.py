"""
Module containing the U-Net model.
"""
import torch
from torch import nn


class DoubleConv(nn.Module):
    """Double convolutional neuronal network for a U-Net."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        Args:
            x: Input tensor.
        Returns:
            Output tensor.
        """
        return self.conv_op(x)


class DownBlock(nn.Module):
    """Downsampling block for a U-Net."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        Args:
            x: Input tensor.
        Returns:
            Output tensor.
        """
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upsampling block for a U-Net."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        Args:
            x1: Input tensor from the previous layer (UpBlock).
            x2: Input tensor from the encoder (DownBlock).
        Returns:
            Output tensor.
        """
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolutional block for a U-Net."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.conv_op = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        Args:
            x: Input tensor.
        Returns:
            Output tensor.
        """
        return self.conv_op(x)


class UNet(nn.Module):
    """Convolutional neuronal network U-Net."""
    def __init__(self, in_channels: int=1, out_channels: int=3):
        """
        Args:
           in_channels: Number of input channels.
           out_channels: Number of output channels (i.e., classes).
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_conv = DoubleConv(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)

        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.out_conv = OutConv(64, out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        Args:
            x: Input tensor.
        Returns:
            Output tensor.
        """
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.out_conv(x)
        return self.softmax(x)
