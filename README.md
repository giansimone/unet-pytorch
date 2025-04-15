# unet-pytorch
PyTorch implementation of a convolutional neural network (U-Net) for semantic segmentation of biomedical images.

## Overview
This repository contains a PyTorch implementation of the U-Net architecture for semantic segmentation tasks.

U-Net is a convolutional neural network architecture that was originally designed for biomedical image segmentation.

## Installation
You can install the U-Net model using pip.

```bash
pip install unet-pytorch
```
We recommend using Python 3.11 or later and PyTorch 2.6 or later.

You can also clone the repository and install it in local development mode using poetry.

```bash
git clone https://github.com/giansimone/unet-pytorch.git

cd unet-pytorch

pip install poetry

poetry install
```

## Usage
To use the U-Net model, you can import the `UNet` class from the `unet_pytorch.model` module and create an instance of the model.

```python
import torch

from unet_pytorch.model import UNet

model = UNet(in_channels=1, out_channels=2)

input_tensor = torch.randn(1, 1, 512, 512)

output_tensor = model(input_tensor)

print(output_tensor.shape)  # Should be (1, 2, 512, 512)
```

## Training
To train the U-Net model, you can use the `unet_pytorch.train` module.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.