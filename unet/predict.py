"""
Module to make predictions using the U-Net model.
"""
import os

import torch

from unet.model import UNet
from unet.utils import im_to_tensor


def predict(path_model: str, im_path: str) -> torch.Tensor:
    """Make a prediction using the U-Net.

    Args:
        path_model: Path to the model weights.
        im_path: Path to the image.
    Returns:
        pred: Prediction done for a single image.
    """
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load image on `device`
    im = im_to_tensor(im_path)
    im = im.unsqueeze(0)
    im = im.to(device)

    # Build the model and load the weights
    model = UNet().to(device)
    model.load_state_dict(
        torch.load(os.path.join(path_model, 'weights.pth'), map_location=device)
    )

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        pred = model(im)

    return pred.squeeze(0).detach().cpu().numpy()


def mask_from_pred(pred: torch.Tensor) -> torch.Tensor:
    """Calculate a mask from the prediction.
    Args:
        pred: Prediction done for a single image.
    Returns:
        mask: Mask calculated from the prediction.
    """

    return pred.argmax(0)
