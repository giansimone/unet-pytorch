"""
Module to make predictions using the U-Net model.
"""
import numpy as np
import torch

from unet_pytorch.model import UNet
from unet_pytorch.utils import im_to_tensor


class UNetPredictor:
    """Predictor class for the U-Net model."""

    def __init__(self, model_file: str) -> None:
        """Initialise the U-Net predictor.

        Args:
            path: Path to the model.
        """
        self.model_file = model_file

        if torch.backends.mps.is_available():
            self.device = torch.device('mps') # Apple silicon
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = UNet().to(self.device)
        self.model.load_state_dict(
            torch.load(self.model_file, map_location=self.device)['model_weights']
        )
        self.model.eval()

    def predict(self, image_file: str) -> np.ndarray:
        """Make a prediction using the U-Net.
        Args:
            image_file: Path to the image.
        Returns:
            pred: Prediction done for a single image.
        """
        im = im_to_tensor(image_file)
        im = im.unsqueeze(0)
        im = im.to(self.device)

        with torch.no_grad():
            pred = self.model(im)

        return pred.squeeze(0).argmax(0).detach().cpu().numpy()

    def predict_batch(self, image_files: list[str]) -> list[np.ndarray]:
        """Make predictions for a batch of images.
        
        Args:
            image_files: List of paths to the images.
        Returns:
            preds: List of predictions for each image.
        """
        preds = []
        for image_file in image_files:
            pred = self.predict(image_file)
            preds.append(pred)

        return preds
