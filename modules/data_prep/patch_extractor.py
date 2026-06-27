# modules/data_prep/patch_extractor.py
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

class TissuePatchExtractor:
    """
    Extracts informative non-background tiles from Whole Slide Images
    and encodes them into low-dimensional representations using DINOv2.
    """
    def __init__(self, patch_size=224):
        self.patch_size = patch_size
        # Load frozen foundational vision model from PyTorch Hub
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.dinov2.eval()
        
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def is_informative_patch(self, patch: Image.Image, threshold=0.8) -> bool:
        """Filters out blank background patches (white space) using grayscale variance."""
        gray = patch.convert('L')
        arr = np.array(gray)
        # If variance is extremely low or mean is near pure white, discard it
        if np.mean(arr) > 220 and np.std(arr) < 15:
            return False
        return True

    @torch.no_grad()
    def extract_features(self, patch: Image.Image) -> torch.Tensor:
        """Encodes a single 224x224 patch into a 384-dimensional DINOv2 feature vector."""
        tensor = self.transform(patch).unsqueeze(0).to(self.device)
        embedding = self.dinov2(tensor) # Output shape: [1, 384]
        return embedding.squeeze(0).cpu()