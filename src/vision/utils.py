import os
from PIL import Image
from torch.utils.data import Dataset
from torch import from_numpy as torch_from_numpy
import numpy as np
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        if image.shape[:2] != (72, 128):
            image = np.array(Image.fromarray(image.astype(np.uint8)).resize((128, 72)), dtype=np.float32)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = np.array(image, dtype=np.float32) / 255.0  # Escala entre 0-1
            image = torch_from_numpy(image).permute(2, 0, 1)  # Reorganiza [H, W, C] -> [C, H, W]
        
        return image