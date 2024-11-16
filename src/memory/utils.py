from PIL import Image
from torch.utils.data import Dataset
from torch import from_numpy as torch_from_numpy
import torch
import numpy as np
import pandas as pd
from pathlib import Path

class SecuenceDataLoader(Dataset):
    def __init__(self, path_csv):
        self.df = pd.read_csv(path_csv)
        self.df['fecha'] = pd.to_datetime(self.df['fecha'], format='%H-%M-%S')
        self.df['time_15s'] = self.df['fecha'].dt.floor('15s')
        
        # Agrupar tanto los paths como las teclas
        self.groups = self.df.groupby('time_15s').agg({'path': list, 'teclas': list}).reset_index()

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group_paths = self.groups.iloc[idx, 1]  # Rutas de las imágenes
        group_teclas = self.groups.iloc[idx, 2]  # Teclas correspondientes
        images = [self._load_image(path) for path in group_paths]
        return images, group_teclas  # Retornar imágenes y teclas
    
    def _load_image(self, path):
        image = Image.open(Path(__file__).parent / path).convert("RGB")
        image = image.resize((64,36))
        image = np.array(image)
        image = torch_from_numpy(image).float()
        image = image.permute(2, 0, 1)  # Convertir a formato de canales primero
        return image

def custom_collate(batch):
    images_batch, teclas_batch = zip(*batch)
    
    # Determinar la longitud máxima de la secuencia en el batch
    max_len = max(len(teclas) for teclas in teclas_batch)
    
    # Rellenar las secuencias para que todas tengan la misma longitud
    padded_images = []
    padded_teclas = []
    for images, teclas in zip(images_batch, teclas_batch):
        # Rellenar imágenes
        num_padding = max_len - len(images)
        if num_padding > 0:
            # Crear imágenes de padding (por ejemplo, imágenes negras)
            padding = [torch.zeros_like(images[0]) for _ in range(num_padding)]
            images = images + padding
        padded_images.append(torch.stack(images))
        
        # Rellenar teclas con "<PAD>"
        teclas = teclas + ["<PAD>"] * (max_len - len(teclas))
        padded_teclas.append(teclas)
    
    # Convertir listas de imágenes a tensores
    images_tensor = torch.stack(padded_images)  # Shape: (batch_size, max_len, C, H, W)
    
    # Mantener teclas como listas de cadenas
    return images_tensor, padded_teclas