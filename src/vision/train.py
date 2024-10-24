import os
from vae import train
import torch

epochs = 100
w, h = 128, 72

image_dir = 'images'  # Directorio donde están las 5 imágenestorch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_dir = 'output'  # Directorio donde se guardará el modelo

print(torch.cuda.is_available())
torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Crear directorio de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Entrenar el modelo
train(epochs, w, h, image_dir, output_dir)