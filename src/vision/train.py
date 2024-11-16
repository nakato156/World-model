import os
from .vae import train, show_images
import torch


def evaluate_model(vae, data_loader, device, output_dir, num_samples=5):
    vae.eval()
    with torch.no_grad():
        samples = []
        for i, x in enumerate(data_loader):
            x = x.to(device) / 255
            _, _, reconstructed = vae(x)
            samples.append((x.cpu(), reconstructed.cpu()))
            if i >= num_samples - 1:
                break

    for idx, (original, reconstructed) in enumerate(samples):
        show_images(original, reconstructed, f"eval_{idx + 1}")

    print("Evaluation complete. Check output directory for results.")
    

def train_model():
    epochs = 100
    w, h = 160, 90

    image_dir = 'vision/images'  # Directorio donde están las 5 imágenestorch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_dir = 'output'  # Directorio donde se guardará el modelo

    print(torch.cuda.is_available())
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Crear directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Entrenar el modelo
    train(epochs, w, h, image_dir, output_dir)

if __name__ == "__main__" :
    train_model()
    print("Training complete. Check output directory for results.")