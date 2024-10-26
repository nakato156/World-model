import torch 
import torch.nn as nn 
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader 
from utils import ImageDataset
import os

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, w, h, input_channels, latent_dim):
        super(Encoder, self).__init__()

        # Definiendo capas
        self.conv_layers = nn.Sequential(
            # 36, 32, 3, 1
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(), # salida => w * h * 32
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(), # salida => 32 * 32 * 64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(), # salida => 32 * 32 * 64
            nn.Flatten() # salida => 64 * w * h
        )
        f = 64
        self.fc_mean = nn.Linear(w * h * f, latent_dim)
        self.fc_var = nn.Linear(w * h * f, latent_dim)
        
    def forward(self, x):
        conv = self.conv_layers(x)
        mean = self.fc_mean(conv)
        logvar = self.fc_var(conv)

        return mean, logvar
    
    def reparameterize(self, mean, logvar): # crea el espacio latente
        """
        Reparametrización de la distribución normal para obtener `z`.
        `z = μ + σ * ϵ`

        Args:
            mean (Tensor): Media de la distribución normal.
            logvar (Tensor): Logaritmo de la varianza de la distribución normal.

        Returns:
            Tensor: Valor reparametrizado `z`.
        """

        std = torch.exp(0.5 * logvar) #desviacion estandar de logvar
        eps = torch.randn_like(std) # epsilon
        return mean + eps * std

class Decoder(nn.Module):
    def __init__(self, w, h, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.w = w
        self.h = h
        self.f = 64
        hidden_dim = w * h * self.f
        self.fc1 = nn.Linear(latent_dim, hidden_dim)

        self.conv_t_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LayerNorm([32, self.h, self.w]),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LayerNorm([16, self.h, self.w]),
            nn.ConvTranspose2d(in_channels=16, out_channels=output_dim, kernel_size=3, padding=1),
        )
    
    def forward(self, z):
        out = torch.relu(self.fc1(z))
        # print("Después de fc1:", out.mean().item(), out.std().item())
        out = out.view(z.size(0), self.f, self.h, self.w)
        a = self.conv_t_layers(out)
        # print("Salida antes de sigmoid:", a.mean().item(), a.std().item())
        reconstructed = torch.sigmoid(a)
        # print("Salida después de sigmoid:", reconstructed.mean().item(), reconstructed.std().item())

        return reconstructed

class VAE(nn.Module):
    def __init__(self, w, h, input_dim, laten_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(w, h, input_dim, laten_dim)
        self.decoder = Decoder(w, h, laten_dim, input_dim)
    
    def forward(self, x):
        # print("Entrada al Encoder - Mean:", x.mean().item(), "Std Dev:", x.std().item())
    
        mean, logvar = self.encoder(x)
        
        z = self.encoder.reparameterize(mean, logvar)
        # print("Mean:", mean.mean().item(), "Std Dev:", z.std().item())

        return mean, logvar, self.decoder(z)

def loss_function(x, x_hat, mu, logvar):
    x_hat_normalized = x_hat
    x_normalized = x
    # print("x_hat_normalized", x_hat_normalized.min().item(), x_hat_normalized.max().item())
    # print("x_normalized", x_normalized.min().item(), x_normalized.max().item())
    beta = 0.1
    BCE = nn.functional.binary_cross_entropy(x_hat_normalized, x_normalized, reduction='sum')
    KLD = -0.5 * beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def show_images(original, reconstructed, epoch):
    original = original.cpu().numpy().transpose(0, 2, 3, 1)  # Pasa a formato HWC
    reconstructed = reconstructed.cpu().detach().numpy().transpose(0, 2, 3, 1)  # Pasa a formato HWC

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    axs[0].imshow(original[0])  # Muestra la primera imagen del batch
    axs[0].set_title('Original')
    axs[0].axis('off')
    
    axs[1].imshow(reconstructed[0])  # Muestra la primera imagen reconstruida
    axs[1].set_title('Reconstruida')
    axs[1].axis('off')
    
    plt.suptitle(f'Epoch {epoch}')
    plt.show()

def train(epochs, w, h, image_dir, output_dir=None):
    # hiperparametros
    input_dim = 3
    latent_dim = 20
    lr=1e-3
    batch_size = 1 # 4
    
    # Verifica si cuda está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # VAE
    vae = VAE(w, h, input_dim, latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    
    vae.train()
    
    transform = transforms.Compose([
        transforms.ToTensor(),           # Convierte a tensor y escala a [0, 1]
    ])

    # Carga del dataset
    train_dataset = ImageDataset(image_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("Training")
    for epoch in range(epochs):
        train_loss = 0
        for x in train_loader:
            x = x.to(device) / 255.0    # Mueve las imágenes al dispositivo
            optimizer.zero_grad()
            mean, logvar, x_hat = vae(x)
            loss = loss_function(x, x_hat, mean, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader)}")

        if epoch % 10 == 0:
            with torch.no_grad():
                sample = x[:1]  # Selecciona una imagen del batch
                _, _, reconstructed = vae(sample)
                show_images(sample, reconstructed, epoch + 1)
            
    if output_dir:
        torch.save(vae.state_dict(), os.path.join(output_dir, 'vae.pth'))
