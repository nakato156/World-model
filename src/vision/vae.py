import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader 
from .utils import ImageDataset
import os

class Encoder(nn.Module):
    def __init__(self, w, h, input_channels, latent_dim):
        super(Encoder, self).__init__()

        # Definiendo capas
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(), # salida => w * h * 32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1),
            nn.ReLU(), # salida => 32 * 32 * 64
            nn.Flatten() # salida => 64 * w * h
        )

        self.fc_mean = nn.Linear(w * h * 64, latent_dim)
        self.fc_var = nn.Linear(w * h * 64, latent_dim)
        
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
        hidden_dim = w * h * 64 

        self.fc1 = nn.Linear(latent_dim, hidden_dim)

        self.conv_t_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, z):
        out = torch.relu(self.fc1(z))
        out = out.view(-1, 64, self.w, self.h)
        return self.conv_t_layers(out)

class VAE(nn.Module):
    def __init__(self, w, h, input_dim, laten_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(w, h, input_dim, laten_dim)
        self.decoder = Decoder(w, h, laten_dim, input_dim)
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mean, logvar)
        return mean, logvar, self.decoder(z)

def loss_function(x, x_hat, mu, logvar):
    x_hat_normalized = x_hat / 255.0
    x_normalized = x / 255.0
    BCE = nn.functional.binary_cross_entropy(x_hat_normalized, x_normalized, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(epochs, w, h, image_dir, output_dir=None):
    # hiperparametros
    input_dim = 784
    hidden_dim = 400
    latent_dim = 20
    lr=1e-3
    batch_size = 128
    
    # VAE
    vae = VAE(w, h, input_dim, latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    vae.train()
    
    # Carga del dataset
    train_dataset = ImageDataset(image_dir, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        train_loss = 0
        for x in train_loader:
            optimizer.zero_grad()
            mean, logvar, x_hat = vae(x)
            loss = loss_function(x, x_hat, mean, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}")
    
    if output_dir:
        torch.save(vae.state_dict(), os.path.join(output_dir, 'vae.pth'))