import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader 
from .utils import ImageDataset
import os

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        # Definiendo capas
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        hidden = self.input_layer(x)
        hidden = F.relu(hidden) # no linealidad entre capas

        mean = self.fc_mean(hidden)
        logvar = self.fc_var(hidden)

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
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h = torch.relu(self.fc1(z))
        return self.fc2(h)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, laten_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, laten_dim)
        self.decoder = Decoder(laten_dim, hidden_dim, input_dim)
    
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

def train(epochs, image_dir, output_dir=None):
    # hiperparametros
    input_dim = 784
    hidden_dim = 400
    latent_dim = 20
    lr=1e-3
    batch_size = 128
    
    # VAE
    vae = VAE(input_dim, hidden_dim, latent_dim)
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