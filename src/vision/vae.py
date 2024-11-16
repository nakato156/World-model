import torch 
import torch.nn as nn 
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader 
from .utils import ImageDataset
import os

import matplotlib.pyplot as plt
import datetime
import json

fecha = datetime.date.today().isoformat()

class Encoder(nn.Module):
    def __init__(self, w, h, input_channels, latent_dim):
        super(Encoder, self).__init__()

        # Definiendo capas
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            # nn.Flatten()
        )

        conv_output_dim = 7 * 14 * 256
        self.fc_mean = nn.Linear(conv_output_dim, latent_dim)
        self.fc_var = nn.Linear(conv_output_dim, latent_dim)
        
    def forward(self, x):
        # print("Input:", x.shape)
        x = self.conv_layers(x)
        # print("X:", x.shape)
        x = torch.flatten(x, start_dim=1)
        mean = self.fc_mean(x)
        logvar = self.fc_var(x)
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
        self.f = 256
        hidden_dim = 7 * 14 * self.f

        self.fc1 = nn.Linear(latent_dim, hidden_dim)

        self.conv_t_layers = nn.Sequential(
            nn.ConvTranspose2d(self.f, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2,),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_dim, kernel_size=4, stride=2),
        )
    
    def forward(self, z):
        # apply view for unflatten
        out = torch.relu(self.fc1(z))
        # print("Out1:", out.shape)

        out = out.view(-1, self.f, 7, 14)
        # print("Out2:", out.shape)

        a = self.conv_t_layers(out)
        # print("convT:", a.shape)
        reconstructed = torch.sigmoid(a)
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
        reconstructed = self.decoder(z)
        return mean, logvar, reconstructed
    
    def loss_function(self, recon_x, x, mu, logvar, kl_tolerance=0.5):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = torch.maximum(kl_loss, torch.tensor(kl_tolerance * mu.size(1)))
        return recon_loss + kl_loss

def show_images(original, reconstructed, epoch):
    if not os.path.exists("im-logs"):
        os.makedirs("im-logs")

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
    plt.savefig(f"im-logs/{epoch}.png")

def train(epochs, w, h, image_dir, output_dir=None):
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # hiperparametros
    input_dim = 3
    latent_dim = 64
    lr=1e-3
    batch_size = 4
    
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

    # Registro de pérdidas
    epoch_losses = []

    print("Training")
    for epoch in range(epochs):
        train_loss = 0
        for x in train_loader:
            x = x.to(device) / 255    # Mueve las imágenes al dispositivo
            optimizer.zero_grad()
            mean, logvar, x_hat = vae(x)
            loss = vae.loss_function(x_hat, x, mean, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

        if epoch % 10 == 0:
            with torch.no_grad():
                sample = x[:1]  # Selecciona una imagen del batch
                _, _, reconstructed = vae(sample)
                show_images(sample, reconstructed, epoch + 1)
                torch.save(vae.state_dict(), os.path.join(output_dir, f'vae_{fecha}.pth'))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"training_loss_{fecha}.png"))
    plt.show()

    if output_dir:
        with open(os.path.join(output_dir, f'training_stats_{fecha}.json'), 'w') as f:
            json.dump({"epoch_losses": epoch_losses}, f)

        torch.save(vae.state_dict(), os.path.join(output_dir, f'vae_final_{fecha}.pth'))
