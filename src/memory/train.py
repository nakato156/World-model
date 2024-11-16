import torch
import torch.optim as optim
from torch import nn
from .transformer import Megatron, default_config
from .utils import SecuenceDataLoader, custom_collate
from torch.utils.data import DataLoader 
from pathlib import Path

def main():
    # Configuración del modelo
    config = default_config()
    key_vocab = {k: i for i, k in enumerate("abcdefghijklmnopqrstuvwxyz")}
    modifier_vocab = {"ctrl": 0, "shift": 1, "alt": 2, "tab": 3, "space": 4, "cap": 5}

    # Instancia del modelo Megatron
    model = Megatron(
        w=config["w"],
        h=config["h"],
        input_dim=config["input_dim"],
        latent_dim=config["latent_dim"],
        key_vocab=key_vocab,
        modifier_vocab=modifier_vocab,
        embedding_dim=config["embedding_dim"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"]
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Optimizador
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Datos ficticios para ilustrar
    # Suponiendo que tienes un dataloader que proporciona batches de (frames, key_lists, target)
    seq_frames = SecuenceDataLoader(Path(__file__).parent / "./videos.csv")
    dataloader = DataLoader(seq_frames, batch_size=33, collate_fn=custom_collate)

    # Bucle de entrenamiento
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # Modo entrenamiento
        total_loss = 0.0
        
        for frames, key_lists in dataloader:
            frames, target = frames[:-1], frames[-1]
            # print(key_lists)
            # exit()
            frames, target = frames.to(model.fc_combined.weight.device), target.to(model.fc_combined.weight.device)
            
            # Forward pass
            output = model(frames, key_lists)
            
            # Cálculo de la pérdida
            loss = nn.functional.mse_loss(output, target)  # Cambia la función de pérdida según tu tarea
            
            # Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

if __name__ == "__main__":
    main()