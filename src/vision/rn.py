import torch
import torch.nn as nn
import torch.optim as optim

class ActionNetwork(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(ActionNetwork, self).__init__()
        
        # Definir una red neuronal simple con 2 capas ocultas
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.fc3(x), dim=-1)  # Distribución de probabilidades de las acciones
        return action_probs

def train_action_network(latent_vectors, actions, action_dim, epochs=10, lr=0.001):
    """
    Entrenamiento de la red neuronal que predice acciones a partir de las representaciones latentes z.
    
    Args:
        latent_vectors (Tensor): Conjunto de representaciones latentes.
        actions (Tensor): Conjunto de acciones verdaderas (one-hot encoded).
        action_dim (int): Tamaño del espacio de acción.
        epochs (int): Número de épocas para el entrenamiento.
        lr (float): Tasa de aprendizaje.
    """
    network = ActionNetwork(latent_vectors.size(1), action_dim)
    optimizer = optim.Adam(network.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = network(latent_vectors)
        loss = loss_fn(output, actions)
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')
    
    return network
