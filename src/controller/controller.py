import torch
from vision.vae import VAE
from rn import ActionNetwork
from vision.utils import ImageDataset
from torch.utils.data import DataLoader

class Controller:
    def __init__(self, vae_model_path, action_space, w, h, input_dim, latent_dim, action_dim):
        self.vae = VAE(w, h, input_dim, latent_dim)
        self.vae.load_state_dict(torch.load(vae_model_path))
        self.vae.eval()  # No entrenamos más el VAE

        self.action_network = ActionNetwork(latent_dim, action_dim)  # Red neuronal para elegir acciones
        self.action_space = action_space
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae.to(self.device)
        self.action_network.to(self.device)

    def choose_action(self, latent_z):
        """
        Escoge una acción basada en la representación latente z utilizando la red neuronal entrenada.
        """
        with torch.no_grad():
            action_probs = self.action_network(latent_z)
            action = torch.argmax(action_probs, dim=-1).item()  # Selecciona la acción con mayor probabilidad
        return action

    def process_observation(self, observation):
        """
        Procesa la observación del entorno y devuelve la representación latente z.
        """
        with torch.no_grad():
            observation = observation.to(self.device)
            mean, logvar = self.vae.encoder(observation)
            latent_z = self.vae.encoder.reparameterize(mean, logvar)
        return latent_z

    def get_reward(self, state, action):
        """
        Devuelve la recompensa y penalización correspondiente al estado y acción del robot
        basado en la tabla proporcionada.
        """
        if state == "Imagen de terreno plano con obstáculos bajos":
            if action == "Paso hacia adelante (0.3 m/s)":
                return 5, -1  # Recompensa y penalización
        elif state == "Imagen de terreno con obstáculos medianos":
            if action == "Paso rápido (0.5 m/s)":
                return 8, -2
        elif state == "Imagen de terreno resbaladizo":
            if action == "Paso cuidadoso (0.2 m/s)":
                return 6, -4
        elif state == "Imagen de terreno con desniveles moderados":
            if action == "Paso hacia adelante con ajuste de balance (0.4 m/s)":
                return 9, -2
        elif state == "Imagen de terreno con vegetación densa":
            if action == "Avanzar lentamente y esquivar obstáculos (0.3 m/s)":
                return 7, -1
        elif state == "Imagen de terreno rocoso":
            if action == "Paso hacia adelante con cuidado (0.2 m/s)":
                return 5, -5
        elif state == "Seguir ruta específica en terreno complicado":
            if action == "Mantenerse en la ruta designada sin desviarse (0.3 m/s)":
                return 15, -5
        elif state == "Evitar colisiones durante 30 segundos en terreno difícil":
            if action == "Navegar por el entorno evitando colisiones con obstáculos móviles":
                return 20, -5
        elif state == "Mala detección de obstáculos":
            if action == "Error al detectar y evitar un obstáculo a tiempo":
                return 0, -10
        return 0, 0  

    def run(self, environment, num_episodes=100):
        """
        Bucle de control principal: Interactúa con el entorno, elige acciones y realiza entrenamiento/optimización.
        """
        for episode in range(num_episodes):
            state = environment.reset()  # Estado inicial
            done = False
            total_reward = 0

            while not done:
                # Convierte la observación del entorno en el formato correcto
                observation = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                # Procesa la observación a través del VAE
                latent_z = self.process_observation(observation)

                # Escoge una acción
                action = self.choose_action(latent_z)

                # Realiza la acción en el entorno
                new_state, reward, done, _ = environment.step(action)

                # Obtener recompensas y penalizaciones
                reward, penalty = self.get_reward(state, action)

                # Actualizar la recompensa total
                total_reward += reward + penalty

                # Actualizar el estado actual
                state = new_state

            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")
