import torch
from vision.vae import VAE
from rn import ActionNetwork
from memory.transformer import Megatron 
from vision.utils import ImageDataset
from torch.utils.data import DataLoader

class Controller:
    def __init__(self, vae_model_path, transformer_config, action_space, w, h, input_dim, latent_dim, action_dim):
        # Cargar VAE
        self.vae = VAE(w, h, input_dim, latent_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae.load_state_dict(torch.load(vae_model_path, map_location=self.device))
        self.vae.eval()  # No entrenamos más el VAE
        self.vae.to(self.device)

        self.action_network = ActionNetwork(latent_dim + transformer_config['d_model'], action_dim)
        self.action_network.to(self.device)
        self.action_space = action_space

        self.transformer = Megatron(
            transformer_config['num_frames'],
            transformer_config['num_actions'],
            transformer_config['d_model'],
            transformer_config['nhead'],
            transformer_config['num_layers']
        )
        self.transformer.to(self.device)

    def choose_action(self, latent_z, transformer_memory):
        with torch.no_grad():
            combined_input = torch.cat([latent_z, transformer_memory], dim=-1)
            action_probs = self.action_network(combined_input)
            action = torch.argmax(action_probs, dim=-1).item()
        return action

    def process_observation(self, observation):
        with torch.no_grad():
            observation = observation.to(self.device)
            mean, logvar = self.vae.encoder(observation)
            latent_z = self.vae.encoder.reparameterize(mean, logvar)
        return latent_z

    def process_transformer(self, frames, actions):
        frames = frames.to(self.device)
        actions = actions.to(self.device)
        
        _, transformer_memory = self.transformer(frames, actions)
        return transformer_memory[-1] 

    def get_reward(self, state, action):
        """
        Devuelve la recompensa y penalización correspondiente al estado y acción del robot
        basado en la tabla proporcionada.
        """
        if state == "Imagen de terreno plano con obstáculos bajos":
            if action == "Paso hacia adelante (0.3 m/s)":
                return 10, -1  
            elif action == "Correr hacia adelante (1.0 m/s)":
                return 12, -2  

        elif state == "Imagen de terreno con obstáculos medianos":
            if action == "Paso rápido (0.5 m/s)":
                return 8, -2 
            elif action == "Esquivar obstáculos (0.4 m/s)":
                return 10, -3  

        elif state == "Imagen de terreno resbaladizo":
            if action == "Paso cuidadoso (0.2 m/s)":
                return 7, -3  
            elif action == "Paso rápido (0.5 m/s)":
                return 2, -6  

        elif state == "Imagen de terreno con desniveles moderados":
            if action == "Paso hacia adelante con ajuste de balance (0.4 m/s)":
                return 9, -2  
            elif action == "Correr con ajuste de balance (0.6 m/s)":
                return 11, -4 

        elif state == "Imagen de terreno con vegetación densa":
            if action == "Avanzar lentamente y esquivar obstáculos (0.3 m/s)":
                return 10, -1 
            elif action == "Empujar a través de la vegetación (0.4 m/s)":
                return 6, -3  

        elif state == "Imagen de terreno rocoso":
            if action == "Paso hacia adelante con cuidado (0.2 m/s)":
                return 7, -3  
            elif action == "Salto entre rocas (0.5 m/s)":
                return 4, -5 

        elif state == "Seguir ruta específica en terreno complicado":
            if action == "Mantenerse en la ruta designada sin desviarse (0.3 m/s)":
                return 15, -2  
            elif action == "Salir de la ruta designada":
                return -5, -5  

        elif state == "Evitar colisiones durante 30 segundos en terreno difícil":
            if action == "Navegar por el entorno evitando colisiones con obstáculos móviles":
                return 20, -3  

        elif state == "Mala detección de obstáculos":
            if action == "Error al detectar y evitar un obstáculo a tiempo":
                return 0, -10  
            elif action == "Corrección rápida después del error":
                return 5, -2  

        return 0, 0

    def run(self, environment, num_episodes=100):
        for episode in range(num_episodes):
            state = environment.reset() 
            done = False
            total_reward = 0

            frames_seq = []
            actions_seq = []

            while not done:
                observation = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                # Obtener la representación latente z del VAE
                latent_z = self.process_observation(observation)

                # Limitar la secuencia a un tamaño máximo (10 en este caso)
                if len(frames_seq) < 10: 
                    frames_seq.append(observation)
                    actions_seq.append(torch.zeros((1, self.transformer.transformer.d_model), device=self.device))
                else:
                    frames_seq.pop(0)
                    actions_seq.pop(0)

                # Crear tensores de secuencias
                frames_tensor = torch.stack(frames_seq).squeeze(1)
                actions_tensor = torch.stack(actions_seq).squeeze(1)

                # Procesar la secuencia de frames y acciones en el Transformer
                transformer_memory = self.process_transformer(frames_tensor, actions_tensor)

                # Elegir acción basada en la latente y la memoria del Transformer
                action = self.choose_action(latent_z, transformer_memory)

                # Ejecutar la acción en el entorno
                new_state, reward, done, _ = environment.step(action)

                # Obtener la recompensa personalizada basada en estado/acción
                reward, penalty = self.get_reward(state, action)

                total_reward += reward + penalty

                state = new_state

            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")
