import unittest
import torch
from src.vision.vae import VAE
from src.vision.rn import ActionNetwork
from src.vision.controller import Controller

class TestVAE(unittest.TestCase):
    
    def setUp(self):
        # Inicializa el modelo VAE para las pruebas
        self.w, self.h = 64, 64
        self.input_dim = 3
        self.latent_dim = 20
        self.vae = VAE(self.w, self.h, self.input_dim, self.latent_dim)
    
    def test_vae_forward_pass(self):
        # Crea un tensor aleatorio que simula una imagen (batch size de 1)
        x = torch.randn(1, self.input_dim, self.w, self.h)
        mean, logvar, decoded = self.vae(x)
        
        # Verifica las dimensiones del espacio latente y la salida
        self.assertEqual(mean.shape, (1, self.latent_dim))
        self.assertEqual(logvar.shape, (1, self.latent_dim))
        self.assertEqual(decoded.shape, x.shape)

class TestActionNetwork(unittest.TestCase):
    
    def setUp(self):
        # Inicializa la red neuronal para acciones
        self.latent_dim = 20
        self.action_dim = 4
        self.action_net = ActionNetwork(self.latent_dim, self.action_dim)
    
    def test_action_network_output(self):
        # Prueba una pasada hacia adelante (forward pass) por la red neuronal
        latent_z = torch.randn(1, self.latent_dim)
        action_probs = self.action_net(latent_z)
        
        # Verifica que la salida tenga el tamaño correcto
        self.assertEqual(action_probs.shape, (1, self.action_dim))

class TestController(unittest.TestCase):
    
    def setUp(self):
        # Inicializa el controlador para las pruebas
        self.w, self.h = 64, 64
        self.input_dim = 3
        self.latent_dim = 20
        self.action_dim = 4
        self.controller = Controller(
            vae_model_path="path/to/vae.pth",
            action_space=[0, 1, 2, 3],
            w=self.w, h=self.h,
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            action_dim=self.action_dim
        )

    def test_controller_action_choice(self):
        # Prueba que el controlador pueda procesar una observación y elegir una acción
        observation = torch.randn(1, self.input_dim, self.w, self.h)
        latent_z = self.controller.process_observation(observation)
        action = self.controller.choose_action(latent_z)
        
        # Verifica que la acción esté dentro del espacio de acciones
        self.assertIn(action, [0, 1, 2, 3])

if __name__ == '__main__':
    unittest.main()
