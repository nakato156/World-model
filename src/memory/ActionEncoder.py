import torch
import torch.nn as nn

class ActionEncoder(nn.Module):
    def __init__(self, key_vocab, modifier_vocab, embedding_dim):
        super(ActionEncoder, self).__init__()
        self.key_vocab = key_vocab  # Teclas base (por ejemplo, letras, números)
        self.modifier_vocab = modifier_vocab  # Teclas modificadoras (Ctrl, Shift, Alt)
        
        # Embeddings para teclas base
        self.base_embedding = nn.Embedding(len(key_vocab), embedding_dim)
        
        # Embeddings para modificadores
        self.modifier_embedding = nn.Embedding(len(modifier_vocab), embedding_dim)
    
    def forward(self, key_lists):
        """
        Args:
            key_lists (List[List[List[str]]]): Lista de listas de listas de teclas.
                                               Ejemplo: [[['Ctrl', 'A']], [['Shift', 'B']], ...]
        
        Returns:
            Tensor: Embeddings para cada paso de tiempo, con modificadores aplicados.
                    Shape: (batch_size, seq_length, embedding_dim)
        """
        batch_size = len(key_lists)
        seq_length = len(key_lists[0])
        device = next(self.parameters()).device
        
        # Inicializar tensor para embeddings de teclas base y modificadores
        base_embedding_tensor = torch.zeros(batch_size, seq_length, self.base_embedding.embedding_dim, device=device)
        modifier_embedding_tensor = torch.zeros(batch_size, seq_length, self.modifier_embedding.embedding_dim, device=device)
        
        # Llenar embeddings para teclas y modificadores
        for b, batch in enumerate(key_lists):
            for t, keys in enumerate(batch):
                base_keys = [k for k in keys if k in self.key_vocab]
                modifiers = [k for k in keys if k in self.modifier_vocab]
                
                # Procesar teclas base
                if base_keys:
                    base_indices = torch.tensor([self.key_vocab[k] for k in base_keys], device=device)
                    base_embedding_tensor[b, t] = self.base_embedding(base_indices).mean(dim=0)
                
                # Procesar modificadores y sumarlos
                if modifiers:
                    modifier_indices = torch.tensor([self.modifier_vocab[m] for m in modifiers], device=device)
                    modifier_embedding_tensor[b, t] = self.modifier_embedding(modifier_indices).sum(dim=0)
        
        # Combinar las teclas base con sus modificadores
        action_embeddings = base_embedding_tensor + modifier_embedding_tensor
        return action_embeddings
