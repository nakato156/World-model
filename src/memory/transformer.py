import torch
import torch.nn as nn
from vision import VAE
from .ActionEncoder import ActionEncoder

class TRANS(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TRANS, self).__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, combined_seq):
        transformer_output = self.transformer(combined_seq, combined_seq)
        output = self.fc_out(transformer_output[:, -1, :])
        return output

class Megatron(nn.Module):
    def __init__(self, w, h, input_dim, latent_dim, key_vocab, modifier_vocab, embedding_dim, d_model, nhead, num_layers):
        super(Megatron, self).__init__()
        self.vae = VAE(w, h, input_dim, latent_dim)
        self.action_encoder = ActionEncoder(key_vocab, modifier_vocab, embedding_dim)
        self.fc_combined = nn.Linear(latent_dim + embedding_dim, d_model)
        self.transformer = TRANS(d_model, nhead, num_layers)
    
    def forward(self, frames, key_lists):
        batch_size, seq_length, C, H, W = frames.size()
        device = frames.device
        
        z_seq = []
        for t in range(seq_length):
            frame = frames[:, t, :, :, :]
            mean, logvar, _ = self.vae(frame)
            z = self.vae.encoder.reparameterize(mean, logvar)
            z_seq.append(z)
        z_seq = torch.stack(z_seq, dim=1)
        
        action_seq = self.action_encoder(key_lists)
        
        combined_seq = torch.cat([z_seq, action_seq], dim=-1)
        combined_seq = self.fc_combined(combined_seq)
        
        output = self.transformer(combined_seq)
        return output

def default_config() -> dict:
    """
    Configuracion default de hyperparametros
    """
    return {
        "w": 64,
        "h": 36,
        "input_dim": 3,
        "latent_dim": 32,
        "embedding_dim": 36,
        "num_frames": 14,
        "num_actions": 36,
        "d_model": 64,
        "nhead": 1,
        "num_layers": 3,
    }