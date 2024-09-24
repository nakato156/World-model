import torch
import torch.nn as nn

class Megatron(nn.Module):
    def __init__(self, num_frames, num_actions, d_model, nhead, num_layers):
        super(Megatron, self).__init__()
        self.embedding_frame = nn.Linear(num_frames, d_model)
        self.embedding_action = nn.Linear(num_actions, d_model)
        
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc_out = nn.Linear(d_model, num_frames)

    def forward(self, frames, actions):
        frame_emb = self.embedding_frame(frames)
        action_emb = self.embedding_action(actions)
        
        # Combinar embeddings
        combined = frame_emb + action_emb
        
        transformer_output = self.transformer(combined)
        
        output = self.fc_out(transformer_output[-1])
        return output

def default_config() -> dict:
    """
    Configuracion default de hyperparametros
    """
    return {
        "num_frames": 10,
        "num_actions": 5,
        "d_model": 64,
        "nhead": 6,
        "num_layers": 3,
    }

if __name__ == "__main__":
    config = default_config()
    num_frames = config["num_frames"]
    num_actions = config["num_actions"]
    d_model = config["d_model"]
    nhead = config["nhead"]
    num_layers = config["num_layers"]

    model = Megatron(num_frames, num_actions, d_model, nhead, num_layers)

    # Dummy data
    frames = torch.rand(num_frames, 1, num_frames)      # (sequence_length, batch_size, num_frames)
    actions = torch.rand(num_frames, 1, num_actions)    # (sequence_length, batch_size, num_actions)

    # Forward pass
    predicted_frame = model(frames, actions)
    print(predicted_frame.shape)
