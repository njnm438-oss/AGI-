# agi/world_model.py
import torch
import torch.nn as nn

class WorldModel(nn.Module):
    """Simple world model: encode state -> hidden -> transition -> next state prediction"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.transition = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """state: (B, state_dim), action: (B, action_dim) -> next_state: (B, state_dim)"""
        h = self.encoder(state)
        x = torch.cat([h, action], dim=-1)
        next_s = self.transition(x)
        return next_s
