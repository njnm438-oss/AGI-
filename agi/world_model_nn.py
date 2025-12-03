import torch
import torch.nn as nn

class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=512):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        # transition predicts next-state in raw state space (or embedding space)
        self.transition = nn.Sequential(
            nn.Linear(hidden + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )

    def forward(self, state, action):
        """
        state: tensor (..., state_dim)
        action: tensor (..., action_dim)
        returns next_state_pred: tensor (..., state_dim)
        """
        h = self.encoder(state)
        x = torch.cat([h, action], dim=-1)
        next_state = self.transition(x)
        return next_state
