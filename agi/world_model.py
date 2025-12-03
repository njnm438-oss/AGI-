import torch
import torch.nn as nn

class WorldModel(nn.Module):
    def __init__(self, input_dim, hidden=512, n_layers=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(n_layers-1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, input_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
