import torch
import numpy as np

class Learner:
    def __init__(self, world_model, replay, lr=1e-4, device='cpu'):
        self.model = world_model
        self.replay = replay
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device

    def step(self, batch_size=64):
        if len(self.replay) < batch_size:
            return None
        s, a, r, s2, done = self.replay.sample(batch_size)
        # Convert to tensors
        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.float32).to(self.device)
        s2 = torch.tensor(s2, dtype=torch.float32).to(self.device)
        # forward
        pred = self.model(s, a)
        loss = torch.nn.functional.mse_loss(pred, s2)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())
