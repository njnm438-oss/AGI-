import torch

class Learner:
    def __init__(self, world_model, replay_buffer, lr=1e-4, device='cpu'):
        self.model = world_model.to(device)
        self.replay = replay_buffer
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device

    def step(self, batch_size=64):
        if len(self.replay) < batch_size:
            return None
        s,a,r,s2,d = self.replay.sample(batch_size)
        # convert to tensors
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        pred = self.model(s, a)
        loss = torch.nn.functional.mse_loss(pred, s2)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(loss.item())
