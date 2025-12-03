# agi/simulator.py
import torch
from .world_model import WorldModel

class Simulator:
    def __init__(self, model: WorldModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device

    def simulate(self, state: torch.Tensor, action_seq: torch.Tensor):
        """
        Simulate a sequence of actions.
        state: (state_dim,) or (1, state_dim)
        action_seq: (T, action_dim) or (1, T, action_dim)
        returns list of states (T+1 items)
        """
        s = state.to(self.device).unsqueeze(0) if state.dim() == 1 else state.to(self.device)
        seq = action_seq.to(self.device)
        out = [s.squeeze(0).cpu()]
        for i in range(seq.shape[0]):
            a = seq[i].unsqueeze(0)
            s = self.model(s, a)
            out.append(s.squeeze(0).cpu())
        return out
