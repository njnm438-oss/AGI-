import torch
from .world_model_nn import WorldModel

class Simulator:
    def __init__(self, model: WorldModel, device='cpu'):
        self.model = model.to(device)
        self.device = device

    def simulate(self, state: torch.Tensor, action_seq):
        """
        state: torch.Tensor shape (batch, state_dim) or (state_dim,)
        action_seq: iterable of action tensors (seq_len, action_dim) or list
        returns: list of states (including initial)
        """
        self.model.eval()
        states = []
        s = state.to(self.device)
        states.append(s)
        with torch.no_grad():
            for a in action_seq:
                a_t = a.to(self.device)
                # ensure batch dims compatible
                if a_t.dim() == 1:
                    a_in = a_t.unsqueeze(0)
                else:
                    a_in = a_t
                s = self.model(s, a_in)
                states.append(s)
        return states
