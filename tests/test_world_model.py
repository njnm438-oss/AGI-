import numpy as np
import torch
from agi.world_model import WorldModel

def test_world_model_forward():
    # small smoke test
    s_dim = 16; a_dim = 8
    model = WorldModel(state_dim=s_dim, action_dim=a_dim)
    s = torch.randn(4, s_dim)
    a = torch.randn(4, a_dim)
    next_s = model(s,a)
    assert next_s.shape == (4, s_dim)

def test_train_smoke():
    # just check learner.step runs without crash on toy data
    from agi.replay_buffer import ReplayBuffer
    from agi.learner import Learner
    buf = ReplayBuffer()
    for i in range(64):
        s = np.random.randn(16).astype('float32')
        a = np.random.randn(8).astype('float32')
        # pad or truncate action to state length to avoid broadcasting errors
        a_p = np.zeros_like(s)
        a_p[:len(a)] = a
        s2 = s + 0.1 * a_p
        buf.add((s,a,0.0,s2,False))
    model = WorldModel(16,8)
    learner = Learner(model, buf)
    loss = learner.step(batch_size=16)
    assert isinstance(loss, float) or loss is None
