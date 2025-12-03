import torch
from agi.world_model_nn import WorldModel

def test_world_model_shapes():
    state_dim = 16
    action_dim = 4
    model = WorldModel(state_dim, action_dim, hidden=64)
    s = torch.randn(2, state_dim)
    a = torch.randn(2, action_dim)
    out = model(s, a)
    assert out.shape == s.shape

def test_world_model_train_step():
    state_dim = 8
    action_dim = 3
    model = WorldModel(state_dim, action_dim, hidden=32)
    s = torch.randn(10, state_dim)
    a = torch.randn(10, action_dim)
    s2 = torch.randn(10, state_dim)
    pred = model(s, a)
    loss = torch.nn.functional.mse_loss(pred, s2)
    assert float(loss) >= 0.0
