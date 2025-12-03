def test_learner_step_smoke():
    from agi.world_model_nn import WorldModel
    from agi.replay_buffer import ReplayBuffer
    from agi.learner import Learner
    import numpy as np

    state_dim = 8
    action_dim = 2
    model = WorldModel(state_dim, action_dim, hidden=32)
    rb = ReplayBuffer()
    # fill
    for _ in range(64):
        s = np.random.randn(state_dim)
        a = np.random.randn(action_dim)
        s2 = np.random.randn(state_dim)
        rb.add((s, a, 0.0, s2, False))
    learner = Learner(model, rb, lr=1e-3, device='cpu')
    loss = learner.step(batch_size=32)
    assert loss is None or isinstance(loss, float)
