import numpy as np
from agi.replay_buffer import ReplayBuffer

def test_replay_add_and_sample():
    buf = ReplayBuffer(capacity=10)
    for i in range(5):
        s = np.zeros(3)
        a = np.zeros(1)
        buf.add((s, a, 0.0, s, False))
    assert len(buf) == 5
    batch = buf.sample(4)
    # sample returns list of arrays: s,a,r,s2,done
    assert isinstance(batch, list) or hasattr(batch, '__iter__')
