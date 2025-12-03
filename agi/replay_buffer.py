import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int = 200000):
        self.buf = deque(maxlen=capacity)

    def add(self, transition):
        # transition: (s, a, r, s_next, done)
        self.buf.append(transition)

    def sample(self, batch_size: int = 64):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        # transpose
        s, a, r, s2, d = zip(*batch)
        return map(lambda x: np.stack(x), (s,a,r,s2,d))

    def __len__(self):
        return len(self.buf)
