import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buf = deque(maxlen=capacity)

    def add(self, transition):
        # transition = (s, a, r, s_next, done)
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        # transpose and stack
        return list(map(lambda arr: np.stack(list(arr)), zip(*batch)))

    def __len__(self):
        return len(self.buf)
