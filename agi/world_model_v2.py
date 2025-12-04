"""
World Model v2 (WM2) — lightweight seq→seq prototype
- Provides `predict_next` to return next_state, reward, done, latent
- Lightweight GRU-like internal state implemented in NumPy for testing/CI
This is a prototype: replace with PyTorch/TensorFlow model for production.
"""

from typing import List, Tuple, Any
import numpy as np
import hashlib
import logging

logger = logging.getLogger(__name__)


class WorldModelV2:
    """Very small seq→seq predictive model using NumPy.
    Interface:
      - encode(sequence_of_embeddings) -> latent vector
      - predict_next(latent, action_embedding) -> next_state, reward, done, new_latent, confidence
      - rollout(latent, actions, max_steps)
      - random_init_state()
    """

    def __init__(self, latent_dim: int = 128, state_dim: int = 64, seed: int = 42):
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.rng = np.random.RandomState(seed)
        # small weight matrices
        self.W_latent = self.rng.randn(self.latent_dim, self.latent_dim) * 0.01
        self.W_action = self.rng.randn(self.latent_dim, self.state_dim) * 0.01
        self.W_out = self.rng.randn(self.state_dim, self.state_dim) * 0.01
        # bias
        self.b = np.zeros(self.latent_dim)

    def encode(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Encode a sequence of embeddings into a latent vector (simple averaging + transform)."""
        if not embeddings:
            return np.zeros(self.latent_dim)
        seq = np.stack(embeddings)
        avg = np.mean(seq, axis=0)
        # project or pad/truncate
        if avg.shape[0] >= self.latent_dim:
            latent = avg[: self.latent_dim]
        else:
            latent = np.zeros(self.latent_dim)
            latent[: avg.shape[0]] = avg
        # simple transform
        latent = np.tanh(latent @ self.W_latent + self.b)
        return latent

    def predict_next(self, latent: np.ndarray, action_embedding: np.ndarray = None) -> Tuple[np.ndarray, float, bool, np.ndarray, float]:
        """Predict next state (state vector), reward, done, new latent, confidence.
        Returns:
          next_state: ndarray shape (state_dim,)
          reward: float
          done: bool
          new_latent: ndarray shape (latent_dim,)
          confidence: float in [0,1]
        """
        if latent is None:
            latent = np.zeros(self.latent_dim)
        if action_embedding is None:
            action_embedding = np.zeros(self.state_dim)

        # update latent
        # GRU-like update (very simplified)
        z = 1 / (1 + np.exp(- (latent @ np.diag(self.W_latent))))  # pretend gate using diagonal
        # project action embedding into latent space
        try:
            action_proj = self.W_action @ action_embedding
        except Exception:
            # fallback: pad/truncate action embedding
            ae = np.zeros(self.W_action.shape[1])
            ae[: min(len(action_embedding), ae.shape[0])] = action_embedding[: ae.shape[0]]
            action_proj = self.W_action @ ae
        new_latent = np.tanh((latent * z) @ self.W_latent + action_proj + self.b)
        # next state
        next_state = np.tanh(new_latent[: self.state_dim] @ self.W_out)
        # reward: small scalar from dot product
        reward = float(np.tanh(np.dot(next_state[: min(len(next_state), 5)], np.arange(1, min(len(next_state),5)+1))))
        # done: low probability
        done = bool((np.abs(reward) > 0.95) and (np.sum(np.abs(new_latent)) > self.latent_dim * 0.5))
        # confidence: based on norm of latent changes
        conf = float(1.0 / (1.0 + np.linalg.norm(new_latent - latent)))
        return next_state, reward, done, new_latent, conf

    def rollout(self, latent: np.ndarray, action_embeddings: List[np.ndarray], max_steps: int = 10) -> List[dict]:
        """Simulate forward using a sequence of action embeddings. Returns list of step dicts."""
        trace = []
        current = latent.copy() if latent is not None else np.zeros(self.latent_dim)
        steps = 0
        for ae in action_embeddings:
            if steps >= max_steps:
                break
            ns, reward, done, current, conf = self.predict_next(current, ae)
            trace.append({"state": ns, "reward": reward, "done": done, "confidence": conf})
            steps += 1
            if done:
                break
        return trace

    def random_init_state(self) -> np.ndarray:
        return self.rng.randn(self.latent_dim)


if __name__ == "__main__":
    # quick sanity check
    wm = WorldModelV2()
    latent = wm.random_init_state()
    ns, r, d, nl, conf = wm.predict_next(latent)
    print("ns", ns.shape, "r", r, "d", d, "conf", conf)
