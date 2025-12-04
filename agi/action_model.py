"""
Action Model (AM1) — prototype
- Predicts probability of success for a given (state, action)
- Suggests alternative actions if predicted failure
- Simple heuristic + small logistic model using NumPy
"""

from typing import List, Tuple, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ActionModel:
    def __init__(self, rng_seed: int = 123, action_dim: int = 32):
        self.rng = np.random.RandomState(rng_seed)
        # small weight for logistic predictor
        self.W = self.rng.randn(action_dim) * 0.01
        self.b = 0.0
        self.action_catalog = [f"action_{i}" for i in range(20)]

    def embed_action(self, action: str) -> np.ndarray:
        """Simple embedding for action strings."""
        h = np.abs(hash(action)) % 1000
        emb = (np.arange(32) * 0.01 + (h % 100) * 0.001)
        return emb

    def predict_success(self, state_vector: np.ndarray, action: str) -> Tuple[float, float]:
        """Return (p_success, confidence) between 0 and 1."""
        a_emb = self.embed_action(action)
        # combine
        vec = np.concatenate([state_vector[: len(a_emb)], a_emb])[: len(self.W)]
        logit = float(np.dot(vec, self.W) + self.b)
        p = 1.0 / (1.0 + np.exp(-logit))
        conf = 0.5 + 0.5 * (1.0 - abs(logit) / (1.0 + abs(logit)))
        return float(p), float(conf)

    def suggest_alternatives(self, state_vector: np.ndarray, action: str, k: int = 3) -> List[Tuple[str, float]]:
        """Return up to k alternative actions with estimated success probabilities."""
        scores = []
        for a in self.action_catalog:
            p, conf = self.predict_success(state_vector, a)
            scores.append((a, p))
        scores.sort(key=lambda x: x[1], reverse=True)
        # filter out original action
        candidates = [(a, p) for a, p in scores if a != action]
        return candidates[:k]

    def update(self, action: str, state_vector: np.ndarray, outcome: float, lr: float = 1e-3):
        """Very small online update for logistic weights using outcome (0/1)."""
        a_emb = self.embed_action(action)
        vec = np.concatenate([state_vector[: len(a_emb)], a_emb])[: len(self.W)]
        logit = float(np.dot(vec, self.W) + self.b)
        pred = 1.0 / (1.0 + np.exp(-logit))
        error = outcome - pred
        # gradient step
        self.W += lr * error * vec
        self.b += lr * error
        # clamp for stability
        self.W = np.clip(self.W, -10, 10)
        self.b = float(np.clip(self.b, -10, 10))


if __name__ == "__main__":
    am = ActionModel()
    sv = np.zeros(32)
    print(am.predict_success(sv, "action_1"))
    print(am.suggest_alternatives(sv, "action_1"))
