import numpy as np
from dataclasses import dataclass

@dataclass
class EmotionalState:
    joy: float = 0.5
    curiosity: float = 0.6
    surprise: float = 0.5
    frustration: float = 0.0
    fear: float = 0.0
    confidence: float = 0.5
    satisfaction: float = 0.5
    determination: float = 0.5
    confusion: float = 0.0
    focus: float = 0.7
    energy: float = 1.0

    def update(self, reward: float = 0.0, pred_error: float = 0.0, success: bool = True, novelty: float = 0.0):
        d = 0.96
        self.joy = np.clip(self.joy * d + 0.12 * reward, 0, 1)
        self.curiosity = np.clip(self.curiosity * d + 0.15 * novelty, 0, 1)
        self.surprise = np.clip(min(1.0, abs(pred_error)), 0, 1)
        self.frustration = np.clip(self.frustration * d + 0.1 * (1 - float(success)), 0, 1)
        self.fear = np.clip(self.fear * 0.98 + 0.05 * max(0, -reward), 0, 1)
        self.confidence = np.clip(self.confidence * 0.95 + 0.05 * float(success), 0, 1)
        self.satisfaction = np.clip(0.7 * self.joy + 0.3 * self.confidence, 0, 1)
        self.determination = np.clip(self.confidence - 0.3 * self.frustration, 0, 1)
        self.confusion = np.clip(self.surprise + 0.5 * self.frustration, 0, 1)
        self.focus = np.clip(self.determination - 0.3 * self.confusion, 0.2, 1)
        self.energy = np.clip(self.energy * 0.995 + 0.001, 0, 1)

    def to_vector(self):
        return [self.joy, self.curiosity, self.surprise, self.frustration, self.fear, self.confidence, self.satisfaction, self.determination, self.confusion, self.focus, self.energy]
