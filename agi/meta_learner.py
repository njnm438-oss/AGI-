"""
Meta-Learner (M1) — AGI v8 Self-Tuning
Analyzes world-model errors and adjusts hyperparameters automatically.
- Learns learning rate, batch size, exploration temperature
- Analyzes performance trends
- Selects exploration actions dynamically
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterState:
    """Current hyperparameter configuration."""
    learning_rate: float = 0.001
    batch_size: int = 32
    exploration_temperature: float = 0.1
    regularization: float = 0.01
    momentum: float = 0.9
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "exploration_temperature": self.exploration_temperature,
            "regularization": self.regularization,
            "momentum": self.momentum,
        }
    
    def copy(self) -> "HyperparameterState":
        return HyperparameterState(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            exploration_temperature=self.exploration_temperature,
            regularization=self.regularization,
            momentum=self.momentum,
        )


@dataclass
class PerformanceMetrics:
    """Performance snapshot."""
    timestamp: float
    loss: float
    accuracy: float = 0.0
    exploration_reward: float = 0.0
    memory_usage: float = 0.0
    catastrophic_forgetting_ratio: float = 0.0  # loss increase on old tasks


class MetaLearner:
    """
    Analyzes world-model training dynamics and adjusts hyperparameters.
    Tracks: loss curves, learning efficiency, exploration effectiveness.
    """
    
    def __init__(self):
        self.hyperparams = HyperparameterState()
        self.performance_history: deque = deque(maxlen=100)  # Last 100 steps
        self.adjustment_history: List[Dict[str, Any]] = []
        
        # Meta-learning state
        self.loss_trend: float = 0.0  # decreasing = good
        self.learning_efficiency: float = 1.0
        self.exploration_effectiveness: float = 0.5
        
    def record_performance(self, loss: float, accuracy: float = 0.0, 
                          exploration_reward: float = 0.0,
                          catastrophic_forgetting: float = 0.0):
        """Record training metrics."""
        import time
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            loss=loss,
            accuracy=accuracy,
            exploration_reward=exploration_reward,
            catastrophic_forgetting_ratio=catastrophic_forgetting
        )
        self.performance_history.append(metrics)
        
        # Update trend
        if len(self.performance_history) >= 2:
            recent_losses = [m.loss for m in list(self.performance_history)[-10:]]
            self.loss_trend = (recent_losses[-1] - recent_losses[0]) / (recent_losses[0] + 1e-6)
    
    def suggest_adjustment(self) -> Tuple[HyperparameterState, str]:
        """Analyze performance and suggest hyperparameter adjustments."""
        if len(self.performance_history) < 5:
            return self.hyperparams.copy(), "Not enough data for adjustment"
        
        suggested = self.hyperparams.copy()
        reasons = []
        
        # Metric 1: Loss trend
        if self.loss_trend > 0.1:  # Loss increasing → problem
            reasons.append("loss_increasing")
            # Try reducing learning rate to stabilize
            suggested.learning_rate *= 0.9
            reasons.append("↓ learning_rate (loss instability)")
        elif self.loss_trend < -0.05:  # Loss decreasing well
            # Can afford to increase learning rate slightly
            suggested.learning_rate = min(0.01, suggested.learning_rate * 1.05)
            reasons.append("↑ learning_rate (stable decrease)")
        
        # Metric 2: Exploration effectiveness
        recent_metrics = list(self.performance_history)[-10:]
        avg_explore_reward = np.mean([m.exploration_reward for m in recent_metrics])
        
        if avg_explore_reward > 0.5:
            # Exploration is productive → increase temperature
            suggested.exploration_temperature = min(1.0, suggested.exploration_temperature * 1.1)
            reasons.append("↑ exploration_temperature (productive)")
        elif avg_explore_reward < 0.1:
            # Exploration not helping → decrease temperature (exploit more)
            suggested.exploration_temperature = max(0.01, suggested.exploration_temperature * 0.8)
            reasons.append("↓ exploration_temperature (low reward)")
        
        # Metric 3: Catastrophic forgetting
        avg_forgetting = np.mean([m.catastrophic_forgetting_ratio for m in recent_metrics])
        if avg_forgetting > 0.2:
            # Forgetting too much → increase regularization
            suggested.regularization = min(0.1, suggested.regularization * 1.2)
            reasons.append("↑ regularization (high forgetting)")
        
        # Metric 4: Batch size adaptation
        if len(self.performance_history) > 20:
            # If loss is very noisy, increase batch size for stability
            loss_std = np.std([m.loss for m in recent_metrics])
            if loss_std > 0.5:
                suggested.batch_size = min(256, int(suggested.batch_size * 1.2))
                reasons.append("↑ batch_size (noisy loss)")
        
        reason_str = " | ".join(reasons) if reasons else "stable"
        logger.info("Meta-learner adjustment: %s", reason_str)
        
        return suggested, reason_str
    
    def apply_adjustment(self, new_hyperparams: HyperparameterState):
        """Apply new hyperparameters and record adjustment."""
        adjustment_record = {
            "old_hyperparams": self.hyperparams.to_dict(),
            "new_hyperparams": new_hyperparams.to_dict(),
            "reason": self.adjustment_history[-1] if self.adjustment_history else "initial",
        }
        self.adjustment_history.append(adjustment_record)
        self.hyperparams = new_hyperparams
    
    def select_exploration_action(self, available_actions: List[str]) -> str:
        """
        Dynamically select exploration action based on effectiveness.
        Returns: action to explore.
        """
        if not available_actions:
            return "random"
        
        # Simple policy: if exploration is productive, pick random; else exploit
        recent_metrics = list(self.performance_history)[-5:] if self.performance_history else []
        
        if recent_metrics:
            avg_reward = np.mean([m.exploration_reward for m in recent_metrics])
            if avg_reward > 0.3:
                # Exploration paying off → continue diverse exploration
                return np.random.choice(available_actions)
            else:
                # Not paying off → shift to exploitation (first action usually safest)
                return available_actions[0]
        
        # Default: random exploration
        return np.random.choice(available_actions)
    
    def should_consolidate_memory(self) -> bool:
        """
        Decide if it's time to consolidate episodic to semantic memory.
        Triggers: high catastrophic forgetting, large performance gap.
        """
        if len(self.performance_history) < 10:
            return False
        
        recent = list(self.performance_history)[-10:]
        avg_forgetting = np.mean([m.catastrophic_forgetting_ratio for m in recent])
        
        return avg_forgetting > 0.15  # If forgetting > 15%, consolidate
    
    def estimate_sample_efficiency(self) -> float:
        """
        Estimate how efficiently the model learns (loss/samples processed).
        Returns: 0.0 to 1.0 (higher = more efficient).
        """
        if len(self.performance_history) < 2:
            return 0.5
        
        first_loss = self.performance_history[0].loss
        last_loss = self.performance_history[-1].loss
        
        # Efficiency = 1 - (last_loss / first_loss), clamped
        efficiency = max(0.0, min(1.0, 1.0 - (last_loss / (first_loss + 1e-6))))
        
        return efficiency
    
    def get_summary(self) -> Dict[str, Any]:
        """Get meta-learner state summary."""
        return {
            "current_hyperparams": self.hyperparams.to_dict(),
            "loss_trend": float(self.loss_trend),
            "exploration_effectiveness": float(self.exploration_effectiveness),
            "num_adjustments": len(self.adjustment_history),
            "sample_efficiency": float(self.estimate_sample_efficiency()),
            "performance_history_size": len(self.performance_history),
        }
    
    def __repr__(self) -> str:
        return f"MetaLearner(lr={self.hyperparams.learning_rate:.4f}, bs={self.hyperparams.batch_size}, temp={self.hyperparams.exploration_temperature:.3f})"
