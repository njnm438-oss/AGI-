"""
Continual Learning (CL3) — AGI v10
Permanent learning without catastrophic forgetting.
- Consolidation by importance (prioritized replay)
- Mixed replay buffer strategy
- Internal world-model distillation
"""

import logging
from typing import List, Dict, Tuple, Any
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class ContinualLearner:
    """Manages continual learning with importance-weighted consolidation."""

    def __init__(self, buffer_capacity: int = 1000, consolidation_interval: int = 100):
        self.replay_buffer = deque(maxlen=buffer_capacity)
        self.importance_weights: Dict[int, float] = {}
        self.consolidated_memory: List[Dict[str, Any]] = []
        self.consolidation_interval = consolidation_interval
        self.samples_seen = 0
        self.task_buffers: Dict[str, deque] = {}  # per-task buffers

    def add_experience(self, task_id: str, experience: Dict[str, Any], importance: float = 0.5):
        """Add experience with importance weight."""
        exp_id = len(self.replay_buffer)
        self.replay_buffer.append(experience)
        self.importance_weights[exp_id] = importance

        # Add to task-specific buffer
        if task_id not in self.task_buffers:
            self.task_buffers[task_id] = deque(maxlen=100)
        self.task_buffers[task_id].append(experience)

        self.samples_seen += 1

        # Trigger consolidation if needed
        if self.samples_seen % self.consolidation_interval == 0:
            self.consolidate()

    def consolidate(self):
        """Consolidate high-importance experiences into semantic memory."""
        if not self.replay_buffer:
            return

        # Sort by importance
        sorted_exps = sorted(
            enumerate(self.replay_buffer),
            key=lambda x: self.importance_weights.get(x[0], 0.0),
            reverse=True,
        )

        # Keep top 30% as consolidated
        top_k = max(1, len(sorted_exps) // 3)
        consolidated = [{"experience": exp, "importance": self.importance_weights.get(i, 0.0)} for i, exp in sorted_exps[:top_k]]

        self.consolidated_memory = consolidated
        logger.info("Consolidated %d high-importance experiences", len(consolidated))

    def mixed_replay(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Sample from both recent and consolidated memory."""
        if len(self.replay_buffer) == 0:
            return []

        # Mix: 70% from recent buffer, 30% from consolidated
        recent_size = max(1, int(batch_size * 0.7))
        consolidated_size = batch_size - recent_size

        batch = []

        # Sample recent
        recent_indices = np.random.choice(len(self.replay_buffer), size=min(recent_size, len(self.replay_buffer)), replace=False)
        for idx in recent_indices:
            batch.append(self.replay_buffer[idx])

        # Sample consolidated
        if self.consolidated_memory:
            consol_indices = np.random.choice(
                len(self.consolidated_memory), size=min(consolidated_size, len(self.consolidated_memory)), replace=False
            )
            for idx in consol_indices:
                batch.append(self.consolidated_memory[idx]["experience"])

        return batch

    def distill_world_model(self, teacher_model, student_model, batch_size: int = 32, steps: int = 10):
        """Distill teacher world-model into student (simplified pseudo-code)."""
        logger.info("Starting world-model distillation (teacher→student)")

        for step in range(steps):
            batch = self.mixed_replay(batch_size=batch_size)
            if not batch:
                break

            # Simulate distillation update (placeholder)
            teacher_loss = np.random.rand() * 0.1
            student_loss = teacher_loss * 1.2  # student slightly worse

            if step % 5 == 0:
                logger.debug("Distillation step %d: teacher_loss=%.4f, student_loss=%.4f", step, teacher_loss, student_loss)

    def estimate_forgetting(self, task_id: str) -> float:
        """Estimate how much a specific task might be forgotten (0.0 to 1.0)."""
        if task_id not in self.task_buffers:
            return 0.0

        task_buffer = self.task_buffers[task_id]
        if not task_buffer:
            return 1.0

        # Estimate: how many high-importance experiences from this task are in consolidated memory?
        task_importance = sum(self.importance_weights.get(i, 0.0) for i, exp in enumerate(self.replay_buffer) if task_id in str(exp))
        total_importance = sum(self.importance_weights.values())

        if total_importance == 0:
            return 0.5

        forgetting_risk = 1.0 - (task_importance / total_importance)
        return float(forgetting_risk)

    def get_consolidation_report(self) -> Dict[str, Any]:
        """Get consolidation status report."""
        return {
            "samples_seen": self.samples_seen,
            "buffer_size": len(self.replay_buffer),
            "consolidated_size": len(self.consolidated_memory),
            "num_tasks": len(self.task_buffers),
            "task_forgetting_risks": {task_id: self.estimate_forgetting(task_id) for task_id in self.task_buffers},
            "avg_importance": float(np.mean(list(self.importance_weights.values())) if self.importance_weights else 0.0),
        }
