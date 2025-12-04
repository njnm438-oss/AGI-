"""
Continual Learning (CL3) prototype
- Prioritized replay buffer
- Consolidation stub (compress important experiences to KG / semantic memory)
- Distillation stub for world-model
"""

from typing import List, Dict, Any, Tuple
import heapq
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PrioritizedReplay:
    """Simple prioritized replay using heap for highest-priority sampling."""
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.heap = []  # min-heap of (-priority, ts, idx, item)
        self.counter = 0

    def add(self, item: dict, priority: float = 0.5):
        ts = time.time()
        entry = (-float(priority), ts, self.counter, item)
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, entry)
        else:
            # if new priority higher than smallest, replace
            if entry > self.heap[0]:
                heapq.heapreplace(self.heap, entry)
        self.counter += 1

    def sample(self, k: int = 32) -> List[dict]:
        # sample top-k highest priority items
        top = sorted(self.heap, reverse=True)[:k]
        return [e[3] for e in top]

    def __len__(self):
        return len(self.heap)


class ContinualLearner:
    def __init__(self, replay_capacity: int = 1000):
        self.replay = PrioritizedReplay(replay_capacity)

    def observe(self, experience: dict, priority: float = 0.5):
        self.replay.add(experience, priority)

    def consolidate(self, knowledge_sink):
        """Consolidate important experiences into semantic memory (knowledge_sink expected to have `add_relation` or similar).
        This is a stub converting highest-priority experiences into simple SVO facts.
        """
        top = self.replay.sample(k=50)
        for e in top:
            # expect e to have 'text' or 'event'
            text = e.get('text') or e.get('event') or str(e)
            # naive SVO: split first three words
            words = text.split()
            if len(words) >= 3 and hasattr(knowledge_sink, 'add_relation'):
                subj = words[0]
                pred = words[1]
                obj = ' '.join(words[2:4])
                try:
                    knowledge_sink.add_relation(subj, pred, obj, weight=0.6, evidence=text[:200])
                except Exception:
                    logger.debug('Failed to add relation during consolidation')
        logger.info('Consolidation complete: %d items processed', len(top))

    def distill_world_model(self, teacher_model, student_model):
        """Distill teacher world-model into student (stub): sample rollouts and fit student to mimic teacher outputs.
        Returns number of distilled samples.
        """
        # Prototype: sample up to 50 experiences and call student's ``fit_on_example`` if available
        examples = self.replay.sample(k=50)
        count = 0
        for ex in examples:
            state = ex.get('state')
            action = ex.get('action')
            if state is None or action is None:
                continue
            # ask teacher for prediction
            try:
                pred = teacher_model.predict_next(state)
                if hasattr(student_model, 'fit_on_example'):
                    student_model.fit_on_example(state, action, pred)
                    count += 1
            except Exception:
                continue
        logger.info('Distilled %d examples into student model', count)
        return count


if __name__ == '__main__':
    cl = ContinualLearner()
    for i in range(20):
        cl.observe({'text': f'agent did action_{i} result ok', 'state': None}, priority=0.5 + i*0.01)
    print('replay len', len(cl.replay))
