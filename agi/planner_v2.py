"""
Planner v2 (PL2) — world-model–guided MCTS prototype
- Uses `WorldModelV2` to simulate rollouts
- Multi-step rollout, scoring with estimated reward, novelty, confidence, cognitive cost
- Returns best action sequence (plan)
"""

from typing import List, Tuple, Any, Dict
import numpy as np
import hashlib
import logging

logger = logging.getLogger(__name__)


class SimpleMCTSNode:
    def __init__(self, latent, parent=None, action=None):
        self.latent = latent
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.action = action

    def add_child(self, node):
        self.children.append(node)


class PlannerV2:
    def __init__(self, world_model, action_model=None, rng_seed: int = 1):
        self.wm = world_model
        self.am = action_model
        self.rng = np.random.RandomState(rng_seed)

    def score_rollout(self, rollout: List[dict], novelty_weight: float = 0.3, confidence_weight: float = 0.2, cost_weight: float = 0.1) -> float:
        """Score a rollout: sum reward + novelty bonus - cost, scaled by confidence."""
        if not rollout:
            return -1e6
        total_reward = sum(s.get('reward', 0.0) for s in rollout)
        avg_conf = float(np.mean([s.get('confidence', 0.0) for s in rollout]))
        # novelty proxy: variance of states
        states = np.stack([s.get('state') for s in rollout if s.get('state') is not None])
        novelty = float(np.mean(np.var(states, axis=0))) if states.size else 0.0
        cost = len(rollout) * cost_weight
        score = total_reward + novelty_weight * novelty + confidence_weight * avg_conf - cost
        return float(score)

    def plan(self, latent, available_actions: List[str], horizon: int = 5, simulations: int = 50) -> List[str]:
        """Return best action sequence of length up to `horizon`.
        Simple MCTS-style: sample action sequences, simulate with world model, score, keep best.
        """
        best_score = -1e9
        best_seq = []
        for sim in range(simulations):
            seq = []
            action_embs = []
            cur = latent.copy() if latent is not None else self.wm.random_init_state()
            # build random or guided sequence
            for t in range(horizon):
                if self.am:
                    # prefer actions with higher predicted success
                    probs = [(a, self.am.predict_success(cur, a)[0]) for a in available_actions]
                    probs.sort(key=lambda x: x[1], reverse=True)
                    a = probs[self.rng.randint(min(3, len(probs)))][0]
                else:
                    a = self.rng.choice(available_actions)
                seq.append(a)
                # build action embedding vector
                a_emb = self.am.embed_action(a) if self.am else self.rng.randn(self.wm.state_dim)
                action_embs.append(a_emb)
            rollout = self.wm.rollout(cur, action_embs, max_steps=horizon)
            sc = self.score_rollout(rollout)
            if sc > best_score:
                best_score = sc
                best_seq = seq
        logger.debug("Planner selected seq (score=%.3f): %s", best_score, best_seq)
        return best_seq

    def plan_with_costs(self, latent, available_actions: List[str], horizon: int = 5, simulations: int = 50) -> Dict[str, Any]:
        seq = self.plan(latent, available_actions, horizon=horizon, simulations=simulations)
        return {"plan": seq, "estimated_value": None}


if __name__ == "__main__":
    from agi.world_model_v2 import WorldModelV2
    from agi.action_model import ActionModel
    wm = WorldModelV2()
    am = ActionModel()
    p = PlannerV2(wm, am)
    latent = wm.random_init_state()
    plan = p.plan(latent, [f"action_{i}" for i in range(6)], horizon=3, simulations=20)
    print(plan)
