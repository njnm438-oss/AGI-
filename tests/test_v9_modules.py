"""
Unit tests for AGI v9 modules: WorldModelV2, ActionModel, PlannerV2, GoalManagerV2
"""

import pytest
import numpy as np
from agi.world_model_v2 import WorldModelV2
from agi.action_model import ActionModel
from agi.planner_v2 import PlannerV2
from agi.goal_manager_v2 import GoalManagerV2


def test_world_model_predict_and_rollout():
    wm = WorldModelV2(latent_dim=64, state_dim=32, seed=1)
    latent = wm.random_init_state()
    next_state, reward, done, new_latent, conf = wm.predict_next(latent)
    assert isinstance(next_state, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(new_latent, np.ndarray)
    assert 0.0 <= conf <= 1.0

    # test rollout with random actions
    action_embs = [np.random.randn(wm.state_dim) for _ in range(3)]
    trace = wm.rollout(latent, action_embs, max_steps=5)
    assert isinstance(trace, list)
    if trace:
        assert all('state' in s and 'reward' in s and 'confidence' in s for s in trace)


def test_action_model_predict_and_alternatives():
    am = ActionModel(rng_seed=2, action_dim=32)
    sv = np.zeros(32)
    p, conf = am.predict_success(sv, 'action_1')
    assert 0.0 <= p <= 1.0
    assert 0.0 <= conf <= 1.0

    alts = am.suggest_alternatives(sv, 'action_1', k=3)
    assert isinstance(alts, list)
    assert len(alts) <= 3

    # online update shouldn't crash
    am.update('action_1', sv, outcome=1.0, lr=0.01)


def test_planner_v2_plan_basic():
    wm = WorldModelV2(latent_dim=48, state_dim=24, seed=3)
    am = ActionModel(rng_seed=3, action_dim=24)
    planner = PlannerV2(wm, am, rng_seed=4)

    latent = wm.random_init_state()
    actions = [f'action_{i}' for i in range(6)]
    plan = planner.plan(latent, actions, horizon=3, simulations=10)
    assert isinstance(plan, list)
    assert len(plan) <= 3


def test_goal_manager_decompose_and_progress():
    gm = GoalManagerV2()
    gid = gm.add_goal('Find the key, open the door and exit the room.', priority=0.8)
    assert gid in gm.goals
    nextg = gm.get_next_goal()
    assert nextg is not None
    # complete first subgoal
    gm.complete_subgoal(gid, 0, success=True)
    # after completing all subgoals the goal should eventually mark complete (simulate)
    for i in range(len(gm.goals[gid].subgoals)):
        gm.complete_subgoal(gid, i, success=True)
    assert gm.goals[gid].status == 'complete'


if __name__ == '__main__':
    pytest.main([__file__])
