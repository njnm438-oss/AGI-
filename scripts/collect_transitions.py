"""Collect transitions (state, action, next_state) during runs and save to numpy file.
Usage: python scripts/collect_transitions.py --out tests/fixtures/transitions_small.npy --steps 200
"""
import argparse, time, numpy as np
from agi.replay_buffer import ReplayBuffer
from agi.agent_pro_en import AGIAgentProEN

# This script runs the agent in demo mode and collects simple synthetic transitions

def random_action_vec(action_dim):
    return np.random.randn(action_dim).astype('float32')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--out', type=str, default='tests/fixtures/transitions_small.npy')
    args = p.parse_args()

    agent = AGIAgentProEN()
    buffer = ReplayBuffer(capacity=10000)

    # Synthetic loop: perceive text -> action vector -> next pseudo-state
    state = np.random.randn(agent.config.state_dim).astype('float32')
    for i in range(args.steps):
        # simple perception + action
        q = f'observation {i}'
        _ = agent.perceive_text(q)
        a = random_action_vec(agent.config.action_dim if hasattr(agent.config, 'action_dim') else 8)
        next_state = state + 0.1 * a[:state.shape[0]] + 0.01 * np.random.randn(*state.shape)
        r = 0.0
        done = False
        buffer.add((state, a, r, next_state, done))
        state = next_state

    # save small fixture
    arr = list(buffer.buf)
    np.save(args.out, arr, allow_pickle=True)
    print('Saved', len(arr), 'transitions to', args.out)
