"""Collect transitions (state, action, next_state) during runs and save to numpy file.
Usage: python scripts/collect_transitions.py --out data/transitions.npy --episodes 10
"""
import argparse
import numpy as np
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='data/transitions.npy')
    p.add_argument('--episodes', type=int, default=10)
    args = p.parse_args()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    transitions = []
    # Dummy collector: in real system, hook into agent environment
    for ep in range(args.episodes):
        # generate random transitions as placeholder
        for t in range(5):
            s = np.random.randn(8)
            a = np.random.randn(2)
            s2 = s + 0.1 * np.random.randn(8)
            transitions.append((s, a, 0.0, s2, False))

    np.save(str(outp), transitions)
    print(f"Saved {len(transitions)} transitions to {outp}")

if __name__ == '__main__':
    main()
