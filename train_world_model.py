# train_world_model.py
import argparse, numpy as np, torch
from agi.world_model import WorldModel
from agi.learner import Learner
from agi.replay_buffer import ReplayBuffer

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='tests/fixtures/transitions_small.npy')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch', type=int, default=32)
    args = p.parse_args()

    data = np.load(args.data, allow_pickle=True)
    transitions = list(data)
    # create replay
    buf = ReplayBuffer(capacity=100000)
    for t in transitions:
        buf.add(t)

    state_dim = transitions[0][0].shape[0]
    action_dim = transitions[0][1].shape[0]
    model = WorldModel(state_dim=state_dim, action_dim=action_dim)
    learner = Learner(model, buf, device='cpu')

    for ep in range(args.epochs):
        loss_acc = 0.0; n=0
        # perform a number of steps (bounded by buffer size)
        while len(buf) >= args.batch:
            l = learner.step(batch_size=args.batch)
            if l is None: break
            loss_acc += l; n += 1
            if n % 10 == 0:
                print(f'ep{ep} iter{n} loss={l:.6f}')
        print(f'epoch {ep} avg loss: {loss_acc/(n+1e-9):.6f}')
