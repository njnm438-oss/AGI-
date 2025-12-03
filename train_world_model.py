"""Training script for world model (smoke run)
Usage: python train_world_model.py --data data/transitions.npy --epochs 1 --batch-size 32
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from agi.world_model_nn import WorldModel

class TransitionDataset(Dataset):
    def __init__(self, transitions):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        s, a, r, s2, done = self.transitions[idx]
        return s.astype('float32'), a.astype('float32'), s2.astype('float32')

def collate_fn(batch):
    s = np.stack([b[0] for b in batch])
    a = np.stack([b[1] for b in batch])
    s2 = np.stack([b[2] for b in batch])
    return torch.from_numpy(s), torch.from_numpy(a), torch.from_numpy(s2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='data/transitions.npy')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', type=str, default='cpu')
    args = p.parse_args()

    arr = np.load(args.data, allow_pickle=True)
    transitions = list(arr)
    if len(transitions) == 0:
        print('No transitions found. Run scripts/collect_transitions.py first')
        return

    s0 = transitions[0][0]
    a0 = transitions[0][1]
    state_dim = s0.shape[0]
    action_dim = a0.shape[0]

    dataset = TransitionDataset(transitions)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = WorldModel(state_dim, action_dim, hidden=128).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        total = 0.0
        n = 0
        for s, a, s2 in loader:
            s = s.to(args.device)
            a = a.to(args.device)
            s2 = s2.to(args.device)
            pred = model(s, a)
            loss = torch.nn.functional.mse_loss(pred, s2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        print(f"Epoch {epoch}: avg loss {total/(n if n else 1):.6f}")

    torch.save(model.state_dict(), 'models/world_model_final.pth')
    print('Model saved to models/world_model_final.pth')

if __name__ == '__main__':
    main()
