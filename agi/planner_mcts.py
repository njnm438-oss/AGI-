import math
import random
from collections import defaultdict

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0

class MCTSPlanner:
    def __init__(self, simulator, action_space, rollout_depth=5, cpuct=1.0):
        self.simulator = simulator
        self.action_space = action_space  # list of possible action tensors
        self.rollout_depth = rollout_depth
        self.cpuct = cpuct

    def uct_score(self, parent, child):
        if child.visits == 0:
            return float('inf')
        return child.value / child.visits + self.cpuct * math.sqrt(math.log(parent.visits + 1) / child.visits)

    def search(self, root_state, n_simulations=50):
        root = MCTSNode(root_state)
        for _ in range(n_simulations):
            node = root
            # selection
            while node.children:
                node = max(node.children.values(), key=lambda c: self.uct_score(node, c))
            # expansion
            if node.visits == 0:
                value = self.rollout(node.state)
            else:
                # expand
                for a in self.action_space:
                    next_states = self.simulator.simulate(node.state, [a])
                    child_state = next_states[-1]
                    node.children[str(a)] = MCTSNode(child_state, parent=node)
                # after expansion, do a rollout from one child
                child = next(iter(node.children.values()))
                value = self.rollout(child.state)
            # backpropagate
            self.backprop(node, value)
        # pick best action by visit count
        if not root.children:
            return None
        best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return best_action

    def rollout(self, state):
        # simple heuristic: simulate random actions for rollout_depth and estimate value  (to be improved)
        actions = [random.choice(self.action_space) for _ in range(self.rollout_depth)]
        states = self.simulator.simulate(state, actions)
        # value: placeholder 0.0
        return 0.0

    def backprop(self, node, value):
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
