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

    # agi/planner_mcts.py
    import math, random
    from collections import defaultdict

    class MCTSNode:
        def __init__(self, state_repr, parent=None, prior=1.0):
            self.state = state_repr
            self.parent = parent
            self.children = {}
            self.visit_count = 0
            self.value_sum = 0.0
            self.prior = prior

        def value(self):
            return 0.0 if self.visit_count == 0 else (self.value_sum / self.visit_count)

    class MCTSPlanner:
        def __init__(self, simulator, action_space, n_sim=50, c_puct=1.0):
            self.simulator = simulator
            self.action_space = action_space  # list or array of discrete action vectors
            self.n_sim = n_sim
            self.c_puct = c_puct

        def search(self, root_state, rollout_policy, value_fn):
            root = MCTSNode(root_state)
            for _ in range(self.n_sim):
                node = root
                path = [node]
                # selection
                while node.children:
                    best = None; best_score = -1e9
                    for a, child in node.children.items():
                        u = self.c_puct * child.prior * math.sqrt(node.visit_count + 1) / (1 + child.visit_count)
                        score = child.value() + u
                        if score > best_score:
                            best_score = score; best = (a, child)
                    if best is None:
                        break
                    node = best[1]
                    path.append(node)
                # expand
                actions, priors = rollout_policy(node.state)
                for a, p in zip(actions, priors):
                    if a not in node.children:
                        node.children[a] = MCTSNode(state_repr=a, parent=node, prior=p)
                # simulate/evaluate
                v = value_fn(node.state)
                # backprop
                for n in reversed(path):
                    n.visit_count += 1
                    n.value_sum += v
            # choose best action by visit count
            best_act = None; best_visits = -1
            for a,c in root.children.items():
                if c.visit_count > best_visits:
                    best_visits = c.visit_count; best_act = a
            return best_act
