import math

class MCTSNode:
    def __init__(self, state_repr, parent=None, prior=1.0):
        self.state = state_repr
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

    def value(self):
        if self.visit_count==0: return 0.0
        return self.value_sum/self.visit_count

class MCTS:
    def __init__(self, policy_fn, value_fn, c_puct=1.0, n_sim=50):
        self.policy_fn = policy_fn
        self.value_fn = value_fn
        self.c_puct = c_puct
        self.n_sim = n_sim

    def search(self, root_state):
        root = MCTSNode(root_state)
        for _ in range(self.n_sim):
            node = root; path=[node]
            # selection
            while node.children:
                best=None; best_score=-1e9
                for a,child in node.children.items():
                    u = self.c_puct * child.prior * math.sqrt(node.visit_count + 1)/(1+child.visit_count)
                    score = child.value() + u
                    if score>best_score:
                        best_score=score; best=child
                node = best; path.append(node)
            # expansion
            actions, priors = self.policy_fn(node.state)
            for a,p in zip(actions, priors):
                if a not in node.children:
                    node.children[a] = MCTSNode(a, parent=node, prior=p)
            v = self.value_fn(node.state)
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += v
        return root
