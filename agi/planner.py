from .mcts import MCTS

class Planner:
    def __init__(self, world_model, policy_fn, value_fn, n_sim=100):
        self.mcts = MCTS(policy_fn, value_fn, n_sim=n_sim)
        self.world_model = world_model

    def plan(self, state_repr):
        root = self.mcts.search(state_repr)
        best_action = None; best_visits = -1
        for a,c in root.children.items():
            if c.visit_count > best_visits:
                best_visits = c.visit_count; best_action = a
        return best_action if best_action is not None else 0
