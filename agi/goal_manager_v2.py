"""
Goal Manager v2 — decompose and manage multiple goals
- Decompose an objective into subgoals (naive NLP heuristic)
- Manage multiple goals with priority, context and statuses
"""

from typing import List, Dict, Any, Tuple
import time
import logging
import heapq

logger = logging.getLogger(__name__)


class Goal:
    def __init__(self, description: str, priority: float = 0.5):
        self.id = f"goal_{int(time.time()*1000)}_{abs(hash(description))%1000}"
        self.description = description
        self.priority = float(priority)
        self.subgoals: List[Dict[str, Any]] = []
        self.status = "pending"
        self.created = time.time()

    def add_subgoal(self, desc: str, priority: float = 0.5):
        self.subgoals.append({"desc": desc, "priority": priority, "status": "pending"})


class GoalManagerV2:
    def __init__(self):
        # priority queue of (priority, timestamp, goal_id)
        self.goals: Dict[str, Goal] = {}
        self.pq: List[Tuple[float, float, str]] = []

    def decompose(self, description: str) -> List[str]:
        """Naive decomposition: split by commas/and; also produce sequential substeps by sentence tokens."""
        parts = []
        # split by common delimiters
        for chunk in description.split(','):
            chunk = chunk.strip()
            if ' and ' in chunk:
                parts.extend([p.strip() for p in chunk.split(' and ') if p.strip()])
            else:
                if chunk:
                    parts.append(chunk)
        # further split by sentences
        subs = []
        for p in parts:
            if '.' in p:
                subs.extend([s.strip() for s in p.split('.') if s.strip()])
            else:
                subs.append(p)
        return subs

    def add_goal(self, description: str, priority: float = 0.5) -> str:
        goal = Goal(description, priority)
        subs = self.decompose(description)
        for s in subs:
            goal.add_subgoal(s, priority)
        self.goals[goal.id] = goal
        heapq.heappush(self.pq, (-goal.priority, time.time(), goal.id))
        logger.info("Added goal %s: %s", goal.id, description)
        return goal.id

    def get_next_goal(self) -> Dict[str, Any]:
        if not self.pq:
            return None
        _, _, gid = self.pq[0]
        g = self.goals.get(gid)
        return {"id": g.id, "description": g.description, "priority": g.priority, "subgoals": g.subgoals}

    def complete_subgoal(self, goal_id: str, sub_idx: int, success: bool = True):
        goal = self.goals.get(goal_id)
        if not goal:
            return
        if 0 <= sub_idx < len(goal.subgoals):
            goal.subgoals[sub_idx]['status'] = 'success' if success else 'failed'
            # if all done, mark goal
            if all(s['status'] == 'success' for s in goal.subgoals):
                goal.status = 'complete'
                # remove from pq
                self.pq = [item for item in self.pq if item[2] != goal_id]
                heapq.heapify(self.pq)

    def repr_active_goals(self) -> List[Dict[str, Any]]:
        # return sorted active goals
        return [
            {"id": g.id, "desc": g.description, "priority": g.priority, "status": g.status}
            for g in sorted(self.goals.values(), key=lambda x: (-x.priority, x.created)) if g.status != 'complete'
        ]


if __name__ == "__main__":
    gm = GoalManagerV2()
    gid = gm.add_goal("Solve the maze, find key and open the door.", priority=0.9)
    print(gm.get_next_goal())
    print(gm.repr_active_goals())
