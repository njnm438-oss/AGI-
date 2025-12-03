import threading, time, hashlib

class GoalManager:
    def __init__(self):
        self.goals = []
        self.lock = threading.RLock()

    def add_goal(self, desc, priority=0.5):
        with self.lock:
            gid = hashlib.md5((desc+str(time.time())).encode()).hexdigest()[:8]
            g = {'id':gid,'desc':desc,'priority':priority,'status':'pending','created':time.time()}
            self.goals.append(g)
            self.goals.sort(key=lambda x: x['priority'], reverse=True)
            return g

    def active(self):
        with self.lock:
            for g in self.goals:
                if g['status']=='pending': return g
            return None

    def complete(self, gid):
        with self.lock:
            for g in self.goals:
                if g['id']==gid:
                    g['status']='completed'; return True
            return False
