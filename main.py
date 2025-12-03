from agi.agent_pro import AGIAgentPro as AGIAgent
import argparse
import threading
import time
import numpy as np

from agi.replay_buffer import ReplayBuffer
from agi.learner import Learner


class TransitionCollector:
    def __init__(self, agent, replay: ReplayBuffer, interval: float = 1.0):
        self.agent = agent
        self.replay = replay
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def _make_action(self):
        dim = getattr(self.agent.config, 'action_dim', 8)
        return np.random.randn(dim).astype('float32')

    def _make_state_from_emb(self, emb):
        sd = getattr(self.agent.config, 'state_dim', 256)
        arr = np.zeros(sd, dtype='float32')
        e = np.array(emb).astype('float32')
        arr[:min(len(e), sd)] = e[:sd]
        return arr

    def _loop(self):
        count = 0
        state = np.random.randn(getattr(self.agent.config, 'state_dim', 256)).astype('float32')
        while not self._stop.is_set():
            try:
                # perceive a short observation to update agent memory
                obs = f"autonomous tick {count}"
                emb = self.agent.perceive_text(obs)
                action = self._make_action()
                next_state = state + 0.1 * action[:state.shape[0]] + 0.01 * np.random.randn(*state.shape)
                self.replay.add((state, action, 0.0, next_state, False))
                state = next_state
                count += 1
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


class BackgroundLearner:
    def __init__(self, learner: Learner, replay: ReplayBuffer, interval: float = 5.0, batch_size: int = 32):
        self.learner = learner
        self.replay = replay
        self.interval = interval
        self.batch_size = batch_size
        self._stop = threading.Event()
        self._thread = None
        self.last_loss = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                if len(self.replay) >= self.batch_size:
                    loss = self.learner.step(batch_size=self.batch_size)
                    self.last_loss = loss
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['chat','demo','autonomous','server','test'], default='demo')
    p.add_argument('--llama', type=str, default=None, help='path to ggml llama model')
    p.add_argument('--learn', action='store_true', help='enable background learning/collection')
    args = p.parse_args()

    agent = AGIAgent(llama_model_path=args.llama)

    # prepare shared replay + learner if requested
    replay = ReplayBuffer(capacity=50000) if args.learn else None
    learner = Learner(agent.world_model, replay, device=getattr(agent.config, 'device', 'cpu')) if args.learn else None
    collector = TransitionCollector(agent, replay) if args.learn else None
    bg_learner = BackgroundLearner(learner, replay) if args.learn else None

    try:
        if args.mode == 'chat':
            print('Starting chat; type exit to quit')
            while True:
                try:
                    q = input('You: ')
                except (EOFError, KeyboardInterrupt):
                    break
                if q.strip().lower() in ('exit','quit'):
                    break
                print('AGI:', agent.ask(q))

        elif args.mode == 'autonomous':
            # start collector and learner if enabled
            if args.learn and collector and bg_learner:
                collector.start(); bg_learner.start();
            print('Running autonomous loop (press Ctrl+C to stop)')
            i = 0
            try:
                while True:
                    emb = agent.perceive_text(f'observation {i}')
                    print('perceived:', i)
                    time.sleep(1.0)
                    i += 1
            except KeyboardInterrupt:
                pass

        elif args.mode == 'server':
            # run Flask app (app.py will create its own agent + background worker)
            from app import app as flask_app
            # run flask directly
            flask_app.run(host='127.0.0.1', port=5000, debug=True)

        else:
            agent.perceive_text('Hello, I am AGI Lab prototype', importance=0.7)
            print(agent.ask('Introduce yourself in one sentence'))
    finally:
        try:
            if args.learn and collector:
                collector.stop()
            if args.learn and bg_learner:
                bg_learner.stop()
        except Exception:
            pass
        try:
            agent.shutdown()
        except Exception:
            pass
