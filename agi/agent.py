import time, threading, hashlib, json
import numpy as np
from collections import deque
from .config import CONFIG
from .utils import ensure_dir
from .embedding import EmbeddingModule
from .memory import EpisodicMemory, MemoryItem
from .llm_backends import make_default_llm_manager
from .world_model import WorldModel
from .planner import Planner
from .perception_v2 import PerceptorV2
from .emotion import EmotionalState
from .coordinator import Coordinator
from .goal_manager import GoalManager

class AGIAgent:
    def __init__(self, config=CONFIG, llama_model_path: str = None):
        self.config = config
        ensure_dir(config.checkpoint_dir); ensure_dir(config.memory_dir)
        self.embedding = EmbeddingModule(dim=config.embedding_dim)
        self.perceptor = PerceptorV2(self.embedding)
        self.memory = EpisodicMemory(dim=config.embedding_dim, capacity=config.episodic_capacity, path=(config.memory_dir+'/episodic.pkl' if config.use_faiss else None))
        self.autobio = []
        self.chat_history = deque(maxlen=64)
        self.emotion = EmotionalState()
        self.llm = make_default_llm_manager(config, llama_model_path)
        self.world_model = WorldModel(input_dim=config.state_dim, hidden=config.hidden_dim, n_layers=3)
        self.planner = Planner(self.world_model, self.policy_fn, self.value_fn, n_sim=80)
        self.goal_manager = GoalManager()
        self.coordinator = Coordinator(self)
        self._running = True
        self._start_background()
        # optional self_model
        self.self_model = {'profile': {'name':'AGI_Lab_V4'}}

    def _start_background(self):
        def loop():
            while self._running:
                try:
                    time.sleep(self.config.consolidation_interval)
                except Exception:
                    time.sleep(1.0)
        t = threading.Thread(target=loop, daemon=True); t.start(); self._bg_thread = t

    def shutdown(self):
        self._running = False
        try: self._bg_thread.join(timeout=1.0)
        except Exception: pass

    def perceive_text(self, text: str, importance: float = 0.5):
        emb = self.embedding.encode_text(text)
        mid = hashlib.sha256(text.encode()).hexdigest()[:16]
        item = MemoryItem(id=mid, content={'type':'text','text':text}, embedding=emb, ts=time.time(), importance=importance)
        self.memory.add(item)
        return emb

    def perceive_image(self, pil_image, importance: float = 0.5):
        emb = self.perceptor.perceive_image(pil_image)
        mid = hashlib.sha256(pil_image.tobytes()).hexdigest()[:16]
        item = MemoryItem(id=mid, content={'type':'image'}, embedding=emb, ts=time.time(), importance=importance)
        self.memory.add(item)
        return emb

    def policy_fn(self, state):
        actions = list(range(self.config.action_dim))
        priors = [1.0/len(actions)]*len(actions)
        return actions, priors

    def value_fn(self, state):
        return 0.5*getattr(self.emotion,'curiosity',0.0) + 0.5*getattr(self.emotion,'confidence',0.0)

    def ask(self, question: str):
        st = self.coordinator.build_context(question)
        return self.coordinator.decide_response(st)

    def autonomous_step(self, observation_emb: np.ndarray):
        state = np.zeros(self.config.state_dim, dtype=np.float32)
        state[:min(len(observation_emb), self.config.state_dim)] = observation_emb[:self.config.state_dim]
        action = self.planner.plan(state)
        reward = 0.0
        try:
            self.emotion.update(reward=reward, novelty=0.01, success=True)
        except Exception:
            pass
        return action

    def _clean_output(self, txt: str) -> str:
        """Remove self-reflections, repeated question blocks, internal traces."""
        bad_patterns = [
            "Question:", 
            "Réponse:", 
            "Q:", 
            "Memories activated:", 
            "Emotion snapshot:", 
            "Keywords:",
            "Objectif actif:",
            "CoT:",
            "Answer:"
        ]
        lines = txt.split("\n")
        cleaned = []
        for line in lines:
            if not any(p in line for p in bad_patterns):
                cleaned.append(line.strip())
        out = " ".join([c for c in cleaned if c])
        return out.strip() if out else txt.strip()

    def generate(self, prompt: str, max_tokens: int = 256):
        raw = self.llm.generate(prompt, max_tokens=max_tokens)
        return self._clean_output(raw)

    def introspect(self):
        vec = None
        try:
            vec = getattr(self.emotion,'to_vector',lambda: None)()
        except Exception:
            vec = None
        return {'emotion': vec, 'mem_count': len(self.memory.items), 'active_goal': self.goal_manager.active()}
