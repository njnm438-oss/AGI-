"""
AGI Agent Pro — Professional / Stable pipeline (English mode)
- Heuristic backend (template-based quick answers)
- Combined scorer (memory, heuristic, LLM)
- Controlled GPT-2 wrapper (short replies, English prompt, anti-hallucination)
Drop-in replacement for previous AGIAgent.ask() behavior.
"""

import time
import threading
import hashlib
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Any

from .config import CONFIG
from .utils import ensure_dir, cosine
from .embedding import EmbeddingModule
from .memory import EpisodicMemory, MemoryItem
from .llm_backends import make_default_llm_manager
from .perception_v2 import PerceptorV2
from .world_model import WorldModel
from .planner import Planner
from .emotion import EmotionalState
from .goal_manager import GoalManager
from .coordinator import Coordinator

# Optional imports for safety
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# ---------------------------
# Heuristic backend (fast)
# ---------------------------
def heuristic_response(question: str) -> str:
    q = question.strip().lower()
    # simple classifiers / templates
    greetings = ["hello", "hi", "hey", "greetings"]
    if any(g in q for g in greetings) and len(q.split()) <= 3:
        return "Hello! How can I help you today?"
    if "who are you" in q or "what are you" in q or "introduce yourself" in q:
        return "I am an AGI prototype — an experimental research system."
    if q.endswith('?') and len(q.split()) <= 4:
        # short factual question -> heuristic safe reply
        return "Good question — I can search my memory or use my language model to find an answer."
    # fallback: no heuristic
    return ""

# ---------------------------
# GPT-2 controlled wrapper
# ---------------------------
class ControlledLLM:
    def __init__(self, llm_manager):
        self.llm = llm_manager
        # if transformers locally available, we can use tokenizer/model directly for more control
        self.local_tok = None
        self.local_model = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.local_tok = GPT2Tokenizer.from_pretrained('distilgpt2')
                self.local_model = GPT2LMHeadModel.from_pretrained('distilgpt2')
                self.local_model.eval()
                # if GPU available, this could be moved to device
            except Exception:
                self.local_tok = None
                self.local_model = None

    def generate(self, question: str, context: str = "", max_new_tokens: int = 60) -> str:
        """
        Generate a short, controlled answer in English. Prefer local model if present,
        otherwise use manager backends (which may be llama_cpp or gpt2).
        Postprocess: keep at most 2 sentences, filter autobiographic patterns.
        """
        # build directive prompt in English to constrain generation
        prompt = (
            "Short, professional answer in English (1-2 sentences). "
            "Do not make up autobiographical experiences or personal claims.\n"
            f"Context: {context}\nQuestion: {question}\nAnswer:"
        )

        # Try local transformers for tighter control if available
        gen_text = ""
        if self.local_model is not None and self.local_tok is not None:
            try:
                toks = self.local_tok.encode(prompt, return_tensors='pt')
                max_len = min(toks.shape[1] + max_new_tokens, 512)
                with torch.no_grad():
                    out = self.local_model.generate(
                        toks,
                        max_length=max_len,
                        do_sample=True,
                        top_k=40,
                        top_p=0.9,
                        temperature=0.8,
                        eos_token_id=self.local_tok.eos_token_id,
                        pad_token_id=self.local_tok.eos_token_id,
                    )
                decoded = self.local_tok.decode(out[0], skip_special_tokens=True)
                gen_text = decoded[len(prompt):].strip()
            except Exception:
                gen_text = ""
        # fallback to manager
        if not gen_text:
            try:
                gen_text = self.llm.generate(prompt, max_tokens=max_new_tokens) or ""
            except Exception:
                gen_text = ""

        # Postprocess: normalize whitespace
        gen_text = gen_text.replace("\r", " ").replace("\n", " ").strip()

        # Keep at most 2 sentences (split by ., !, ?)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', gen_text)
        if len(sentences) > 2:
            gen_text = " ".join(sentences[:2]).strip()
        # safety filters — ban autobiographical telltales
        BAN_PATTERNS = ["i was", "when i was", "my parents", "my job", "since 20", "i've been", "i am a", "my childhood", "my experience"]
        low = gen_text.lower()
        if any(p in low for p in BAN_PATTERNS):
            # fallback safe answer
            return "I am an AI agent. How can I assist you?"
        # ensure short: if too long, truncate politely
        if len(gen_text.split()) > 60:
            gen_text = " ".join(gen_text.split()[:60]) + "..."
        return gen_text.strip()

# ---------------------------
# Scoring function
# ---------------------------
def score_candidate(text: str, question: str, q_emb: np.ndarray, emb_mod: EmbeddingModule, meta: dict, emotion: EmotionalState) -> float:
    """
    Combined scoring:
    - semantic similarity to question (embedding)
    - length penalty (prefer short)
    - source priority (heuristic > memory > llm)
    - penalize autobiographical / hallucination markers
    - reward curiosity if creative answers allowed
    """
    s = 0.0
    # similarity
    try:
        t_emb = emb_mod.encode_text(text[:400]) if emb_mod else None
        sim = cosine(t_emb, q_emb) if (t_emb is not None and q_emb is not None) else 0.0
    except Exception:
        sim = 0.0
    s += 0.5 * sim

    # length: prefer concise answers (ideal up to ~40 tokens)
    tokens = len(text.split())
    if tokens <= 40:
        s += 0.2
    else:
        s += 0.2 * max(0.0, 1.0 - (tokens - 40)/200.0)  # decreasing

    # source priorities
    source = meta.get('source', '')
    if source == 'heuristic':
        s += 0.45   # strong boost
    elif source == 'memory':
        s += 0.25
    elif source == 'gpt2' or source == 'llm':
        s += 0.15

    # emotion influence (curiosity increases tolerance for creative LLM answers)
    s += 0.1 * float(getattr(emotion, 'curiosity', 0.0))

    # penalize autobiographic or hallucinatory markers
    hallo_tokens = ["i was", "when i was", "my book", "my homework", "i've been", "my family", "my school"]
    low = text.lower()
    if any(h in low for h in hallo_tokens):
        s -= 0.9  # heavy penalty for autobiographical content

    # penalize repetitions or question-like dumps
    if text.count("Question:") > 0 or text.count("Answer:") > 0 or text.count("Response:") > 0:
        s -= 1.0  # avoid meta-question pollution

    # small random tie-breaker
    s += np.random.RandomState(int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % 1000).randn() * 1e-4
    return float(s)

# ---------------------------
# AGI Agent Pro
# ---------------------------
class AGIAgentPro:
    def __init__(self, config=CONFIG, llama_model_path: str = None):
        self.config = config
        ensure_dir(config.checkpoint_dir); ensure_dir(config.memory_dir)
        self.embedding = EmbeddingModule(dim=config.embedding_dim)
        self.perceptor = PerceptorV2(self.embedding)
        self.memory = EpisodicMemory(dim=config.embedding_dim, capacity=config.episodic_capacity, path=(config.memory_dir + "/episodic.pkl" if config.use_faiss else None))
        self.chat_history = deque(maxlen=128)
        self.emotion = EmotionalState()
        self.llm_manager = make_default_llm_manager(config, llama_model_path)
        self.llm = ControlledLLM(self.llm_manager)
        self.goal_manager = GoalManager()
        self.coordinator = Coordinator(self)
        self.lock = threading.Lock()
        # start minimal background tasks if needed
        self._running = True

    # perception convenience
    def perceive_text(self, text: str, importance: float = 0.5):
        emb = self.embedding.encode_text(text)
        mid = hashlib.sha256(text.encode()).hexdigest()[:16]
        item = MemoryItem(id=mid, content={'type':'text','text':text}, embedding=emb, ts=time.time(), importance=importance)
        self.memory.add(item)
        return emb

    # main ask pipeline (public)
    def ask(self, question: str) -> str:
        # build context
        q = (question or "").strip()
        if q == "":
            return "I did not receive any question."

        # thread-safe single call
        with self.lock:
            # 1) Heuristic quick check
            h_resp = heuristic_response(q)
            # prepare embedding for scoring
            q_emb = None
            try:
                q_emb = self.embedding.encode_text(q)
            except Exception:
                q_emb = None

            candidates: List[Tuple[str, dict]] = []
            # Add heuristic candidate if present
            if h_resp:
                candidates.append((h_resp, {'source': 'heuristic'}))

            # 2) Memory-based candidate: retrieve top items and create a synthetic short answer
            mem_items = []
            try:
                mem_items = self.memory.search(q_emb, k=6) if q_emb is not None else []
            except Exception:
                mem_items = []
            if mem_items:
                # produce concise memory summary candidate
                facts = []
                for m in mem_items[:4]:
                    c = m.content
                    if isinstance(c, dict) and 'text' in c:
                        facts.append(c['text'][:200])
                    else:
                        facts.append(str(c)[:200])
                mem_text = "Based on my memories: " + " | ".join(facts)
                candidates.append((mem_text, {'source': 'memory'}))

            # 3) LLM candidate (controlled)
            # Only call LLM if question is open-ended or no good heuristic/memory answer
            call_llm = True
            if h_resp and len(h_resp) < 80:
                # heuristic answers short questions; avoid calling LLM unnecessarily
                call_llm = False
            if call_llm:
                try:
                    gen = self.llm.generate(question=q, context=(" | ".join([str(m.content) for m in mem_items[:3]])), max_new_tokens=80)
                    if gen:
                        candidates.append((gen, {'source': 'gpt2'}))
                except Exception:
                    pass

            # If no candidates at all, default safe reply
            if not candidates:
                return "I apologize, but I cannot provide an answer at this moment."

            # Score all candidates
            scored = []
            for txt, meta in candidates:
                try:
                    s = score_candidate(txt, q, q_emb, self.embedding, meta, self.emotion)
                except Exception:
                    s = -9e9
                scored.append((s, txt, meta))
            scored.sort(key=lambda x: x[0], reverse=True)

            # Best candidate
            best_score, best_text, best_meta = scored[0]

            # Final cleaning: remove internal markers and ensure not empty
            def clean_output(txt: str) -> str:
                if not isinstance(txt, str):
                    return str(txt)
                # drop "Question:" / "Answer:" blocks and other metadata
                lines = [L.strip() for L in txt.splitlines()]
                keep = []
                for L in lines:
                    if any(p in L for p in ["Question:", "Answer:", "Response:", "Memories activated:", "Emotion snapshot:", "Keywords:"]):
                        continue
                    keep.append(L)
                out = " ".join([k for k in keep if k])
                out = out.strip()
                if out == "":
                    return "I apologize, I do not have a relevant answer."
                return out

            final = clean_output(best_text)

            # record dialogue safely (avoid storing raw LLM prompts/responses that contain "Question:" etc.)
            try:
                safe_q = q if len(q) < 500 and "Question:" not in q else q[:200]
                self.memory.add(MemoryItem(id=hashlib.md5(safe_q.encode()).hexdigest()[:16], content={'type':'dialogue','text': safe_q}, embedding=(q_emb if q_emb is not None else self.embedding.encode_text(safe_q)), ts=time.time(), importance=0.4))
            except Exception:
                pass

            # update emotion lightly
            try:
                self.emotion.update(reward=0.02, pred_error=0.0, success=True, novelty=0.01)
            except Exception:
                pass

            return final
