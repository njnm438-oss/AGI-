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
import logging
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
# v10 proto-AGI modules
from .meta_reasoning_engine import MetaReasoningEngine
from .self_modification import SelfModificationFramework
from .continual_learner import ContinualLearner
from .llm_adapter import LLMAdapter
from .unified_memory import UnifiedMemory

# module logger
logger = logging.getLogger(__name__)

# Note: heavy libraries (transformers/torch) are imported lazily inside ControlledLLM

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
        # lazy-load flag for local transformers model
        self._tried_loading_local = False

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
        if not self._tried_loading_local:
            self._tried_loading_local = True
            try:
                from transformers import GPT2LMHeadModel, GPT2Tokenizer
                import torch
                try:
                    self.local_tok = GPT2Tokenizer.from_pretrained('distilgpt2')
                    self.local_model = GPT2LMHeadModel.from_pretrained('distilgpt2')
                    self.local_model.eval()
                except Exception:
                    self.local_tok = None
                    self.local_model = None
            except Exception:
                self.local_tok = None
                self.local_model = None

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
        s += 0.20
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
        # metric: how often memory dominated the final reply
        self._memory_dominance_count = 0
        
        # ===== V10 Proto-AGI Modules =====
        # Meta-Reasoning Engine: introspection on reasoning quality
        self.meta_reasoning = MetaReasoningEngine()
        # Self-Modification Framework: safe module variant creation and testing
        self.self_modifier = SelfModificationFramework()
        # Continual Learner: importance-weighted consolidation + mixed replay
        self.continual_learner = ContinualLearner(episodic_capacity=config.episodic_capacity)
        # LLM Adapter: dynamic LLM adaptation and fine-tuning
        self.llm_adapter = LLMAdapter()
        # Unified Memory: merge episodic + semantic + KG
        try:
            self.unified_memory = UnifiedMemory(self.embedding)
        except Exception as e:
            logger.warning("Failed to initialize UnifiedMemory: %s", e)
            self.unified_memory = None

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
            # Add heuristic candidate if present (kept but deprioritized for open questions)
            if h_resp:
                candidates.append((h_resp, {'source': 'heuristic'}))

            # 2) Memory-based candidate: retrieve top items and create a synthetic short answer
            mem_items = []
            try:
                mem_items = self.memory.search(q_emb, k=6) if q_emb is not None else []
            except Exception:
                mem_items = []
            memory_summary = ""
            # Only use memory if we have at least 2 relevant items
            if mem_items and len(mem_items) >= 2:
                # compute similarity to top memory item; disable memory if weak match
                try:
                    top_emb = getattr(mem_items[0], 'embedding', None)
                    similarity = 0.0
                    if top_emb is not None and q_emb is not None:
                        similarity = float(cosine(top_emb, q_emb))
                    else:
                        similarity = 0.0
                except Exception:
                    similarity = 0.0

                if similarity >= 0.4:
                    # produce concise memory summary candidate
                    facts = []
                    for m in mem_items[:4]:
                        c = m.content
                        if isinstance(c, dict) and 'text' in c:
                            facts.append(c['text'][:200])
                        else:
                            facts.append(str(c)[:200])
                    memory_summary = " | ".join(facts)
                    # Only include memory candidate if we have a couple of attended items
                    if len(mem_items) >= 2:
                        mem_text = "Based on my memories: " + memory_summary
                        candidates.append((mem_text, {'source': 'memory'}))
                else:
                    # weak similarity -> ignore memory
                    memory_summary = ""

            # 3) LLM candidate (controlled) - ENSURE LLM always present
            llm_resp = None
            try:
                gen = self.llm.generate(question=q, context=(memory_summary if memory_summary else ""), max_new_tokens=120)
                if gen and len(gen.strip()) > 0:
                    llm_resp = gen.strip()
                # else fallthrough
            except Exception:
                # If LLM generation failed, still continue with other candidates
                llm_resp = None

            # V10: Meta-Reasoning verification on LLM response
            llm_quality_score = 1.0  # default: trust LLM
            try:
                if llm_resp and self.meta_reasoning:
                    # Trace the reasoning chain: Question -> Context -> Answer
                    reasoning_steps = [
                        f"Question: {q[:100]}",
                        f"Context: {(memory_summary if memory_summary else 'N/A')[:100]}",
                        f"Answer: {llm_resp[:100]}"
                    ]
                    # Record reasoning trace
                    trace = self.meta_reasoning.trace_reasoning(reasoning_steps, conclusion=llm_resp, confidence=0.85)
                    # Verify conclusion vs premises
                    quality = self.meta_reasoning.verify_conclusion(trace)
                    llm_quality_score = quality.get('quality_score', 1.0)
                    # If quality is low, flag for later adjustment
                    if llm_quality_score < 0.5:
                        logger.debug("V10: LLM response quality low (%.2f), may use memory instead", llm_quality_score)
            except Exception as e:
                logger.debug("V10: Meta-reasoning verification failed: %s", e)
                llm_quality_score = 1.0  # fallback to trusting LLM

            # If LLM produced nothing but we have enough memory items, offer a memory candidate as a fallback
            try:
                min_items_req = getattr(self.config, 'memory_min_items', 2)
                if (not llm_resp) and mem_items and len(mem_items) >= min_items_req and not any(m.get('source') == 'memory' for _, m in candidates):
                    facts = []
                    for m in mem_items[:4]:
                        c = m.content
                        if isinstance(c, dict) and 'text' in c:
                            facts.append(c['text'][:200])
                        else:
                            facts.append(str(c)[:200])
                    memory_summary_fallback = " | ".join(facts)
                    mem_text = "Based on my memories: " + memory_summary_fallback
                    try:
                        logger.debug("Using memory fallback: %s (items=%d)", memory_summary_fallback[:200], len(mem_items))
                    except Exception:
                        pass
                    candidates.append((mem_text, {'source': 'memory'}))
            except Exception:
                pass

            # Decision: Priority LLM > heuristic > memory (with blending)
            # evaluate memory relevance
            best_mem_sim = 0.0
            try:
                if mem_items and q_emb is not None:
                    best_mem_sim = max((float(cosine(getattr(m, 'embedding', None), q_emb)) for m in mem_items if getattr(m, 'embedding', None) is not None), default=0.0)
            except Exception:
                best_mem_sim = 0.0

            mem_relevant = (len(mem_items) >= getattr(self.config, 'memory_min_items', 2)) and (best_mem_sim >= getattr(self.config, 'memory_similarity_threshold', 0.4))

            # If LLM produced a usable response, prefer it, optionally attach memory summary
            if llm_resp:
                final = llm_resp
                if getattr(self.config, 'attach_memory_to_llm', True) and mem_relevant and memory_summary:
                    # build short memory note
                    facts = []
                    for m in mem_items[:4]:
                        txt = m.content['text'] if isinstance(m.content, dict) and 'text' in m.content else str(m.content)
                        facts.append(txt[:getattr(self.config, 'memory_attach_max_chars', 160)])
                    mem_summary_short = " | ".join(facts[:3])
                    final = f"{final}  (Based on memory: {mem_summary_short})"
                # record dialogue and return
                final = final.strip()
                # record dialogue safely (avoid storing raw LLM prompts/responses that contain "Question:" etc.)
                try:
                    safe_q = q if len(q) < 500 and "Question:" not in q else q[:200]
                    emb_for_mem = q_emb if q_emb is not None else self.embedding.encode_text(safe_q)
                    self.memory.add(MemoryItem(id=hashlib.md5(safe_q.encode()).hexdigest()[:16], content={'type':'dialogue','text': safe_q}, embedding=emb_for_mem, ts=time.time(), importance=0.4))
                except Exception:
                    pass
                try:
                    self.emotion.update(reward=0.03, pred_error=0.0, success=True, novelty=0.02)
                except Exception:
                    pass
                return final

            # no LLM answer: fall back to candidates (heuristic or memory)
            if not candidates:
                return "I apologize, but I cannot provide an answer at this moment."

            # Score all candidates
            scored = []
            for txt, meta in candidates:
                try:
                    s = score_candidate(txt, q, q_emb, self.embedding, meta, self.emotion)
                except Exception:
                    s = -9e9
                # Ensure LLM candidate cannot be completely zeroed out
                if meta.get('source') in ('gpt2', 'llm'):
                    s = max(s, 1e-4)
                scored.append((s, txt, meta))
            scored.sort(key=lambda x: x[0], reverse=True)

            # If question is clearly open (personal/identity/opinion), prefer LLM over pure memory
            lowq = q.lower()
            open_q = False
            open_markers = ("who", "what", "how", "where", "when", "why", "do you", "your")
            if lowq.split()[0] in open_markers or any(phr in lowq for phr in ["how do you feel", "who are you", "what do you think", "what is your name"]):
                open_q = True

            # If top candidate is memory but an LLM candidate exists and question is open, choose LLM
            best_score, best_text, best_meta = scored[0]
            if open_q and best_meta.get('source') == 'memory':
                # find best llm candidate
                llm_candidates = [s for s in scored if s[2].get('source') in ('gpt2', 'llm')]
                if llm_candidates:
                    _, llm_text, llm_meta = llm_candidates[0]
                    # merge: prefer LLM text, but append brief memory context if present
                    if memory_summary:
                        final = f"{llm_text.strip()}\n\n(Based on memory: {memory_summary})"
                    else:
                        final = llm_text.strip()
                else:
                    final = best_text.strip()
                    self._memory_dominance_count += 1
                    try:
                        logger.debug("Memory dominated final answer (no LLM candidate). memory_summary=%s", memory_summary[:200] if memory_summary else "(none)")
                    except Exception:
                        pass
            else:
                # If LLM exists and memory also exists, blend them: prefer LLM as main answer
                if best_meta.get('source') in ('gpt2', 'llm') and memory_summary:
                    final = f"{best_text.strip()}\n\n(Based on memory: {memory_summary})"
                else:
                    final = best_text.strip()
                    if best_meta.get('source') == 'memory':
                        self._memory_dominance_count += 1
                        try:
                            logger.debug("Memory dominated final answer (best_meta==memory). memory_summary=%s", memory_summary[:200] if memory_summary else "(none)")
                        except Exception:
                            pass

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

            final = clean_output(final)

            # record dialogue safely (avoid storing raw LLM prompts/responses that contain "Question:" etc.)
            try:
                safe_q = q if len(q) < 500 and "Question:" not in q else q[:200]
                emb_for_mem = q_emb if q_emb is not None else self.embedding.encode_text(safe_q)
                self.memory.add(MemoryItem(id=hashlib.md5(safe_q.encode()).hexdigest()[:16], content={'type':'dialogue','text': safe_q}, embedding=emb_for_mem, ts=time.time(), importance=0.4))
            except Exception:
                pass

            # V10: Continual Learning & Unified Memory integration
            try:
                if self.continual_learner:
                    # Record this QA pair as an experience with importance based on meta-reasoning quality
                    experience = {
                        'question': q[:300],
                        'answer': final[:300],
                        'quality_score': llm_quality_score,
                        'memory_relevant': len(mem_items) >= 2,
                        'timestamp': time.time()
                    }
                    # Importance = combination of quality + novelty
                    importance = min(1.0, max(0.3, llm_quality_score * 0.7 + np.random.rand() * 0.3))
                    self.continual_learner.add_experience(experience, importance=importance)
                    
                    # Periodically consolidate (every 50 experiences)
                    if self.continual_learner.buffer_size % 50 == 0:
                        self.continual_learner.consolidate()
                        logger.debug("V10: Continual learning consolidation triggered")
            except Exception as e:
                logger.debug("V10: Continual learning failed: %s", e)

            # V10: Unified Memory update
            try:
                if self.unified_memory:
                    # Add to unified memory (episodic + KG fusion)
                    self.unified_memory.add_event(f"Q: {q[:100]} A: {final[:100]}", importance=importance)
            except Exception as e:
                logger.debug("V10: Unified memory update failed: %s", e)

            # update emotion lightly and modulate tone (small influence preserved)
            try:
                self.emotion.update(reward=0.03, pred_error=0.0, success=True, novelty=0.02)
            except Exception:
                pass

            return final

    def ask_chat(self, question: str, conversational: bool = True) -> str:
        """ChatGPT-like conversational mode.
        - Uses `chat_history` (short-term memory)
        - Calls LLM primarily and appends a brief memory summary
        - Updates emotion and chat history
        Returns a plain text answer.
        """
        q = (question or "").strip()
        if q == "":
            return "I did not receive any question."

        with self.lock:
            # record user message in short-term chat history
            try:
                self.chat_history.append({'role': 'user', 'content': q})
            except Exception:
                pass

            # prepare embedding and memory summary
            q_emb = None
            try:
                q_emb = self.embedding.encode_text(q)
            except Exception:
                q_emb = None

            mem_items = []
            try:
                mem_items = self.memory.search(q_emb, k=6) if q_emb is not None else []
            except Exception:
                mem_items = []

            mem_summary = ""
            if mem_items:
                parts = []
                for m in mem_items[:4]:
                    c = m.content
                    if isinstance(c, dict) and 'text' in c:
                        parts.append(c['text'][:160])
                    else:
                        parts.append(str(c)[:160])
                mem_summary = " | ".join(parts)

            # build conversational prompt including recent chat history
            recent = list(self.chat_history)[-10:]
            history_text = "\n".join([f"{h['role']}: {h['content']}" for h in recent])
            prompt = (
                f"Conversation style answer in English. Keep it conversational and concise (1-3 sentences).\n"
                f"Short memory: {mem_summary}\nConversation history:\n{history_text}\nUser: {q}\nAssistant:"
            )

            # Ask the LLM
            try:
                gen = self.llm.generate(question=q, context=prompt, max_new_tokens=160)
            except Exception:
                gen = ""

            answer = gen.strip() if gen else "I apologize, I cannot answer that right now."

            # append assistant reply to short-term history
            try:
                self.chat_history.append({'role': 'assistant', 'content': answer})
            except Exception:
                pass

            # store a compact memory of the exchange
            try:
                summary_for_mem = (answer[:300] + ("..." if len(answer) > 300 else ""))
                mid = hashlib.sha256((q + summary_for_mem).encode()).hexdigest()[:16]
                emb = self.embedding.encode_text(summary_for_mem)
                self.memory.add(MemoryItem(id=mid, content={'type': 'dialogue', 'text': summary_for_mem}, embedding=emb, ts=time.time(), importance=0.35))
            except Exception:
                pass

            # small emotion update for conversational tone
            try:
                self.emotion.update(reward=0.05, pred_error=0.0, success=True, novelty=0.03)
            except Exception:
                pass

            return answer

    def adapt_llm_v10(self, task_name: str, examples: List[Tuple[str, str]]):
        """V10: Fine-tune LLM adapter for specific task.
        Args:
            task_name: Name of the task (e.g., "math", "creative", "analytical")
            examples: List of (input, expected_output) pairs for fine-tuning
        """
        try:
            if self.llm_adapter:
                # Register task-specific prompt template
                task_template = f"You are helping with {task_name} tasks. Keep responses concise and accurate."
                self.llm_adapter.register_prompt_template(task_name, task_template)
                # Fine-tune on examples (if batch available)
                if len(examples) > 0:
                    self.llm_adapter.fine_tune_on_task(task_name, examples)
                    logger.info("V10: LLM adapted for task '%s' with %d examples", task_name, len(examples))
                    return True
        except Exception as e:
            logger.warning("V10: LLM adaptation failed: %s", e)
        return False

    def get_v10_metrics(self) -> Dict[str, Any]:
        """Collect v10 proto-AGI metrics for monitoring."""
        metrics = {
            'meta_reasoning_traces': len(self.meta_reasoning.traces) if self.meta_reasoning else 0,
            'self_modification_variants': len(self.self_modifier.history) if self.self_modifier else 0,
            'continual_learner_buffer_size': self.continual_learner.buffer_size if self.continual_learner else 0,
            'unified_memory_size': len(self.unified_memory.episodic.items) if self.unified_memory else 0,
        }
        return metrics

