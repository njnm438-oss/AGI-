import time, threading, pickle, os
from dataclasses import dataclass, field
from typing import Any, List, Optional
import numpy as np
from .utils import ensure_dir
try:
    import faiss
    FAISS = True
except Exception:
    FAISS = False

@dataclass
class MemoryItem:
    id: str
    content: Any
    embedding: np.ndarray
    ts: float = field(default_factory=time.time)
    importance: float = 0.5

class EpisodicMemory:
    def __init__(self, dim=512, capacity=200000, path:Optional[str]=None):
        self.dim = dim; self.capacity = capacity; self.lock = threading.RLock()
        self.items: List[MemoryItem] = []
        self.index = None; self.matrix = None; self.path = path
        self.use_faiss = FAISS and path is not None
        if self.use_faiss:
            try:
                self.index = faiss.IndexFlatIP(dim)
                self.matrix = np.zeros((0,dim), dtype=np.float32)
            except Exception as e:
                self.use_faiss = False

    def add(self, item: MemoryItem):
        with self.lock:
            self.items.append(item)
            if len(self.items) > self.capacity:
                self.items.sort(key=lambda x: x.importance)
                self.items = self.items[-self.capacity:]
            if self.use_faiss and self.index is not None:
                v = item.embedding.reshape(1,-1).astype('float32')
                if self.matrix.size==0:
                    self.matrix = v; self.index.add(v)
                else:
                    self.matrix = np.vstack([self.matrix, v]); self.index.add(v)

    def search(self, q_emb: np.ndarray, k=10):
        with self.lock:
            if self.use_faiss and self.index is not None and self.index.ntotal>0:
                q = q_emb.reshape(1,-1).astype('float32')
                D,I = self.index.search(q, min(k, self.index.ntotal))
                return [self.items[i] for i in I[0] if 0<=i<len(self.items)]
            sims = [(m, float(np.dot(m.embedding, q_emb)/(np.linalg.norm(m.embedding)*(np.linalg.norm(q_emb)+1e-9)))) for m in self.items]
            sims.sort(key=lambda x: x[1], reverse=True)
            return [m for m,_ in sims[:k]]

    def save(self, path:str):
        ensure_dir(os.path.dirname(path))
        with open(path,'wb') as f: pickle.dump(self.items, f)

    def load(self, path:str):
        with open(path,'rb') as f: self.items = pickle.load(f)
