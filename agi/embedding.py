import hashlib, numpy as np
from typing import Optional
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    ST_AVAILABLE = False

class EmbeddingModule:
    def __init__(self, dim=512):
        self.dim = dim
        self.model = None
        if ST_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-mpnet-base-v2')
            except Exception:
                self.model = None

    def encode_text(self, text: str) -> np.ndarray:
        if self.model is not None:
            v = self.model.encode(text)
            v = np.asarray(v, dtype=np.float32)
            if v.shape[0] != self.dim:
                v = self._resize(v)
            return v / (np.linalg.norm(v)+1e-9)
        h = hashlib.sha256(text.encode('utf-8')).digest()
        seed = int.from_bytes(h[:4], 'little')
        rng = np.random.RandomState(seed)
        v = rng.randn(self.dim).astype(np.float32)
        return v / (np.linalg.norm(v)+1e-9)

    def encode_image(self, pil_image) -> np.ndarray:
        arr = np.array(pil_image).astype(np.float32)/255.0
        base = np.concatenate([arr.mean(axis=(0,1)), np.histogram(arr.flatten(), bins=32, range=(0,1))[0].astype(np.float32)])
        vec = base
        if vec.shape[0] < self.dim:
            pad = np.zeros(self.dim-vec.shape[0], dtype=np.float32)
            vec = np.concatenate([vec, pad])
        return vec / (np.linalg.norm(vec)+1e-9)

    def _resize(self, v):
        if v.shape[0] > self.dim:
            return v[:self.dim]
        pad = np.zeros(self.dim - v.shape[0], dtype=np.float32)
        return np.concatenate([v, pad])
