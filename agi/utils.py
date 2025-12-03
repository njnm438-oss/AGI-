import os, time, logging
import numpy as np
from typing import Any
logger = logging.getLogger('agi.utils')

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def normalize(v: np.ndarray):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / (n + 1e-9)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na==0 or nb==0: return 0.0
    return float(np.dot(a,b)/(na*nb))
