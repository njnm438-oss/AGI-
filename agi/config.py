from dataclasses import dataclass
from typing import Optional

# lightweight checks for optional libs
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False


@dataclass
class Config:
    embedding_dim: int = 384
    state_dim: int = 128
    action_dim: int = 16
    hidden_dim: int = 512
    device: str = 'cuda' if TORCH_AVAILABLE and getattr(torch, 'cuda', None) and torch.cuda.is_available() else 'cpu'
    checkpoint_dir: str = 'checkpoints'
    memory_dir: str = 'memory'
    working_capacity: int = 32
    episodic_capacity: int = 50000
    semantic_capacity: int = 20000
    autobio_capacity: int = 20000
    save_freq_steps: int = 500
    consolidation_interval: float = 60.0
    seed: Optional[int] = 42
    use_faiss: bool = FAISS_AVAILABLE
    llm_backend_preference: tuple = ('llama_cpp','gpt2')

    # Nouveaux paramètres configurables
    memory_similarity_threshold: float = 0.40   # seuil cosine min pour considérer mémoire pertinente
    memory_min_items: int = 2                   # nombre min d'items pertinents pour produire un résumé mémoire
    attach_memory_to_llm: bool = True           # si True, on ajoute bref résumé mémoire après la réponse LLM
    memory_attach_max_chars: int = 160          # longueur max du résumé mémoire si attaché


CONFIG = Config()
