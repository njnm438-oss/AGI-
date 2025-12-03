from dataclasses import dataclass

@dataclass
class Config:
    embedding_dim: int = 512
    state_dim: int = 256
    action_dim: int = 32
    hidden_dim: int = 768
    device: str = 'cpu'
    checkpoint_dir: str = 'checkpoints'
    memory_dir: str = 'memory'
    working_capacity: int = 128
    episodic_capacity: int = 200000
    autobio_capacity: int = 50000
    consolidation_interval: float = 60.0
    seed: int = 42
    use_faiss: bool = True
    llm_backend_preference: tuple = ('llama_cpp','gpt2')

CONFIG = Config()
