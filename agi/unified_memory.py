"""
Unified Memory System (UM3)
- Combine EpisodicMemory + KnowledgeGraph into unified interface
- Provide add_event, query, consolidate and summary
"""

from typing import List, Dict, Any, Optional
import logging
from .memory import EpisodicMemory, MemoryItem
from .knowledge_graph import KnowledgeGraph
from .embedding import EmbeddingModule
import time

logger = logging.getLogger(__name__)


class UnifiedMemory:
    def __init__(self, embedding: EmbeddingModule, kg_persist_path: Optional[str] = None, episodic_capacity: int = 5000):
        self.embedding = embedding
        self.episodic = EpisodicMemory(dim=embedding.dim if hasattr(embedding,'dim') else getattr(embedding,'dim',128), capacity=episodic_capacity)
        self.kg = KnowledgeGraph(persist_path=kg_persist_path)

    def add_event(self, text: str, importance: float = 0.5):
        emb = None
        try:
            emb = self.embedding.encode_text(text)
        except Exception:
            emb = None
        item = MemoryItem(id=f'mem_{int(time.time()*1000)}', content={'type':'text','text':text}, embedding=emb, ts=time.time(), importance=importance)
        try:
            self.episodic.add(item)
        except Exception:
            # fallback to memory.add
            try:
                self.episodic.add(item)
            except Exception:
                logger.debug('Episodic memory add failed')
        # extract naive relations into KG
        try:
            rels = self.kg.extract_relations(text)
            logger.debug('Extracted %d relations from event', len(rels))
        except Exception:
            logger.debug('KG extraction failed')

    def query(self, query_text: str, k: int = 5) -> List[MemoryItem]:
        q_emb = None
        try:
            q_emb = self.embedding.encode_text(query_text)
        except Exception:
            q_emb = None
        try:
            results = self.episodic.search(q_emb, k=k) if q_emb is not None else []
            return results
        except Exception:
            return []

    def consolidate(self):
        # Move top episodic items into KG summary facts
        try:
            top = self.episodic.get_top_k(50)
        except Exception:
            top = []
        for m in top:
            text = m.content['text'] if isinstance(m.content, dict) and 'text' in m.content else str(m.content)
            try:
                self.kg.extract_relations(text)
            except Exception:
                continue
        logger.info('UnifiedMemory consolidation done: processed %d items', len(top))

    def summary(self) -> Dict[str, Any]:
        return {'episodic_size': len(self.episodic), 'kg_summary': self.kg.summary()}


if __name__ == '__main__':
    from .embedding import EmbeddingModule
    emb = EmbeddingModule(dim=64)
    um = UnifiedMemory(emb)
    um.add_event('Water is essential for life')
    print(um.summary())
