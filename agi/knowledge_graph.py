"""
Knowledge Graph (KM1) — AGI v8 Semantic Memory
Structured knowledge with concepts, relations, and clustering.
- Automatic relation extraction (subject-verb-object)
- Concept clustering
- Query API (nearest concepts, paths, etc.)
"""

import logging
from typing import List, Dict, Set, Tuple, Optional, Any
import networkx as nx
import json
import os
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    """A concept node in the KG."""
    id: str
    label: str
    frequency: int = 1
    importance: float = 0.5
    definition: str = ""
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Relation:
    """A relation edge in the KG."""
    source: str  # concept ID
    target: str  # concept ID
    predicate: str  # verb/relation type
    weight: float = 1.0  # strength of relation
    evidence: List[str] = None  # source facts
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


class KnowledgeGraph:
    """NetworkX-based knowledge graph with relation extraction and clustering."""
    
    def __init__(self, persist_path: str = None):
        self.graph = nx.DiGraph()
        self.concepts: Dict[str, Concept] = {}
        self.persist_path = persist_path
        self._load()
    
    def _load(self):
        """Load KG from disk if available."""
        if self.persist_path and os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, 'r') as f:
                    data = json.load(f)
                    # Reconstruct graph
                    for concept_data in data.get('concepts', []):
                        c = Concept(**concept_data)
                        self.concepts[c.id] = c
                        self.graph.add_node(c.id, **asdict(c))
                    
                    for rel_data in data.get('relations', []):
                        self.graph.add_edge(
                            rel_data['source'],
                            rel_data['target'],
                            predicate=rel_data['predicate'],
                            weight=rel_data.get('weight', 1.0),
                            evidence=rel_data.get('evidence', [])
                        )
                
                logger.info("Loaded KG: %d concepts, %d relations", len(self.concepts), self.graph.number_of_edges())
            except Exception as e:
                logger.warning("Failed to load KG: %s", e)
    
    def _save(self):
        """Persist KG to disk."""
        if not self.persist_path:
            return
        
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        
        try:
            data = {
                "concepts": [asdict(c) for c in self.concepts.values()],
                "relations": []
            }
            
            for source, target, attrs in self.graph.edges(data=True):
                data["relations"].append({
                    "source": source,
                    "target": target,
                    "predicate": attrs.get('predicate', 'related_to'),
                    "weight": attrs.get('weight', 1.0),
                    "evidence": attrs.get('evidence', [])
                })
            
            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error("Failed to save KG: %s", e)
    
    def add_concept(self, label: str, importance: float = 0.5, definition: str = "") -> str:
        """Add or update a concept, return its ID."""
        concept_id = hashlib.md5(label.encode()).hexdigest()[:12]
        
        if concept_id in self.concepts:
            self.concepts[concept_id].frequency += 1
            self.concepts[concept_id].importance = max(self.concepts[concept_id].importance, importance)
        else:
            concept = Concept(id=concept_id, label=label, importance=importance, definition=definition)
            self.concepts[concept_id] = concept
            self.graph.add_node(concept_id, **asdict(concept))
        
        return concept_id
    
    def add_relation(self, source_label: str, predicate: str, target_label: str, weight: float = 1.0, evidence: str = ""):
        """Add a subject-verb-object relation."""
        source_id = self.add_concept(source_label)
        target_id = self.add_concept(target_label)
        
        # Update or add edge
        if self.graph.has_edge(source_id, target_id):
            current_weight = self.graph[source_id][target_id].get('weight', 1.0)
            new_weight = (current_weight + weight) / 2
            self.graph[source_id][target_id]['weight'] = new_weight
            if evidence:
                self.graph[source_id][target_id]['evidence'].append(evidence)
        else:
            self.graph.add_edge(source_id, target_id, predicate=predicate, weight=weight, evidence=[evidence] if evidence else [])
        
        self._save()
    
    def extract_relations(self, text: str) -> List[Relation]:
        """
        Extract subject-verb-object relations from text (naive implementation).
        Uses simple heuristics (stop words, verb detection).
        """
        relations = []
        
        # Naive SVO extraction
        words = text.split()
        
        # Common verbs
        verbs = ["is", "has", "causes", "precedes", "related_to", "includes", "defines"]
        
        for i, word in enumerate(words):
            if any(verb in word.lower() for verb in verbs):
                # Simple: prev word = subject, word = verb, next words = object
                if i > 0 and i < len(words) - 1:
                    subject = words[i - 1].strip('.,!?;:')
                    predicate = word.lower().strip('.,!?;:')
                    obj = " ".join(words[i + 1:i + 3]).strip('.,!?;:')
                    
                    if len(subject) > 2 and len(obj) > 2:
                        rel = Relation(
                            source=self.add_concept(subject),
                            target=self.add_concept(obj),
                            predicate=predicate,
                            weight=0.7,
                            evidence=[text[:100]]
                        )
                        relations.append(rel)
                        self.add_relation(subject, predicate, obj, weight=0.7, evidence=text[:100])
        
        return relations
    
    def query_concept(self, label: str) -> Optional[Concept]:
        """Find a concept by label."""
        for concept in self.concepts.values():
            if concept.label.lower() == label.lower():
                return concept
        return None
    
    def find_nearest_concepts(self, label: str, k: int = 5) -> List[Concept]:
        """Find k nearest concepts (by label similarity)."""
        query_concept = self.query_concept(label)
        if not query_concept:
            return []
        
        # Find neighbors in graph
        neighbors = list(self.graph.successors(query_concept.id)) + list(self.graph.predecessors(query_concept.id))
        neighbor_concepts = [self.concepts[n] for n in neighbors if n in self.concepts]
        
        return neighbor_concepts[:k]
    
    def find_path(self, source_label: str, target_label: str, max_length: int = 5) -> List[str]:
        """Find path between two concepts (BFS)."""
        source_id = None
        target_id = None
        
        for concept in self.concepts.values():
            if concept.label.lower() == source_label.lower():
                source_id = concept.id
            if concept.label.lower() == target_label.lower():
                target_id = concept.id
        
        if not source_id or not target_id:
            return []
        
        try:
            path = nx.shortest_path(self.graph, source_id, target_id, weight=lambda u, v, d: 2.0 - d.get('weight', 1.0))
            return [self.concepts[node].label for node in path]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def cluster_concepts(self, method: str = "simple") -> Dict[int, List[Concept]]:
        """Cluster concepts by connectivity or similarity."""
        clusters = {}
        
        if method == "simple":
            # Connected components
            components = nx.weakly_connected_components(self.graph)
            for i, component in enumerate(components):
                clusters[i] = [self.concepts[node] for node in component if node in self.concepts]
        
        return clusters
    
    def infer_new_facts(self) -> List[Relation]:
        """
        Infer new relations from existing ones (transitivity and closure).
        If A -> B and B -> C, infer A -> C.
        """
        new_relations = []
        
            # Transitivity: if has_part and located_in, then infer
            # Collect edges to add first (avoid modifying during iteration)
            edges_to_add = []
        
            nodes_list = list(self.graph.nodes())
            for source in nodes_list:
                successors_list = list(self.graph.successors(source))
                for intermediate in successors_list:
                    intermediate_successors = list(self.graph.successors(intermediate))
                    for target in intermediate_successors:
                        if not self.graph.has_edge(source, target):
                            edges_to_add.append((source, target))
        
            # Now add edges
            for source, target in edges_to_add:
                self.graph.add_edge(source, target, predicate="inferred", weight=0.5, evidence=[])
            
                source_label = self.concepts[source].label if source in self.concepts else "?"
                target_label = self.concepts[target].label if target in self.concepts else "?"
            
                rel = Relation(source=source, target=target, predicate="inferred", weight=0.5)
                new_relations.append(rel)
        
        if new_relations:
            self._save()
            logger.debug("Inferred %d new relations", len(new_relations))
        
        return new_relations
    
    def summary(self) -> Dict[str, Any]:
        """Get KG summary statistics."""
        return {
            "num_concepts": len(self.concepts),
            "num_relations": self.graph.number_of_edges(),
            "num_clusters": len(self.cluster_concepts()),
            "avg_degree": np.mean([self.graph.degree(n) for n in self.graph.nodes()]) if self.graph.nodes() else 0,
            "top_concepts": [
                {"label": c.label, "importance": c.importance, "frequency": c.frequency}
                for c in sorted(self.concepts.values(), key=lambda x: x.importance * x.frequency, reverse=True)[:5]
            ]
        }
    
    def __repr__(self) -> str:
        return f"KnowledgeGraph(concepts={len(self.concepts)}, relations={self.graph.number_of_edges()})"


# Import numpy for summary
import numpy as np
