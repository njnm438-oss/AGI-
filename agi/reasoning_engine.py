"""
Reasoning Engine (R1) — AGI v8 Core
Structured chain-of-thought reasoning with deduction, abduction, and induction.
- Deduction: from premises to conclusion (rule-based)
- Abduction: hypothesis formation (best explanation)
- Induction: pattern recognition from observations
"""

import logging
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import hashlib
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Premise:
    """A logical premise (fact or rule)."""
    content: str
    confidence: float = 1.0  # 0.0 to 1.0
    source: str = "user"  # user, memory, inference, etc.
    
    def __hash__(self):
        return hash(self.content)
    
    def __eq__(self, other):
        return isinstance(other, Premise) and self.content == other.content


@dataclass
class Conclusion:
    """A reasoned conclusion with supporting evidence."""
    claim: str
    confidence: float = 0.5
    reasoning_type: str = "unknown"  # deduction, abduction, induction
    premises: List[Premise] = field(default_factory=list)
    explanation: str = ""
    steps: List[str] = field(default_factory=list)  # step-by-step trace


class DeductionModule:
    """Rule-based deduction: if (premises) then (conclusion)."""
    
    def __init__(self):
        self.rules: List[Tuple[List[str], str]] = [
            # Classic rules
            (["X is mortal", "X is human"], "X is mortal"),
            (["X causes Y", "Y causes Z"], "X causes Z"),
            (["X is instance of Y", "Y has property P"], "X has property P"),
        ]
    
    def deduce(self, premises: List[Premise]) -> List[Conclusion]:
        """Apply deduction rules to premises."""
        conclusions = []
        premise_strs = [p.content.lower() for p in premises]
        
        for rule_premises, rule_conclusion in self.rules:
                # Check if all rule premises are satisfied
                matched = 0
                for rp in rule_premises:
                    rp_lower = rp.lower()
                    if any(rp_lower in ps or ps in rp_lower for ps in premise_strs):
                        matched += 1
            
                # Allow if most premises match
                if matched >= len(rule_premises) - 1:
                    min_conf = min([p.confidence for p in premises])
                    conc = Conclusion(
                        claim=rule_conclusion,
                        confidence=min(0.85, min_conf * 0.9),
                        reasoning_type="deduction",
                        premises=premises,
                        explanation=f"Applied rule: {' AND '.join(rule_premises)} → {rule_conclusion}",
                        steps=[f"Premise: {p.content}" for p in premises]
                    )
                    conclusions.append(conc)
        
        return conclusions


class AbductionModule:
    """Hypothesis formation: find best explanation for observations."""
    
    def __init__(self):
        self.hypotheses_db = {
            "sick": ["fever", "cough", "fatigue"],
            "learned": ["asks questions", "remembers", "improves"],
            "causal": ["event A preceded event B"],
        }
    
    def abduce(self, observations: List[str]) -> List[Conclusion]:
        """Generate best hypotheses for observations."""
        conclusions = []
        obs_lower = [o.lower() for o in observations]
        
        for hypothesis, indicators in self.hypotheses_db.items():
            match_count = sum(1 for ind in indicators if any(ind.lower() in o for o in obs_lower))
            if match_count >= 1:
                confidence = min(0.9, match_count / len(indicators))
                conc = Conclusion(
                    claim=f"Hypothesis: {hypothesis}",
                    confidence=confidence,
                    reasoning_type="abduction",
                    premises=[Premise(o, 0.8) for o in observations],
                    explanation=f"Best explanation for observations: {match_count}/{len(indicators)} indicators match",
                    steps=[f"Observation: {o}" for o in observations]
                )
                conclusions.append(conc)
        
        return conclusions


class InductionModule:
    """Pattern recognition from observations."""
    
    def __init__(self):
        self.pattern_memory: List[Dict[str, Any]] = []
    
    def induce(self, observations: List[str]) -> List[Conclusion]:
        """Identify patterns and generalizations."""
        conclusions = []
        
        if len(observations) < 2:
            return conclusions
        
        # Simple pattern: repeated words/concepts
        word_freq = {}
        for obs in observations:
            for word in obs.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        frequent_patterns = [w for w, freq in word_freq.items() if freq >= len(observations) // 2 and len(w) > 3]
        
        if frequent_patterns:
            conc = Conclusion(
                claim=f"Pattern detected: {', '.join(frequent_patterns)} appear frequently",
                confidence=0.7,
                reasoning_type="induction",
                premises=[Premise(o, 0.7) for o in observations],
                explanation=f"Generalization: concept appears in {len(frequent_patterns)} observations",
                steps=[f"Observation {i}: {o}" for i, o in enumerate(observations)]
            )
            conclusions.append(conc)
        
        return conclusions


class ReasoningEngine:
    """Orchestrates structured reasoning with deduction, abduction, induction."""
    
    def __init__(self):
        self.deduction = DeductionModule()
        self.abduction = AbductionModule()
        self.induction = InductionModule()
        self.reasoning_trace: List[Conclusion] = []
    
    def reason(self, premises: List[str], observations: List[str] = None) -> Conclusion:
        """
        Perform structured reasoning:
        1. Convert inputs to Premise objects
        2. Apply deduction, abduction, induction
        3. Score conclusions by coherence
        4. Return best conclusion with trace
        """
        if observations is None:
            observations = []
        
        premise_objs = [Premise(p, confidence=0.85) for p in premises]
        
        # Apply all reasoning modules
        deductive = self.deduction.deduce(premise_objs)
        abductive = self.abduction.abduce(observations if observations else premises)
        inductive = self.induction.induce(observations if observations else premises)
        
        all_conclusions = deductive + abductive + inductive
        
        if not all_conclusions:
            # Fallback: simple concatenation
            return Conclusion(
                claim=" AND ".join(premises),
                confidence=0.3,
                reasoning_type="fallback",
                premises=premise_objs,
                explanation="No structured reasoning applied; concatenating premises.",
                steps=premises
            )
        
        # Score conclusions by coherence
        best = max(all_conclusions, key=lambda c: c.confidence)
        self.reasoning_trace.append(best)
        
        logger.debug("Reasoning result: %s (confidence=%.2f, type=%s)", best.claim, best.confidence, best.reasoning_type)
        
        return best
    
    def chain_of_thought(self, question: str, knowledge_base: List[str]) -> Dict[str, Any]:
        """
        Generate step-by-step CoT explanation.
        Returns: {steps: [...], final_claim: ..., confidence: ...}
        """
        steps = []
        
        # Step 1: identify key concepts
        concepts = [w for w in question.lower().split() if len(w) > 3]
        steps.append(f"[Step 1] Key concepts: {', '.join(concepts)}")
        
        # Step 2: search knowledge base
        relevant_kb = [k for k in knowledge_base if any(c in k.lower() for c in concepts)]
        steps.append(f"[Step 2] Relevant facts: {len(relevant_kb)} items")
        
        # Step 3: apply reasoning
        conclusion = self.reason(relevant_kb[:3], knowledge_base[3:6] if len(knowledge_base) > 3 else [])
        steps.extend(conclusion.steps)
        steps.append(f"[Final] Conclusion: {conclusion.claim} (confidence: {conclusion.confidence:.2f})")
        
        return {
            "steps": steps,
            "final_claim": conclusion.claim,
            "confidence": conclusion.confidence,
            "reasoning_type": conclusion.reasoning_type
        }
    
    def coherence_score(self, conclusions: List[Conclusion]) -> float:
        """Score coherence of a set of conclusions (0.0 to 1.0)."""
        if not conclusions:
            return 0.0
        
        # Coherence = avg confidence
        avg_conf = np.mean([c.confidence for c in conclusions])
        
        # Bonus if reasoning types are diverse
        types = set(c.reasoning_type for c in conclusions)
        diversity_bonus = min(0.2, len(types) * 0.05)
        
        return min(1.0, avg_conf + diversity_bonus)
    
    def verify_consistency(self, conclusions: List[Conclusion]) -> Tuple[bool, str]:
        """Check if conclusions are internally consistent."""
        if len(conclusions) <= 1:
            return True, "Single or no conclusions; trivially consistent."
        
        # Simple check: no direct contradictions in claims
        claims = [c.claim.lower() for c in conclusions]
        
        contradictory_pairs = [
                ("is not", "is"),
                ("impossible", "possible"),
                ("false", "true"),
                ("reject", "accept"),
        ]
        
        for pair in contradictory_pairs:
                 pair0_found = any(f" {pair[0]} " in f" {claim} " or claim.startswith(pair[0]) for claim in claims)
                 pair1_found = any(f" {pair[1]} " in f" {claim} " or claim.startswith(pair[1]) for claim in claims)
             
                 if pair0_found and pair1_found:
                     return False, f"Contradiction detected: {pair[0]} vs {pair[1]}"
        
        return True, "Conclusions appear consistent."
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """Return reasoning trace for debugging/logging."""
        return [
            {
                "claim": c.claim,
                "confidence": c.confidence,
                "type": c.reasoning_type,
                "explanation": c.explanation,
            }
            for c in self.reasoning_trace[-10:]  # Last 10 conclusions
        ]
