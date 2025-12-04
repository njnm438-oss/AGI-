"""
Meta-Reasoning Engine (MR2)
- Compare multiple reasoning traces
- Verify conclusions, detect contradictions
- Propose corrective actions (remove weak premises, re-run reasoning)
- Score alternative reasoning chains and pick best
"""

from typing import List, Dict, Any, Tuple
import logging
from .reasoning_engine import ReasoningEngine, Conclusion, Premise

logger = logging.getLogger(__name__)


class MetaReasoner:
    def __init__(self, reasoning_engine: ReasoningEngine = None):
        self.engine = reasoning_engine or ReasoningEngine()

    def evaluate_traces(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Given multiple reasoning traces (dicts with 'final_claim','confidence','steps'),
        compute coherence, consistency and rank them.
        Returns a report with best trace and scores.
        """
        scored = []
        for t in traces:
            conf = float(t.get('confidence', 0.0))
            # simple coherence proxy: confidence + len(steps)/10
            steps = t.get('steps', [])
            coherence = conf + min(0.2, len(steps) / 50.0)
            scored.append((coherence, t))
        scored.sort(key=lambda x: x[0], reverse=True)

        best = scored[0][1] if scored else None
        report = {
            'best': best,
            'ranked': [{'score': s, 'trace': t} for s, t in scored]
        }
        logger.debug('MetaReasoner evaluated %d traces, best confidence=%.3f', len(traces), float(best.get('confidence',0.0)) if best else 0.0)
        return report

    def verify_and_repair(self, premises: List[str], knowledge_base: List[str]) -> Tuple[Conclusion, List[str]]:
        """Run reasoning, verify consistency; if inconsistent or low confidence, attempt simple repairs:
        - drop lowest-confidence premise (simulated) and re-run
        Returns final conclusion and list of applied repairs.
        """
        # initial run
        initial = self.engine.reason(premises, knowledge_base)
        repairs = []

        # quick consistency check using engine.verify_consistency
        ok, msg = self.engine.verify_consistency([initial])
        if not ok or initial.confidence < 0.4:
            # attempt repair: remove the least informative premise (shortest text heuristic)
            if premises:
                weakest = min(premises, key=lambda p: len(p))
                premises2 = [p for p in premises if p != weakest]
                repairs.append(f'removed_weak_premise:{weakest}')
                repaired = self.engine.reason(premises2, knowledge_base)
                # choose between initial and repaired by confidence
                chosen = repaired if repaired.confidence >= initial.confidence else initial
                logger.info('Repair attempted, initial_conf=%.3f, repaired_conf=%.3f', initial.confidence, repaired.confidence)
                return chosen, repairs
        return initial, repairs

    def self_criticize(self, premises: List[str], knowledge_base: List[str]) -> Dict[str, Any]:
        """Produce a self-critique report: strengths, weaknesses, suggestions."""
        conclusion = self.engine.reason(premises, knowledge_base)
        issues = []
        if conclusion.confidence < 0.5:
            issues.append('low_confidence')
        # check consistency with simple verify
        ok, msg = self.engine.verify_consistency([conclusion])
        if not ok:
            issues.append('inconsistency')

        suggestions = []
        if 'low_confidence' in issues:
            suggestions.append('gather_more_evidence')
        if 'inconsistency' in issues:
            suggestions.append('reexamine_premises')

        return {
            'conclusion': conclusion.claim,
            'confidence': conclusion.confidence,
            'issues': issues,
            'suggestions': suggestions
        }


if __name__ == '__main__':
    mr = MetaReasoner()
    p = ['Water is necessary for life', 'Plants need water']
    kb = ['Water is a chemical', 'Plants need water']
    print(mr.self_criticize(p, kb))
