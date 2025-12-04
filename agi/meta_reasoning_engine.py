"""
Meta-Reasoning Engine (MR2) — AGI v10
Reasons about reasoning: verifies own conclusions, compares alternatives, self-critique.
- Traces reasoning chains and evaluates them
- Compares multiple solution paths
- Detects inconsistencies and corrects
- Assigns confidence scores to conclusions
"""

import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


@dataclass
class ReasoningTrace:
    """A single reasoning chain with metadata."""
    steps: List[str] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.5
    logical_consistency: float = 0.5
    alternative_score: float = 0.0
    contradictions: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class MetaReasoningEngine:
    """Meta-cognition: agent reasons about its own reasoning."""

    def __init__(self):
        self.reasoning_traces: List[ReasoningTrace] = []
        self.meta_insights: List[Dict[str, Any]] = []
        self.reasoning_quality_history: List[float] = []

    def trace_reasoning(self, steps: List[str], conclusion: str) -> ReasoningTrace:
        """Record a reasoning chain."""
        trace = ReasoningTrace(steps=steps, conclusion=conclusion)
        self.reasoning_traces.append(trace)
        return trace

    def verify_conclusion(self, trace: ReasoningTrace) -> Tuple[bool, str]:
        """Check if conclusion logically follows from premises (basic verification)."""
        # Check for key keywords consistency
        conclusion_lower = trace.conclusion.lower()
        steps_lower = " ".join([s.lower() for s in trace.steps])

        # Simple heuristic: if key words from premises appear in conclusion
        words_in_conclusion = set(conclusion_lower.split())
        words_in_steps = set(steps_lower.split())
        overlap = len(words_in_conclusion & words_in_steps)

        if overlap > 0:
            trace.logical_consistency = 0.7 + 0.3 * (overlap / max(len(words_in_conclusion), 1))
            return True, "Conclusion logically consistent with premises"
        else:
            trace.logical_consistency = 0.3
            return False, "Conclusion may not follow logically from premises"

    def compare_reasoning_paths(self, traces: List[ReasoningTrace]) -> ReasoningTrace:
        """Compare multiple reasoning chains and select the best."""
        if not traces:
            return ReasoningTrace()

        # Score each trace
        for trace in traces:
            self.verify_conclusion(trace)
            # Score based on: consistency, length (prefer concise), confidence
            length_score = 1.0 / (1.0 + len(trace.steps) / 10.0)
            trace.alternative_score = trace.logical_consistency * 0.6 + length_score * 0.4

        # Select best
        best = max(traces, key=lambda t: t.alternative_score)
        logger.debug("Selected best reasoning path (score=%.3f): %s", best.alternative_score, best.conclusion)

        return best

    def detect_contradictions(self, traces: List[ReasoningTrace]) -> List[str]:
        """Find logical contradictions between traces."""
        contradictions = []

        for i, trace1 in enumerate(traces):
            for j, trace2 in enumerate(traces[i + 1 :], start=i + 1):
                # Simple contradiction check: opposite conclusions
                conc1_lower = trace1.conclusion.lower()
                conc2_lower = trace2.conclusion.lower()

                # Check for negation patterns
                negation_markers = [("is not", "is"), ("cannot", "can"), ("false", "true"), ("impossible", "possible")]

                for neg, pos in negation_markers:
                    if (neg in conc1_lower and pos in conc2_lower) or (neg in conc2_lower and pos in conc1_lower):
                        contradictions.append(f"Trace {i} and {j} contradict: '{trace1.conclusion}' vs '{trace2.conclusion}'")
                        trace1.contradictions.append(f"Contradicts trace {j}")
                        trace2.contradictions.append(f"Contradicts trace {i}")

        return contradictions

    def self_critique(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """Generate constructive self-critique of a reasoning chain."""
        critique = {
            "trace_id": id(trace),
            "reasoning_quality": 0.0,
            "weaknesses": [],
            "strengths": [],
            "improvements": [],
        }

        # Assess reasoning quality
        is_valid, msg = self.verify_conclusion(trace)
        critique["is_logically_valid"] = is_valid
        critique["validity_message"] = msg

        # Identify weaknesses
        if len(trace.steps) < 2:
            critique["weaknesses"].append("Reasoning chain too short; lacks intermediate steps")
        if trace.logical_consistency < 0.6:
            critique["weaknesses"].append("Low logical consistency detected")
        if trace.contradictions:
            critique["weaknesses"].extend(trace.contradictions)

        # Identify strengths
        if len(trace.steps) > 3:
            critique["strengths"].append("Multi-step reasoning shows good depth")
        if trace.logical_consistency > 0.8:
            critique["strengths"].append("High logical consistency")

        # Generate improvements
        if critique["weaknesses"]:
            critique["improvements"].append("Add more intermediate steps to improve clarity")
            critique["improvements"].append("Verify each premise before drawing conclusions")
            critique["improvements"].append("Compare with alternative reasoning paths")

        # Calculate overall quality
        critique["reasoning_quality"] = (
            0.4 * trace.logical_consistency + 0.3 * (1.0 - len(trace.contradictions) / 10.0) + 0.3 * (0.7 if is_valid else 0.3)
        )

        self.meta_insights.append(critique)
        self.reasoning_quality_history.append(critique["reasoning_quality"])

        return critique

    def meta_learn(self) -> Dict[str, Any]:
        """Learn from past reasoning traces to improve future reasoning."""
        if not self.reasoning_quality_history:
            return {"improvement_trend": 0.0, "insights": []}

        # Calculate trend
        if len(self.reasoning_quality_history) >= 2:
            recent = self.reasoning_quality_history[-5:]
            trend = (recent[-1] - recent[0]) / max(1, len(recent) - 1) if len(recent) > 1 else 0.0
        else:
            trend = 0.0

        # Extract insights
        insights = []
        avg_quality = sum(self.reasoning_quality_history) / len(self.reasoning_quality_history)

        if avg_quality < 0.5:
            insights.append("Overall reasoning quality low; recommend more careful premise verification")
        elif avg_quality > 0.8:
            insights.append("High-quality reasoning; maintain current approach")

        if trend > 0.05:
            insights.append("Reasoning improving over time")
        elif trend < -0.05:
            insights.append("Reasoning quality declining; review recent traces for errors")

        return {
            "improvement_trend": float(trend),
            "average_quality": float(avg_quality),
            "num_traces": len(self.reasoning_traces),
            "insights": insights,
        }

    def get_meta_report(self) -> Dict[str, Any]:
        """Generate comprehensive meta-reasoning report."""
        return {
            "num_traces": len(self.reasoning_traces),
            "num_meta_insights": len(self.meta_insights),
            "meta_learning": self.meta_learn(),
            "recent_traces": [
                {
                    "conclusion": t.conclusion,
                    "consistency": float(t.logical_consistency),
                    "steps_count": len(t.steps),
                }
                for t in self.reasoning_traces[-5:]
            ],
        }
