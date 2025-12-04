"""
Self-Modification Framework (SMF1)
- Register candidate module implementations (callables)
- Run a set of unit testcases / evaluation scenarios
- Score implementations and select the best version
This is a safe prototype: it executes callables provided programmatically (no arbitrary code execution from untrusted sources).
"""

from typing import Callable, List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class Candidate:
    def __init__(self, id: str, func: Callable, meta: Dict[str, Any] = None):
        self.id = id
        self.func = func
        self.meta = meta or {}
        self.score = None


class SelfModificationFramework:
    def __init__(self):
        self.candidates: Dict[str, Candidate] = {}
        self.evaluations: Dict[str, Dict[str, Any]] = {}

    def register(self, id: str, func: Callable, meta: Dict[str, Any] = None):
        self.candidates[id] = Candidate(id, func, meta)
        logger.info('Registered candidate %s', id)

    def evaluate(self, id: str, testcases: List[Tuple[Any, Any]]) -> Dict[str, Any]:
        """Evaluate candidate `id` on provided testcases.
        testcases: list of (input, expected_output) pairs. Candidate.func should accept input and return output.
        Returns evaluation metrics.
        """
        cand = self.candidates.get(id)
        if not cand:
            raise KeyError(id)
        total = 0
        correct = 0
        details = []
        for inp, expected in testcases:
            try:
                out = cand.func(inp)
                ok = out == expected
            except Exception as e:
                out = None
                ok = False
                logger.exception('Candidate %s raised exception', id)
            details.append({'input': inp, 'expected': expected, 'output': out, 'ok': ok})
            total += 1
            if ok:
                correct += 1
        score = correct / max(1, total)
        cand.score = score
        ev = {'score': score, 'correct': correct, 'total': total, 'details': details}
        self.evaluations[id] = ev
        logger.info('Evaluated candidate %s: score=%.3f', id, score)
        return ev

    def select_best(self) -> Tuple[str, float]:
        """Return (id, score) of best candidate."""
        best_id = None
        best_score = -1.0
        for id, cand in self.candidates.items():
            s = cand.score if cand.score is not None else -1.0
            if s > best_score:
                best_score = s
                best_id = id
        return best_id, best_score

    def run_evaluation_suite(self, test_suite: Dict[str, List[Tuple[Any, Any]]]) -> Dict[str, Any]:
        """Run multiple named test suites against all candidates and summarize."""
        summary = {}
        for id in list(self.candidates.keys()):
            agg_correct = 0
            agg_total = 0
            for name, cases in test_suite.items():
                ev = self.evaluate(id, cases)
                agg_correct += ev['correct']
                agg_total += ev['total']
            summary[id] = {'score': agg_correct / max(1, agg_total), 'correct': agg_correct, 'total': agg_total}
        return summary


if __name__ == '__main__':
    smf = SelfModificationFramework()
    # example: register two candidates for a toy function
    smf.register('v1', lambda x: x*2)
    smf.register('v2', lambda x: x+ x)
    suite = {'basic': [(2,4), (3,6), (5,10)]}
    print(smf.run_evaluation_suite(suite))
