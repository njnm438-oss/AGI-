"""
Self-Modification Framework (SMF1) — AGI v10
Safe module rewriting, version testing, and selection.
- Creates variants of a module implementation
- Tests multiple versions
- Selects best performing variant
- Maintains version history for rollback
"""

import logging
import hashlib
import time
from typing import List, Dict, Tuple, Any, Callable, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModuleVariant:
    """A variant of a module with performance history."""
    variant_id: str
    name: str
    code: str  # simplified: store as string
    description: str = ""
    creation_time: float = 0.0
    test_results: List[Dict[str, float]] = None
    avg_performance: float = 0.5
    is_active: bool = False

    def __post_init__(self):
        if self.test_results is None:
            self.test_results = []
        if self.creation_time == 0.0:
            self.creation_time = time.time()


class SelfModificationFramework:
    """Manages safe self-modification through variant testing."""

    def __init__(self):
        self.module_variants: Dict[str, List[ModuleVariant]] = {}
        self.active_modules: Dict[str, ModuleVariant] = {}
        self.modification_history: List[Dict[str, Any]] = []

    def create_variant(self, module_name: str, code: str, description: str = "") -> ModuleVariant:
        """Create a new variant of a module."""
        variant_id = hashlib.md5(f"{module_name}_{time.time()}_{code}".encode()).hexdigest()[:12]
        variant = ModuleVariant(
            variant_id=variant_id,
            name=module_name,
            code=code,
            description=description,
        )

        if module_name not in self.module_variants:
            self.module_variants[module_name] = []

        self.module_variants[module_name].append(variant)
        logger.info("Created variant %s for module %s", variant_id, module_name)

        return variant

    def test_variant(self, variant: ModuleVariant, test_suite: List[Tuple[str, float]]) -> Dict[str, float]:
        """Run tests on a variant. test_suite: list of (test_name, expected_result)."""
        results = {}

        for test_name, expected in test_suite:
            # Simulate test execution (placeholder: would run actual test)
            score = 0.5 + (hash(test_name + variant.code) % 100) / 200.0  # pseudo-random in [0.5, 1.0]
            passed = score > 0.6
            results[test_name] = float(score)

            if passed:
                logger.debug("Test %s PASSED for variant %s", test_name, variant.variant_id)
            else:
                logger.debug("Test %s FAILED for variant %s", test_name, variant.variant_id)

        variant.test_results.append(results)
        variant.avg_performance = sum(results.values()) / max(1, len(results))

        return results

    def select_best_variant(self, module_name: str) -> Optional[ModuleVariant]:
        """Select the best performing variant for a module."""
        if module_name not in self.module_variants or not self.module_variants[module_name]:
            return None

        variants = self.module_variants[module_name]
        best = max(variants, key=lambda v: v.avg_performance if v.avg_performance > 0 else 0.0)

        if best.is_active:
            logger.info("Best variant %s already active for module %s", best.variant_id, module_name)
            return best

        # Deactivate previous active variant
        if module_name in self.active_modules:
            old_active = self.active_modules[module_name]
            old_active.is_active = False
            logger.info("Deactivated variant %s", old_active.variant_id)

        best.is_active = True
        self.active_modules[module_name] = best

        self.modification_history.append(
            {
                "timestamp": time.time(),
                "module": module_name,
                "selected_variant": best.variant_id,
                "performance": float(best.avg_performance),
            }
        )

        logger.info("Selected best variant %s for module %s (performance=%.3f)", best.variant_id, module_name, best.avg_performance)

        return best

    def rollback_variant(self, module_name: str, variant_id: str) -> bool:
        """Revert to a previous variant."""
        if module_name not in self.module_variants:
            logger.warning("Module %s not found", module_name)
            return False

        variants = self.module_variants[module_name]
        target = next((v for v in variants if v.variant_id == variant_id), None)

        if not target:
            logger.warning("Variant %s not found for module %s", variant_id, module_name)
            return False

        # Deactivate current
        if module_name in self.active_modules:
            self.active_modules[module_name].is_active = False

        target.is_active = True
        self.active_modules[module_name] = target

        logger.info("Rolled back to variant %s for module %s", variant_id, module_name)

        return True

    def suggest_modification(self, module_name: str) -> Dict[str, Any]:
        """Suggest improvements to a module based on test results."""
        if module_name not in self.module_variants:
            return {"suggestion": "No variants found for module", "recommendations": []}

        variants = self.module_variants[module_name]
        failed_tests = []

        for variant in variants:
            for test_result in variant.test_results:
                for test_name, score in test_result.items():
                    if score < 0.6:
                        failed_tests.append(test_name)

        recommendations = []

        if failed_tests:
            recommendations.append(f"Focus on fixing failing tests: {set(failed_tests)}")

        if len(variants) < 3:
            recommendations.append("Create more variants to explore solution space better")

        perf_trend = [v.avg_performance for v in variants if v.avg_performance > 0]
        if perf_trend and perf_trend[-1] < 0.6:
            recommendations.append("Current module performance is low; consider major redesign")

        return {
            "module": module_name,
            "num_variants": len(variants),
            "best_variant": max(variants, key=lambda v: v.avg_performance).variant_id if variants else None,
            "failed_tests": list(set(failed_tests)),
            "recommendations": recommendations,
        }

    def get_modification_history(self) -> List[Dict[str, Any]]:
        """Return history of modifications."""
        return self.modification_history[-20:]  # Last 20 modifications
