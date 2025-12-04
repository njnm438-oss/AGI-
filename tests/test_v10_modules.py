"""
Unit tests for AGI v10 proto-AGI modules:
- MetaReasoningEngine (MR2)
- SelfModificationFramework (SMF1)
- ContinualLearner (CL3)
- Unified Memory (UM3) - note: already exists, so test it
- LLMAdapter (L4)
"""

import pytest
import numpy as np
from agi.meta_reasoning_engine import MetaReasoningEngine, ReasoningTrace
from agi.self_modification import SelfModificationFramework
from agi.continual_learner import ContinualLearner
from agi.unified_memory import UnifiedMemory
from agi.llm_adapter import LLMAdapter, AdaptationConfig
from agi.embedding import EmbeddingModule


class TestMetaReasoningEngine:
    """Tests for MR2."""

    def test_trace_reasoning(self):
        mre = MetaReasoningEngine()
        trace = mre.trace_reasoning(
            steps=["Premise A", "Premise B"], conclusion="Conclusion C"
        )
        assert isinstance(trace, ReasoningTrace)
        assert len(mre.reasoning_traces) == 1

    def test_verify_conclusion(self):
        mre = MetaReasoningEngine()
        trace = mre.trace_reasoning(
            steps=["All humans are mortal", "Socrates is human"],
            conclusion="Socrates is mortal",
        )
        is_valid, msg = mre.verify_conclusion(trace)
        assert isinstance(is_valid, bool)
        assert trace.logical_consistency > 0.0

    def test_compare_reasoning_paths(self):
        mre = MetaReasoningEngine()
        trace1 = mre.trace_reasoning(
            steps=["A", "B", "C"], conclusion="Valid conclusion"
        )
        trace2 = mre.trace_reasoning(
            steps=["X", "Y"], conclusion="Invalid conclusion"
        )
        best = mre.compare_reasoning_paths([trace1, trace2])
        assert best is not None

    def test_detect_contradictions(self):
        mre = MetaReasoningEngine()
        trace1 = mre.trace_reasoning(steps=["A"], conclusion="It is true")
        trace2 = mre.trace_reasoning(steps=["B"], conclusion="It is false")
        contras = mre.detect_contradictions([trace1, trace2])
        assert len(contras) > 0

    def test_self_critique(self):
        mre = MetaReasoningEngine()
        trace = mre.trace_reasoning(
            steps=["Premise 1", "Premise 2", "Premise 3"],
            conclusion="Conclusion",
        )
        critique = mre.self_critique(trace)
        assert "reasoning_quality" in critique
        assert critique["reasoning_quality"] >= 0.0

    def test_meta_learn(self):
        mre = MetaReasoningEngine()
        for i in range(5):
            trace = mre.trace_reasoning(
                steps=[f"Step {j}" for j in range(i + 2)], conclusion=f"Conclusion {i}"
            )
            mre.self_critique(trace)
        report = mre.meta_learn()
        assert "improvement_trend" in report


class TestSelfModificationFramework:
    """Tests for SMF1."""

    def test_create_variant(self):
        smf = SelfModificationFramework()
        var = smf.create_variant("module_a", "code_v1", "First variant")
        assert var.variant_id is not None
        assert "module_a" in smf.module_variants

    def test_test_variant(self):
        smf = SelfModificationFramework()
        var = smf.create_variant("module_a", "code_v1")
        tests = [("test_1", 1.0), ("test_2", 1.0)]
        results = smf.test_variant(var, tests)
        assert len(results) > 0
        assert var.avg_performance > 0.0

    def test_select_best_variant(self):
        smf = SelfModificationFramework()
        v1 = smf.create_variant("mod", "code1")
        v2 = smf.create_variant("mod", "code2")
        smf.test_variant(v1, [("t1", 1.0)])
        smf.test_variant(v2, [("t2", 1.0)])
        best = smf.select_best_variant("mod")
        assert best is not None
        assert best.is_active

    def test_rollback_variant(self):
        smf = SelfModificationFramework()
        v1 = smf.create_variant("mod", "v1")
        v2 = smf.create_variant("mod", "v2")
        smf.select_best_variant("mod")
        success = smf.rollback_variant("mod", v1.variant_id)
        assert success is True

    def test_suggest_modification(self):
        smf = SelfModificationFramework()
        v = smf.create_variant("mod", "code")
        smf.test_variant(v, [("test", 1.0)])
        sug = smf.suggest_modification("mod")
        assert "recommendations" in sug


class TestContinualLearner:
    """Tests for CL3."""

    def test_add_experience(self):
        cl = ContinualLearner()
        exp = {"action": "move", "reward": 1.0}
        cl.add_experience("task_1", exp, importance=0.8)
        assert len(cl.replay_buffer) == 1

    def test_consolidate(self):
        cl = ContinualLearner()
        for i in range(10):
            exp = {"step": i, "reward": 0.5}
            cl.add_experience("task_1", exp, importance=0.5 + i * 0.05)
        cl.consolidate()
        assert len(cl.consolidated_memory) > 0

    def test_mixed_replay(self):
        cl = ContinualLearner()
        for i in range(20):
            cl.add_experience("task_1", {"step": i}, importance=0.5)
        batch = cl.mixed_replay(batch_size=10)
        assert len(batch) > 0

    def test_distill_world_model(self):
        cl = ContinualLearner()
        for i in range(5):
            cl.add_experience("task_1", {"data": i}, importance=0.5)
        # Placeholder models
        cl.distill_world_model(None, None, batch_size=2, steps=3)

    def test_get_consolidation_report(self):
        cl = ContinualLearner()
        cl.add_experience("task_1", {"x": 1}, importance=0.7)
        cl.add_experience("task_2", {"y": 2}, importance=0.6)
        report = cl.get_consolidation_report()
        assert "samples_seen" in report
        assert report["num_tasks"] == 2


class TestUnifiedMemory:
    """Tests for UM3 (existing UnifiedMemory)."""

    def test_add_event(self):
        emb = EmbeddingModule(dim=128)
        um = UnifiedMemory(emb)
        um.add_event("Test event", importance=0.5)
        # Check that memory was added (access items list)
        assert len(um.episodic.items) > 0 or len(um.kg.concepts) > 0

    def test_query(self):
        emb = EmbeddingModule(dim=128)
        um = UnifiedMemory(emb)
        um.add_event("Test event about AGI")
        results = um.query("AGI", k=5)
        assert isinstance(results, list)

    def test_consolidate(self):
        emb = EmbeddingModule(dim=128)
        um = UnifiedMemory(emb)
        um.add_event("Event 1")
        um.add_event("Event 2")
        um.consolidate()


class TestLLMAdapter:
    """Tests for L4."""

    def test_init(self):
        llm = LLMAdapter(model_name="gpt2")
        assert llm.model_name == "gpt2"

    def test_set_config(self):
        llm = LLMAdapter()
        config = AdaptationConfig(temperature=0.5, learning_rate=1e-3)
        llm.set_config(config)
        assert llm.config.temperature == 0.5

    def test_register_prompt_template(self):
        llm = LLMAdapter()
        llm.register_prompt_template("task_1", "Template: {query}")
        assert "task_1" in llm.prompt_templates

    def test_adapt_prompt(self):
        llm = LLMAdapter()
        llm.register_prompt_template("task_1", "Query: {query}\nContext: {context}")
        adapted = llm.adapt_prompt("task_1", "What is X?", "Background info")
        assert "What is X?" in adapted

    def test_generate(self):
        llm = LLMAdapter()
        response = llm.generate("Test prompt")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_add_fine_tuning_data(self):
        llm = LLMAdapter()
        examples = [("input1", "output1"), ("input2", "output2")]
        llm.add_fine_tuning_data("task_1", examples)
        assert len(llm.fine_tuning_data) == 2

    def test_fine_tune_on_task(self):
        llm = LLMAdapter()
        examples = [("q", "a") for _ in range(5)]
        llm.add_fine_tuning_data("task_1", examples)
        results = llm.fine_tune_on_task("task_1", steps=5)
        assert "final_loss" in results

    def test_real_time_adaptation(self):
        llm = LLMAdapter()
        old_temp = llm.config.temperature
        llm.real_time_adaptation("task_1", feedback=0.8, direction="increase")
        assert llm.config.temperature > old_temp

    def test_batch_generate(self):
        llm = LLMAdapter()
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = llm.batch_generate(prompts)
        assert len(responses) == 3

    def test_rank_responses(self):
        llm = LLMAdapter()
        prompt = "What is artificial intelligence?"
        candidates = [
            "AI is machine learning",
            "AI is intelligence by machines",
            "AI is something else",
        ]
        ranked = llm.rank_responses(prompt, candidates)
        assert len(ranked) == 3
        assert ranked[0][1] >= ranked[1][1]

    def test_get_adaptation_summary(self):
        llm = LLMAdapter()
        summary = llm.get_adaptation_summary()
        assert "model_name" in summary
        assert "usage_stats" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
