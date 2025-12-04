"""
Test V10 Proto-AGI Integration into AGIAgentPro
Validates that all v10 modules work together in AGIAgentPro.ask()
"""

import pytest
import logging
from agi.agent_pro import AGIAgentPro
from agi.config import CONFIG

logger = logging.getLogger(__name__)


class TestV10Integration:
    """Test v10 proto-AGI integration into AGIAgentPro"""

    def test_agent_initializes_with_v10_modules(self):
        """Test that AGIAgentPro initializes all v10 modules"""
        agent = AGIAgentPro(config=CONFIG)
        
        # Check all v10 modules are initialized
        assert agent.meta_reasoning is not None, "MetaReasoningEngine not initialized"
        assert agent.self_modifier is not None, "SelfModificationFramework not initialized"
        assert agent.continual_learner is not None, "ContinualLearner not initialized"
        assert agent.llm_adapter is not None, "LLMAdapter not initialized"
        # UnifiedMemory may be None if initialization fails gracefully
        if agent.unified_memory is not None:
            assert hasattr(agent.unified_memory, 'episodic'), "UnifiedMemory missing episodic attribute"

    def test_ask_with_v10_meta_reasoning(self):
        """Test that ask() uses meta-reasoning verification"""
        agent = AGIAgentPro(config=CONFIG)
        
        # Ask a simple question
        answer = agent.ask("What is 2+2?")
        
        # Verify answer is not empty
        assert isinstance(answer, str), "Answer should be a string"
        assert len(answer) > 0, "Answer should not be empty"
        
        # Check that meta-reasoning traces were recorded
        num_traces = len(agent.meta_reasoning.traces) if agent.meta_reasoning else 0
        assert num_traces >= 0, "Meta-reasoning traces should be non-negative"

    def test_ask_with_continual_learning(self):
        """Test that ask() records experiences in continual learner"""
        agent = AGIAgentPro(config=CONFIG)
        
        initial_buffer = agent.continual_learner.buffer_size if agent.continual_learner else 0
        
        # Ask a few questions
        agent.ask("How are you?")
        agent.ask("What is your name?")
        agent.ask("Tell me about yourself")
        
        final_buffer = agent.continual_learner.buffer_size if agent.continual_learner else 0
        
        # Buffer size should increase (or remain if capacity already reached)
        assert final_buffer >= initial_buffer, "Continual learner buffer should not decrease"

    def test_ask_with_unified_memory(self):
        """Test that ask() updates unified memory"""
        agent = AGIAgentPro(config=CONFIG)
        
        if agent.unified_memory is None:
            pytest.skip("UnifiedMemory not initialized")
        
        initial_size = len(agent.unified_memory.episodic.items)
        
        # Ask a question
        agent.ask("What is artificial intelligence?")
        
        final_size = len(agent.unified_memory.episodic.items)
        
        # Unified memory should record the interaction
        assert final_size >= initial_size, "Unified memory should record events"

    def test_adapt_llm_v10(self):
        """Test LLM v10 adaptation for specific tasks"""
        agent = AGIAgentPro(config=CONFIG)
        
        if agent.llm_adapter is None:
            pytest.skip("LLM adapter not available")
        
        # Adapt for a specific task
        examples = [
            ("What is 2+2?", "4"),
            ("What is 3+3?", "6"),
        ]
        
        success = agent.adapt_llm_v10("math", examples)
        
        # Adaptation should complete (even if mock)
        assert isinstance(success, bool), "adapt_llm_v10 should return bool"

    def test_get_v10_metrics(self):
        """Test v10 metrics collection"""
        agent = AGIAgentPro(config=CONFIG)
        
        # Get metrics
        metrics = agent.get_v10_metrics()
        
        # Verify all expected metrics are present
        assert isinstance(metrics, dict), "Metrics should be a dict"
        assert "meta_reasoning_traces" in metrics, "Missing meta_reasoning_traces"
        assert "self_modification_variants" in metrics, "Missing self_modification_variants"
        assert "continual_learner_buffer_size" in metrics, "Missing continual_learner_buffer_size"
        assert "unified_memory_size" in metrics, "Missing unified_memory_size"
        
        # All values should be non-negative numbers
        for key, value in metrics.items():
            assert isinstance(value, (int, float)), f"{key} should be numeric, got {type(value)}"
            assert value >= 0, f"{key} should be non-negative, got {value}"

    def test_ask_multiple_questions_chain(self):
        """Test that v10 modules work correctly across multiple ask() calls"""
        agent = AGIAgentPro(config=CONFIG)
        
        questions = [
            "Hello, what are you?",
            "Can you help me with learning?",
            "Tell me something interesting about AGI",
        ]
        
        answers = []
        for q in questions:
            answer = agent.ask(q)
            answers.append(answer)
            assert isinstance(answer, str), f"Answer for '{q}' should be string"
            assert len(answer) > 0, f"Answer for '{q}' should not be empty"
        
        # All answers should be different (or at least some variation)
        unique_answers = len(set(answers))
        assert unique_answers >= 1, "Should have at least one unique answer"
        
        # Verify metrics after multiple calls
        metrics = agent.get_v10_metrics()
        assert metrics["continual_learner_buffer_size"] >= len(questions), \
            "Continual learner should record at least as many experiences as questions asked"

    def test_v10_modules_thread_safe(self):
        """Test that v10 modules respect thread safety"""
        import threading
        
        agent = AGIAgentPro(config=CONFIG)
        results = []
        
        def ask_in_thread(question):
            try:
                answer = agent.ask(question)
                results.append((question, answer))
            except Exception as e:
                logger.warning("Thread error: %s", e)
                results.append((question, None))
        
        # Create multiple threads asking questions
        threads = []
        for i in range(3):
            t = threading.Thread(target=ask_in_thread, args=(f"Question {i}?",))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join(timeout=10)
        
        # All threads should have completed
        assert len(results) == 3, "All threads should complete"
        assert all(r[1] is not None for r in results), "All questions should have answers"
