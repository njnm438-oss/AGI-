"""
Unit tests for AGI v8 modules:
- Reasoning Engine (R1)
- Knowledge Graph (KM1)
- Meta-Learner (M1)
- Self-Model v2 (SM2)
"""

import pytest
import tempfile
import os
from agi.reasoning_engine import ReasoningEngine, Premise, DeductionModule, AbductionModule, InductionModule
from agi.knowledge_graph import KnowledgeGraph
from agi.meta_learner import MetaLearner, HyperparameterState
from agi.self_model_v2 import SelfModel, TaskStatus


class TestReasoningEngine:
    """Tests for R1 reasoning modules."""
    
    def test_deduction_module(self):
        """Test deduction with simple rules."""
        deduction = DeductionModule()
        premises = [
            Premise("Socrates is human"),
            Premise("Humans are mortal"),
        ]
        
        conclusions = deduction.deduce(premises)
        assert len(conclusions) > 0
        assert any("mortal" in c.claim for c in conclusions)
    
    def test_abduction_module(self):
        """Test abduction (hypothesis formation)."""
        abduction = AbductionModule()
        observations = ["has fever", "has cough", "feels tired"]
        
        conclusions = abduction.abduce(observations)
        assert len(conclusions) > 0
        assert any("sick" in c.claim.lower() for c in conclusions)
    
    def test_induction_module(self):
        """Test induction (pattern recognition)."""
        induction = InductionModule()
        observations = [
            "The sky is blue",
            "The ocean is blue",
            "The river is blue",
        ]
        
        conclusions = induction.induce(observations)
        assert len(conclusions) > 0
        assert any("blue" in c.claim.lower() for c in conclusions)
    
    def test_reasoning_engine_chain_of_thought(self):
        """Test full CoT generation."""
        engine = ReasoningEngine()
        question = "Is water essential for life?"
        knowledge_base = [
            "Water is a chemical compound",
            "Life requires water",
            "Plants need water",
            "Animals need water",
        ]
        
        cot = engine.chain_of_thought(question, knowledge_base)
        
        assert "steps" in cot
        assert "final_claim" in cot
        assert "confidence" in cot
        assert len(cot["steps"]) > 0
        assert cot["confidence"] > 0.0
    
    def test_coherence_score(self):
        """Test coherence scoring."""
        engine = ReasoningEngine()
        
        c1 = Premise("water is essential")
        c2 = Premise("water is necessary")
        
        deduction = DeductionModule()
        conclusions = deduction.deduce([c1, c2])
        
        score = engine.coherence_score(conclusions)
        assert 0.0 <= score <= 1.0
    
    def test_consistency_verification(self):
        """Test contradiction detection."""
        engine = ReasoningEngine()
        
        from agi.reasoning_engine import Conclusion
        conclusions = [
            Conclusion("It is raining", confidence=0.9, reasoning_type="deduction"),
            Conclusion("It is not raining", confidence=0.8, reasoning_type="observation"),
        ]
        
        consistent, msg = engine.verify_consistency(conclusions)
        assert not consistent  # Should detect contradiction


class TestKnowledgeGraph:
    """Tests for KM1 knowledge graph."""
    
    def test_add_concept(self):
        """Test concept addition."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraph(persist_path=os.path.join(tmpdir, "kg.json"))
            
            concept_id = kg.add_concept("AGI", importance=0.9)
            assert concept_id is not None
            assert "AGI" in kg.concepts[concept_id].label
    
    def test_add_relation(self):
        """Test relation addition."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraph(persist_path=os.path.join(tmpdir, "kg.json"))
            
            kg.add_relation("AGI", "is", "intelligent", weight=0.9)
            assert kg.graph.number_of_edges() > 0
    
    def test_extract_relations(self):
        """Test SVO extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraph(persist_path=os.path.join(tmpdir, "kg.json"))
            
            text = "Water is essential for life"
            relations = kg.extract_relations(text)
            
            assert len(relations) > 0
    
    def test_find_nearest_concepts(self):
        """Test concept neighbor search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraph(persist_path=os.path.join(tmpdir, "kg.json"))
            
            kg.add_relation("machine learning", "is", "AI", weight=0.9)
            kg.add_relation("AI", "includes", "neural networks", weight=0.8)
            
            neighbors = kg.find_nearest_concepts("AI", k=2)
            assert len(neighbors) > 0
    
    def test_find_path(self):
        """Test path finding between concepts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraph(persist_path=os.path.join(tmpdir, "kg.json"))
            
            kg.add_relation("AGI", "can_do", "reasoning")
            kg.add_relation("reasoning", "enables", "problem_solving")
            
            path = kg.find_path("AGI", "problem_solving")
            assert len(path) >= 2
    
    def test_cluster_concepts(self):
        """Test concept clustering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraph(persist_path=os.path.join(tmpdir, "kg.json"))
            
            kg.add_relation("cat", "is", "animal")
            kg.add_relation("dog", "is", "animal")
            
            clusters = kg.cluster_concepts()
            assert len(clusters) > 0
    
    def test_infer_new_facts(self):
        """Test transitive inference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraph(persist_path=os.path.join(tmpdir, "kg.json"))
            
            kg.add_relation("A", "causes", "B")
            kg.add_relation("B", "causes", "C")
            
            initial_edges = kg.graph.number_of_edges()
            new_rels = kg.infer_new_facts()
            
            # Should infer A -> C
            assert kg.graph.number_of_edges() >= initial_edges


class TestMetaLearner:
    """Tests for M1 meta-learner."""
    
    def test_hyperparameter_state(self):
        """Test hyperparameter tracking."""
        hps = HyperparameterState()
        assert hps.learning_rate > 0
        assert hps.batch_size > 0
        
        hps2 = hps.copy()
        hps2.learning_rate = 0.1
        assert hps.learning_rate != hps2.learning_rate
    
    def test_record_performance(self):
        """Test recording performance metrics."""
        ml = MetaLearner()
        
        ml.record_performance(loss=0.5, accuracy=0.9)
        assert len(ml.performance_history) == 1
        
        ml.record_performance(loss=0.3, accuracy=0.95)
        assert len(ml.performance_history) == 2
    
    def test_suggest_adjustment(self):
        """Test hyperparameter adjustment."""
        ml = MetaLearner()
        
        # Record improving loss
        for i in range(10):
            loss = 1.0 - i * 0.05
            ml.record_performance(loss=loss, accuracy=0.5 + i * 0.05)
        
        adjusted, reason = ml.suggest_adjustment()
        
        # Should suggest increasing learning rate (loss improving)
        assert adjusted.learning_rate > ml.hyperparams.learning_rate or adjusted.exploration_temperature > 0
    
    def test_select_exploration_action(self):
        """Test dynamic action selection."""
        ml = MetaLearner()
        
        for i in range(5):
            ml.record_performance(loss=0.5, exploration_reward=0.6)
        
        actions = ["explore_random", "exploit_best", "curiosity_driven"]
        selected = ml.select_exploration_action(actions)
        
        assert selected in actions
    
    def test_should_consolidate_memory(self):
        """Test memory consolidation trigger."""
        ml = MetaLearner()
        
        # High forgetting
        for i in range(10):
            ml.record_performance(loss=0.5, catastrophic_forgetting=0.2)
        
        should_consolidate = ml.should_consolidate_memory()
        assert bool(should_consolidate) is True
    
    def test_estimate_sample_efficiency(self):
        """Test learning efficiency calculation."""
        ml = MetaLearner()
        
        ml.record_performance(loss=1.0)
        ml.record_performance(loss=0.5)
        ml.record_performance(loss=0.3)
        
        efficiency = ml.estimate_sample_efficiency()
        assert 0.0 <= efficiency <= 1.0
        assert efficiency > 0.5  # Should be high


class TestSelfModel:
    """Tests for SM2 self-model."""
    
    def test_start_mission(self):
        """Test mission creation."""
        sm = SelfModel("TestAgent")
        
        mission_id = sm.start_mission("Solve a puzzle")
        assert mission_id in sm.missions
        assert sm.total_tasks_attempted == 1
    
    def test_complete_mission(self):
        """Test mission completion."""
        sm = SelfModel()
        
        mission_id = sm.start_mission("Test task")
        sm.complete_mission(mission_id, TaskStatus.SUCCESS, learning_gain=0.1)
        
        assert sm.missions[mission_id].status == TaskStatus.SUCCESS
        assert sm.total_tasks_succeeded == 1
    
    def test_record_performance(self):
        """Test performance recording."""
        sm = SelfModel()
        
        sm.record_performance("reasoning", accuracy=0.9, response_time=0.1, confidence=0.85)
        
        assert "reasoning" in sm.performance_records
        assert len(sm.performance_records["reasoning"]) == 1
    
    def test_get_learning_curve(self):
        """Test learning curve extraction."""
        sm = SelfModel()
        
        for i in range(5):
            sm.record_performance("memory_recall", accuracy=0.5 + i * 0.1, response_time=0.5 - i * 0.05, confidence=0.6)
        
        curve = sm.get_learning_curve("memory_recall")
        assert len(curve) == 5
        assert curve[-1] > curve[0]  # Improving
    
    def test_estimate_task_difficulty(self):
        """Test difficulty estimation."""
        sm = SelfModel()
        
        easy_difficulty = sm.estimate_task_difficulty("simple factual retrieval")
        hard_difficulty = sm.estimate_task_difficulty("complex multi-step causal reasoning")
        
        assert easy_difficulty < hard_difficulty
    
    def test_estimate_success_probability(self):
        """Test success probability prediction."""
        sm = SelfModel()
        
        sm.capabilities["reasoning"] = 0.9
        
        prob = sm.estimate_success_probability("reason about causality")
        assert 0.0 <= prob <= 1.0
        assert prob > 0.5  # Should be fairly high given strong capability
    
    def test_get_self_assessment(self):
        """Test self-assessment generation."""
        sm = SelfModel("AGI-v8")
        
        sm.start_mission("Task 1")
        sm.complete_mission(list(sm.missions.keys())[0], TaskStatus.SUCCESS)
        
        assessment = sm.get_self_assessment()
        
        assert assessment["agent_name"] == "AGI-v8"
        assert "overall_success_rate" in assessment
        assert "capabilities" in assessment
    
    def test_identify_strengths_weaknesses(self):
        """Test strength/weakness identification."""
        sm = SelfModel()
        
        sm.capabilities["reasoning"] = 0.8
        sm.capabilities["memory_recall"] = 0.9
        sm.capabilities["planning"] = 0.3
        
        analysis = sm.identify_strengths_and_weaknesses()
        
        assert len(analysis["strengths"]) >= 2
        assert len(analysis["weaknesses"]) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
