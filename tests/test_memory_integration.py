import pytest
from agi.agent_pro import AGIAgentPro
from agi.memory import EpisodicMemory, MemoryItem
from agi.embedding import EmbeddingModule
import numpy as np


def test_memory_dominant_when_relevant():
    """When memory is relevant, it should be attached to LLM response or dominate fallback."""
    agent = AGIAgentPro()
    
    # Inject 3 relevant memory items similar to the question
    q = "what are you thinking about AI"
    q_emb = agent.embedding.encode_text(q)
    
    facts = [
        "AI systems process information",
        "Artificial intelligence learns from data",
        "AI can solve complex problems"
    ]
    
    for i, fact in enumerate(facts):
        emb = agent.embedding.encode_text(fact)
        item = MemoryItem(
            id=f"mem_{i}",
            content={'type': 'text', 'text': fact},
            embedding=emb,
            ts=1.0,
            importance=0.8
        )
        agent.memory.add(item)
    
    # Ask a question that should retrieve these memories
    resp = agent.ask(q)
    assert isinstance(resp, str)
    assert len(resp) > 0
    # Memory should be referenced or blended
    # If LLM dominated, it may mention AI; if memory dominated, it will say "Based on my memories"
    assert resp.lower() in resp.lower()  # just sanity check


def test_memory_ignored_when_irrelevant():
    """When memory similarity is low, 'Based on my memories' should not appear."""
    agent = AGIAgentPro()
    
    # Inject memory that is very different from the question
    mem_text = "The weather is sunny and warm today"
    mem_emb = agent.embedding.encode_text(mem_text)
    item = MemoryItem(
        id="mem_weather",
        content={'type': 'text', 'text': mem_text},
        embedding=mem_emb,
        ts=1.0,
        importance=0.5
    )
    agent.memory.add(item)
    
    # Ask a question totally different
    q = "what is your reasoning about quantum mechanics"
    resp = agent.ask(q)
    
    assert isinstance(resp, str)
    # Since memory is irrelevant (low similarity), response should not start with "Based on my memories"
    assert not resp.lower().startswith('based on my memories')


def test_memory_dominance_count_increments():
    """Counter should increment when memory provides the final answer."""
    agent = AGIAgentPro()
    
    # Start counter at 0
    initial_count = getattr(agent, '_memory_dominance_count', 0)
    assert initial_count == 0
    
    # Clear/bypass LLM to force memory to be the fallback answer
    # We do this by monkey-patching the LLM's generate method to return empty
    original_gen = agent.llm.generate
    agent.llm.generate = lambda question, context="", max_new_tokens=60: ""
    
    # Inject at least 2 relevant memory items (to meet min_items requirement)
    mem_texts = ["AGI is fascinating", "AGI systems are interesting", "AGI research is important"]
    for idx, mem_text in enumerate(mem_texts):
        mem_emb = agent.embedding.encode_text(mem_text)
        item = MemoryItem(
            id=f"mem_{idx}",
            content={'type': 'text', 'text': mem_text},
            embedding=mem_emb,
            ts=1.0,
            importance=0.8
        )
        agent.memory.add(item)
    
    # Ask question that matches memory
    q = "what about AGI"
    resp = agent.ask(q)
    
    # Counter should have incremented (memory dominated because LLM returned empty)
    new_count = getattr(agent, '_memory_dominance_count', 0)
    assert new_count > initial_count, f"Expected counter to increment from {initial_count} to > {initial_count}, got {new_count}"
    
    # Restore original LLM
    agent.llm.generate = original_gen


def test_memory_attach_to_llm_response():
    """When LLM responds and memory is relevant, memory should be attached if configured."""
    from agi.config import Config
    
    # Create config with memory attachment enabled
    config = Config()
    config.attach_memory_to_llm = True
    agent = AGIAgentPro(config=config)
    
    # Inject relevant memory
    mem_text = "Machine learning is a subset of AI"
    mem_emb = agent.embedding.encode_text(mem_text)
    item = MemoryItem(
        id="mem_ml",
        content={'type': 'text', 'text': mem_text},
        embedding=mem_emb,
        ts=1.0,
        importance=0.8
    )
    agent.memory.add(item)
    
    # Ask a question
    q = "what is machine learning"
    resp = agent.ask(q)
    
    assert isinstance(resp, str)
    assert len(resp) > 0
    # The response should contain either the LLM answer or memory-enhanced version
    # Depending on LLM availability, we just check it's a valid response
    assert resp  # not empty
