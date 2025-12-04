import pytest

from agi.agent_pro import AGIAgentPro


def test_llm_blend_memory():
    agent = AGIAgentPro()
    # clear memory to simulate no relevant memory
    try:
        agent.memory = type('EmptyMem', (), {'search': lambda self, q, k=6: []})()
    except Exception:
        pass
    q = "what are you thinking about human"
    resp = agent.ask(q)
    assert isinstance(resp, str)
    assert not resp.lower().startswith('based on my memories')
*** End Patch