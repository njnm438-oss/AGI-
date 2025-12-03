from typing import List

class Reasoner:
    def __init__(self, llm, critic_fn):
        self.llm = llm
        self.critic = critic_fn

    def generate_chains(self, question: str, n=3) -> List[str]:
        chains = []
        for i in range(n):
            prompt = f"Think step-by-step (private). Question: {question}\nChain:"
            out = self.llm.generate(prompt, max_new_tokens=80)
            chains.append(out or "")
        return chains

    def score_and_select(self, chains, question, q_emb, emb_mod, emotion):
        scored = [(self.critic(c, question, q_emb, emb_mod, {}, emotion), c) for c in chains]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]  # best chain
