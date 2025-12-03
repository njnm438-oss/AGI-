from typing import Dict

class Coordinator:
    def __init__(self, agent):
        self.agent = agent

    def build_context(self, question:str)->Dict:
        emb = self.agent.embedding.encode_text(question)
        attended = self.agent.memory.search(emb, k=12) if hasattr(self.agent.memory,'search') else []
        state = {'question': question, 'embedding': emb, 'attended': attended,
                 'emotion': getattr(self.agent.emotion,'to_vector',lambda:None)(),
                 'profile': getattr(self.agent,'self_model',None)}
        return state

    def decide_response(self, state: Dict):
        goal = self.agent.goal_manager.active()
        goal_desc = goal['desc'] if goal else None
        # Use a concise instruction to encourage short, useful answers in English
        prompt = f"Provide a brief, helpful answer to the following question.\nQuestion: {state.get('question','')}\nBrief answer:"
        resp = self.agent.generate(prompt, max_tokens=256)
        
        # Only store valid, clean responses (not internal traces)
        if resp and len(resp.strip()) > 10:
            try:
                self.agent.chat_history.append({'role': 'user', 'content': state.get('question', '')})
                self.agent.chat_history.append({'role': 'assistant', 'content': resp})
            except Exception:
                pass
        
        if not resp:
            return "I apologize, I do not have a complete answer at this time."
        return resp

