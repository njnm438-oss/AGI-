import os, logging
logger = logging.getLogger('agi.llm')

class LLMManager:
    def __init__(self, config):
        self.config = config
        self.backends = {}

    def register(self, name: str, fn):
        self.backends[name] = fn

    def generate(self, prompt: str, max_tokens: int = 256, priority: tuple = None) -> str:
        order = list(priority or self.config.llm_backend_preference)
        # If we have a 'heuristic' backend, prefer it before other fallbacks
        if 'heuristic' in self.backends and 'heuristic' not in order:
            order.insert(0, 'heuristic')

        for b in order:
            fn = self.backends.get(b)
            if not fn: continue
            try:
                out = fn(prompt, max_tokens=max_tokens)
                if not out:
                    continue

                # Post-process GPT-2 outputs: penalize wild personal/temporal claims
                if b == 'gpt2':
                    low = out.lower()
                    bad_patterns = ['i was', 'when i was', 'my job', 'since 2013']
                    if any(p in low for p in bad_patterns):
                        logger.info('gpt2 output penalized for pattern match, skipping')
                        continue

                return out
            except Exception as e:
                logger.warning('Backend %s failed: %s', b, e)
        return ''

# llama-cpp adapter
try:
    from llama_cpp import Llama
    def llama_cpp_gen(prompt, max_tokens=256, model_path=None):
        if model_path is None or not os.path.exists(model_path):
            return ''
        llm = Llama(model_path=model_path)
        res = llm(prompt, max_tokens=max_tokens)
        return res['choices'][0]['text'].strip()
    LLAMA_CPP_AVAILABLE = True
except Exception:
    LLAMA_CPP_AVAILABLE = False

# GPT-2 fallback via transformers (lazy-imported inside factory)
GPT2_AVAILABLE = False
def _register_gpt2_if_available(mgr):
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        import torch

        def gpt2_gen(prompt, max_tokens=128):
            try:
                tok = GPT2Tokenizer.from_pretrained('distilgpt2')
                model = GPT2LMHeadModel.from_pretrained('distilgpt2')
                model.eval()
                toks = tok.encode(prompt, return_tensors='pt')
                out = model.generate(toks, max_length=min(toks.shape[1]+max_tokens, 512), do_sample=True, top_k=40, top_p=0.92)
                s = tok.decode(out[0], skip_special_tokens=True)
                gen = s[len(prompt):].strip()
                # keep only 1-2 sentences to avoid long, off-topic generations
                try:
                    import re
                    parts = re.split(r'(?<=[.!?])\s+', gen)
                    if len(parts) > 2:
                        gen = ' '.join(parts[:2]).strip()
                except Exception:
                    parts = gen.split('.')
                    gen = ('.'.join(parts[:2])).strip()
                return gen
            except Exception as e:
                logger.warning('gpt2_gen failed: %s', e)
                return ''

        mgr.register('gpt2', gpt2_gen)
        return True
    except Exception:
        return False

def make_default_llm_manager(config, llama_model_path: str = None):
    m = LLMManager(config)
    if LLAMA_CPP_AVAILABLE and llama_model_path:
        m.register('llama_cpp', lambda p, max_tokens=256: llama_cpp_gen(p, max_tokens=max_tokens, model_path=llama_model_path))
    # attempt to register GPT-2 lazily only if explicitly enabled via env
    try:
        import os as _os
        if _os.environ.get('AGI_ENABLE_GPT2', '').lower() in ('1', 'true', 'yes'):
            _register_gpt2_if_available(m)
    except Exception:
        pass
    return m
