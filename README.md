## AGI Laboratory v4.0

Modular AGI prototype (lab-oriented). Copy files in `agi/` and run `main.py`.

## Quickstart
1. Create venv: `python -m venv venv && source venv/bin/activate`
2. Install: `pip install -r requirements.txt` (optional models: LLaMA, CLIP)
3. Demo: `python main.py --mode demo`
4. Chat: `python main.py --mode chat`
5. Autonomous demo: `python main.py --mode autonomous`

## Notes
- LLaMA requires `llama-cpp-python` + ggml model file.
- CLIP optional: if not installed, image fallback used.
- FAISS optional: if installed and path provided, episodic memory uses FAISS.
- This repo is a research scaffold — tune components for your tasks.

## Roadmap
- Trainable world model, MuZero-style planner, LLM backends local, perception upgrades, benchmarks.
