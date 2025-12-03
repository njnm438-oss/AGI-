## AGI Laboratory v4.0

Modular AGI prototype (lab-oriented). Copy files in `agi/` and run `main.py`.

## Quickstart
1. Create venv: `python -m venv venv && source venv/bin/activate`
2. Install: `pip install -r requirements.txt` (optional models: LLaMA, CLIP)
3. Demo: `python main.py --mode demo`
4. Chat: `python main.py --mode chat`
5. Autonomous demo: `python main.py --mode autonomous`

## Notes

## Roadmap
## World Model v6
This feature adds a small learnable world model, simulator and MCTS planner.
Use `scripts/collect_transitions.py` to generate small fixtures and `train_world_model.py` to train a smoke model.

