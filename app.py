from flask import Flask, render_template, request, jsonify
import json
from agi.agent_pro import AGIAgentPro
import threading
import atexit
import time
from agi.replay_buffer import ReplayBuffer
from agi.learner import Learner

app = Flask(__name__)

# Create only ONE agent instance
agent = AGIAgentPro()

# Background learning/collection (lightweight, daemon)
_replay = ReplayBuffer(capacity=50000)
_learner = Learner(getattr(agent, 'world_model', None) or None, _replay, device=getattr(agent.config, 'device', 'cpu'))
_collector_stop = threading.Event()
_learner_stop = threading.Event()
_collector_count = 0
_last_loss = None
_world_model = getattr(agent, 'world_model', None)
if _world_model is None:
    _learner = None
else:
    _learner = Learner(_world_model, _replay, device=getattr(agent.config, 'device', 'cpu'))

def _collector_loop(interval=1.0):
    global _collector_count
    state = None
    try:
        state = (0.0 * 1)  # placeholder
    except Exception:
        state = None
    while not _collector_stop.is_set():
        try:
            obs = f"web tick {_collector_count}"
            emb = agent.perceive_text(obs)
            sd = getattr(agent.config, 'state_dim', 256)
            st = [0.0]*sd
            # convert embedding to state-like vector if possible
            try:
                import numpy as _np
                e = _np.array(emb, dtype=_np.float32)
                st = _np.zeros(sd, dtype=_np.float32)
                st[:min(len(e), sd)] = e[:sd]
            except Exception:
                pass
            import numpy as _np
            action_dim = getattr(agent.config, 'action_dim', 8)
            a = _np.random.randn(action_dim).astype('float32')
            next_st = _np.array(st) + 0.1 * a[:len(st)] + 0.01 * _np.random.randn(len(st)).astype('float32')
            _replay.add((st, a, 0.0, next_st, False))
            _collector_count += 1
        except Exception:
            pass
        _collector_stop.wait(interval)

def _learner_loop(interval=5.0, batch_size=32):
    global _last_loss
    while not _learner_stop.is_set():
        try:
            if len(_replay) >= batch_size:
                l = _learner.step(batch_size=batch_size)
                _last_loss = l
        except Exception:
            pass
        _learner_stop.wait(interval)

_collector_thread = threading.Thread(target=_collector_loop, daemon=True)
_learner_thread = threading.Thread(target=_learner_loop, daemon=True)
_collector_thread.start()
_learner_thread.start()


def _shutdown_background():
    try:
        _collector_stop.set()
        _learner_stop.set()
        _collector_thread.join(timeout=1.0)
        _learner_thread.join(timeout=1.0)
    except Exception:
        pass
    try:
        agent.shutdown()
    except Exception:
        pass

atexit.register(_shutdown_background)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    # Always ensure a JSON result is returned.
    try:
        raw = request.get_data(as_text=True) or ""
        if raw.strip() == "":
            return jsonify({"answer": "Empty request body"}), 400

        # try to parse JSON
        try:
            data = json.loads(raw)
        except Exception:
            return jsonify({"answer": "Invalid JSON format"}), 400

        # Extract question/message
        msg = data.get("message") or data.get("question")
        if not msg:
            return jsonify({"answer": "Missing 'message' in request"}), 400

        # Ask AGI safely
        try:
            response = agent.ask(msg)
            # If agent returns a dict/complex structure, extract the textual answer
            if isinstance(response, dict):
                final_text = response.get("answer") or response.get("text") or str(response)
            else:
                final_text = str(response)
            if not final_text or final_text.strip() == "":
                final_text = "…"
            answer = final_text
        except Exception as e:
            answer = f"Internal error: {e}"

        # final return ALWAYS valid JSON
        return jsonify({"answer": answer})

    except Exception as e:
        # This ALWAYS returns JSON even on fatal error
        return jsonify({"answer": f"Fatal server error: {e}"}), 500


@app.route('/status')
def status():
    return jsonify({
        'replay_size': len(_replay),
        'collector_count': _collector_count,
        'last_loss': _last_loss,
        'agent_running': getattr(agent, '_running', True)
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
