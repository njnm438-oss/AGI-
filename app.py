from flask import Flask, render_template, request, jsonify
import json
from agi.agent_pro import AGIAgentPro

app = Flask(__name__)

# Create only ONE agent instance
agent = AGIAgentPro()

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
            answer = agent.ask(msg)
            if not isinstance(answer, str) or answer.strip() == "":
                answer = "…"
        except Exception as e:
            answer = f"Internal error: {e}"

        # final return ALWAYS valid JSON
        return jsonify({"answer": answer})

    except Exception as e:
        # This ALWAYS returns JSON even on fatal error
        return jsonify({"answer": f"Fatal server error: {e}"}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
