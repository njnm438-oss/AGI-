from flask import Flask, render_template, request, jsonify
from agi.agent import AGIAgent

app = Flask(__name__)

# Create only ONE agent instance
agent = AGIAgent()

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/api/chat", methods=["POST"])
def chat_api():
    # Robust JSON parsing: accept either 'message' or 'question'
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"answer": "Invalid request format"}), 400
        user_msg = ""
        if "message" in data:
            user_msg = data.get("message", "")
        elif "question" in data:
            user_msg = data.get("question", "")
        user_msg = user_msg.strip() if isinstance(user_msg, str) else ""
    except Exception as e:
        return jsonify({"answer": f"Error reading request: {e}"}), 400

    if not user_msg:
        # Always return valid JSON
        return jsonify({"answer": "⚠️ Message vide."}), 400

    # ---- CALL AGI SAFELY ----
    try:
        answer = agent.ask(user_msg)
        if not isinstance(answer, str) or len(answer.strip()) == 0:
            answer = "…"
    except Exception as e:
        answer = f"Internal error: {str(e)}"

    # ---- GUARANTEED SAFE JSON OUTPUT ----
    try:
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"JSON serialization error: {e}"}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
