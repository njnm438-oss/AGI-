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
    try:
        data = request.get_json()
        user_msg = data.get("message", "").strip() if data else ""
    except Exception:
        return jsonify({"answer": "Invalid request format"}), 400

    if not user_msg:
        return jsonify({"answer": "⚠️ Message vide."}), 400

    try:
        answer = agent.ask(user_msg)
        # Ensure answer is always a valid non-empty string
        if not isinstance(answer, str):
            answer = str(answer) if answer else "..."
        if len(answer.strip()) == 0:
            answer = "..."  # fallback safe value
    except Exception as e:
        answer = f"Erreur interne: {str(e)[:100]}"

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
