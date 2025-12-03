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
    data = request.get_json()
    user_msg = data.get("message", "").strip()

    if not user_msg:
        return jsonify({"answer": "⚠️ Message vide."})

    # Generate answer
    ai_answer = agent.ask(user_msg)

    # Return clean JSON response
    return jsonify({"answer": ai_answer})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
