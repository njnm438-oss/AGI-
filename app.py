from flask import Flask, render_template, request, jsonify
import logging

from agi.agent import AGIAgent

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# Create a single agent instance for the server lifetime
agent = AGIAgent()


@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.get_json(force=True)
        user = data.get('user', '')
        if not isinstance(user, str) or not user.strip():
            return jsonify(ok=False, error='empty input'), 400
        resp = agent.ask(user)
        status = agent.introspect()
        return jsonify(ok=True, response=resp, status=status)
    except Exception as e:
        logging.exception('api_chat error')
        return jsonify(ok=False, error=str(e)), 500


@app.route('/api/health')
def health():
    return jsonify(ok=True)


if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=5000, debug=False)
    finally:
        try:
            agent.shutdown()
        except Exception:
            pass
