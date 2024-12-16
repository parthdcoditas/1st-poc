from flask import Flask, render_template, request, jsonify
from helper_functions import compiled_app, QueryState

app = Flask(__name__)

session_state = {
    "waiting_for_feedback": False,
    "last_query": None,
    "cached_query": None
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get('query', '').strip()
    if not user_query:
        return jsonify({"error": "Query cannot be empty."}), 400

    if session_state["waiting_for_feedback"]:
        feedback = user_query.lower()
        if feedback not in ['yes', 'no']:
            return jsonify({"response": "Please answer with 'yes' or 'no'."})

        if feedback == 'yes':
            session_state["waiting_for_feedback"] = False
            session_state["last_query"] = None
            session_state["cached_query"] = None
            return jsonify({"response": "Glad you are satisfied! You can enter a new query now."})

        elif feedback == 'no':
            state_dict = compiled_app.invoke({"user_query": session_state["last_query"], "force_llm_call": True})
            state = QueryState(**state_dict)  
            session_state["waiting_for_feedback"] = False  
            return jsonify({"response": state.response}) 

    state_dict = compiled_app.invoke({"user_query": user_query})
    state = QueryState(**state_dict)  
    
    if state.cached:
        session_state["waiting_for_feedback"] = True
        session_state["last_query"] = user_query
        session_state["cached_query"] = user_query
        return jsonify({
            "response": state.response,
            "cached": True,
            "follow_up": "Are you satisfied with the response? (yes/no)"
        })
    else:
        session_state["last_query"] = user_query
        session_state["cached_query"] = None
        return jsonify({"response": state.response, "cached": False})

if __name__ == "__main__":
    app.run(debug=True)
