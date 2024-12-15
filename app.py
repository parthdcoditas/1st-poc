from flask import Flask, render_template, request, jsonify
from helper_functions import cache_search, get_context_and_generate_response,store_query_and_response
from embedding_generate import embeddings_model
import numpy as np
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
            user_query_embedding = embeddings_model.embed_query(session_state["last_query"])
            embedding_list = np.array(user_query_embedding).tolist()

            regenerated_response = get_context_and_generate_response(
                session_state["last_query"], embedding_list, session_state["cached_query"]
            )

            store_query_and_response(
                session_state["last_query"], embedding_list, regenerated_response, session_state["cached_query"]
            )

            session_state["waiting_for_feedback"] = False
            return jsonify({"response": regenerated_response})

    search_result = cache_search(user_query)
    if search_result["cached"]:
        session_state["waiting_for_feedback"] = True
        session_state["last_query"] = user_query
        session_state["cached_query"] = user_query
        return jsonify({
            "response": search_result["response"],
            "cached": True,
            "follow_up": "Are you satisfied with the response? (yes/no)"
        })
    else:
        session_state["last_query"] = user_query
        session_state["cached_query"] = None
        return jsonify({"response": search_result["response"], "cached": False})

if __name__ == "__main__":
    app.run(debug=True)