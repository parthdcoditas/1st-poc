from dotenv import load_dotenv
from flask import Flask, render_template, request, session
from flask_session import Session
from helper_functions import cache_search
import os
load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")  
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route("/", methods=["GET", "POST"])
def chat():
    if "chat_history" not in session:
        session["chat_history"] = []  

    if request.method == "POST":
        user_query = request.form["user_query"]
        response = cache_search(user_query)

        # Append the new query and response to chat history
        session["chat_history"].append({"query": user_query, "response": response})
        session.modified = True  # Mark session as modified to save changes

    return render_template("index.html", chat_history=session.get("chat_history", []))

if __name__ == "__main__":
    app.run(debug=True)
