from dotenv import load_dotenv
from flask import Flask, render_template, request
from helper_functions import cache_search

load_dotenv(override=True)
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def chat():
    user_query = ""
    response = ""
    if request.method == "POST":
        user_query = request.form["user_query"]
        response = cache_search(user_query)
    return render_template("index.html", user_query=user_query, response=response)

if __name__ == "__main__":
    app.run(debug=True)
