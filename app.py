from flask import Flask, render_template, request, jsonify
from chat import get_response  # Ensure 'get_response' is correctly implemented

app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    data = request.get_json()
    print("Received data:", data)  # Debugging input

    text = data.get("messages")  # Ensure the key matches the frontend

    if not text:
        return jsonify({"error": "No input provided"}), 400

    response = get_response(text)
    print("Chatbot response:", response)  # Debugging chatbot output

    if not response:
        return jsonify({"error": "No response generated"}), 500

    return jsonify({"answer": response})


if __name__ == "__main__":
    app.run(debug=True)
