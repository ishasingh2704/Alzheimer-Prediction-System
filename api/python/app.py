from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    # Placeholder: Replace with actual model inference
    data = request.json
    # Example: {'prediction': 'Alzheimer', 'confidence': 0.95}
    return jsonify({'prediction': 'Alzheimer', 'confidence': 0.95})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
