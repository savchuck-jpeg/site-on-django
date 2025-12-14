import asyncio
from flask import Flask, jsonify, request
import requests
from model import MoexModel
app = Flask(__name__)

MODEL_API_URL = "http://flask_moex_api:5000/get_ticker_data_28"

model = MoexModel(training_mode=False)

@app.route('/predict', methods=['GET'])
def predict():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Тикер не указан"}), 400

    response = requests.get(f"{MODEL_API_URL}?ticker={ticker}")
    
    if response.status_code != 200:
        return jsonify({"error": "Ошибка получения данных из модели"}), response.status_code
    
    data = response.json()
    next_data = model.predict_next_values(data)
    return jsonify(next_data)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)