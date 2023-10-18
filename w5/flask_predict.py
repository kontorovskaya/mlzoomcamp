from flask import Flask
from flask import request
from flask import jsonify
import pickle
import os 
import inspect
from predict_model import predict

app = Flask('bank')
@app.route('/predict', methods=['POST'])
def ping():
    client = request.get_json()
    y_pred = predict(client)
    result = {
        'probability': y_pred,
        'credit': bool(y_pred >= 0.5)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)