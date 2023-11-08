import pickle
import os 
import inspect
import numpy as np
import sklearn
import glob
from flask import Flask
from flask import request
from flask import jsonify

model_name = glob.glob('model*.bin')[0]
script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
model_file = os.path.join(script_dir, model_name)
with open(model_file, 'rb') as f_in: 
    model = pickle.load(f_in)


def predict(X):
    return model.predict(X)[0]

app = Flask('smoking')
@app.route('/predict', methods=['POST'])
def ping():
    patient = request.get_json()
    y_pred = predict(patient)
    result = {
        'probability': y_pred,
        'smoking': bool(y_pred >= 0.5)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9797)