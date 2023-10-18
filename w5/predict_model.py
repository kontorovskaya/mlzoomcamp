import pickle
import os 
import inspect
import numpy as np
import sklearn
import glob

model_name = glob.glob('model*.bin')[0]
script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
model_file = os.path.join(script_dir, model_name)
with open(model_file, 'rb') as f_in: 
    model = pickle.load(f_in)

dv_file = os.path.join(script_dir, 'dv.bin')
with open(dv_file, 'rb') as f_in: 
    dv = pickle.load(f_in)

def predict(client):
    X = dv.transform([client])
    return model.predict_proba(X)[0, 1]

print('Question 1')
print(sklearn.__version__)

print('Question 3') 
client = {"job": "retired", "duration": 445, "poutcome": "success"}
y_pred = predict(client)
print('probability:', np.round(y_pred, 3))