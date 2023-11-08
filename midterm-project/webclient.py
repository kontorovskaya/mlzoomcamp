import requests
import os 
import inspect
import pandas as pd
import random

script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
smoking_test_df = pd.read_csv(os.path.join(script_dir, 'data/test.csv'))

patientNumber = random.randint(0, smoking_test_df.shape[0])

patient = smoking_test_df.iloc[patientNumber]
print(f'patient: {patient}')
r = requests.post('http://0.0.0.0:9797/predict', json=[patient.to_list()])
print(r.content)