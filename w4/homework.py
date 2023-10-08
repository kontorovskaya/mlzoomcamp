import os 
import inspect
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

def split_sets(dataForSplit, y_name):
    df_full_train, df_test = train_test_split(dataForSplit, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    y_train = df_train[y_name].values
    y_val = df_val[y_name].values
    y_test = df_test[y_name].values
    del df_train[y_name]
    del df_val[y_name]
    del df_test[y_name]
    return df_train, y_train, df_val, y_val, df_test, y_test

script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
data = pd.read_csv(os.path.join(script_dir, 'data.csv'))[['Make', 'Model', 'Year', 'Engine HP', 'Engine Cylinders', 'Transmission Type',
                    'Vehicle Style', 'highway MPG', 'city mpg', 'MSRP']]
data.columns = data.columns.str.replace(' ', '_').str.replace('MSRP', 'price').str.lower()
data = data.fillna(0)
data['above_average'] = (data['price'] > data['price'].mean()).astype(int)
numerical = ['year', 'engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg']
categorical = ['make', 'model', 'transmission_type', 'vehicle_style']

print('Question 1') 

for num_variable in numerical:
    score = roc_auc_score(data['above_average'], data[num_variable])
    if score < 0.5:
        score = roc_auc_score(data['above_average'], -data[num_variable])
    print(f'AUC {num_variable}: {score}')

print('Question 2')

df_train, y_train, df_val, y_val, df_test, y_test = split_sets(data[numerical + categorical + ['above_average']], 'above_average')
dv = DictVectorizer(sparse=False)
train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)
val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_val)[:, 1]
score = roc_auc_score(y_val, y_pred)
print(f'AUC predicted: {np.round(score, 3)}')

print('Question 3')

thresholds = np.linspace(0, 1, 101)
print(thresholds)
precision = []
recall = []

actual_positive = (y_val == 1)
actual_negative = (y_val == 0)

for t in thresholds:
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)
    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()
    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    p = tp / (tp + fp)
    precision.append(p)
    r = tp / (tp + fn)
    recall.append(r)

plt.plot(thresholds, precision, label='precision')
plt.plot(thresholds, recall, label='recall')
plt.xlabel('thresholds')
plt.show()

print('Question 4')

precision = pd.array(precision)
precision = precision.fillna(1)
recall = pd.array(recall)
thresholds = pd.array(thresholds)
f1 = 2 * (precision * recall) / (precision + recall)
print(f'F1 max = {f1.max()}, threshold = {thresholds[f1 == f1.max()][0]}')

print('Question 5')
print('Question 6')

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)  
    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
scores = []
df_full_train, df_test = train_test_split(data[numerical + categorical + ['above_average']], test_size=0.2, random_state=1)

for C in tqdm([1.0, 0.01, 0.1, 0.5, 10]):                     
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.above_average.values
        y_val = df_val.above_average.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

