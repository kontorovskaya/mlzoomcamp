import os 
import inspect
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
data = pd.read_csv(os.path.join(script_dir, 'data.csv'))[['Make', 'Model', 'Year', 'Engine HP', 'Engine Cylinders', 'Transmission Type',
                    'Vehicle Style', 'highway MPG', 'city mpg', 'MSRP']]
data.columns = data.columns.str.replace(' ', '_').str.replace('MSRP', 'price').str.lower()
print(data.head())
print('Question 1')
print(data.transmission_type.value_counts())

print('Question 2')
numerical = ['year', 'engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg']
categorical = ['make', 'model', 'transmission_type', 'vehicle_style']
correlation_matrix = data[numerical].corr()
print(correlation_matrix)

data['above_average'] = (data['price'] > data['price'].mean()).astype(int)
print(data.isnull().sum())
data = data.fillna(0)
print(data.head())

def split_sets(dataForSplit, y_name):
    df_full_train, df_test = train_test_split(dataForSplit, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
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

df_train, y_train, df_val, y_val, df_test, y_test = split_sets(data[numerical + categorical + ['above_average']], 'above_average')

print('Question 3')
def mutual_score(series):
    return mutual_info_score(series, y_train)

mi = df_train[categorical].apply(mutual_score)
print(mi.sort_values(ascending=False))

print('Question 4')
dv = DictVectorizer(sparse=False)
train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)
val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
model.fit(X_train, y_train)
pred_val = model.predict(X_val)
accuracy = (pred_val == y_val).mean()
print(np.round(accuracy, 2))

print('Question 5')
all_features = numerical + categorical
for feature in all_features:
    features = numerical + categorical
    features.remove(feature)
    train_dict_f = df_train[features].to_dict(orient='records')
    X_train_f = dv.fit_transform(train_dict_f)
    val_dict_f = df_val[features].to_dict(orient='records')
    X_val_f = dv.transform(val_dict_f)

    model.fit(X_train_f, y_train)
    pred_val_f = model.predict(X_val_f)
    accuracy_f = (pred_val_f == y_val).mean()
    print(f'{feature}: {accuracy_f}, {accuracy - accuracy_f}')

print('Question 6')
dv = DictVectorizer(sparse=True)
df_train, y_train, df_val, y_val, df_test, y_test = split_sets(data[numerical + categorical + ['price']], 'price')
y_train, y_val, y_test = np.log1p(y_train), np.log1p(y_val), np.log1p(y_test)
train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)
val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

for alpha in [0, 0.01, 0.1, 1, 10]:
    model = Ridge(solver='sag', random_state=42, alpha=alpha)
    model.fit(X_train, y_train)
    pred_val = model.predict(X_val)
    rmse_val = np.round(rmse(y_val, pred_val), 3)
    print(f'{alpha}: {rmse_val}')

