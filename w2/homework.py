import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import inspect

def prepare_sets(df_to_prepare, seed = 42, use_mean_for_nulls = False):
    df = df_to_prepare.copy()
    df.median_house_value = np.log1p(df.median_house_value)
    if use_mean_for_nulls:
        df.total_bedrooms = df.total_bedrooms.fillna(housing_train.total_bedrooms.mean())
    else:
        df.total_bedrooms = df.total_bedrooms.fillna(0)

    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
    df_val = df.iloc[idx[n_train:n_train+n_val]].reset_index(drop=True)
    df_test = df.iloc[idx[n_train+n_val:]].reset_index(drop=True)
    
    df_y_train = df_train.median_house_value.values
    df_y_val = df_val.median_house_value.values
    df_y_test = df_test.median_house_value.values

    del df_train['median_house_value']
    del df_val['median_house_value']
    del df_test['median_house_value']

    return df_train, df_y_train, df_val, df_y_val, df_test, df_y_test

def train_linear_regression(X_base, y):
    ones = np.ones(X_base.shape[0])
    X = np.column_stack([ones, X_base])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]

def train_linear_regression_reg(X_base, y, r=0.001):
    ones = np.ones(X_base.shape[0])
    X = np.column_stack([ones, X_base])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]

def make_prediction(X, w0, w):
    return w0 + X.dot(w)

def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
print(script_dir)
housing_df = pd.read_csv(os.path.join(script_dir, 'housing.csv'))
sns.histplot(housing_df.median_house_value, bins=50)
plt.show()

housing_df = housing_df[(housing_df.ocean_proximity == '<1H OCEAN') | (housing_df.ocean_proximity == 'INLAND')]
housing_df = housing_df[['latitude', 'longitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                         'population', 'households', 'median_income', 'median_house_value']]

print('Question 1')
print(housing_df.isnull().sum()[housing_df.isnull().sum() > 0])
print('Question 2')
print(housing_df.population.median())

housing_train, y_train, housing_val, y_val, housing_test, y_test = prepare_sets(housing_df)
w0, w = train_linear_regression(housing_train, y_train)
rmse_0 = round(rmse(y_val, make_prediction(housing_val, w0, w)), 2)

housing_train, y_train, housing_val, y_val, housing_test, y_test = prepare_sets(housing_df, use_mean_for_nulls=True)
w0, w = train_linear_regression(housing_train, y_train)
rmse_mean = round(rmse(y_val, make_prediction(housing_val, w0, w)), 2)
print('Question 3')
print(f'With 0: {rmse_0}')
print(f'With mean {rmse_mean}')

print('Question 4')
housing_train, y_train, housing_val, y_val, housing_test, y_test = prepare_sets(housing_df)
for r in [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    w0, w = train_linear_regression_reg(housing_train, y_train, r)
    rmse_val = round(rmse(y_val, make_prediction(housing_val, w0, w)), 2)
    print(f'{r}: {rmse_val}')

print('Question 5')
rmse_vals = []
for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    housing_train, y_train, housing_val, y_val, housing_test, y_test = prepare_sets(housing_df, seed=seed)
    w0, w = train_linear_regression(housing_train, y_train)
    rmse_vals.append(rmse(y_val, make_prediction(housing_val, w0, w)))

print(np.round(np.std(rmse_vals), 3))

print('Question 6')
housing_train, y_train, housing_val, y_val, housing_test, y_test = prepare_sets(housing_df, seed=9)
housing_full_train = pd.concat([housing_train, housing_val])
housing_full_train = housing_full_train.reset_index(drop=True)
y_full_train = np.concatenate([y_train, y_val])
w0, w = train_linear_regression_reg(housing_full_train, y_full_train, r=0.001)
print(rmse(y_test, make_prediction(housing_test, w0, w)))
