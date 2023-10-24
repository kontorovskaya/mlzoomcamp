import numpy as np
import pandas as pd
import os 
import inspect
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.tree import export_text
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
import xgboost as xgb

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

def prepare_sets(df_to_prepare):
    df = df_to_prepare.copy()
    df.median_house_value = np.log1p(df.median_house_value)
    df = df.fillna(0)   
    return split_sets(df, 'median_house_value')

script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
housing_df = pd.read_csv(os.path.join(script_dir, 'housing.csv'))
print(housing_df.head())
housing_df = housing_df[(housing_df.ocean_proximity == '<1H OCEAN') | (housing_df.ocean_proximity == 'INLAND')]
housing_train, y_train, housing_val, y_val, housing_test, y_test = prepare_sets(housing_df)

print("Question 1")
dv = DictVectorizer(sparse=True)
train_dict = housing_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)
val_dict = housing_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

dt = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train, y_train)
print(export_text(dt, feature_names=list(dv.get_feature_names_out())))

print("Question 2")
rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = sqrt(mse)
accuracy = explained_variance_score(y_val, y_pred)
print(f'accuracy={accuracy}, rmse={rmse}')

print("Question 3")
scores = []
for n in range(10, 201, 10):
    rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.round(sqrt(mse), 3)
    accuracy = np.round(explained_variance_score(y_val, y_pred), 3)
    print(f'n_estimators={n}, accuracy={accuracy}, rmse={rmse}')
    scores.append((n, rmse))

df_scores = pd.DataFrame(scores, columns=['n_estimators', 'rmse'])
plt.plot(df_scores.n_estimators, df_scores.rmse)
plt.show()

print("Question 4")
mean = []
for d in [10, 15, 20, 25]:
    scores = []
    for n in range(10, 201, 10):
        rf = RandomForestRegressor(n_estimators=n, random_state=1, max_depth=d, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.round(sqrt(mse), 3)
        accuracy = np.round(explained_variance_score(y_val, y_pred), 3)
        print(f'd={d}, n_estimators={n}, accuracy={accuracy}, rmse={rmse}')
        scores.append(rmse)
    mean_rmse = np.mean(scores)
    print(f'd={d}, mean rmse={mean_rmse}')
    mean.append((d, mean_rmse))

print(mean)

print("Question 5")
rf = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)

feature_importance = []
for feature, importance in zip(dv.get_feature_names_out(), rf.feature_importances_):
    feature_importance.append((feature, importance))

feature_importance = pd.DataFrame(feature_importance)
print(feature_importance.sort_values(1, ascending=False).iloc[0])


print("Question 6")
features = dv.get_feature_names_out()
features[features == 'ocean_proximity=<1H OCEAN'] = 'ocean_proximity=1H OCEAN'
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features.tolist())
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features.tolist())
xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}
watchlist = [(dtrain, 'train'), (dval, 'val')]
progress = dict()
model = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=watchlist, evals_result=progress, verbose_eval=False)
print(f"eta={xgb_params['eta']}, rmse={progress['val']['rmse'][-1]}")

xgb_params["eta"] = 0.1
watchlist = [(dtrain, 'train'), (dval, 'val')]
progress = dict()
model = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=watchlist, evals_result=progress, verbose_eval=False)
print(f"eta={xgb_params['eta']}, rmse={progress['val']['rmse'][-1]}")


