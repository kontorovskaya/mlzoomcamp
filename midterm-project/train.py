from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
import os 
import inspect
import pickle

def split_dataset(df, y_name, features = []):
    if features:
        dataForSplit = df[features + [y_name]]
    else:
        dataForSplit = df
        
    df_train, df_val = train_test_split(dataForSplit, test_size=0.2, random_state=1)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    
    y_train = df_train[y_name].values
    y_val = df_val[y_name].values
    del df_train[y_name]
    del df_val[y_name]
    
    return df_train, y_train, df_val, y_val

script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
smoking_train_df = pd.read_csv(os.path.join(script_dir, 'data/train.csv'))
df_train, y_train, df_val, y_val = split_dataset(smoking_train_df, 'smoking')

best_d = 10
best_n = 110

rf = RandomForestRegressor(n_estimators=best_n, random_state=1, max_depth=best_d, n_jobs=-1)
rf.fit(df_train, y_train)

with open('model.bin', 'wb') as f_out:
    pickle.dump(rf, f_out)