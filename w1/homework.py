import pandas as pd
import numpy as np

print('Question 1')
print(pd.__version__)
housing = pd.read_csv('housing.csv')
print('Question 2')
print(housing.columns.size)
print('Question 3')
print(housing.isnull().sum()[housing.isnull().sum() > 0])
print('Question 4')
print(housing['ocean_proximity'].nunique())
print('Question 5')
print(housing['median_house_value'].mean())
print('Question 6')
avr_total_bedrooms = housing['total_bedrooms'].mean().__round__(3)
print(avr_total_bedrooms)
housing['total_bedrooms'] = housing['total_bedrooms'].fillna(avr_total_bedrooms)
avr_total_bedrooms_updated = housing['total_bedrooms'].mean().__round__(3)
print(avr_total_bedrooms_updated)
print(avr_total_bedrooms != avr_total_bedrooms_updated)
print('Question 7')
X = housing[housing['ocean_proximity'] == 'ISLAND'][['housing_median_age', 'total_rooms', 'total_bedrooms']].values
XTX = X.T.dot(X)
inverse_XTX = np.linalg.inv(XTX)
y = [950, 1300, 800, 1000, 1300]
w = inverse_XTX.dot(X.T).dot(y)
print(w[-1])
