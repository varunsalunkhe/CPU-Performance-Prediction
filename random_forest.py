from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

# read dataset
dataset = pd.read_csv('machine.csv')
x = dataset.iloc[:, 2:-2].values
y = dataset.iloc[:, -2].values

# split dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)

# Random Forest
Rr = RandomForestRegressor(n_estimators=50, max_features=None, random_state=0)

Rr.fit(x_train, y_train)

y_pred = Rr.predict(x_test)
r2_score = Rr.score(x_test, y_test)
print("Accuracy1:", r2_score*100, '%')
print("RMSE1: ", sqrt(mean_squared_error(y_test, y_pred)))

# predicting value
new_prediction = Rr.predict(np.array([[1100, 768, 2000, 0, 1, 1]]))
print("Prediction performance:", float(new_prediction))

new_prediction = Rr.predict(np.array([[143, 512, 5000, 0, 7, 32]]))
print("Prediction performance:", float(new_prediction))

new_prediction = Rr.predict((np.array([[64, 5240, 20970, 30, 12, 24]])))
print("Prediction performance:", float(new_prediction))
new_prediction = Rr.predict((np.array([[700, 256, 2000, 0, 1, 1]])))
print("Prediction performance:", float(new_prediction))

'''
OUTPUT:
Accuracy1: 82.3363526342748 %
RMSE1:  48.398874036330604   
Prediction performance: 14.92
Prediction performance: 38.92
Prediction performance: 190.83333333333337
Prediction performance: 22.04666666666667 
'''
