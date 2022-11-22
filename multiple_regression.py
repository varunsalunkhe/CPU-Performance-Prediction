from sklearn.linear_model import LinearRegression
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

# Linear Regression
lr = LinearRegression()

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
r2_score = lr.score(x_test, y_test)
print("Accuracy1:", r2_score*100, '%')
print("RMSE1: ", sqrt(mean_squared_error(y_test, y_pred)))

# predicting value
new_prediction = lr.predict(np.array([[1100, 768, 2000, 0, 1, 1]]))
print("Prediction performance:", float(new_prediction))


new_prediction = lr.predict(np.array([[143, 512, 5000, 0, 7, 32]]))
print("Prediction performance:", float(new_prediction))

new_prediction = lr.predict(np.array([[64, 5240, 20970, 30, 12, 24]]))
print("Prediction performance:", float(new_prediction))
new_prediction = lr.predict(np.array([[700, 256, 2000, 0, 1, 1]]))
print("Prediction performance:", float(new_prediction))

'''
OUTPUT:
Accuracy1: 73.34625718871324 %
RMSE1:  59.453053748648685
Prediction performance: 26.735479830340026
Prediction performance: 45.48104224752923
Prediction performance: 186.95419502672712
Prediction performance: -4.773525537188831
'''
