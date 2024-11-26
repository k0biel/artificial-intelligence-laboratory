import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
x_train_bias = np.c_[np.ones((len(x_train), 1)), x_train]
x_test_bias = np.c_[np.ones((len(x_test), 1)), x_test]
theta_best = np.linalg.inv(x_train_bias.T.dot(x_train_bias)).dot(x_train_bias.T).dot(y_train)

# TODO: calculate error
MSE_train = np.mean((x_train_bias.dot(theta_best) - y_train) ** 2)
print("MSE_train:", MSE_train)
MSE_test = np.mean((x_test_bias.dot(theta_best) - y_test) ** 2)
print("MSE_test:", MSE_test)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
x_train_standardized = (x_train - np.mean(y_train))/np.std(x_train)
x_test_standardized = (x_test - np.mean(y_train))/np.std(x_train)
y_train_standardized = (y_train - np.mean(y_train))/np.std(y_train)
y_test_standardized = (y_test - np.mean(y_train))/np.std(y_train)

# TODO: calculate theta using Batch Gradient Descent
x_train_standardized_bias = np.c_[np.ones((len(x_train_standardized), 1)), x_train_standardized]

lr = 0.01
n_iterations = 10000
theta_bgd = np.zeros(2)

for iteration in range(n_iterations):
    gradients = 2 / len(x_train_standardized_bias) * x_train_standardized_bias.T.dot((x_train_standardized_bias.dot(theta_bgd)) - y_train_standardized)
    theta_bgd = theta_bgd - lr * gradients
print("theta_bgd:", theta_bgd)

# TODO: calculate error
x_test_standardized_bias = np.c_[np.ones_like(x_test_standardized), x_test_standardized]
MSE_bgd = np.mean((y_test_standardized - x_test_standardized_bias.dot(theta_bgd)) ** 2)
print("MSE_bgd:", MSE_bgd)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()