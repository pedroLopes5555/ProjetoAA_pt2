import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def cMSE(y, y_hat, c = 1):
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]


def gradient_descent_censored(x, y, iterations=100000, learning_rate=0.0001, stopping_threshold=1e-6, regularization = "ridge"):
    current_weight = np.ones(x.shape[1]) * 0.1
    current_bias = 0.01
    lambda_ = 0.0001

    previous_cost = None
    n = float(len(x))

    for i in range(iterations):

        y_predicted = (x @ current_weight) + current_bias

        current_cost = cMSE(y, y_predicted)

        if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
            break

        previous_cost = current_cost

        weight_derivative = None


        if regularization is None:
            weight_derivative = -(2/n) * (x.T @ (y - y_predicted))
        if regularization == "lasso":
            weight_derivative = -(2 / n) * (x.T @ (y - y_predicted)) + (lambda_ * current_weight)
        if regularization == "ridge":
            weight_derivative = -(2 / n) * (x.T @ (y - y_predicted)) + (lambda_ * (current_weight ** 2))


        bias_derivative = -(2/n) * sum(y-y_predicted)

        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)

        print(f"Iteration {i + 1}: Cost {current_cost}")

    return current_weight, current_bias



def submit_base_model():
    X_train = pd.read_csv('data/X_data.csv')
    y_train = pd.read_csv('data/y_data.csv')
    X_test = pd.read_csv('data/test_data.csv')

    X_train.drop(columns=['id'], inplace=True)
    X_test.drop(columns=['id', 'GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse'], inplace=True)

    X_train = X_train.values
    y_train = y_train.values.ravel()
    X_test = X_test.values

    weight, bias = gradient_descent_censored(X_train, y_train, regularization="lasso")
    y_pred = np.dot(X_test, weight) + bias

    y_pred_df = pd.DataFrame(y_pred, columns=['SurvivalTime'])
    y_pred_df.insert(0, 'id', range(len(y_pred_df)))
    y_pred_df.to_csv('data/cMSE-baseline-submission-05.csv', index=False)

    return

submit_base_model()