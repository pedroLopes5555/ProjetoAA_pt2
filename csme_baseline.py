import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

def censored_mean_squared_error(y_true, y_predicted, indicator):
    cost = np.sum(indicator * (y_true - y_predicted)*2) / np.sum(indicator)
    return cost

def gradient_descent_censored(X, y, indicator, iterations=1000, learning_rate=0.0001, stopping_threshold=1e-6, regularization=None, reg_param=0.1):
    num_features = X.shape[1]
    current_weights = np.zeros(num_features)
    current_bias = 0.0
    n = np.sum(indicator)

    costs = []
    previous_cost = None

    for i in range(iterations):
        y_predicted = np.dot(X, current_weights) + current_bias
        current_cost = censored_mean_squared_error(y, y_predicted, indicator)

        if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
            break

        previous_cost = current_cost
        costs.append(current_cost)

        weight_derivative = -(2 / n) * np.dot(indicator * (y - y_predicted), X)
        bias_derivative = -(2 / n) * np.sum(indicator * (y - y_predicted))

        if regularization == "lasso":
            weight_derivative += reg_param * np.sign(current_weights)
        elif regularization == "ridge":
            weight_derivative += 2 * reg_param * current_weights

        current_weights -= learning_rate * weight_derivative
        current_bias -= learning_rate * bias_derivative

        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}: Cost {current_cost}, Weights {current_weights}, Bias {current_bias}")

    return current_weights, current_bias

def submit_base_model():
    X_train = pd.read_csv('data/X_data.csv')
    y_train = pd.read_csv('data/y_data.csv')
    X_test = pd.read_csv('data/test_data.csv')

    X_train.drop(columns=['id'], inplace=True)
    X_test.drop(columns=['id', 'GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse'], inplace=True)

    X_train = X_train.values
    y_train = y_train.values.ravel()
    X_test = X_test.values

    indicator = np.ones(X_train.shape[0], dtype=int)

    weight, bias = gradient_descent_censored(X_train, y_train, indicator, regularization="lasso", reg_param=0.1)
    y_pred = np.dot(X_test, weight) + bias

    y_pred_df = pd.DataFrame(y_pred, columns=['SurvivalTime'])
    y_pred_df.insert(0, 'id', range(len(y_pred_df)))
    y_pred_df.to_csv('data/cMSE-baseline-submission-03.csv', index=False)

    return

submit_base_model()