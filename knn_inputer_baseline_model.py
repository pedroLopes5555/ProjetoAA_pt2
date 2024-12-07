import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def cMSE(y, y_hat, c ):
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]

def slip(X,y, c):
    train_ratio = 0.8

    n_samples = len(X)

    split_point = int(n_samples * train_ratio)

    X_train_manual = X[:split_point]
    X_test_manual = X[split_point:]
    y_train_manual = y[:split_point]
    y_test_manual = y[split_point:]
    c_manual = c[split_point:split_point + len(y_test_manual)].reset_index(drop=True)

    return X_train_manual, X_test_manual, y_train_manual, y_test_manual, c_manual

def test_base_model():
    X = pd.read_csv('data/X_data.csv')
    y = pd.read_csv('data/y_data.csv')
    c_full = pd.read_csv('data/c.csv')


    n_values = list(range(1, 31))
    knn_scores = []

    for n in n_values:
        imp_mean = KNNImputer(n_neighbors=n)
        X_train = imp_mean.fit_transform(X)
        y_train = imp_mean.fit_transform(y)

        X_train, X_test, y_train, y_test, c = slip(X_train, y_train, c_full)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        cMSE_error = cMSE(y_test, y_pred, c )
        print(f'k: {n}. cMSE: {cMSE_error}')
        knn_scores.append(cMSE_error)

    plt.plot(n_values, knn_scores, marker='o')
    plt.xlabel('n_Neighbors')
    plt.ylabel('cMSE')
    plt.show()




test_base_model()