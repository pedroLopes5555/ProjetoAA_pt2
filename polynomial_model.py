import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def submit_polynomial_model():
    X_train = pd.read_csv('data/X_data.csv')
    y_train = pd.read_csv('data/y_data.csv')
    X_test = pd.read_csv('data/test_data.csv')

    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
            ('regressor', LinearRegression())
        ])

    X_test.drop(columns=['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse'], inplace=True)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.insert(0, 'id', range(len(y_pred_df)))

    y_pred_df.to_csv('data/polynomial-submission-02.csv', index=False)

def test_polynomial_model():
    X_train = pd.read_csv('data/X_data.csv')
    y_train = pd.read_csv('data/y_data.csv')

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    results = []

    for degree in range(1, 11):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),
            ('regressor', LinearRegression())
        ])

        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        mse_scores = -scores
        mean_mse = mse_scores.mean()

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        results.append((degree, mean_mse, mse))

        print(f'Degree {degree}: Cross-validated Mean Squared Error: {mean_mse}, Test Set Mean Squared Error: {mse}')

    return


#test_polynomial_model()
#submit_polynomial_model()
