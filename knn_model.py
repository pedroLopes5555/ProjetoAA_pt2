import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor


def submit_knn_model():
    X_train = pd.read_csv('data/X_data.csv')
    y_train = pd.read_csv('data/y_data.csv')
    X_test = pd.read_csv('data/test_data.csv')

    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor(n_neighbors=20))
        ])

    X_test.drop(columns=['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse'], inplace=True)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.insert(0, 'id', range(len(y_pred_df)))

    y_pred_df.to_csv('data/knn-model-01.csv', index=False)

def test_knn_model():
    X_train = pd.read_csv('data/X_data.csv')
    y_train = pd.read_csv('data/y_data.csv')

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    results = []

    for k in range(1, 101):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor(n_neighbors=k))
        ])

        # Cross-validation to calculate the mean squared error
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        mse_scores = -scores
        mean_mse = mse_scores.mean()

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        results.append((k, mse_scores, mse))

    results_df = pd.DataFrame(results, columns=['k', 'Cross-validated MSE', 'Test MSE'])

    stats = results_df['Cross-validated MSE'].apply(lambda x: pd.Series({
        'Max Error': x.max(),
        'Min Error': x.min(),
        'Mean Error': x.mean(),
        'Std Error': x.std()
    }))

    stats['k'] = results_df['k']
    stats = stats[['k', 'Max Error', 'Min Error', 'Mean Error', 'Std Error']]

    print("")
    print(stats)


#test_knn_model()
submit_knn_model()
