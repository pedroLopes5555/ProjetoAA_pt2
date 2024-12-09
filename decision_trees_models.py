import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def test_HistGradientBoostingRegressor():
    X = pd.read_csv('data/X_data.csv')
    y = pd.read_csv('data/y_data.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    hgb_model = HistGradientBoostingRegressor(random_state=42)

    hgb_model.fit(X_train, y_train)

    y_pred_hgb = hgb_model.predict(X_test)

    rmse_hgb = np.sqrt(mean_squared_error(y_test, y_pred_hgb))
    print(f"HistGradientBoostingRegressor RMSE: {rmse_hgb}")

def submit_HistGradientBoostingRegressor():
    X = pd.read_csv('data/X_data.csv')
    y = pd.read_csv('data/y_data.csv')
    X_test = pd.read_csv('data/test_data.csv')

    # Initialize the model
    hgb_model = HistGradientBoostingRegressor(random_state=42)

    # Train the model
    hgb_model.fit(X, y)

    # Make predictions
    y_pred = hgb_model.predict(X_test)

    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.insert(0, 'id', range(len(y_pred_df)))

    y_pred_df.to_csv('data/handle-missing-submission-01.csv', index=False)

def test_CatBoostRegressor():
    X = pd.read_csv('data/X_data.csv')
    y = pd.read_csv('data/y_data.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize the CatBoost model
    catboost_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',
        verbose=100,
        random_state=42
    )

    # Train the model
    catboost_model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

    # Make predictions
    y_pred_catboost = catboost_model.predict(X_test)

    # Calculate RMSE
    rmse_catboost = np.sqrt(mean_squared_error(y_test, y_pred_catboost))
    print(f"CatBoostRegressor RMSE: {rmse_catboost}")

def submit_CatBoostRegressor():
    X = pd.read_csv('data/X_data.csv')
    y = pd.read_csv('data/y_data.csv')

    X_test = pd.read_csv('data/test_data.csv')
    catboost_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',
        verbose=100,
        random_state=42
    )

    catboost_model.fit(X, y, eval_set=(X, y), use_best_model=True)

    y_pred = catboost_model.predict(X_test)

    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.insert(0, 'id', range(len(y_pred_df)))

    y_pred_df.to_csv('data/handle-missing-submission-01.csv', index=False)


submit_HistGradientBoostingRegressor()