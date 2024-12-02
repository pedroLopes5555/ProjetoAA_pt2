import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

# Custom function to compute the gradient of cMSE
def cMSE_derivative(X, y_true, y_pred, mask):
    error = (y_true - y_pred)
    gradient = -2 * np.dot((mask * error), X) / len(y_true)
    return gradient

# Custom gradient descent function
def gradient_descent(X, y, mask, learning_rate=0.01, epochs=1000):
    weights = np.zeros(X.shape[1])  # Initialize weights as zeros
    
    for epoch in range(epochs):
        predictions = X @ weights  # Linear model prediction
        gradient = cMSE_derivative(X, y, predictions, mask)  # Compute gradient
        weights -= learning_rate * gradient  # Update weights using gradient descent
        
        # Compute and print cMSE loss for monitoring
        cMSE_loss = np.mean(mask * (y - predictions) ** 2)
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch}: cMSE Loss = {cMSE_loss}')
    
    return weights  # Return learned weights

# Main function to test and train the model with gradient descent
def test_base_model():
    X_train = pd.read_csv('data/X_data.csv')
    y_train = pd.read_csv('data/y_data.csv').values.flatten()  # Ensure y_train is a 1D array
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create a mask for censored data (assume non-censored data is present)
    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    
    # Apply StandardScaler to the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Run gradient descent on the training data
    weights = gradient_descent(X_train_scaled, y_train, mask_train, learning_rate=0.01, epochs=1000)
    
    # Predict on the test set using learned weights
    y_pred = X_test_scaled @ weights  # Dot product to get predictions
    mse = mean_squared_error(y_test[mask_test], y_pred[mask_test])
    print(f'Test Set Mean Squared Error: {mse}')
    
    # Experiment with Lasso and Ridge regularization
    lasso = Lasso(alpha=0.1)
    ridge = Ridge(alpha=1.0)
    
    lasso.fit(X_train_scaled, y_train)
    ridge.fit(X_train_scaled, y_train)
    
    y_pred_lasso = lasso.predict(X_test_scaled)
    y_pred_ridge = ridge.predict(X_test_scaled)
    
    mse_lasso = mean_squared_error(y_test[mask_test], y_pred_lasso[mask_test])
    mse_ridge = mean_squared_error(y_test[mask_test], y_pred_ridge[mask_test])
    
    print(f'Lasso Test Set Mean Squared Error: {mse_lasso}')
    print(f'Ridge Test Set Mean Squared Error: {mse_ridge}')
    
    # Save predictions to CSV for Kaggle submission
    submission = pd.DataFrame({'Id': np.arange(len(y_pred)), 'Prediction': y_pred})
    submission.to_csv('cMSE-baseline-submission-01.csv', index=False)
    
    return

def submit_base_model():
    # Load data
    X_train = pd.read_csv('data/X_data.csv')
    y_train = pd.read_csv('data/y_data.csv').values.flatten()  # Ensure y_train is a 1D array
    X_test = pd.read_csv('data/test_data.csv')
    
    # Preprocess the test data
    X_test.drop(columns=['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse'], inplace=True)
    
    # Apply StandardScaler to the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create a mask for censored data (assume non-censored data is present)
    mask_train = ~np.isnan(y_train)
    
    # Run gradient descent on the training data
    weights = gradient_descent(X_train_scaled, y_train, mask_train, learning_rate=0.01, epochs=1000)
    
    # Predict on the test set using learned weights
    y_pred_gradient = X_test_scaled @ weights  # Dot product to get predictions
    
    y_pred_df = pd.DataFrame(y_pred_gradient)
    y_pred_df.insert(0, 'id', range(len(y_pred_df)))

    # Save gradient descent predictions to CSV
    y_pred_df.to_csv('data/baseline-gradient-model.csv', index=False)
    
    # Fit Lasso and Ridge models for comparison
    lasso = Lasso(alpha=0.1)
    ridge = Ridge(alpha=1.0)
    
    lasso.fit(X_train_scaled, y_train)
    ridge.fit(X_train_scaled, y_train)
    
    y_pred_lasso = lasso.predict(X_test_scaled)
    y_pred_ridge = ridge.predict(X_test_scaled)

    y_pred_lasso_df = pd.DataFrame(y_pred_lasso)
    y_pred_lasso_df.insert(0, 'id', range(len(y_pred_lasso_df)))

    y_pred_ridge_df = pd.DataFrame(y_pred_ridge)
    y_pred_ridge_df.insert(0, 'id', range(len(y_pred_ridge_df)))
    
    y_pred_lasso_df.to_csv('data/baseline-lasso-model.csv', index=False)
    y_pred_ridge_df.to_csv('data/baseline-ridge-model.csv', index=False)
    
    # Print MSE for each model for comparison
    print("Mean Squared Error for Gradient Descent:", mean_squared_error(y_train[mask_train], X_train_scaled @ weights))
    print("Mean Squared Error for Lasso:", mean_squared_error(y_train, y_pred_lasso))
    print("Mean Squared Error for Ridge:", mean_squared_error(y_train, y_pred_ridge))
    
    return

# Run the test
#test_base_model()

# Run the function
submit_base_model()