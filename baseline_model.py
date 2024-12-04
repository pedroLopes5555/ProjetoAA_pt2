import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def plot_y_yhat(y_test, y_pred, plot_title="plot"):
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()
    MAX = 500  # Limit the number of points to plot if there are many
    if len(y_test) > MAX:
        idx = np.random.choice(len(y_test), MAX, replace=False)
    else:
        idx = np.arange(len(y_test))

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[idx], y_pred[idx], color='blue', label='Predicted vs Actual')

    # Add the identity line (y = y_hat) for reference
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='y = y_hat')

    # Set labels and title
    plt.xlabel('True Values (Survival Time)')
    plt.ylabel('Predicted Values (Survival Time)')
    plt.title(plot_title)

    # Show the legend
    plt.legend()

    # Show the plot
    plt.show()



def submit_base_model():
    X_train = pd.read_csv('data/X_data.csv')
    y_train = pd.read_csv('data/y_data.csv')
    X_test = pd.read_csv('data/test_data.csv')

    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])

    X_test.drop(columns=['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse'], inplace=True)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.insert(0, 'id', range(len(y_pred_df)))

    y_pred_df.to_csv('data/baseline-model.csv', index=False)


def test_base_model():
    X_train = pd.read_csv('data/X_data.csv')
    y_train = pd.read_csv('data/y_data.csv')

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mse_scores = -scores 
    mean_mse = mse_scores.mean()

    print(f'Cross-validated Mean Squared Error: {mean_mse}')

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    plot_y_yhat(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Test Set Mean Squared Error: {mse}')

    return


def main(option):
    if option == 'test':
        test_base_model()
    if option == 'submit':
        submit_base_model()


if __name__ == "__main__":
    #option = 'submit'
    option = 'test'
    main(option)