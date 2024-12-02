import pandas as pd

def start():
    train_data = pd.read_csv('data/train_data.csv')

    old_name = train_data.columns[0]
    train_data.drop(columns=old_name, inplace=True)

    train_data=delete_missing_columns(train_data)
    train_data=delete_censored_survival_time(train_data)
    train_data.insert(0, 'id', range(0, len(train_data)))

    y_data = create_y_data(train_data)
    X_data = create_X_data(train_data)

    X_data.to_csv('data/X_data.csv', index=False)
    y_data.to_csv('data/y_data.csv', index=False)


def create_y_data(df):
    return df[['SurvivalTime']]

def create_X_data(df):
    return df.drop(columns=['SurvivalTime'], inplace=False)


def delete_missing_columns(df):
    df.drop(columns=['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse', 'Censored'], inplace=True)
    return df


def delete_censored_survival_time(df):
    df = df.dropna(subset=['SurvivalTime'])
    return df


#start()