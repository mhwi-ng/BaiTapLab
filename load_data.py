import pandas as pd
import numpy as np

def load_data(train_path='fashion-mnist_train.csv', test_path='fashion-mnist_test.csv'):
    # Load training data
    train_data = pd.read_csv(train_path)
    X_train = train_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0  # Normalize
    y_train = train_data.iloc[:, 0].values

    # Load testing data
    test_data = pd.read_csv(test_path)
    X_test = test_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
    y_test = test_data.iloc[:, 0].values

    return (X_train, y_train), (X_test, y_test)
