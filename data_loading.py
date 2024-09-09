from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd


def load_data():
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data["target"] = iris.target
    return data


def preprocess_data(data):
    # Example preprocessing steps
    return data


def split_data(data, target_column="target"):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    data = load_data()
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data)
