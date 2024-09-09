import joblib
from sklearn.metrics import accuracy_score, classification_report


def load_model(file_path):
    return joblib.load(file_path)


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report


if __name__ == "__main__":
    from data_loading import load_data, preprocess_data, split_data

    data = load_data()
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data)

    model = load_model("model.pkl")
    accuracy, report = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
