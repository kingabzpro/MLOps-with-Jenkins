from sklearn.ensemble import RandomForestClassifier
import joblib


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def save_model(model, file_path):
    joblib.dump(model, file_path)


if __name__ == "__main__":
    from data_loading import load_data, preprocess_data, split_data

    data = load_data()
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data)

    model = train_model(X_train, y_train)
    save_model(model, "model.pkl")
