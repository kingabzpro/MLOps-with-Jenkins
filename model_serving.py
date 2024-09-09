import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()


class Features(BaseModel):
    features: List[float]


def load_model(file_path):
    return joblib.load(file_path)


model = load_model("model.pkl")


@app.post("/predict")
def predict(features: Features):
    prediction = model.predict([features.features])
    return {"prediction": prediction.tolist()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
