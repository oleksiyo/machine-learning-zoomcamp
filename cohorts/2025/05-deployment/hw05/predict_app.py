import pickle
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Scoring Prediction API")

with open('pipeline_v1.bin', 'rb') as f_in:
        pipeline = pickle.load(f_in)

def predict_single(client):
    result = pipeline.predict_proba(client)[0, 1]
    return float(result)


@app.post("/predict")
def predict(client: dict) -> dict:
    prob = predict_single(client)

    return {
        "convert_probability": prob,
        "converted" : prob >= 0.5
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696, log_level="info")


# from fastapi import FastAPI
# import pickle
#
# # 1. Load model on startup
# with open("pipeline_v1.bin", "rb") as f_in:
#     model = pickle.load(f_in)
#
# # 2. Create FastAPI instance
# app = FastAPI()
#
# # 3. Define POST endpoint
# @app.post("/predict")
# def predict(lead: dict):
#     # Predict probability
#     proba = model.predict_proba([lead])[0, 1]
#     return {"conversion_probability": round(float(proba), 3)}