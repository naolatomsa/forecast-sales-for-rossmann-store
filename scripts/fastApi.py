from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Sales Forecasting API", description="A FastAPI to serve a trained ML model for sales predictions", version="1.0")

# Load the trained model
MODEL_PATH = "random_forest_model_superstar.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print(f"🎉 Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Define input schema using Pydantic
class PredictionInput(BaseModel):
    # Define the features as required by the model
    Feature1: float
    Feature2: float
    Feature3: float
    Feature4: float
    Feature5: float

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sales Forecasting API!"}

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Convert input data to a format suitable for the model
        features = np.array([[input_data.Feature1, input_data.Feature2, input_data.Feature3, input_data.Feature4, input_data.Feature5]])

        # Make prediction
        prediction = model.predict(features)

        # Return the prediction
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
