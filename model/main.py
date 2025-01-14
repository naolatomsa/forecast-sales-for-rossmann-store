from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model  # type: ignore # For deep learning model
# import os

# Get the path to the models folder
# os.chdir('..')

# Load the models using the relative paths
machine_learning_model = joblib.load('./random_forest_model_2025-01-14-15-00-24_superstar.pkl')
deep_learning_model = load_model("./lstm_model.h5", custom_objects={'mse': 'mean_squared_error'})    # Replace with your deep learning model path

# Define the feature names
FEATURES = [
    'Store',
    'DayOfWeek',
    'Open',
    'Promo',
    'SchoolHoliday',
    'CompetitionDistance',
    'Promo2',
    'Promo2SinceWeek',
    'Promo2SinceYear',
    'Year',
    'Month',
    'Day',
    'WeekOfYear',
    'CompetitionOpenSince',
    'Promo2ActiveMonths',
    'StateHoliday_0',
    'StateHoliday_a',
    'StoreType_b',
    'StoreType_c',
    'StoreType_d',
    'Assortment_b',
    'Assortment_c',
    'PromoInterval_Jan_Apr_Jul_Oct',  # Updated name to match the attribute
    'PromoInterval_Mar_Jun_Sept_Dec',  # Updated name to match the attribute
    'PromoInterval_None'
]


# Create FastAPI app
app = FastAPI()

# FastAPI App
app = FastAPI()

# CORS settings
origins = [ # Localhost without a port
    "*",  # Localhost with a port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow frontend origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, OPTIONS)
    allow_headers=["*"],  # Allow all headers
)

# Define the request body schema
class PredictionRequest(BaseModel):
    Store: int
    DayOfWeek: int
    Open: int
    Promo: int
    SchoolHoliday: int
    CompetitionDistance: float
    Promo2: int
    Promo2SinceWeek: int
    Promo2SinceYear: int
    Year: int
    Month: int
    Day: int
    WeekOfYear: int
    CompetitionOpenSince: int
    Promo2ActiveMonths: int
    StateHoliday_0: int
    StateHoliday_a: int
    StoreType_b: int
    StoreType_c: int
    StoreType_d: int
    Assortment_b: int
    Assortment_c: int
    PromoInterval_Jan_Apr_Jul_Oct: int  # Updated name to match the FEATURES list
    PromoInterval_Mar_Jun_Sept_Dec: int  # Updated name to match the FEATURES list
    PromoInterval_None: int
    model_type: str 


# Define the prediction endpoint
@app.post("/predict/")
async def predict(request: PredictionRequest):
    # Validate model type
    if request.model_type not in ["ml", "dl"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose 'ml' or 'dl'.")
    
    # Convert the incoming request data into a numpy array
    input_data = np.array([[getattr(request, feature) for feature in FEATURES]])

    if request.model_type == "ml":
        # Predict using the machine learning model
        prediction = machine_learning_model.predict(input_data)
    else:
        # Reshape input_data to match the model's expected shape (1, 1, 25)
        input_data = input_data.reshape((input_data.shape[0], 1, input_data.shape[1]))  # Reshaped to (1, 1, 25)
        
        # Predict using the deep learning model
        prediction = deep_learning_model.predict(input_data)
        
        # Flatten the prediction if necessary
        prediction = prediction.flatten()

        # Return the prediction


    # Return the prediction
    return {"model_type": request.model_type, "prediction": prediction.tolist()}






