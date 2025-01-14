import joblib
import datetime

from sklearn import pipeline

# Save the trained pipeline
def serialize_model(pipeline, prefix="random_forest_model"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_filename = f"{prefix}_{timestamp}_superstar.pkl"
    joblib.dump(pipeline, model_filename)


    return model_filename