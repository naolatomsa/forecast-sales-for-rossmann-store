from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_columns(trainData, testData, numeric_columns):

    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit the scaler on the training data for the specified columns
    trainData[numeric_columns] = scaler.fit_transform(trainData[numeric_columns])
    
    # Transform the test data using the same scaler
    testData[numeric_columns] = scaler.transform(testData[numeric_columns])
    
    return trainData, testData
