import pandas as pd
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_split(trainData):
    X = trainData.drop(columns=['Sales'])
    y = trainData['Sales']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val




def build_pipeline(X_train, y_train):
    pipeline = Pipeline(steps=[
        ('model', RandomForestRegressor(n_estimators=50, max_depth=2, n_jobs=-1))
    ])

    pipeline.fit(X_train, y_train)
    
    return pipeline



def model_evaluation(pipeline,X_val,y_val):
    y_pred = pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"RMSE: {rmse}")




def post_prediction_analysis(pipeline,X_train):
    # Extract feature importance
    feature_importances = pipeline.named_steps['model'].feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Feature Importance')
    plt.show()
