import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import logging

# # Setup logging
logging.basicConfig(level=logging.INFO, filename='eda.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def check_missing_data(df):
    """Check for missing values in the dataset."""
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    logging.info(f"Missing data: {missing_data}")
    return missing_data



def handle_missing_values_for_store(df):
    """Handle missing values and prepare the dataset."""
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
    df['PromoInterval'] = df['PromoInterval'].fillna('None')
    return df

def handle_missing_values_for_train(df):
    """Handle missing values in the dataset."""
    logging.info("Handling missing values for train")
    df.dropna(subset=['Open'], inplace=True)
    return df



def detect_outliers(data, columns):
    outliers = {}
    
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers[column] = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data[column], color='skyblue')
        plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
        plt.axvline(upper_bound, color='red', linestyle='--', label='Upper Bound')
        plt.title(f"Box Plot for {column} with Outlier Bounds")
        plt.xlabel(column)
        plt.legend()
        plt.show()
        
    logging.info("Detecting outliers")
    return outliers

# Cap and floor outliers for all numeric columns
def cap_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    logging.info("Detecting outliers")
    return data

# Feature Engineering
def add_date_features(df):
    """Add date-related features."""
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Day'] = pd.to_datetime(df['Date']).dt.day
    df['WeekOfYear'] = pd.to_datetime(df['Date']).dt.isocalendar().week
    logging.info("Feature Engineering")
    return df