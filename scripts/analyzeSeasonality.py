import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import logging

# # Setup logging
logging.basicConfig(level=logging.INFO, filename='eda.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_seasonality(df):
    """Analyze seasonal purchasing behaviors."""
    logging.info("Analyze seasonal purchasing behaviors.")

    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    seasonal_sales = df.groupby('Month')['Sales'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Month', y='Sales', data=seasonal_sales, hue='Month', palette='coolwarm')
    plt.title("Average Sales Per Month")
    plt.show()