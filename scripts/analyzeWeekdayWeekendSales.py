import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import logging

# # Setup logging
logging.basicConfig(level=logging.INFO, filename='eda.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_weekday_weekend_sales(df):

    """Analyze sales trends for stores open all weekdays and their weekend performance."""
    weekday_open_stores = df[df['DayOfWeek'].isin([1, 2, 3, 4, 5]) & (df['Open'] == 1)]['Store'].unique()
    weekend_sales = df[(df['Store'].isin(weekday_open_stores)) & (df['DayOfWeek'].isin([6, 7]))]

    weekend_avg_sales = weekend_sales.groupby('Store')['Sales'].mean().reset_index()
    logging.info("Weekend sales for stores open on weekdays:")
    logging.info(weekend_avg_sales)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Store', y='Sales', data=weekend_avg_sales, hue='Store', palette='coolwarm')
    plt.title("Average Weekend Sales for Weekday-Open Stores")
    plt.show()