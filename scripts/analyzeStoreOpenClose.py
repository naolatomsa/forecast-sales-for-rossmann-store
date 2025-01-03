import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import logging

# # Setup logging
logging.basicConfig(level=logging.INFO, filename='eda.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')


def analyze_store_open_close(df):
    logging.info("Trends During Store Open/Close")

    """Analyze customer behavior during store opening and closing times."""
    open_sales = df.groupby('Open')['Sales'].mean().reset_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Open', y='Sales', data=open_sales, hue='Open', palette='coolwarm')
    plt.title("Average Sales When Store Is Open vs Closed")
    plt.show()