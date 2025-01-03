import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import logging

# # Setup logging
logging.basicConfig(level=logging.INFO, filename='eda.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def correlation_sales_customers(df):
    """Calculate and visualize the correlation between sales and customers."""
    logging.info("correlation between sales and customers.")

    correlation = df[['Sales', 'Customers']].corr()
    logging.info("Correlation Between Sales and Customers:")
    logging.info(correlation)

    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title("Correlation Between Sales and Customers")
    plt.show()