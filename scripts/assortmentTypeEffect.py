import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import logging

# # Setup logging
logging.basicConfig(level=logging.INFO, filename='eda.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')


def assortment_type_effect(df):
    """Check how assortment type affects sales."""
    logging.info("Assortment Type and Sales")

    assortment_sales = df.groupby('Assortment')['Sales'].mean().reset_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Assortment', y='Sales', data=assortment_sales,hue='Assortment', palette='coolwarm')
    plt.title("Effect of Assortment Type on Sales")
    plt.show()