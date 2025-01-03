import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import logging

# # Setup logging
logging.basicConfig(level=logging.INFO, filename='eda.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def promo_effect_on_sales(df):
    """Analyze the effect of promos on sales."""
    logging.info("effect of promos on sales.")

    promo_sales = df.groupby('Promo')['Sales'].mean().reset_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Promo', y='Sales', data=promo_sales, hue='Promo', palette='viridis')
    plt.title("Effect of Promo on Sales")
    plt.show()
