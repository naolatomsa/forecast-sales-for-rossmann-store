import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import logging

# # Setup logging
logging.basicConfig(level=logging.INFO, filename='eda.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def promo_deployment_recommendations(df):
    """Recommend which stores should deploy promos based on sales data."""
    logging.info("Promo Deployment Strategies")

    promo_effectiveness = df.groupby(['Store', 'Promo'])['Sales'].mean().unstack()
    promo_effectiveness['Difference'] = promo_effectiveness[1] - promo_effectiveness[0]
    top_stores = promo_effectiveness.sort_values(by='Difference', ascending=False).head(10)

    logging.info("Top stores for promo deployment:")
    logging.info(top_stores)
    return top_stores