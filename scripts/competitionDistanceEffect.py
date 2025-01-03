import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import logging

# # Setup logging
logging.basicConfig(level=logging.INFO, filename='eda.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')



def competition_distance_effect(df):
    logging.info("Competition Distance Effect")

    """Analyze how competition distance affects sales."""
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=df, alpha=0.6)
    plt.title("Effect of Competition Distance on Sales")
    plt.xlabel("Competition Distance")
    plt.ylabel("Sales")
    plt.show()