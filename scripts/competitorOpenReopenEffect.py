import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import logging

# # Setup logging
logging.basicConfig(level=logging.INFO, filename='eda.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')


def competitor_open_reopen_effect(df):
    logging.info("Competitor Open/Reopen Impact")

    """Analyze the impact of competitor openings or reopenings on sales."""
    df['CompetitorOpen'] = pd.to_datetime(
        df['CompetitionOpenSinceYear'].fillna(0).astype(int).astype(str) + '-' +
        df['CompetitionOpenSinceMonth'].fillna(1).astype(int).astype(str) + '-01', errors='coerce'
    )
    df['MonthsSinceCompetition'] = ((df['Date'] - df['CompetitorOpen']).dt.days / 30).fillna(-1)

    sns.lineplot(x='MonthsSinceCompetition', y='Sales', data=df, errorbar=None)
    plt.title("Sales Impact by Months Since Competitor Open")
    plt.xlabel("Months Since Competition Opened")
    plt.ylabel("Sales")
    plt.show()