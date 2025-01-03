import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import logging

# # Setup logging
logging.basicConfig(level=logging.INFO, filename='eda.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def check_promo_distribution(train, test):
    """Plot the distribution of promotions in train and test sets."""
    logging.info("Promo Distribution")

    trian_promo_dist = train['Promo'].value_counts(normalize=True)
    test_promo_dist = test['Promo'].value_counts(normalize=True)
    print("Train Promo Distribution:")
    print(trian_promo_dist)
    print("Test Promo Distribution:")
    print(test_promo_dist)
    
    plt.bar(['Train - No Promo', 'Train - Promo'], trian_promo_dist, label='Train', alpha=0.6)
    plt.bar(['Test - No Promo', 'Test - Promo'], test_promo_dist, label='Test', alpha=0.6)
    plt.title("Promo Distribution in Train and Test Sets")
    plt.legend()
    plt.show()
