import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import logging

# # Setup logging
logging.basicConfig(level=logging.INFO, filename='eda.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(train_path, test_path, store_path):
    """Load training and test datasets."""
    logging.info("Loading datasets")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    store = pd.read_csv(store_path)
    logging.info(f"Train shape: {train.shape}, Test shape: {test.shape}, Store shape: {store.shape}")
    return train, test, store

# Data Merging
def merge_data(train, test, store):
    """Merge train and test datasets with store data."""
    trian_merged = pd.merge(train, store, on='Store', how='left')
    test_merged = pd.merge(test, store, on='Store', how='left')
    logging.info(f"trian merged shape: {trian_merged.shape}, test merged shape: {test_merged.shape}")
    return trian_merged, test_merged