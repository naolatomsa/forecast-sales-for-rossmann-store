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

def check_missing_data(df):
    """Check for missing values in the dataset."""
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    logging.info(f"Missing data: {missing_data}")
    return missing_data

# Data Merging
def merge_data(train, test, store):
    """Merge train and test datasets with store data."""
    trian_merged = pd.merge(train, store, on='Store', how='left')
    test_merged = pd.merge(test, store, on='Store', how='left')
    logging.info(f"trian merged shape: {trian_merged.shape}, test merged shape: {test_merged.shape}")
    return trian_merged, test_merged


def handle_missing_values_for_store(df):
    """Handle missing values and prepare the dataset."""
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
    df['PromoInterval'] = df['PromoInterval'].fillna('None')
    return df

def handle_missing_values_for_train(df):
    """Handle missing values in the dataset."""
    logging.info("Handling missing values for train")
    df.dropna(subset=['Open'], inplace=True)
    return df



def detect_outliers(data, columns):
    outliers = {}
    
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers[column] = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data[column], color='skyblue')
        plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
        plt.axvline(upper_bound, color='red', linestyle='--', label='Upper Bound')
        plt.title(f"Box Plot for {column} with Outlier Bounds")
        plt.xlabel(column)
        plt.legend()
        plt.show()
        
    logging.info("Detecting outliers")
    return outliers

# Cap and floor outliers for all numeric columns
def cap_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    logging.info("Detecting outliers")
    return data

# Feature Engineering
def add_date_features(df):
    """Add date-related features."""
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Day'] = pd.to_datetime(df['Date']).dt.day
    df['WeekOfYear'] = pd.to_datetime(df['Date']).dt.isocalendar().week
    logging.info("Feature Engineering")
    return df

# Task 1: Check for distribution in both training and test sets - Promo Distribution


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

# Task 2: Compare Sales Before, During, and After Holidays

def analyze_holiday_sales(df):
    """Analyze sales behavior before, during, and after holidays."""
    logging.info("Analyze sales behavior before, during, and after holidays.")

    df['Date'] = pd.to_datetime(df['Date'])
    df['StateHoliday'] = df['StateHoliday'].astype(str)
    df = df.sort_values(by='Date')

    df['IsHoliday'] = df['StateHoliday'].isin(['a', 'b', 'c'])

    for holiday in ['a', 'b', 'c']:
        df[f'Before_{holiday}'] = df['StateHoliday'].shift(-1, fill_value='0') == holiday
        df[f'After_{holiday}'] = df['StateHoliday'].shift(1, fill_value='0') == holiday

    df['HolidayCategory'] = 'Normal'
    for holiday in ['a', 'b', 'c']:
        df.loc[df['StateHoliday'] == holiday, 'HolidayCategory'] = f'During {holiday.upper()}'
        df.loc[df[f'Before_{holiday}'], 'HolidayCategory'] = f'Before {holiday.upper()}'
        df.loc[df[f'After_{holiday}'], 'HolidayCategory'] = f'After {holiday.upper()}'

    plt.figure(figsize=(12, 6))
    sns.barplot(x='HolidayCategory', y='Sales', data=df, hue='HolidayCategory', palette='viridis')
    plt.title("Average Sales Before, During, and After Holidays")
    plt.xticks(rotation=45)
    plt.show()

# Task 3: Seasonal Behaviors

def analyze_seasonality(df):
    """Analyze seasonal purchasing behaviors."""
    logging.info("Analyze seasonal purchasing behaviors.")

    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    seasonal_sales = df.groupby('Month')['Sales'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Month', y='Sales', data=seasonal_sales, hue='Month', palette='coolwarm')
    plt.title("Average Sales Per Month")
    plt.show()

# Task 4: Correlation Between Sales and Customers

def correlation_sales_customers(df):
    """Calculate and visualize the correlation between sales and customers."""
    logging.info("correlation between sales and customers.")

    correlation = df[['Sales', 'Customers']].corr()
    logging.info("Correlation Between Sales and Customers:")
    logging.info(correlation)

    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title("Correlation Between Sales and Customers")
    plt.show()

# Task 5: How Promo Affects Sales

def promo_effect_on_sales(df):
    """Analyze the effect of promos on sales."""
    logging.info("effect of promos on sales.")

    promo_sales = df.groupby('Promo')['Sales'].mean().reset_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Promo', y='Sales', data=promo_sales, hue='Promo', palette='viridis')
    plt.title("Effect of Promo on Sales")
    plt.show()

# Task 6: Promo Deployment Strategies

def promo_deployment_recommendations(df):
    """Recommend which stores should deploy promos based on sales data."""
    logging.info("Promo Deployment Strategies")

    promo_effectiveness = df.groupby(['Store', 'Promo'])['Sales'].mean().unstack()
    promo_effectiveness['Difference'] = promo_effectiveness[1] - promo_effectiveness[0]
    top_stores = promo_effectiveness.sort_values(by='Difference', ascending=False).head(10)

    logging.info("Top stores for promo deployment:")
    logging.info(top_stores)
    return top_stores

# Task 7: Trends During Store Open/Close

def analyze_store_open_close(df):
    logging.info("Trends During Store Open/Close")

    """Analyze customer behavior during store opening and closing times."""
    open_sales = df.groupby('Open')['Sales'].mean().reset_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Open', y='Sales', data=open_sales, hue='Open', palette='coolwarm')
    plt.title("Average Sales When Store Is Open vs Closed")
    plt.show()

# Task 8: Stores Open All Weekdays and Weekend Impact

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

# Task 9: Assortment Type and Sales

def assortment_type_effect(df):
    """Check how assortment type affects sales."""
    logging.info("Assortment Type and Sales")

    assortment_sales = df.groupby('Assortment')['Sales'].mean().reset_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Assortment', y='Sales', data=assortment_sales,hue='Assortment', palette='coolwarm')
    plt.title("Effect of Assortment Type on Sales")
    plt.show()

# Task 10: Competition Distance Effect

def competition_distance_effect(df):
    logging.info("Competition Distance Effect")

    """Analyze how competition distance affects sales."""
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=df, alpha=0.6)
    plt.title("Effect of Competition Distance on Sales")
    plt.xlabel("Competition Distance")
    plt.ylabel("Sales")
    plt.show()

# Task 11: Competitor Open/Reopen Impact

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