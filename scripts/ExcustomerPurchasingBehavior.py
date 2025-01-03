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
    return df

# Exploratory Data Analysis
def plot_promo_distribution(train, test):
    """Plot the distribution of promotions in train and test sets."""
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

# Analyze sales behavior before, during, and after state holidays.
def analyze_holiday_sales(df):
    """Analyze sales behavior before, during, and after state holidays."""
    logging.info("Analyzing sales behavior around state holidays")

    # Ensure 'StateHoliday' is treated as a categorical variable
    df['StateHoliday'] = df['StateHoliday'].astype(str)

    # Filter holiday and non-holiday sales
    holiday_sales = df[df['StateHoliday'].isin(['a', 'b', 'c'])]
    non_holiday_sales = df[df['StateHoliday'] == '0']

    # Boxplot: Sales distribution for different holidays
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='StateHoliday', y='Sales', data=df)
    plt.title("Sales Distribution During Holidays and Non-Holidays")
    plt.xlabel("State Holiday (0 = None, a = Public, b = Easter, c = Christmas)")
    plt.ylabel("Sales")
    plt.show()

    # Barplot: Average sales by holiday type
    sales_summary = df.groupby('StateHoliday')['Sales'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='StateHoliday', y='Sales', data=sales_summary, order=['0', 'a', 'b', 'c'])
    plt.title("Average Sales During Holidays and Non-Holidays")
    plt.xlabel("State Holiday (0 = None, a = Public, b = Easter, c = Christmas)")
    plt.ylabel("Average Sales")
    plt.show()

    # Logging average sales
    logging.info("Sales Analysis Summary:")
    for holiday_type in ['0', 'a', 'b', 'c']:
        avg_sales = df[df['StateHoliday'] == holiday_type]['Sales'].mean()
        logging.info(f"Average sales for holiday type '{holiday_type}': {avg_sales}")

    logging.info("Holiday sales analysis complete")


# Seasonal Behavior
def analyze_seasonality(df):
    logging.info("Analyzing seasonal purchasing behavior")
    seasonal_sales = df.groupby('Month')['Sales'].mean()
    plt.figure(figsize=(12, 6))
    seasonal_sales.plot(kind='bar', color='skyblue')
    plt.title("Average Sales Per Month")
    plt.show()
    logging.info("Seasonal analysis complete")

# Correlation Analysis
def analyze_correlation(df):
    logging.info("Analyzing correlation between Sales and Customers")
    corr = df[['Sales', 'Customers']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()
    logging.info(f"Correlation Matrix: {corr}")

# Promo Effectiveness
def analyze_promo_effectiveness(df):
    logging.info("Analyzing promotion effectiveness")
    promo_sales = df[df['Promo'] == 1]['Sales']
    no_promo_sales = df[df['Promo'] == 0]['Sales']
    plt.figure(figsize=(10, 6))
    sns.histplot(promo_sales, kde=True, color='blue', label='Promo')
    sns.histplot(no_promo_sales, kde=True, color='orange', label='No Promo')
    plt.title("Promo vs No Promo Sales")
    plt.legend()
    plt.show()
    logging.info(f"Average Sales with Promo: {promo_sales.mean():.2f}")
    logging.info(f"Average Sales without Promo: {no_promo_sales.mean():.2f}")

# Competitor Effect
def analyze_competitor_effect(df):
    logging.info("Analyzing competitor distance effect")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=df)
    plt.title("Sales vs Competition Distance")
    plt.xlabel("Competition Distance")
    plt.ylabel("Sales")
    plt.show()
    
#Promo Deployment Analysis
def identify_promo_effective_stores(df):
    logging.info("Identifying stores with effective promotions")
    promo_effect = df.groupby('Store')['Sales'].mean().reset_index()
    promo_effect.columns = ['Store', 'AvgSales']
    high_performance_stores = promo_effect.sort_values(by='AvgSales', ascending=False).head(10)
    logging.info(f"Top 10 stores for promo deployment: {high_performance_stores}")
    return high_performance_stores

#Customer Behavior During Open/Close
def analyze_open_close_behavior(df):
    logging.info("Analyzing customer behavior during store open/close times")
    open_sales = df[df['Open'] == 1]['Sales']
    closed_sales = df[df['Open'] == 0]['Sales']
    logging.info(f"Average sales when open: {open_sales.mean():.2f}")
    logging.info(f"Sales when closed: {closed_sales.mean():.2f}")
    plt.figure(figsize=(10, 6))
    sns.histplot(open_sales, kde=True, label='Open', color='green')
    sns.histplot(closed_sales, kde=True, label='Closed', color='red')
    plt.legend()
    plt.title("Sales During Open vs Closed Times")
    plt.show()

#Analyze Weekday vs Weekend Sales
def analyze_weekday_weekend_sales(df):
    logging.info("Analyzing weekday vs weekend sales")
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6])  # Assuming 5 = Saturday, 6 = Sunday
    weekend_sales = df[df['IsWeekend'] == True]['Sales']
    weekday_sales = df[df['IsWeekend'] == False]['Sales']
    logging.info(f"Average weekday sales: {weekday_sales.mean():.2f}")
    logging.info(f"Average weekend sales: {weekend_sales.mean():.2f}")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=['Weekday', 'Weekend'], y=[weekday_sales.mean(), weekend_sales.mean()])
    plt.title("Weekday vs Weekend Sales")
    plt.show()

#Effect of Assortment Type
def analyze_assortment_effect(df):
    logging.info("Analyzing effect of assortment type on sales")
    assortment_sales = df.groupby('Assortment')['Sales'].mean()
    plt.figure(figsize=(10, 6))
    assortment_sales.plot(kind='bar', color='purple')
    plt.title("Sales by Assortment Type")
    plt.xlabel("Assortment Type")
    plt.ylabel("Average Sales")
    plt.show()
    logging.info(f"Assortment effect: {assortment_sales}")
    
#Competitor Effect Over Time
def analyze_new_competitors(df):
    logging.info("Analyzing effect of new competitors on sales")
    df['NewCompetitor'] = df['CompetitionDistance'].isna().astype(int)  # 1 if no competitors initially
    competitor_sales = df.groupby('NewCompetitor')['Sales'].mean()
    logging.info(f"Sales with new competitors: {competitor_sales}")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=competitor_sales.index, y=competitor_sales.values)
    plt.title("Effect of New Competitors on Sales")
    plt.xlabel("New Competitor (1 = Yes)")
    plt.ylabel("Average Sales")
    plt.show()
