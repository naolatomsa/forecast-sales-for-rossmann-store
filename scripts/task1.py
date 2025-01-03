# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import logging

# # # Setup logging
# # logging.basicConfig(level=logging.INFO, filename='eda_task1.log',
# #                     format='%(asctime)s - %(levelname)s - %(message)s')
# # # Data Loading
# # def load_data(trian_path, test_path, store_path):
# #     """Load train, test, and store data."""
# #     train = pd.read_csv(trian_path)
# #     test = pd.read_csv(test_path)
# #     store = pd.read_csv(store_path)
# #     return train, test, store

# # # Data Merging
# # def merge_data(train, test, store):
# #     """Merge train and test datasets with store data."""
# #     trian_merged = pd.merge(train, store, on='Store', how='left')
# #     test_merged = pd.merge(test, store, on='Store', how='left')
# #     return trian_merged, test_merged


# # Data Cleaning
# def clean_data(df):
#     """Handle missing values and prepare the dataset."""
#     df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
#     df['Promo2SinceWeek'].fillna(0, inplace=True)
#     df['Promo2SinceYear'].fillna(0, inplace=True)
#     df['PromoInterval'].fillna('None', inplace=True)
#     return df

# # Feature Engineering
# def add_date_features(df):
#     """Add date-related features."""
#     df['Year'] = pd.to_datetime(df['Date']).dt.year
#     df['Month'] = pd.to_datetime(df['Date']).dt.month
#     df['Day'] = pd.to_datetime(df['Date']).dt.day
#     df['WeekOfYear'] = pd.to_datetime(df['Date']).dt.isocalendar().week
#     return df

# # Exploratory Data Analysis
# def plot_promo_distribution(train, test):
#     """Plot the distribution of promotions in train and test sets."""
#     trian_promo_dist = train['Promo'].value_counts(normalize=True)
#     test_promo_dist = test['Promo'].value_counts(normalize=True)
#     print("Train Promo Distribution:")
#     print(trian_promo_dist)
#     print("Test Promo Distribution:")
#     print(test_promo_dist)
    
#     plt.bar(['Train - No Promo', 'Train - Promo'], trian_promo_dist, label='Train', alpha=0.6)
#     plt.bar(['Test - No Promo', 'Test - Promo'], test_promo_dist, label='Test', alpha=0.6)
#     plt.title("Promo Distribution in Train and Test Sets")
#     plt.legend()
#     plt.show()


# # Distribution of Promotions
# def analyze_promo_distribution(train, test):
#     logging.info("Analyzing promotion distribution in train and test sets")
#     train_promo = train['Promo'].value_counts(normalize=True)
#     test_promo = test['Promo'].value_counts(normalize=True)
#     logging.info(f"Promo distribution (Train): {train_promo}")
#     logging.info(f"Promo distribution (Test): {test_promo}")
#     plt.bar(['Train - No Promo', 'Train - Promo'], train_promo, label='Train', alpha=0.6)
#     plt.bar(['Test - No Promo', 'Test - Promo'], test_promo, label='Test', alpha=0.6)
#     plt.title("Promo Distribution in Train and Test Sets")
#     plt.legend()
#     plt.show()

# # Sales Before, During, and After Holidays
# def analyze_holiday_sales(df):
#     logging.info("Analyzing sales behavior around holidays")
#     holiday_sales = df[df['StateHoliday'] > 0]
#     non_holiday_sales = df[df['StateHoliday'] == 0]
#     plt.figure(figsize=(12, 6))
#     sns.boxplot(x='StateHoliday', y='Sales', data=holiday_sales)
#     plt.title("Sales During Holidays")
#     plt.show()
#     logging.info("Holiday sales analysis complete")

# # Seasonal Behavior
# def analyze_seasonality(df):
#     logging.info("Analyzing seasonal purchasing behavior")
#     seasonal_sales = df.groupby('Month')['Sales'].mean()
#     plt.figure(figsize=(12, 6))
#     seasonal_sales.plot(kind='bar', color='skyblue')
#     plt.title("Average Sales Per Month")
#     plt.show()
#     logging.info("Seasonal analysis complete")

# # Correlation Analysis
# def analyze_correlation(df):
#     logging.info("Analyzing correlation between Sales and Customers")
#     corr = df[['Sales', 'Customers']].corr()
#     sns.heatmap(corr, annot=True, cmap='coolwarm')
#     plt.title("Correlation Matrix")
#     plt.show()
#     logging.info(f"Correlation Matrix: {corr}")

# # Promo Effectiveness
# def analyze_promo_effectiveness(df):
#     logging.info("Analyzing promotion effectiveness")
#     promo_sales = df[df['Promo'] == 1]['Sales']
#     no_promo_sales = df[df['Promo'] == 0]['Sales']
#     plt.figure(figsize=(10, 6))
#     sns.histplot(promo_sales, kde=True, color='blue', label='Promo')
#     sns.histplot(no_promo_sales, kde=True, color='orange', label='No Promo')
#     plt.title("Promo vs No Promo Sales")
#     plt.legend()
#     plt.show()
#     logging.info(f"Average Sales with Promo: {promo_sales.mean():.2f}")
#     logging.info(f"Average Sales without Promo: {no_promo_sales.mean():.2f}")

# # Competitor Effect
# def analyze_competitor_effect(df):
#     logging.info("Analyzing competitor distance effect")
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(x='CompetitionDistance', y='Sales', data=df)
#     plt.title("Sales vs Competition Distance")
#     plt.xlabel("Competition Distance")
#     plt.ylabel("Sales")
#     plt.show()
    
# #Promo Deployment Analysis
# def identify_promo_effective_stores(df):
#     logging.info("Identifying stores with effective promotions")
#     promo_effect = df.groupby('Store')['Sales'].mean().reset_index()
#     promo_effect.columns = ['Store', 'AvgSales']
#     high_performance_stores = promo_effect.sort_values(by='AvgSales', ascending=False).head(10)
#     logging.info(f"Top 10 stores for promo deployment: {high_performance_stores}")
#     return high_performance_stores

# #Customer Behavior During Open/Close
# def analyze_open_close_behavior(df):
#     logging.info("Analyzing customer behavior during store open/close times")
#     open_sales = df[df['Open'] == 1]['Sales']
#     closed_sales = df[df['Open'] == 0]['Sales']
#     logging.info(f"Average sales when open: {open_sales.mean():.2f}")
#     logging.info(f"Sales when closed: {closed_sales.mean():.2f}")
#     plt.figure(figsize=(10, 6))
#     sns.histplot(open_sales, kde=True, label='Open', color='green')
#     sns.histplot(closed_sales, kde=True, label='Closed', color='red')
#     plt.legend()
#     plt.title("Sales During Open vs Closed Times")
#     plt.show()

# #Analyze Weekday vs Weekend Sales
# def analyze_weekday_weekend_sales(df):
#     logging.info("Analyzing weekday vs weekend sales")
#     df['IsWeekend'] = df['DayOfWeek'].isin([5, 6])  # Assuming 5 = Saturday, 6 = Sunday
#     weekend_sales = df[df['IsWeekend'] == True]['Sales']
#     weekday_sales = df[df['IsWeekend'] == False]['Sales']
#     logging.info(f"Average weekday sales: {weekday_sales.mean():.2f}")
#     logging.info(f"Average weekend sales: {weekend_sales.mean():.2f}")
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=['Weekday', 'Weekend'], y=[weekday_sales.mean(), weekend_sales.mean()])
#     plt.title("Weekday vs Weekend Sales")
#     plt.show()

# #Effect of Assortment Type
# def analyze_assortment_effect(df):
#     logging.info("Analyzing effect of assortment type on sales")
#     assortment_sales = df.groupby('Assortment')['Sales'].mean()
#     plt.figure(figsize=(10, 6))
#     assortment_sales.plot(kind='bar', color='purple')
#     plt.title("Sales by Assortment Type")
#     plt.xlabel("Assortment Type")
#     plt.ylabel("Average Sales")
#     plt.show()
#     logging.info(f"Assortment effect: {assortment_sales}")
    
# #Competitor Effect Over Time
# def analyze_new_competitors(df):
#     logging.info("Analyzing effect of new competitors on sales")
#     df['NewCompetitor'] = df['CompetitionDistance'].isna().astype(int)  # 1 if no competitors initially
#     competitor_sales = df.groupby('NewCompetitor')['Sales'].mean()
#     logging.info(f"Sales with new competitors: {competitor_sales}")
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=competitor_sales.index, y=competitor_sales.values)
#     plt.title("Effect of New Competitors on Sales")
#     plt.xlabel("New Competitor (1 = Yes)")
#     plt.ylabel("Average Sales")
#     plt.show()

# # # Main Execution
# # def main():
# #     # Load Data
# #     train = pd.read_csv('train.csv')
# #     test = pd.read_csv('test.csv')
# #     store = pd.read_csv('store.csv')

# #     # Merge Datasets
# #     train = pd.merge(train, store, on='Store', how='left')
# #     test = pd.merge(test, store, on='Store', how='left')

# #     # Clean Data
# #     train = clean_data(train)
# #     test = clean_data(test)

# #     # Perform Analysis
# #     analyze_promo_distribution(train, test)
# #     analyze_holiday_sales(train)
# #     analyze_seasonality(train)
# #     analyze_correlation(train)
# #     analyze_promo_effectiveness(train)
# #     analyze_competitor_effect(train)

# #     logging.info("Exploratory analysis complete")



# # # Main Function

# # def main():
# #     # Load data
# #     train, test, store = load_data('train.csv', 'test.csv', 'store.csv')

# #     # Merge data
# #     trian_merged, test_merged = merge_data(train, test, store)

# #     # Clean data
# #     trian_merged = clean_data(trian_merged)
# #     test_merged = clean_data(test_merged)

# #     # Add features
# #     trian_merged = add_date_features(trian_merged)
# #     test_merged = add_date_features(test_merged)

# #     # EDA
# #     plot_promo_distribution(trian_merged, test_merged)
# #     plot_holiday_sales(trian_merged)
# #     plot_competition_effect(trian_merged)

# #     # Prepare data for modeling
# #     X_trian, y_trian, X_test = prepare_data_for_model(trian_merged, test_merged)

# #     # Train and predict
# #     predictions = trian_and_predict(X_trian, y_trian, X_test)

# #     # Save predictions
# #     save_predictions(test, predictions, 'sales_predictions.csv')

# # if __name__ == "__main__":
# #     main()


# # Data Cleaning
# # def clean_data(df):
# #     logging.info("Starting data cleaning")
# #     # Handle missing values
# #     df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
# #     df['Promo2SinceYear'].fillna(0, inplace=True)
# #     df['Promo2SinceWeek'].fillna(0, inplace=True)
# #     df['PromoInterval'].fillna('None', inplace=True)
# #     logging.info("Missing values handled")
# #     # Add new features
# #     df['Year'] = pd.to_datetime(df['Date']).dt.year
# #     df['Month'] = pd.to_datetime(df['Date']).dt.month
# #     df['Week'] = pd.to_datetime(df['Date']).dt.isocalendar().week
# #     df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
# #     df['IsHolidaySeason'] = df['Month'].isin([11, 12, 1])
# #     logging.info("Feature engineering complete")
# #     return df





# def plot_sales_trends(df, group_by_column):
#     """Plot sales trends grouped by a specific column."""
#     logging.info(f"Plotting sales trends grouped by {group_by_column}")
#     plt.figure(figsize=(12, 6))
#     sns.lineplot(data=df, x='Date', y='Sales', hue=group_by_column)
#     plt.title(f"Sales Trends by {group_by_column}")
#     plt.xlabel("Date")
#     plt.ylabel("Sales")
#     plt.show()

# def analyze_promotions(df):
#     """Analyze the impact of promotions on sales."""
#     logging.info("Analyzing promotions")
#     promo_sales = df[df['Promo'] == 1]['Sales']
#     no_promo_sales = df[df['Promo'] == 0]['Sales']

#     plt.figure(figsize=(10, 6))
#     sns.histplot(promo_sales, kde=True, label='Promo', color='blue')
#     sns.histplot(no_promo_sales, kde=True, label='No Promo', color='orange')
#     plt.legend()
#     plt.title("Sales Distribution: Promo vs No Promo")
#     plt.xlabel("Sales")
#     plt.ylabel("Frequency")
#     plt.show()

#     logging.info(f"Average sales with promo: {promo_sales.mean():.2f}, without promo: {no_promo_sales.mean():.2f}")

# def correlation_analysis(df, columns):
#     """Analyze correlation between specified columns."""
#     logging.info("Performing correlation analysis")
#     corr_matrix = df[columns].corr()
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
#     plt.title("Correlation Matrix")
#     plt.show()
#     return corr_matrix

# def main():
#     # Load datasets
#     train, test = load_data('train.csv', 'test.csv')

#     # Check and handle missing values
#     check_missing_data(train)
#     train = handle_missing_values(train)

#     # Detect and log outliers
#     outliers = detect_outliers(train, 'Sales')

#     # Encode categorical features
#     train = encode_categorical_features(train)

#     # Create new features
#     train = create_new_features(train)

#     # Perform EDA
#     plot_sales_trends(train, 'Promo')
#     analyze_promotions(train)
#     corr_matrix = correlation_analysis(train, ['Sales', 'Customers', 'CompetitionDistance'])

#     # Log summary of insights
#     logging.info("EDA completed successfully")




# def encode_categorical_features(df):
#     """Encode categorical features into numerical values."""
#     logging.info("Encoding categorical features")
#     df['StateHoliday'] = df['StateHoliday'].replace({'0': 0, 'a': 1, 'b': 2, 'c': 3}).astype(int)
#     df['StoreType'] = df['StoreType'].map({'a': 1, 'b': 2, 'c': 3, 'd': 4})
#     df['Assortment'] = df['Assortment'].map({'a': 1, 'b': 2, 'c': 3})
#     logging.info("Categorical features encoded")
#     return df

# def create_new_features(df):
#     """Create new features for analysis."""
#     logging.info("Creating new features")
#     df['Year'] = pd.to_datetime(df['Date']).dt.year
#     df['Month'] = pd.to_datetime(df['Date']).dt.month
#     df['Week'] = pd.to_datetime(df['Date']).dt.isocalendar().week
#     df['Day'] = pd.to_datetime(df['Date']).dt.day
#     df['IsHolidaySeason'] = df['Month'].isin([11, 12, 1])
#     logging.info("New features created")
#     return df