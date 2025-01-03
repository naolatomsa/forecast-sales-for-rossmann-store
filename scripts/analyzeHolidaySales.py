import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import logging

# # Setup logging
logging.basicConfig(level=logging.INFO, filename='eda.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')


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