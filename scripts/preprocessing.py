import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def preprocessing(trainData, testData):

    trainData['PromoInterval'] = trainData['PromoInterval'].fillna('None')
    testData['PromoInterval'] = testData['PromoInterval'].fillna('None')
    
    trainData.drop('Date', axis=1, inplace=True)
    testData.drop('Date', axis=1, inplace=True)
    
    
    # future engineering
    trainData['CompetitionOpenSince'] = (
    12 * (trainData['Year'] - trainData['CompetitionOpenSinceYear']) +
    (trainData['Month'] - trainData['CompetitionOpenSinceMonth'])
    )
    trainData['CompetitionOpenSince'] = trainData['CompetitionOpenSince'].clip(lower=0)

    trainData['Promo2ActiveMonths'] = (
        12 * (trainData['Year'] - trainData['Promo2SinceYear']) +
        (trainData['WeekOfYear'] - trainData['Promo2SinceWeek']) // 4
    )
    trainData['Promo2ActiveMonths'] = trainData['Promo2ActiveMonths'].clip(lower=0)


    testData['CompetitionOpenSince'] = (
    12 * (testData['Year'] - testData['CompetitionOpenSinceYear']) +
    (testData['Month'] - testData['CompetitionOpenSinceMonth'])
    )
    testData['CompetitionOpenSince'] = testData['CompetitionOpenSince'].clip(lower=0)

    testData['Promo2ActiveMonths'] = (
        12 * (testData['Year'] - testData['Promo2SinceYear']) +
        (testData['WeekOfYear'] - testData['Promo2SinceWeek']) // 4
    )
    testData['Promo2ActiveMonths'] = testData['Promo2ActiveMonths'].clip(lower=0)
    
    
    # Encoding Categorical Data
    categorical_columns = trainData.select_dtypes(include=['object']).columns
    trainData = pd.get_dummies(trainData, columns=categorical_columns, drop_first=True)
    
    categorical_columns = testData.select_dtypes(include=['object']).columns
    testData = pd.get_dummies(testData, columns=categorical_columns, drop_first=True)
    
    trainData = trainData.drop(columns=['Customers', 'StateHoliday_b', 'StateHoliday_c', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'])
    
    testData = testData.drop(columns=['Id','CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'])
    
    return trainData, testData;
