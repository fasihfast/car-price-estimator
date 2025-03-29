from scraper import get_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from numpy import sqrt
import joblib
import sys
import os

if __name__ == '__main__':
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == 'new':
            if not os.path.exists('data'):
                os.makedirs('data')
                
            data = get_data(num_pages=15)
            df = pd.DataFrame(data)
            
            # ------ DATA CLEANING ------
            print("Preprocessing data")

            # cleaning 'mileage' column
            df['mileage'] = df['mileage'].astype(str).str.replace(r'[a-zA-Z,\s]+', '', regex=True)
            df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')

            # cleaning 'engine_capacity' column
            df['engine_capacity'] = df['engine_capacity'].astype(str).str.replace(r'[a-zA-Z,\s]+', '', regex=True)
            df['engine_capacity'] = pd.to_numeric(df['engine_capacity'], errors='coerce')

            # ensuring correct data types
            df['year'] = df['year'].astype(int)
            df['mileage'] = df['mileage'].astype(int)
            df['engine_capacity'] = df['engine_capacity'].astype(int)

            # filling empty prices
            df['price'].fillna(df['price'].mean(), inplace=True)

            df.to_csv('data/data.csv', index=False)
        elif arg == 'old':
            print("Loading data")
            df = pd.read_csv('data/data.csv')
            print("Preprocessing data")
        else:
            raise ValueError('Provide "new" or "old" as argument')
    else:
        raise ValueError('Provide "new" or "old" as argument')


    df.dropna(inplace=True)
    print(f'Number of records after dropping nulls: {df.shape[0]}')
    
    # encoding
    categorical_cols = ["model", "make", "transmission", "fuel_type", "city"]
    encoders = {} # storing encoders to use later
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str)) # important to cast to string to avoid errors.
        encoders[col] = le
    joblib.dump(encoders, 'data/encoders.joblib')

    print(df.head())

    # splitting
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # scaling
    # scaler = StandardScaler()
    # numerical_cols = ["year", "mileage", "engine_capacity"]
    # X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    # X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    # y_train[['price']] = scaler.fit_transform(y_train[['price']])
    # y_test[['price']] = scaler.transform(y_test[['price']])

    # training
    print("Training model")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    # model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'data/model.joblib')

    # evaluating model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    print(f"MSE: {mse}  RMSE: {rmse}")

    print("The script has finished successfully")