from scraper import get_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
import sys
import os

if __name__ == '__main__':
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == 'new':
            if not os.path.exists('data'):
                os.makedirs('data')
                
            data = get_data(num_pages=5)
            new_df = pd.DataFrame(data)

            try:
                df_old = pd.read_csv('data/data.csv')
                df = pd.concat([new_df, df_old], ignore_index=True)
            except FileNotFoundError:
                df = new_df
                
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


    print(f'Number of records before: {df.shape[0]}')

    # outlier removal
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lb = Q1 - 1.5 * IQR
    ub = Q3 + 1.5 * IQR
    condition = (df['price'] >= lb) & (df['price'] <= ub)
    df = df[condition]
    print(f'Number of records after dropping outliers: {df.shape[0]}')
    
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True, keep='first')
    df.reset_index(drop=True, inplace=True)
    # df['price'] = df['price'].astype('float64')
    print(f'Number of records after dropping nulls and duplicates: {df.shape[0]}')

    # df['price'] = df['price'] / 1000000 # in 10 lakhs or 1 million
    # df['mileage'] = np.log1p(df['mileage'])

    df['age'] = 2025 - df['year']  # Convert year to car age
    df['mileage_per_year'] = df['mileage'] / (df['age'] + 1)
    df['engine_per_mileage'] = df['engine_capacity'] / (df['mileage'] + 1)  # avoid division by zero
    # df['mileage_engine'] = df['mileage'] * df['engine_capacity']
    # df['age_mileage'] = df['age'] * df['mileage']
    # df['age_engine'] = df['age'] * df['engine_capacity']

    df['price'] = np.log1p(df['price'])
    df['mileage'] = np.log1p(df['mileage'])
    df['mileage_per_year'] = np.log1p(df['mileage_per_year'])
    df['engine_per_mileage'] = np.log1p(df['engine_per_mileage'])
    # df['mileage_engine'] =  np.log1p(df['mileage_engine'])
    # df['age_mileage'] = np.log1p(df['age_mileage'])
    # df['age_engine'] =  np.log1p(df['age_engine'])


    # target encoding
    from category_encoders import TargetEncoder
    target_cols= ['make', 'model', 'city']
    encoder = TargetEncoder()
    df_encoded = encoder.fit_transform(df[target_cols], df['price'])
    df[target_cols] = df_encoded
    # pd.DataFrame(df_encoded, columns=target_cols, index=df.index)
    joblib.dump(encoder, 'data/target_encoder.joblib')
   
    # one hot encoding
    one_hot_cols = ["transmission", "fuel_type"]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[one_hot_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(one_hot_cols))
    df = pd.concat([df, encoded_df], axis=1).drop(columns=one_hot_cols)
    joblib.dump(encoder, 'data/onehot_encoder.joblib')

    # label encoding
    # encoders = {} # storing encoders to use later
    # for col in categorical_cols:
    #     le = LabelEncoder()
    #     df[col] = le.fit_transform(df[col].astype(str)) # important to cast to string to avoid errors.
    #     encoders[col] = le
    # joblib.dump(encoders, 'data/label_encoders.joblib')



    # splitting
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # scaling
    # scaler = StandardScaler()
    scaler = RobustScaler()
    numerical_cols = ["year", "mileage", "engine_capacity", "age", "mileage_per_year", "engine_per_mileage"]
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols].copy())
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols].copy())
    joblib.dump(scaler, 'data/scaler.joblib')
    # y_train[['price']] = scaler.fit_transform(y_train[['price']])
    # y_test[['price']] = scaler.transform(y_test[['price']])

    # training
    print("Training model")

    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=300, learning_rate=0.05)
    # model = RandomForestRegressor(n_estimators=300, random_state=42)
    # model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'data/model.joblib')

    importances = model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    print(feature_importance_df.sort_values(by='Importance', ascending=False))

    # evaluating model
    y_pred = model.predict(X_test)
    y_pred = np.expm1(y_pred)
    y_test = np.expm1(y_test)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # print(X_test['make'].iloc[0])
    # print(X_test['model'].iloc[0])
    # print(X_test['year'].iloc[0])
    # print(X_test['mileage'].iloc[0])
    # print(X_test['city'].iloc[0])
    # print(y_pred[:20])
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"MSE: {mse}  RMSE: {rmse}")

    print("The script has finished successfully")