from scraper import get_data
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    model = joblib.load('data/model.joblib')
    
    st.set_page_config(page_title='Car Price Estimator', page_icon='🚘')
    st.title('🚘 Car Price Estimator (Pakistan)')
    st.write("Enter car details to get an accurate estimate of price.")

    col1, col2 = st.columns(2)
    with col1:
        make = st.text_input('Make', value='Honda')
        trim = st.text_input('Trim/Variant', value='CX Eco') # todo: not passing in model
        transmission = st.selectbox('Transmission', ['Hybrid', 'Automatic', 'Manual'])
        min_mileage = st.number_input('Minimum Mileage (km)', min_value=0, max_value=1000000, value=50000)
        engine_capacity = st.number_input('Engine Capacity (cc)', min_value=0, max_value=100000, value=847)
    with col2:
        model_field = st.text_input('Model', value='Civic')
        year = st.number_input('Year', min_value=1980, max_value=2025, placeholder='2016', value=2016)
        fuel_type = st.selectbox('Fuel Type', ['Petrol', 'CNG'])
        max_mileage = st.number_input('Maximum Mileage (km)', min_value=0, max_value=1000000, value=1000000)
        city = st.text_input('City', value='Karachi')
    
    # engine_type = st.text_input('Engine Type/Size', placeholder='e.g. 1.3 VVTi')

    if st.button('Estimate Price', use_container_width=True):
        df = pd.DataFrame([{
            'make': make,
            'model': model_field,
            'year': year,
            # 'trim': trim,
            'mileage': (min_mileage + max_mileage) // 2, # todo: fix this
            'fuel_type': fuel_type,
            'engine_capacity': engine_capacity,
            'transmission': transmission,
            'city': city
        }])

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

        # scaling
        numerical_cols = ["year", "mileage", "engine_capacity"]
        scaler = StandardScaler()
        df_original = pd.read_csv('data/data.csv')
        df[numerical_cols] = scaler.fit(df_original[numerical_cols]).transform(df[numerical_cols]) # important to fit on the original training data


        #price conversion

        def price_convert(value):
            if value >= 10000000:  # 1 crore = 10 million
                short_value = round(value / 10000000, 1)  # Convert to crore
                return f"{short_value}Cr"  # Append 'Cr' for Crore
        
            elif value >= 100000:  # Convert to lakh if less than crore
            
                short_value = round(value / 100000, 1)
                return f"{short_value}Lac"  # Append 'L' for Lakh

        # encoding
        categorical_cols = ["model", "make", "transmission", "fuel_type", "city"]
        encoders = joblib.load('data/encoders.joblib')
        for col in categorical_cols:
            le = encoders[col]
            df[col] = le.transform(df[col])

        prediction = model.predict(df)

        if prediction and len(prediction) > 0:
            price = int(prediction[0])
            new_price=price_convert(price)
            st.title(f'PKR {new_price}')
        else:
            st.title('An error occurred')

    st.write('---')
    st.write('Created by [mafgit](https://github.com/mafgit) & [fasihfast](https://github.com/fasihfast)')
    st.write('Data Source: [PakWheels](https://www.pakwheels.com/)')