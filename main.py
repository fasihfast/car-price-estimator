from scraper import get_data
import pandas as pd
import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy import expm1


df_original = pd.read_csv('data/data.csv')  # original dataset

def submit_fn(make,model_field,year,min_mileage,max_mileage,fuel_type,engine_capacity,transmission,city):
    # make,model,year,mileage,fuel_type,engine_capacity,transmission,city,price

    if make:
        make = make.strip()
    if model_field:
        model_field = model_field.strip()
    # if year:
    #     input_data['year'] = year
    if not min_mileage:
        min_mileage = 0
    if not max_mileage:
        max_mileage = 1000000
    # input_data['mileage'] = (min_mileage + max_mileage) / 2        
    # if fuel_type != 'Any':
    #     input_data['fuel_type'] = fuel_type
    # if engine_capacity:
    #     input_data['engine_capacity'] = engine_capacity
    # if trim:
    #     input_data['trim'] = trim
    # if transmission != 'Any':
    #     input_data['transmission'] = transmission
    if city:
        city = city.strip()
        # input_data['city'] = city

    if not make or not model_field or not year or not max_mileage or not city or not fuel_type or not transmission or not engine_capacity:
        st.error('Please fill in all the fields.')
        return
    
    input_data = {
        'make': make,
        'model': model_field,
        'year': year,
        'mileage': (min_mileage + max_mileage) / 2,
        'fuel_type': fuel_type,
        'engine_capacity': engine_capacity,
        'transmission': transmission,
        'city': city
    }

    df = pd.DataFrame([input_data])

    # cleaning 'mileage' column
    df['mileage'] = df['mileage'].astype(str).str.replace(r'[a-zA-Z,\s]+', '', regex=True)
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
    df['mileage'] = df['mileage'].astype(int) # ensuring correct data type

    # cleaning 'engine_capacity' column
    if engine_capacity:
        df['engine_capacity'] = df['engine_capacity'].astype(str).str.replace(r'[a-zA-Z,\s]+', '', regex=True)
        df['engine_capacity'] = pd.to_numeric(df['engine_capacity'], errors='coerce')
        df['engine_capacity'] = df['engine_capacity'].astype(int) # ensuring correct data type

    if year:
        df['year'] = df['year'].astype(int)

    # scaling
    numerical_cols = ["year", "mileage", "engine_capacity"]
    # if year:
    #     numerical_cols.append("year")
    # if engine_capacity:
        # numerical_cols.append("engine_capacity")
    scaler = StandardScaler()
    # df_original = pd.read_csv('data/data.csv')
    df[numerical_cols] = scaler.fit(df_original[numerical_cols]).transform(df[numerical_cols]) # important to fit on the original training data


    # price conversion
    def price_convert(value):
        if value >= 10000000:
            short_value = round(value / 10000000, 1)
            return f"{short_value} Crore" 
    
        elif value >= 100000:
            short_value = round(value / 100000, 1)
            return f"{short_value} Lakh"

    # encoding
    categorical_cols = ["model", "make", "transmission", "fuel_type", "city"]
    # if fuel_type != 'Any':
    #     categorical_cols.append('fuel_type')
    # if transmission != 'Any':
    #     categorical_cols.append('transmission')
    # if make:
    #     categorical_cols.append('make')
    # if model_field:
    #     categorical_cols.append('model')
    # if city:
    #     categorical_cols.append('city')

    encoder = joblib.load('data/onehot_encoder.joblib')
    encoded_features = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
    df = pd.concat([df, encoded_df], axis=1).drop(columns=categorical_cols)

    prediction = model.predict(df)

    if prediction and len(prediction) > 0:
        price = int(prediction[0])
        price = expm1(price)
        new_price=price_convert(price)
        st.title(f'PKR {new_price}')
    else:
        st.title('An error occurred')


if __name__ == '__main__':
    model = joblib.load('data/model.joblib')

    make_list=df_original['make'].unique()
    make_list=np.sort(make_list)

    city_list=df_original['city'].unique()
    city_list=np.sort(city_list)

    model_list=df_original['model'].unique()


    
    st.set_page_config(page_title='Car Price Estimator', page_icon='🚘')
    st.title('🚘 Car Price Estimator (Pakistan)')
    st.info("Enter car details to get an accurate estimate of price. You can skip fields.")
    
    with st.form("car_details", border=False):
        st.header("Car Details")
        col1, col2 = st.columns(2)
        with col1:
            make = st.selectbox('Make', make_list) # created a dropdown for make 

            model_field = st.selectbox('Model', model_list) # created a dropdown for model 
            # trim = st.text_input('Trim/Variant', value='CX Eco') # todo: not passing in model
            transmission = st.selectbox('Transmission', ['Any', 'Automatic', 'Manual', 'Hybrid'])
            min_mileage = st.number_input('Minimum Mileage (km)', min_value=0, max_value=1000000, value=0)
            city = st.selectbox('City', city_list) # created a dropdown for city
        with col2:
            engine_capacity = st.number_input('Engine Capacity (cc)', min_value=0, max_value=100000, value=None, placeholder='e.g. 1500')
            year = st.number_input('Year', min_value=1980, max_value=2025, placeholder='e.g. 2016', value=None) # todo: max min year
            fuel_type = st.selectbox('Fuel Type', ['Any', 'Petrol', 'Diesel', 'Electric', 'CNG', 'Hybrid'])
            max_mileage = st.number_input('Maximum Mileage (km)', min_value=0, max_value=1000000, value=1000000)
            # engine_type = st.text_input('Engine Type/Size', placeholder='e.g. 1.3 VVTi')

        submit = st.form_submit_button("Estimate Price", use_container_width=True)

    if submit: 
        with st.spinner('Estimating price...'):
            submit_fn(make,model_field,year,min_mileage,max_mileage,fuel_type,engine_capacity,transmission,city)

    st.write('---')
    st.write('-> Created by [mafgit](https://github.com/mafgit) & [fasihfast](https://github.com/fasihfast)')
    st.write('-> Made using machine learning')
    st.write('-> Data source: [PakWheels](https://www.pakwheels.com/)')

# todo: for both min and max mileage, calculate separate estimate and show average. Do the same for year
# todo: or remove min and max because what if min is set to 0? it will show wrong estimate at 0
# todo: if city not selected, show average for three most popular cities?
# todo: if transmission or fuel type not selected, show average of estimates of all possibilities
# todo: or just remove these extra features?
# todo: but if we remove them, then only make model and year will remain, and we can just look up csv file to get estimates instead of using a model

# todo: dropdowns for make model etc
# todo: CI/CD
# todo: feature engineering
# todo: gradient booster