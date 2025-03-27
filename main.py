from scraper import get_data
import pandas as pd
import streamlit as st

def update_data_and_model():
    data = get_data(num_pages=10)
    df = pd.DataFrame(data)
    # todo: data cleaning
    df.to_csv('data.csv')
    # todo: train model


if __name__ == '__main__':
    # update_data_and_model()
    st.markdown("<h1 style='font-size:60px'>🚗</h1>", unsafe_allow_html=True)
    st.title('Car Price Estimator (Pakistan)')
    st.write("Enter car details to get an accurate estimate of price.")

    make = st.text_input('Make', placeholder='e.g. Honda')
    model = st.text_input('Model', placeholder='e.g. Civic')
    trim = st.text_input('Trim/Variant', placeholder='e.g. CX Eco')
    year = st.number_input('Year', min_value=1980, max_value=2025, placeholder='2016', value=2016)
    transmission = st.selectbox('Transmission', ['Any', 'Automatic', 'Manual'])
    fuel_type = st.selectbox('Fuel Type', ['Any', 'Petrol', 'CNG'])
    engine_type = st.text_input('Engine Type/Size', placeholder='e.g. 1.3 VVTi')
    city = st.text_input('City', placeholder='e.g. Karachi')

    col1, col2 = st.columns(2)
    with col1:
        min_mileage = st.number_input('Minimum Mileage', min_value=0, max_value=1000000, value=0)
    with col2:
        max_mileage = st.number_input('Maximum Mileage', min_value=0, max_value=1000000, value=1000000)

    if st.button('Estimate Price', use_container_width=True):
        st.title('PKR 200,000')