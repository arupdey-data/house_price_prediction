import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("🏡 Indian House Price Predictor AI")
st.write("Type in the details below to estimate the property value.")

@st.cache_data
def train_model():
    df = pd.read_csv('synthetic_house_prices.csv')
    X = df[['area_sqft', 'bedrooms', 'bathrooms', 'age_years', 'parking_spaces', 'distance_to_city_km']]
    y = df['price_inr']
    model = LinearRegression()
    model.fit(X, y)
    return model

ai_brain = train_model()

area = st.number_input("Area (Square Feet)", min_value=500, max_value=5000, value=1500)
beds = st.slider("Number of Bedrooms", 1, 6, 3)
baths = st.slider("Number of Bathrooms", 1, 5, 2)
age = st.slider("Age of the Property (Years)", 0, 40, 5)
parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)
distance = st.slider("Distance to City Center (km)", 0.5, 25.0, 10.0)

if st.button("Calculate Predicted Price"):
    input_data = pd.DataFrame([[area, beds, baths, age, parking, distance]], 
                               columns=['area_sqft', 'bedrooms', 'bathrooms', 'age_years', 'parking_spaces', 'distance_to_city_km'])
    prediction = ai_brain.predict(input_data)
    st.success(f"💰 Estimated Price: ₹{prediction[0]:,.2f}")
