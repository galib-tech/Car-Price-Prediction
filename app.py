import streamlit as st
import pickle
import  pandas as pd
from numpy.ma.extras import unique

# Page title
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Price Prediction")
st.write("Predict the resale price of your car.")

# Load dataset
car = pd.read_csv("clean_car.csv")

# Model load
pipe = pickle.load(open("pipe.pkl", "rb"))

# Company Dorpdown

company = st.selectbox(
    "Select Company",
    sorted(car["company"].unique())
)

# Car name dropdown

name = st.selectbox(
    "Select Name",
    sorted(car["name"].unique())
)

# Year dropdown

year = st.selectbox(
    "Select Year",
    sorted(car["year"].unique(), reverse= True)
)

# Fuel Type

fuel_type = st.selectbox(
    "Select Fuel Type",
    sorted(car["fuel_type"].unique())
)

# Kilometers Driven

kms = st.selectbox(
    "Kilometers Driven",
    sorted(car["kms_driven"].unique())
)

# Prediction Button

if st.button("Predict Price"):

    # Create dataframe for model

    input_df = pd.DataFrame([[company, name, year, fuel_type, kms]],
                            columns=["company", "name", "year", "fuel_type", "kms_driven"])



    # Prediction
    prediction = pipe.predict(input_df)

    # Result
    st.success(f"Predicted Price: ₹ {int(prediction[0]):,}")
