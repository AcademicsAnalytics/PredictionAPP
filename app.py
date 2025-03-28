import streamlit as st
import joblib
import numpy as np
import sklearn

# Load the model
model = joblib.load("model.pkl")

# Title
st.title("Erik's Simple House Price Prediction App")

st.divider()

# App description
st.write("This app uses machine learning for predicting house prices with given features of the house. For using this app you can enter the inputs from this user interface and the use predict button.")

st.divider()

# Input fields for house features
bedrooms = st.number_input("Number of bedrooms", min_value=0, value=0)
bathrooms = st.number_input("Number of bathrooms", min_value=0, value=0)
livingarea = st.number_input("Living area", min_value=0, value=2000)
condition = st.number_input("Condition", min_value=0, value=3)
numberofschools = st.number_input("Number of schools nearby", min_value=0, value=0)

st.divider()

# Collecting input features
x = [bedrooms, bathrooms, livingarea, condition, numberofschools]

# Predict button
predictbutton = st.button("Predict!")

if predictbutton:
    st.balloons()
    
    # Prepare the input array and reshape it to be 2D
    X_array = np.array(x).reshape(1, -1)

    # Make prediction
    prediction = model.predict(X_array)

    # Display prediction
    st.write(f"Price prediction is ${prediction[0]:,.2f}")

else:
    st.write("Please use predict button after entering values")






#Order of X ['number of bedrooms', 'number of bathrooms', 'living area',
#'condition of the house', 'Number of schools nearby']