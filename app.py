import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load the trained model and scaler
model = tf.keras.models.load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

# App UI
st.title("Stock Price Prediction")
input_price = st.number_input("Enter the last closing price:")

if st.button("Predict"):
    scaled_input = scaler.transform(np.array([[input_price]]))
    prediction = model.predict(np.array([scaled_input]))
    predicted_price = scaler.inverse_transform(prediction)
    st.write(f"Predicted Price: {predicted_price[0][0]:.2f}")
