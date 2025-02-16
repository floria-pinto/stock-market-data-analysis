import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load the trained model and scaler
model = tf.keras.models.load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

# UI Title
st.title("Stock Price Prediction")

# User input for last 10 closing prices
st.write("Enter the last 10 closing prices:")
price_inputs = []
for i in range(10):
    price_inputs.append(st.number_input(f"Day {i+1} Price:", min_value=0.0, format="%.2f"))

if st.button("Predict Next Price"):
    if len(price_inputs) < 10 or any(p == 0 for p in price_inputs):
        st.error("Please enter valid prices for all 10 days.")
    else:
        # Scale input prices
        scaled_input = scaler.transform(np.array(price_inputs).reshape(-1, 1)).reshape(1, 10, 1)

        # Predict next price
        prediction = model.predict(scaled_input)
        predicted_price = scaler.inverse_transform(prediction)[0][0]

        # Display Prediction
        st.write(f"Predicted Next Closing Price: **${predicted_price:.2f}**")

        # Generate Buy/Sell Signal
        signal = "Buy 📈" if predicted_price > price_inputs[-1] else "Sell 📉"
        st.write(f"**Signal:** {signal}")

        # Plot the stock trend
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, 11), price_inputs, marker="o", linestyle="-", label="Actual Prices")
        plt.plot(11, predicted_price, marker="*", markersize=12, color="red", label="Predicted Price")
        plt.xlabel("Days")
        plt.ylabel("Price ($)")
        plt.title("Stock Price Prediction")
        plt.legend()
        st.pyplot(plt)
