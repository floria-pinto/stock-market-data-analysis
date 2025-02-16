import streamlit as st
import pandas as pd
import sqlite3
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained LSTM model
model_path = "/mnt/data/lstm_model.h5"
model = load_model(model_path)

# Connect to SQLite database
db_path = "/mnt/data/db.sqlite3"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Fetch stock names and symbols
cursor.execute("SELECT name, symbol FROM Stocks_stocks;")
stocks_data = cursor.fetchall()
conn.close()

# Convert to DataFrame
stocks_df = pd.DataFrame(stocks_data, columns=["Stock", "Symbol"])

# Simulate EMA predictions using LSTM (Dummy example, replace with actual logic)
def predict_ema(stock_symbol):
    dummy_input = np.random.rand(1, 10, 1)  # Assuming 10 time steps
    prediction = model.predict(dummy_input)
    return "Buy ↑" if prediction[0][0] > 0.5 else "Sell ↓"

# Generate predictions
stocks_df["EMA 20"] = stocks_df["Symbol"].apply(predict_ema)
stocks_df["EMA 50"] = stocks_df["Symbol"].apply(predict_ema)
stocks_df["EMA 100"] = stocks_df["Symbol"].apply(predict_ema)
stocks_df["EMA 200"] = stocks_df["Symbol"].apply(predict_ema)

# Streamlit UI
st.sidebar.title("Stocky AI")
st.sidebar.button("Home")
st.sidebar.button("Charts")
st.sidebar.button("Sectors")
st.sidebar.button("Stocks")
st.sidebar.button("Alerts")
st.sidebar.button("Portfolio")
st.sidebar.button("Settings")
st.sidebar.button("Watchlist")
st.sidebar.button("Help")

st.title("Stock Predictions")
st.dataframe(stocks_df.style.applymap(lambda x: "background-color: red; color: white" if "Sell" in x else "background-color: green; color: white"))
