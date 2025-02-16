import streamlit as st
import pandas as pd
import sqlite3
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained LSTM model
model_path = "lstm_model.h5"
model = load_model(model_path)

# Connect to SQLite database
db_path = "db.sqlite3"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Fetch stock names and symbols
cursor.execute("""
    SELECT s.name, f.symbol, f.ema20, f.ema30, f.ema50, f.ema100, f.ema200
    FROM Stocks_financialdata f
    JOIN Stocks_stocks s ON f.symbol = s.symbol;
""")
stocks_data = cursor.fetchall()
conn.close()

# Convert to DataFrame
stocks_df = pd.DataFrame(stocks_data, columns=["Sector", "Symbol", "EMA 20", "EMA 30", "EMA 50", "EMA 100", "EMA 200"])

# Handle missing data
stocks_df.fillna("", inplace=True)

# Simulate EMA predictions using LSTM (Dummy example, replace with actual logic)
def predict_ema(stock_symbol):
    dummy_input = np.random.rand(1, 10, 1)  # Assuming 10 time steps
    prediction = model.predict(dummy_input)
    return "Buy ↑" if prediction[0][0] > 0.5 else "Sell ↓"

# Generate predictions
for ema_col in ["EMA 20", "EMA 30", "EMA 50", "EMA 100", "EMA 200"]:
    stocks_df[ema_col] = stocks_df["Symbol"].apply(predict_ema)

# Define cell styling function
def highlight_cells(val):
    if isinstance(val, str) and "Sell" in val:
        return "background-color: red; color: white"
    elif isinstance(val, str) and "Buy" in val:
        return "background-color: green; color: white"
    return ""

# Streamlit UI
st.sidebar.title("Stocky AI")
menu_options = ["Home", "Charts", "Sectors", "Stocks", "Alerts", "Portfolio", "Settings", "Watchlist", "Help", "About Us", "Admin"]
for option in menu_options:
    st.sidebar.button(option)

st.title("Stock Predictions")

# Display DataFrame with styling
st.dataframe(stocks_df.style.applymap(highlight_cells))
