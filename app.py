import streamlit as st
import pandas as pd
import numpy as np
import sqlite3

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

# Function to simulate Buy/Sell predictions
def predict_ema(value):
    return "Buy ↑" if value and value > np.random.uniform(0, 1) else "Sell ↓"

# Apply predictions to EMA columns
for ema_col in ["EMA 20", "EMA 30", "EMA 50", "EMA 100", "EMA 200"]:
    stocks_df[ema_col] = stocks_df[ema_col].apply(predict_ema)

# Streamlit Sidebar
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

# Main Title
st.title("Stock Predictions")

# Apply Conditional Formatting
def highlight_cells(val):
    return "background-color: red; color: white" if "Sell" in val else "background-color: green; color: white"

st.dataframe(stocks_df.style.applymap(highlight_cells))
