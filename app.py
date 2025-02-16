import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained LSTM model
model_path = "lstm_model.h5"
model = load_model(model_path)

# Connect to SQLite database
db_path = "db.sqlite3"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Fetch stock names, symbols, and financial data
query = """
    SELECT s.name, f.symbol, f.date, f.close_price, f.ema20, f.ema50, f.ema100, f.ema200
    FROM Stocks_financialdata f
    JOIN Stocks_stocks s ON f.symbol = s.symbol;
"""
cursor.execute(query)
stocks_data = cursor.fetchall()
conn.close()

# Convert to DataFrame
stocks_df = pd.DataFrame(stocks_data, columns=["Stock", "Symbol", "Date", "Close Price", "EMA 20", "EMA 50", "EMA 100", "EMA 200"])

# Generate Buy/Sell signals
stocks_df['Signal'] = stocks_df.apply(lambda row: "Buy" if row['EMA 20'] > row['EMA 100'] else "Sell", axis=1)

# Keep only the latest entry for each stock
stocks_df = stocks_df.sort_values("Date").drop_duplicates(subset="Symbol", keep="last")

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

# Search bar
search_query = st.text_input("Search for a stock:", "")
filtered_df = stocks_df[stocks_df["Stock"].str.contains(search_query, case=False, na=False)]

# Color Buy/Sell recommendations
def highlight_cells(val):
    color = 'green' if val == 'Buy' else 'red'
    return f'background-color: {color}; color: white'

st.dataframe(filtered_df.style.applymap(highlight_cells, subset=['Signal']))
