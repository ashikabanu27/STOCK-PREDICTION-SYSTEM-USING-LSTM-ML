import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Streamlit App Title
st.title("📈 Stock Price Prediction using LSTM")

# Sidebar for User Input
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, GOOGL)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

# Fetch Stock Data
@st.cache_data
def load_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

data = load_stock_data(ticker, start_date, end_date)

st.subheader(f"Stock Data for {ticker}")
st.write(data.tail())

# Plot Closing Price
st.subheader("Stock Price Chart")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data["Close"], label="Closing Price", color='blue')
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Preprocess Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']].values)

# Create Sequences
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Train Model
def build_lstm_model():
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

st.subheader("🔄 Training Model...")
model = build_lstm_model()
model.fit(X, y, epochs=10, batch_size=32, verbose=0)

# Predict
predicted_prices = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot Predictions
st.subheader("📊 Actual vs Predicted Stock Prices")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index[60:], data["Close"][60:], label="Actual Price", color="red")
ax.plot(data.index[60:], predicted_prices, label="Predicted Price", color="blue")
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.legend()
st.pyplot(fig)

st.success("✅ Stock Prediction Completed!")
