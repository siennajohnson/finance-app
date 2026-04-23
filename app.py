import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Finance Dashboard", layout="wide")

# -----------------------
# USER INPUTS
# -----------------------
st.title("📊 Financial Data Dashboard")

ticker = st.text_input("Enter Stock Ticker", "MSFT")
period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

# -----------------------
# DOWNLOAD DATA
# -----------------------
data = yf.download(ticker, period=period, auto_adjust=False)

if data.empty:
    st.error("No data found for this ticker.")
    st.stop()

close = data["Close"].squeeze()

# -----------------------
# MOVING AVERAGES
# -----------------------
data["MA5"] = close.rolling(5).mean()
data["MA20"] = close.rolling(20).mean()
data["MA50"] = close.rolling(50).mean()

# -----------------------
# CURRENT PRICE
# -----------------------
current_price = close.iloc[-1]

# -----------------------
# RSI FUNCTION
# -----------------------
delta = close.diff()
gains = delta.clip(lower=0)
losses = -delta.clip(upper=0)

avg_gain = gains.rolling(14).mean().iloc[-1]
avg_loss = losses.rolling(14).mean().iloc[-1]

if avg_loss == 0:
    rsi = 100
else:
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

# -----------------------
# VOLATILITY
# -----------------------
daily_returns = close.pct_change().dropna()
volatility = daily_returns.std() * np.sqrt(252)

# -----------------------
# TREND SIGNAL
# -----------------------
ma_20 = data["MA20"].iloc[-1]
ma_50 = data["MA50"].iloc[-1]

if current_price > ma_20 and current_price > ma_50:
    trend = "📈 Upward Trend"
elif current_price < ma_20 and current_price < ma_50:
    trend = "📉 Downward Trend"
else:
    trend = "🔄 Mixed Trend"

# -----------------------
# DISPLAY METRICS
# -----------------------
st.subheader(f"{ticker} Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("RSI", f"{rsi:.2f}")
col3.metric("Volatility", f"{volatility:.2%}")

st.write("### Trend Signal:", trend)

# -----------------------
# CHART
# -----------------------
st.subheader("Price + Moving Averages")

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(data["Close"], label="Price")
ax.plot(data["MA5"], label="MA5")
ax.plot(data["MA20"], label="MA20")
ax.plot(data["MA50"], label="MA50")

ax.legend()
ax.set_title(f"{ticker} Moving Averages")

st.pyplot(fig)
