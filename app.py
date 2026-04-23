import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Financial Analytics Project")

st.title("📊 Financial Data Analytics Dashboard")

# -----------------------------
# USER INPUTS
# -----------------------------
stock = st.text_input("Enter Stock Ticker", "AAPL")

portfolio_input = st.text_input(
    "Enter Portfolio Tickers (comma separated)",
    "AAPL,MSFT,NVDA"
)

# -----------------------------
# STEP 1 STOCK ANALYSIS
# -----------------------------
st.header("📈 Step 1: Individual Stock Analysis")

# Pull 6 months data
df = yf.download(stock, period="6mo")

if not df.empty:

    # Use Close Price
    df = df[["Close"]].dropna()

    # Moving Averages
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # Graph
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        name="Close Price"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["MA20"],
        name="20-Day MA"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["MA50"],
        name="50-Day MA"
    ))

    fig.update_layout(
        title=f"{stock} Price + Moving Averages (6 Months)",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # RSI
    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    latest_rsi = df["RSI"].iloc[-1]

    st.subheader("RSI Indicator")
    st.line_chart(df["RSI"])

    # Determine Trend
    if latest_rsi > 55:
        st.success("Trend: Upward 📈")
    elif latest_rsi < 45:
        st.error("Trend: Downward 📉")
    else:
        st.info("Trend: Mixed ↔️")

# -----------------------------
# STEP 2 PORTFOLIO
# -----------------------------
st.header("💼 Step 2: Portfolio Analysis")

tickers = [x.strip() for x in portfolio_input.split(",")]

prices = yf.download(tickers, period="6mo")["Close"]

returns = prices.pct_change().dropna()

weights = np.array([1/len(tickers)] * len(tickers))

portfolio_returns = returns.dot(weights)

portfolio_growth = (1 + portfolio_returns).cumprod()

# Benchmark
spy = yf.download("SPY", period="6mo")["Close"]
spy_returns = spy.pct_change().dropna()
spy_growth = (1 + spy_returns).cumprod()

# Compare Chart
compare = pd.DataFrame({
    "Portfolio": portfolio_growth,
    "SPY": spy_growth
})

st.line_chart(compare)

# Metrics
total_portfolio_return = (portfolio_growth.iloc[-1] - 1) * 100
total_spy_return = (spy_growth.iloc[-1] - 1) * 100

st.write("Portfolio Return:", round(total_portfolio_return,2), "%")
st.write("SPY Return:", round(total_spy_return,2), "%")

if total_portfolio_return > total_spy_return:
    st.success("Portfolio Outperformed SPY")
else:
    st.warning("Portfolio Underperformed SPY")
