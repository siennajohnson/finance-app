import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("📈 Financial Analytics Dashboard")

# Inputs
stock = st.text_input("Enter Stock Ticker", "AAPL")
portfolio = st.text_input("Enter Portfolio Tickers (comma separated)", "AAPL,MSFT,NVDA")
start = st.date_input("Start Date", pd.to_datetime("2024-01-01"))
end = st.date_input("End Date", pd.to_datetime("today"))

# Download stock data
df = yf.download(stock, start=start, end=end)

if not df.empty:

    st.header("📊 Individual Stock Analysis")

    # Moving averages
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50"))
    st.plotly_chart(fig)

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    st.subheader("RSI")
    st.line_chart(df["RSI"])

    # Returns
    df["Returns"] = df["Close"].pct_change()

    volatility = df["Returns"].std() * np.sqrt(252)

    st.write("Annualized Volatility:", round(volatility,4))

    # Trading Signals
    latest_rsi = df["RSI"].iloc[-1]

    if latest_rsi < 30:
        st.success("BUY Signal: Oversold")
    elif latest_rsi > 70:
        st.error("SELL Signal: Overbought")
    else:
        st.info("HOLD Signal")

# Portfolio
st.header("💼 Portfolio Analysis")

tickers = [x.strip() for x in portfolio.split(",")]

prices = yf.download(tickers, start=start, end=end)["Close"]

returns = prices.pct_change()

weights = np.array([1/len(tickers)] * len(tickers))

portfolio_returns = returns.dot(weights)

cum_returns = (1 + portfolio_returns).cumprod()

st.line_chart(cum_returns)

port_vol = portfolio_returns.std() * np.sqrt(252)
port_return = portfolio_returns.mean() * 252
sharpe = port_return / port_vol

st.write("Portfolio Return:", round(port_return,4))
st.write("Portfolio Volatility:", round(port_vol,4))
st.write("Sharpe Ratio:", round(sharpe,4))

# Benchmark
spy = yf.download("SPY", start=start, end=end)["Close"]
spy_returns = spy.pct_change()
spy_growth = (1 + spy_returns).cumprod()

st.subheader("Portfolio vs SPY")

compare = pd.DataFrame({
    "Portfolio": cum_returns,
    "SPY": spy_growth
})

st.line_chart(compare)
