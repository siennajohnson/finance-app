import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock & Portfolio Analyzer", layout="wide")

st.title("📈 Stock & Portfolio Trend Analyzer")

# ---------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------

def load_data(ticker):
    df = yf.download(ticker, period="6mo", auto_adjust=True)
    return df

def calculate_rsi(data, window=14):
    delta = data["Close"].diff()

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain = pd.Series(gain, index=data.index).rolling(window=window).mean()
    loss = pd.Series(loss, index=data.index).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def stock_trend(ma20, ma50):
    if ma20.iloc[-1] > ma50.iloc[-1]:
        return "📈 Upward Trend"
    elif ma20.iloc[-1] < ma50.iloc[-1]:
        return "📉 Downward Trend"
    else:
        return "➡ Mixed Trend"

def portfolio_metrics(returns, benchmark_returns):
    total_return = (1 + returns).prod() - 1
    benchmark_return = (1 + benchmark_returns).prod() - 1

    volatility = returns.std() * np.sqrt(252)

    risk_free = 0.02
    sharpe = ((returns.mean() * 252) - risk_free) / volatility

    difference = total_return - benchmark_return

    return total_return, benchmark_return, difference, volatility, sharpe

# ---------------------------------------------------
# PART 1 STOCK ANALYSIS
# ---------------------------------------------------

st.header("Part 1: Individual Stock Analysis")

ticker = st.text_input("Enter Stock Ticker (Example: AAPL)").upper()

if ticker:
    try:
        df = load_data(ticker)

        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        df["RSI"] = calculate_rsi(df)

        trend = stock_trend(df["MA20"], df["MA50"])

        st.subheader(f"{ticker} Moving Average Chart")

        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df.index, df["Close"], label="Close Price")
        ax.plot(df.index, df["MA20"], label="20 Day MA")
        ax.plot(df.index, df["MA50"], label="50 Day MA")
        ax.legend()
        st.pyplot(fig)

        latest_rsi = df["RSI"].iloc[-1]

        st.metric("Current RSI", round(latest_rsi,2))
        st.success(trend)

    except:
        st.error("Invalid ticker symbol")

# ---------------------------------------------------
# PART 2 PORTFOLIO ANALYSIS
# ---------------------------------------------------

st.header("Part 2: Portfolio Analysis")

st.write("Enter up to 5 tickers and portfolio weights.")

tickers = []
weights = []

for i in range(5):
    col1, col2 = st.columns(2)

    with col1:
        t = st.text_input(f"Ticker {i+1}", key=f"ticker{i}")

    with col2:
        w = st.number_input(f"Weight % {i+1}", min_value=0.0, max_value=100.0, key=f"weight{i}")

    if t:
        tickers.append(t.upper())
        weights.append(w / 100)

benchmark = st.text_input("Benchmark Ticker (Example: SPY)", value="SPY").upper()

if st.button("Analyze Portfolio"):

    if len(tickers) == 0:
        st.error("Please enter portfolio tickers")
    elif round(sum(weights),2) != 1.00:
        st.error("Weights must equal 100%")
    else:
        try:
            portfolio_prices = pd.DataFrame()

            for t in tickers:
                temp = load_data(t)["Close"]
                portfolio_prices[t] = temp

            portfolio_returns = portfolio_prices.pct_change().dropna()

            weighted_returns = (portfolio_returns * weights).sum(axis=1)

            benchmark_data = load_data(benchmark)["Close"]
            benchmark_returns = benchmark_data.pct_change().dropna()

            benchmark_returns = benchmark_returns.loc[weighted_returns.index]

            total_return, benchmark_return, difference, volatility, sharpe = portfolio_metrics(
                weighted_returns, benchmark_returns
            )

            st.subheader("Portfolio Results")

            st.metric("Total Return", f"{total_return*100:.2f}%")
            st.metric("Benchmark Return", f"{benchmark_return*100:.2f}%")
            st.metric("Outperformance / Underperformance", f"{difference*100:.2f}%")
            st.metric("Annualized Volatility", f"{volatility*100:.2f}%")
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")

            if total_return > benchmark_return:
                st.success("✅ Portfolio Meets / Beats Benchmark")
            else:
                st.error("❌ Portfolio Fails to Meet Benchmark")

        except:
            st.error("Error loading portfolio data")
