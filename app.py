# Stock Analysis & Portfolio Dashboard
# Streamlit App
# -------------------------------------------------
# FEATURES:
# Part 1: Individual Stock Analysis
# - 6 months historical data (yfinance)
# - Moving averages (20/50)
# - Trend detection
# - RSI (14-day)
# - Volatility (20-day annualized)
# - Buy/Sell/Hold recommendation
#
# Part 2: Portfolio Dashboard
# - 5-stock portfolio with weights
# - Benchmark: SPY
# - 1-year performance
# - Returns, volatility, Sharpe ratio
# -------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Stock & Portfolio Analyzer", layout="wide")

# ------------------------------
# Helper Functions
# ------------------------------

def get_data(ticker, period="6mo"):
    df = yf.download(ticker, period=period)
    df = df[['Close']].dropna()
    return df


def rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def analyze_stock(ticker):
    df = get_data(ticker, "6mo")

    df['20MA'] = df['Close'].rolling(20).mean()
    df['50MA'] = df['Close'].rolling(50).mean()
    df['RSI'] = rsi(df['Close'], 14)

    latest = df.iloc[-1]

    price = latest['Close']
    ma20 = latest['20MA']
    ma50 = latest['50MA']
    rsi_val = latest['RSI']

    # Trend
    if price > ma20 > ma50:
        trend = "Strong Uptrend"
    elif price < ma20 < ma50:
        trend = "Strong Downtrend"
    else:
        trend = "Mixed Trend"

    # RSI Signal
    if rsi_val > 70:
        rsi_signal = "Overbought (Sell signal)"
    elif rsi_val < 30:
        rsi_signal = "Oversold (Buy signal)"
    else:
        rsi_signal = "Neutral"

    # Volatility (20-day)
    returns = df['Close'].pct_change()
    vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)

    if vol > 0.40:
        vol_level = "High"
    elif vol > 0.25:
        vol_level = "Medium"
    else:
        vol_level = "Low"

    # Recommendation logic
    if trend == "Strong Uptrend" and rsi_val < 70:
        rec = "Buy"
        reason = "Uptrend with no overbought signal"
    elif trend == "Strong Downtrend" or rsi_val > 70:
        rec = "Sell"
        reason = "Downtrend or overbought conditions"
    else:
        rec = "Hold"
        reason = "Mixed signals"

    return {
        "Price": price,
        "Trend": trend,
        "RSI": rsi_val,
        "RSI Signal": rsi_signal,
        "Volatility": vol,
        "Vol Level": vol_level,
        "Recommendation": rec,
        "Reason": reason
    }


def portfolio_analysis(tickers, weights):
    data = yf.download(tickers + ["SPY"], period="1y")["Close"]
    data = data.dropna()

    returns = data.pct_change().dropna()

    portfolio_returns = (returns[tickers] * weights).sum(axis=1)
    benchmark_returns = returns["SPY"]

    total_port_return = (1 + portfolio_returns).prod() - 1
    benchmark_return = (1 + benchmark_returns).prod() - 1

    vol = portfolio_returns.std() * np.sqrt(252)

    sharpe = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))

    return {
        "Portfolio Return": total_port_return,
        "Benchmark Return": benchmark_return,
        "Outperformance": total_port_return - benchmark_return,
        "Volatility": vol,
        "Sharpe Ratio": sharpe
    }

# ------------------------------
# UI
# ------------------------------

st.title("📊 Stock Analysis & Portfolio Dashboard")

menu = st.sidebar.radio("Select Mode", ["Individual Stock", "Portfolio Analysis"])

# ------------------------------
# Part 1
# ------------------------------

if menu == "Individual Stock":
    st.header("Individual Stock Analysis")

    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)")

    if st.button("Analyze") and ticker:
        result = analyze_stock(ticker.upper())

        st.subheader(f"Results for {ticker.upper()}")

        st.write(f"**Price:** {result['Price']:.2f}")
        st.write(f"**Trend:** {result['Trend']}")
        st.write(f"**RSI:** {result['RSI']:.2f} → {result['RSI Signal']}")
        st.write(f"**Volatility:** {result['Volatility']:.2%} ({result['Vol Level']})")

        st.success(f"Recommendation: {result['Recommendation']}")
        st.info(result['Reason'])

# ------------------------------
# Part 2
# ------------------------------

if menu == "Portfolio Analysis":
    st.header("Portfolio Performance Dashboard")

    st.write("Enter 5 stocks and weights (must sum to 1.0)")

    tickers = []
    weights = []

    for i in range(5):
        col1, col2 = st.columns(2)
        with col1:
            t = st.text_input(f"Stock {i+1}", key=f"t{i}")
        with col2:
            w = st.number_input(f"Weight {i+1}", key=f"w{i}", min_value=0.0, max_value=1.0, step=0.05)
        if t:
            tickers.append(t.upper())
            weights.append(w)

    if st.button("Run Portfolio Analysis"):
        if abs(sum(weights) - 1.0) > 0.01:
            st.error("Weights must sum to 1.0")
        else:
            result = portfolio_analysis(tickers, weights)

            st.subheader("Results")
            st.write(f"Portfolio Return: {result['Portfolio Return']:.2%}")
            st.write(f"Benchmark (SPY): {result['Benchmark Return']:.2%}")
            st.write(f"Outperformance: {result['Outperformance']:.2%}")
            st.write(f"Volatility: {result['Volatility']:.2%}")
            st.write(f"Sharpe Ratio: {result['Sharpe Ratio']:.2f}")

            if result['Outperformance'] > 0:
                st.success("Portfolio outperformed benchmark")
            else:
                st.warning("Portfolio underperformed benchmark")

            if result['Sharpe Ratio'] > 1:
                st.success("Efficient portfolio (good risk-adjusted returns)")
            else:
                st.info("Lower risk-adjusted efficiency")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("Built with Streamlit + yfinance")





