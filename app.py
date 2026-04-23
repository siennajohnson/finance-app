import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator

st.set_page_config(page_title="Finance Dashboard", layout="wide")

# =========================
# DATA FUNCTIONS
# =========================

def get_stock_data(ticker, period="6mo"):
    df = yf.download(ticker, period=period)
    df = df[['Close']].dropna()
    return df

def add_moving_averages(df):
    df['20MA'] = df['Close'].rolling(20).mean()
    df['50MA'] = df['Close'].rolling(50).mean()
    return df

def add_rsi(df):
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    return df

def detect_trend(df):
    price = df['Close'].iloc[-1]
    ma20 = df['20MA'].iloc[-1]
    ma50 = df['50MA'].iloc[-1]

    if price > ma20 > ma50:
        return "Strong Uptrend"
    elif price < ma20 < ma50:
        return "Strong Downtrend"
    else:
        return "Mixed Trend"

def rsi_signal(df):
    rsi = df['RSI'].iloc[-1]
    if rsi > 70:
        return "Overbought (Sell)"
    elif rsi < 30:
        return "Oversold (Buy)"
    return "Neutral"

def volatility(df):
    returns = df['Close'].pct_change()
    vol = returns.std() * np.sqrt(252)

    if vol > 0.40:
        level = "High"
    elif vol > 0.25:
        level = "Medium"
    else:
        level = "Low"

    return vol, level

def recommendation(trend, rsi_sig):
    if trend == "Strong Uptrend" and rsi_sig != "Overbought (Sell)":
        return "BUY"
    elif trend == "Strong Downtrend" or rsi_sig == "Overbought (Sell)":
        return "SELL"
    return "HOLD"

# =========================
# PORTFOLIO FUNCTIONS
# =========================

def get_portfolio_data(tickers):
    return yf.download(tickers, period="1y")['Close']

def compute_returns(data):
    return data.pct_change().dropna()

def portfolio_returns(returns, weights):
    return returns.dot(weights)

def metrics(port_ret, bench_ret):
    total_port = (1 + port_ret).prod() - 1
    total_bench = (1 + bench_ret).prod() - 1
    vol = port_ret.std() * np.sqrt(252)
    sharpe = (port_ret.mean() * 252) / (port_ret.std() * np.sqrt(252))
    return total_port, total_bench, vol, sharpe

# =========================
# UI
# =========================

st.title("📊 Finance Analytics Dashboard")

# -------------------------
# STOCK ANALYSIS
# -------------------------
st.header("Part 1: Stock Analysis")

ticker = st.text_input("Stock Ticker", "AAPL")

if st.button("Analyze Stock"):

    df = get_stock_data(ticker)
    df = add_moving_averages(df)
    df = add_rsi(df)

    trend = detect_trend(df)
    rsi_sig = rsi_signal(df)
    vol, vol_level = volatility(df)
    rec = recommendation(trend, rsi_sig)

    st.subheader("Results")
    st.write("Trend:", trend)
    st.write("RSI:", rsi_sig)
    st.write(f"Volatility: {vol:.2%} ({vol_level})")
    st.write("Recommendation:", rec)

    st.line_chart(df[['Close', '20MA', '50MA']])

# -------------------------
# PORTFOLIO ANALYSIS
# -------------------------
st.header("Part 2: Portfolio Analysis")

tickers_input = st.text_input("5 tickers", "AAPL,MSFT,GOOGL,AMZN,TSLA")
weights_input = st.text_input("Weights", "0.2,0.2,0.2,0.2,0.2")

if st.button("Analyze Portfolio"):

    tickers = [t.strip() for t in tickers_input.split(",")]
    weights = np.array([float(w) for w in weights_input.split(",")])

    data = get_portfolio_data(tickers)
    returns = compute_returns(data)
    port_ret = portfolio_returns(returns, weights)

    bench = yf.download("SPY", period="1y")['Close'].pct_change().dropna()

    total_port, total_bench, vol, sharpe = metrics(port_ret, bench)

    st.subheader("Results")

    st.write("Portfolio Return:", f"{total_port:.2%}")
    st.write("Benchmark Return (SPY):", f"{total_bench:.2%}")
    st.write("Volatility:", f"{vol:.2%}")
    st.write("Sharpe Ratio:", f"{sharpe:.2f}")

    if total_port > total_bench:
        st.success("Outperformed Benchmark")
    else:
        st.warning("Underperformed Benchmark")
