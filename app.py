import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock & Portfolio Analyzer", layout="wide")

st.title("📈 Stock & Portfolio Trend Analyzer")

# -----------------------------
# FUNCTIONS
# -----------------------------

def load_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", auto_adjust=False, progress=False)
        return df
    except Exception as e:
        st.error(f"Download error for {ticker}: {e}")
        return pd.DataFrame()

def calculate_rsi(data, window=14):
    delta = data["Close"].diff()

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain = pd.Series(gain, index=data.index).rolling(window).mean()
    loss = pd.Series(loss, index=data.index).rolling(window).mean()

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

# -----------------------------
# PART 1: STOCK ANALYSIS
# -----------------------------

st.header("Part 1: Individual Stock Analysis")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)").upper().strip()

if ticker:

    df = load_data(ticker)

    # 🔥 FIX: Proper validation instead of fake "invalid ticker"
    if df is None or df.empty:
        st.error("❌ No data found. Check ticker symbol (e.g., AAPL, TSLA, MSFT).")
    else:
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        df["RSI"] = calculate_rsi(df)

        trend = stock_trend(df["MA20"], df["MA50"])

        st.subheader(f"{ticker} Moving Average Chart")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df["Close"], label="Close Price")
        ax.plot(df.index, df["MA20"], label="20-Day MA")
        ax.plot(df.index, df["MA50"], label="50-Day MA")
        ax.legend()
        ax.set_title(f"{ticker} Price + Moving Averages")
        st.pyplot(fig)

        st.metric("RSI", round(df["RSI"].iloc[-1], 2))
        st.success(trend)

# -----------------------------
# PART 2: PORTFOLIO ANALYSIS
# -----------------------------

st.header("Part 2: Portfolio Analysis")

st.write("Enter up to 5 tickers and their weights (must equal 100%).")

tickers = []
weights = []

for i in range(5):
    col1, col2 = st.columns(2)

    with col1:
        t = st.text_input(f"Ticker {i+1}", key=f"t{i}").upper().strip()

    with col2:
        w = st.number_input(f"Weight % {i+1}", 0.0, 100.0, key=f"w{i}")

    if t:
        tickers.append(t)
        weights.append(w / 100)

benchmark = st.text_input("Benchmark (e.g., SPY)", "SPY").upper().strip()

if st.button("Analyze Portfolio"):

    if len(tickers) == 0:
        st.error("Please enter at least one ticker.")

    elif round(sum(weights), 2) != 1.00:
        st.error("Weights must sum to 100%.")

    else:
        try:
            price_data = pd.DataFrame()

            # Load each stock safely
            valid_tickers = []

            for t in tickers:
                df = load_data(t)

                if df is None or df.empty:
                    st.warning(f"Skipping {t}: no data found")
                    continue

                price_data[t] = df["Close"]
                valid_tickers.append(t)

            if len(valid_tickers) == 0:
                st.error("No valid tickers to analyze.")
            else:
                returns = price_data.pct_change().dropna()

                # match weights to valid tickers only
                weights_filtered = [weights[tickers.index(t)] for t in valid_tickers]

                weighted_returns = (returns[valid_tickers] * weights_filtered).sum(axis=1)

                bench_df = load_data(benchmark)

                if bench_df.empty:
                    st.error("Benchmark ticker invalid.")
                else:
                    bench_returns = bench_df["Close"].pct_change().dropna()
                    bench_returns = bench_returns.loc[weighted_returns.index]

                    total_return, benchmark_return, diff, vol, sharpe = portfolio_metrics(
                        weighted_returns, bench_returns
                    )

                    st.subheader("Results")

                    st.metric("Portfolio Return", f"{total_return*100:.2f}%")
                    st.metric("Benchmark Return", f"{benchmark_return*100:.2f}%")
                    st.metric("Difference", f"{diff*100:.2f}%")
                    st.metric("Volatility", f"{vol*100:.2f}%")
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")

                    if total_return > benchmark_return:
                        st.success("✅ Portfolio beats benchmark")
                    else:
                        st.error("❌ Portfolio underperforms benchmark")

        except Exception as e:
            st.error(f"Error: {e}")
