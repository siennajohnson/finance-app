## ============================================================
## FIN 330 — Final Project: Stock Analytics & Portfolio Dashboard
## ============================================================
## This Streamlit app covers two parts:
##   Part 1: Individual Stock Analysis (Trend, RSI, Volatility, Recommendation)
##   Part 2: Portfolio Performance Dashboard (Returns, Volatility, Sharpe Ratio)
##
## To run locally:   streamlit run fin330_app.py
## To deploy:        Push to GitHub → connect to Streamlit Community Cloud
## ============================================================


# ── Import Libraries ──────────────────────────────────────────
import streamlit as st          # Web app framework
import pandas as pd             # Data manipulation
import numpy as np              # Numerical calculations
import matplotlib.pyplot as plt # Plotting
import yfinance as yf           # Yahoo Finance data


# ── Page Configuration ────────────────────────────────────────
# Must be the first Streamlit command in the script
st.set_page_config(
    page_title="FIN 330 | Stock & Portfolio Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── Custom CSS Styling ────────────────────────────────────────
# Injects CSS directly into the Streamlit app for visual polish
st.markdown("""
<style>
    /* Main background and font */
    .main { background-color: #0f1117; }
    h1, h2, h3 { color: #4fc3f7; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 12px;
    }

    /* Signal badge styles */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 15px;
        margin: 4px 0;
    }
    .badge-buy    { background:#1b5e20; color:#a5d6a7; }
    .badge-sell   { background:#b71c1c; color:#ef9a9a; }
    .badge-hold   { background:#e65100; color:#ffe0b2; }
    .badge-up     { background:#1b5e20; color:#a5d6a7; }
    .badge-down   { background:#b71c1c; color:#ef9a9a; }
    .badge-mixed  { background:#37474f; color:#b0bec5; }
    .badge-ob     { background:#b71c1c; color:#ef9a9a; }   /* Overbought */
    .badge-os     { background:#1b5e20; color:#a5d6a7; }   /* Oversold   */
    .badge-neutral{ background:#37474f; color:#b0bec5; }
    .badge-high   { background:#b71c1c; color:#ef9a9a; }
    .badge-medium { background:#e65100; color:#ffe0b2; }
    .badge-low    { background:#1b5e20; color:#a5d6a7; }

    /* Section divider */
    hr { border-color: #2e3250; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR — User Inputs
# ══════════════════════════════════════════════════════════════
# The sidebar holds all user-configurable parameters so the
# main panel stays clean and focused on results.

with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    # ── Part 1 Inputs ─────────────────────────────────────────
    st.subheader("📊 Part 1: Individual Stock")

    # User types in any valid ticker symbol (e.g. AAPL, TSLA)
    ticker_input = st.text_input(
        "Stock Ticker Symbol",
        value="AAPL",
        help="Enter any valid Yahoo Finance ticker (e.g. AAPL, MSFT, TSLA)"
    ).upper().strip()

    st.markdown("---")

    # ── Part 2 Inputs ─────────────────────────────────────────
    st.subheader("💼 Part 2: Portfolio")

    # Five stock tickers for the portfolio
    st.markdown("**Enter 5 portfolio stocks:**")
    p1 = st.text_input("Stock 1", "AAPL").upper().strip()
    p2 = st.text_input("Stock 2", "MSFT").upper().strip()
    p3 = st.text_input("Stock 3", "GOOGL").upper().strip()
    p4 = st.text_input("Stock 4", "AMZN").upper().strip()
    p5 = st.text_input("Stock 5", "NVDA").upper().strip()

    # Portfolio weights — must sum to 1.00
    st.markdown("**Assign Weights (must sum to 1.00):**")
    w1 = st.number_input("Weight 1", 0.0, 1.0, 0.20, 0.05)
    w2 = st.number_input("Weight 2", 0.0, 1.0, 0.20, 0.05)
    w3 = st.number_input("Weight 3", 0.0, 1.0, 0.20, 0.05)
    w4 = st.number_input("Weight 4", 0.0, 1.0, 0.20, 0.05)
    w5 = st.number_input("Weight 5", 0.0, 1.0, 0.20, 0.05)

    # Validate weight sum
    total_weight = round(w1 + w2 + w3 + w4 + w5, 2)
    if total_weight != 1.00:
        st.error(f"⚠️ Weights sum to {total_weight:.2f} — must equal 1.00")
    else:
        st.success("✅ Weights sum to 1.00")

    # Benchmark ETF
    benchmark = st.text_input("Benchmark ETF", "SPY").upper().strip()

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True)


# ══════════════════════════════════════════════════════════════
# APP HEADER
# ══════════════════════════════════════════════════════════════
st.title("📈 FIN 330 — Stock Analytics & Portfolio Dashboard")
st.caption("Real-time financial analysis powered by Yahoo Finance · FIN 330 Final Project")
st.markdown("---")


# ══════════════════════════════════════════════════════════════
# PART 1: INDIVIDUAL STOCK ANALYSIS
# ══════════════════════════════════════════════════════════════
st.header("Part 1: Individual Stock Analysis")

if run_btn:

    # ── Step 1: Data Collection ───────────────────────────────
    ## Step 1 — Download 6 months of daily closing price data
    # yfinance pulls OHLCV data directly from Yahoo Finance
    st.subheader(f"📥 Step 1: Data Collection — {ticker_input}")

    with st.spinner(f"Downloading data for {ticker_input}..."):
        try:
            # Download 6 months of data; auto_adjust=True accounts for splits/dividends
            raw = yf.download(ticker_input, period="6mo", auto_adjust=True)

            # Validate that data was returned
            if raw.empty:
                st.error(f"No data found for '{ticker_input}'. Check the ticker symbol.")
                st.stop()

            # Extract the closing price as a 1-D Series for calculations
            close = raw["Close"].squeeze()

            st.success(f"✅ Downloaded {len(close)} trading days of data.")

            # Show a preview of the raw data table
            with st.expander("📋 View Raw Data (first 5 rows)"):
                st.dataframe(raw.head(), use_container_width=True)

        except Exception as e:
            st.error(f"Error downloading data: {e}")
            st.stop()

    st.markdown("---")

    # ── Step 2: Trend Analysis (Moving Averages) ──────────────
    ## Step 2 — Calculate Moving Averages and determine trend direction
    # MA20 = average closing price over last 20 trading days
    # MA50 = average closing price over last 50 trading days
    st.subheader("📉 Step 2: Trend Analysis (Moving Averages)")

    # Calculate rolling moving averages on the full series
    ma_20_series = close.rolling(20).mean()
    ma_50_series = close.rolling(50).mean()

    # Most recent values (scalar floats for comparison)
    current_price = float(close.iloc[-1])
    ma_20         = float(ma_20_series.iloc[-1])
    ma_50         = float(ma_50_series.iloc[-1])

    # Display key metrics in a clean row of cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price",  f"${current_price:.2f}")
    col2.metric("20-Day MA",      f"${ma_20:.2f}",
                delta=f"{current_price - ma_20:+.2f} vs price")
    col3.metric("50-Day MA",      f"${ma_50:.2f}",
                delta=f"{current_price - ma_50:+.2f} vs price")

    # ── Trend Signal Logic ────────────────────────────────────
    # Strong Uptrend:   Price > MA20 > MA50
    # Strong Downtrend: Price < MA20 < MA50
    # Mixed Trend:      Anything else
    if current_price > ma_20 and ma_20 > ma_50:
        trend        = "Strong Uptrend"
        trend_class  = "badge-up"
        trend_detail = "Price is above both moving averages — bullish momentum."
    elif current_price < ma_20 and ma_20 < ma_50:
        trend        = "Strong Downtrend"
        trend_class  = "badge-down"
        trend_detail = "Price is below both moving averages — bearish momentum."
    else:
        trend        = "Mixed Trend"
        trend_class  = "badge-mixed"
        trend_detail = "Price is between the moving averages — no clear direction."

    st.markdown(f"**Trend Signal:** <span class='badge {trend_class}'>{trend}</span>",
                unsafe_allow_html=True)
    st.caption(trend_detail)

    # ── Moving Average Chart ──────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    fig1.patch.set_facecolor("#0f1117")
    ax1.set_facecolor("#1e2130")

    ax1.plot(close.index,          close.values,              color="#4fc3f7", linewidth=2,   label="Close Price")
    ax1.plot(ma_20_series.index,   ma_20_series.values,       color="#ffb300", linewidth=1.5, label="20-Day MA",   linestyle="--")
    ax1.plot(ma_50_series.index,   ma_50_series.values,       color="#ef5350", linewidth=1.5, label="50-Day MA",   linestyle="--")

    ax1.set_title(f"{ticker_input} — Price vs Moving Averages", color="white", fontsize=14)
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#1e2130", labelcolor="white")
    ax1.spines[["top","right","left","bottom"]].set_color("#2e3250")
    ax1.yaxis.label.set_color("white")
    ax1.xaxis.label.set_color("white")

    st.pyplot(fig1)
    plt.close(fig1)   # Free memory after rendering

    st.markdown("---")

    # ── Step 3: Momentum — RSI ────────────────────────────────
    ## Step 3 — Relative Strength Index (RSI) over 14 days
    # RSI measures momentum: how fast and how much the price is changing.
    # Formula:  RSI = 100 − (100 / (1 + RS))   where RS = avg_gain / avg_loss
    st.subheader("⚡ Step 3: Momentum (RSI)")

    # Daily price changes
    delta  = close.diff()

    # Separate gains (positive changes) and losses (negative changes)
    gains  = delta.clip(lower=0)          # Keep only positive values; 0 elsewhere
    losses = -delta.clip(upper=0)         # Flip negatives to positives; 0 elsewhere

    # Simple 14-period averages (project uses simple average, not Wilder's smoothing)
    avg_gain = float(gains.rolling(14).mean().iloc[-1])
    avg_loss = float(losses.rolling(14).mean().iloc[-1])

    # RSI calculation — avoid division by zero
    if avg_loss == 0:
        rsi = 100.0    # Pure upward move; RSI pegged at 100
    else:
        rs  = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

    # ── RSI Signal Logic ──────────────────────────────────────
    # Overbought  (RSI > 70): price may be too high, possible sell signal
    # Oversold    (RSI < 30): price may be too low,  possible buy signal
    # Neutral     (30–70)   : no strong momentum signal
    if rsi > 70:
        rsi_signal      = "Overbought — Possible Sell Signal"
        rsi_class       = "badge-ob"
        rsi_description = "RSI above 70 suggests the stock may be overvalued in the short term."
    elif rsi < 30:
        rsi_signal      = "Oversold — Possible Buy Signal"
        rsi_class       = "badge-os"
        rsi_description = "RSI below 30 suggests the stock may be undervalued in the short term."
    else:
        rsi_signal      = "Neutral"
        rsi_class       = "badge-neutral"
        rsi_description = "RSI between 30–70 indicates no extreme momentum in either direction."

    st.metric("14-Day RSI", f"{rsi:.2f}")
    st.markdown(f"**RSI Signal:** <span class='badge {rsi_class}'>{rsi_signal}</span>",
                unsafe_allow_html=True)
    st.caption(rsi_description)

    # ── RSI Chart ─────────────────────────────────────────────
    rsi_series = 100 - (100 / (1 + gains.rolling(14).mean() / losses.rolling(14).mean().replace(0, np.nan)))

    fig2, ax2 = plt.subplots(figsize=(12, 3))
    fig2.patch.set_facecolor("#0f1117")
    ax2.set_facecolor("#1e2130")

    ax2.plot(rsi_series.index, rsi_series.values, color="#ce93d8", linewidth=1.8, label="RSI (14)")
    ax2.axhline(70,  color="#ef5350", linestyle="--", linewidth=1, label="Overbought (70)")
    ax2.axhline(30,  color="#66bb6a", linestyle="--", linewidth=1, label="Oversold (30)")
    ax2.fill_between(rsi_series.index, rsi_series.values, 70,
                     where=(rsi_series >= 70), alpha=0.2, color="#ef5350")
    ax2.fill_between(rsi_series.index, rsi_series.values, 30,
                     where=(rsi_series <= 30), alpha=0.2, color="#66bb6a")
    ax2.set_ylim(0, 100)
    ax2.set_title(f"{ticker_input} — RSI (14-Day)", color="white", fontsize=12)
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#1e2130", labelcolor="white", fontsize=9)
    ax2.spines[["top","right","left","bottom"]].set_color("#2e3250")

    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown("---")

    # ── Step 4: Volatility ────────────────────────────────────
    ## Step 4 — Annualized Volatility (20-day rolling)
    # Daily log returns are more statistically stable than simple returns.
    # Annualize by multiplying std dev by √252 (trading days per year).
    st.subheader("🌊 Step 4: Volatility")

    # Calculate daily log returns
    daily_returns = np.log(close / close.shift(1))

    # 20-day rolling standard deviation, annualized
    vol_series        = daily_returns.rolling(20).std() * np.sqrt(252)
    current_vol       = float(vol_series.iloc[-1])
    current_vol_pct   = current_vol * 100   # Express as percentage

    # ── Volatility Signal ─────────────────────────────────────
    # High:   > 40% — large price swings, higher risk
    # Medium: 25%–40% — moderate fluctuation
    # Low:    < 25% — relatively stable
    if current_vol_pct > 40:
        vol_signal = "High Volatility"
        vol_class  = "badge-high"
        vol_detail = "Annualized volatility above 40% — significant price swings expected."
    elif current_vol_pct >= 25:
        vol_signal = "Medium Volatility"
        vol_class  = "badge-medium"
        vol_detail = "Annualized volatility between 25–40% — moderate risk level."
    else:
        vol_signal = "Low Volatility"
        vol_class  = "badge-low"
        vol_detail = "Annualized volatility below 25% — relatively stable price movement."

    st.metric("20-Day Annualized Volatility", f"{current_vol_pct:.2f}%")
    st.markdown(f"**Volatility Level:** <span class='badge {vol_class}'>{vol_signal}</span>",
                unsafe_allow_html=True)
    st.caption(vol_detail)

    # Volatility chart
    fig3, ax3 = plt.subplots(figsize=(12, 3))
    fig3.patch.set_facecolor("#0f1117")
    ax3.set_facecolor("#1e2130")

    ax3.plot(vol_series.index, vol_series.values * 100, color="#ffb300", linewidth=1.8)
    ax3.axhline(40, color="#ef5350", linestyle="--", linewidth=1, label="High (40%)")
    ax3.axhline(25, color="#66bb6a", linestyle="--", linewidth=1, label="Low (25%)")
    ax3.set_title(f"{ticker_input} — 20-Day Annualized Volatility (%)", color="white", fontsize=12)
    ax3.tick_params(colors="white")
    ax3.legend(facecolor="#1e2130", labelcolor="white", fontsize=9)
    ax3.spines[["top","right","left","bottom"]].set_color("#2e3250")

    st.pyplot(fig3)
    plt.close(fig3)

    st.markdown("---")

    # ── Step 5: Trading Recommendation ───────────────────────
    ## Step 5 — Combine all signals into one final recommendation
    # Logic:
    #   BUY  → Uptrend + Oversold RSI + Low/Medium Volatility
    #   SELL → Downtrend + Overbought RSI OR High Volatility
    #   HOLD → All other combinations
    st.subheader("🎯 Step 5: Trading Recommendation")

    # Encode signals as numeric scores for decision logic
    uptrend   = 1 if "Uptrend"   in trend    else (-1 if "Downtrend" in trend else 0)
    rsi_score = 1 if rsi < 30    else (-1 if rsi > 70 else 0)
    vol_ok    = current_vol_pct < 40   # True if volatility is not extreme

    score = uptrend + rsi_score   # Range: -2 to +2

    # Final recommendation
    if score >= 1 and vol_ok:
        recommendation = "BUY"
        rec_class      = "badge-buy"
        reasoning      = (
            f"{ticker_input} shows bullish momentum. "
            f"The price is {'above' if uptrend == 1 else 'near'} its moving averages, "
            f"RSI is {'oversold' if rsi_score == 1 else 'neutral'} at {rsi:.1f}, "
            f"and volatility is manageable at {current_vol_pct:.1f}%."
        )
    elif score <= -1 or not vol_ok:
        recommendation = "SELL"
        rec_class      = "badge-sell"
        reasoning      = (
            f"{ticker_input} shows bearish signals. "
            f"{'Price is below moving averages.' if uptrend == -1 else ''} "
            f"{'RSI is overbought at ' + str(round(rsi, 1)) + '.' if rsi_score == -1 else ''} "
            f"{'High volatility of ' + str(round(current_vol_pct, 1)) + '% increases risk.' if not vol_ok else ''}"
        ).strip()
    else:
        recommendation = "HOLD"
        rec_class      = "badge-hold"
        reasoning      = (
            f"{ticker_input} shows mixed signals. "
            f"Trend is {trend.lower()}, RSI is {rsi:.1f} (neutral zone), "
            f"and volatility is {current_vol_pct:.1f}%. No clear entry or exit point."
        )

    # Display recommendation prominently
    st.markdown(
        f"### Final Recommendation: <span class='badge {rec_class}' "
        f"style='font-size:20px;padding:6px 20px;'>{recommendation}</span>",
        unsafe_allow_html=True
    )
    st.info(f"**Reasoning:** {reasoning}")

    # Summary table
    summary_df = pd.DataFrame({
        "Indicator":  ["Current Price", "20-Day MA", "50-Day MA", "Trend", "RSI (14)", "RSI Signal", "Volatility (Ann.)", "Volatility Level", "Recommendation"],
        "Value":      [f"${current_price:.2f}", f"${ma_20:.2f}", f"${ma_50:.2f}", trend, f"{rsi:.2f}", rsi_signal, f"{current_vol_pct:.2f}%", vol_signal, recommendation]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")


    # ══════════════════════════════════════════════════════════
    # PART 2: PORTFOLIO PERFORMANCE DASHBOARD
    # ══════════════════════════════════════════════════════════
    st.header("Part 2: Portfolio Performance Dashboard")

    # Validate weights before continuing
    if total_weight != 1.00:
        st.error("⚠️ Portfolio weights do not sum to 1.00. Adjust in the sidebar.")
        st.stop()

    portfolio_tickers = [p1, p2, p3, p4, p5]
    weights           = np.array([w1, w2, w3, w4, w5])

    # ── Step 1: Download Portfolio Data ───────────────────────
    ## Step 1 — Download 1 year of price data for all 5 stocks + benchmark
    st.subheader("📥 Step 1: Portfolio Data Collection")

    with st.spinner("Downloading portfolio and benchmark data..."):
        try:
            all_tickers  = portfolio_tickers + [benchmark]
            port_raw     = yf.download(all_tickers, period="1y", auto_adjust=True)["Close"]

            # Drop any tickers that returned no data
            port_raw.dropna(axis=1, how="all", inplace=True)

            missing = [t for t in all_tickers if t not in port_raw.columns]
            if missing:
                st.warning(f"Could not fetch data for: {missing}. They will be excluded.")

            st.success(f"✅ Downloaded 1 year of data for: {', '.join(port_raw.columns.tolist())}")

            with st.expander("📋 View Portfolio Price Data (first 5 rows)"):
                st.dataframe(port_raw.head(), use_container_width=True)

        except Exception as e:
            st.error(f"Error downloading portfolio data: {e}")
            st.stop()

    st.markdown("---")

    # ── Step 2: Return Calculations ───────────────────────────
    ## Step 2 — Calculate daily returns for each stock, the portfolio, and the benchmark
    # Daily return = (today's price − yesterday's price) / yesterday's price
    st.subheader("📊 Step 2: Return Calculations")

    # Daily simple returns for all tickers
    daily_ret = port_raw.pct_change().dropna()

    # Separate benchmark returns from stock returns
    bench_ret = daily_ret[benchmark] if benchmark in daily_ret.columns else None

    # Filter to only portfolio tickers that exist in the data
    valid_port = [t for t in portfolio_tickers if t in daily_ret.columns]
    valid_wts  = np.array([weights[portfolio_tickers.index(t)] for t in valid_port])

    # Re-normalize weights if any tickers were dropped
    if valid_wts.sum() != 1.0:
        valid_wts = valid_wts / valid_wts.sum()

    # Portfolio daily return = weighted sum of individual stock daily returns
    port_daily_ret = (daily_ret[valid_port] * valid_wts).sum(axis=1)

    with st.expander("📋 View Daily Returns (first 5 rows)"):
        st.dataframe(daily_ret.head(), use_container_width=True)

    st.markdown("---")

    # ── Step 3: Performance Metrics ───────────────────────────
    ## Step 3 — Compute key performance metrics
    # Total Return     = cumulative product of (1 + daily_return) − 1
    # Annualized Vol.  = std(daily_returns) × √252
    # Sharpe Ratio     = (total_return / years) / annualized_vol  [simplified, rf=0]
    st.subheader("📈 Step 3: Performance Metrics")

    # Total (cumulative) return over the 1-year period
    port_total_ret  = float((1 + port_daily_ret).prod() - 1)
    bench_total_ret = float((1 + bench_ret).prod() - 1)    if bench_ret is not None else None

    # Outperformance relative to benchmark
    outperf = port_total_ret - bench_total_ret if bench_total_ret is not None else None

    # Annualized volatility: std of daily returns scaled to yearly
    port_vol  = float(port_daily_ret.std() * np.sqrt(252))
    bench_vol = float(bench_ret.std() * np.sqrt(252))       if bench_ret is not None else None

    # Sharpe Ratio = annualized return / annualized volatility (risk-free rate = 0)
    # Higher is better: measures return earned per unit of risk
    sharpe = port_total_ret / port_vol if port_vol > 0 else 0.0

    # ── Display Metrics ───────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Portfolio Return",   f"{port_total_ret*100:.2f}%")
    m2.metric("Benchmark Return",   f"{bench_total_ret*100:.2f}%" if bench_total_ret is not None else "N/A",
              delta=f"{outperf*100:+.2f}% vs {benchmark}" if outperf is not None else None)
    m3.metric("Portfolio Vol.",     f"{port_vol*100:.2f}%")
    m4.metric("Benchmark Vol.",     f"{bench_vol*100:.2f}%" if bench_vol is not None else "N/A")
    m5.metric("Sharpe Ratio",       f"{sharpe:.2f}")

    st.markdown("---")

    # ── Cumulative Returns Chart ───────────────────────────────
    ## Visualize how $1 invested at start would grow over the year
    st.subheader("📉 Cumulative Returns: Portfolio vs Benchmark")

    cum_port  = (1 + port_daily_ret).cumprod()
    cum_bench = (1 + bench_ret).cumprod() if bench_ret is not None else None

    fig4, ax4 = plt.subplots(figsize=(12, 5))
    fig4.patch.set_facecolor("#0f1117")
    ax4.set_facecolor("#1e2130")

    ax4.plot(cum_port.index,  cum_port.values,  color="#4fc3f7", linewidth=2,   label="Portfolio")
    if cum_bench is not None:
        ax4.plot(cum_bench.index, cum_bench.values, color="#ef5350", linewidth=2, linestyle="--", label=benchmark)

    ax4.axhline(1.0, color="#555", linestyle=":", linewidth=1)
    ax4.set_title("Cumulative Returns (1 Year)", color="white", fontsize=14)
    ax4.tick_params(colors="white")
    ax4.legend(facecolor="#1e2130", labelcolor="white")
    ax4.spines[["top","right","left","bottom"]].set_color("#2e3250")
    ax4.set_ylabel("Growth of $1", color="white")

    st.pyplot(fig4)
    plt.close(fig4)

    st.markdown("---")

    # ── Step 4: Interpretation ────────────────────────────────
    ## Step 4 — Translate numbers into plain-English interpretation
    # This section answers the three graded interpretation questions.
    st.subheader("💬 Step 4: Interpretation")

    # Question 1: Did the portfolio outperform?
    if outperf is not None:
        if outperf > 0:
            st.success(f"✅ **Outperformance:** Your portfolio returned **{port_total_ret*100:.2f}%**, "
                       f"beating {benchmark} by **{outperf*100:.2f} percentage points**.")
        else:
            st.error(f"❌ **Underperformance:** Your portfolio returned **{port_total_ret*100:.2f}%**, "
                     f"lagging {benchmark} by **{abs(outperf)*100:.2f} percentage points**.")

    # Question 2: Was it more or less risky?
    if bench_vol is not None:
        if port_vol > bench_vol:
            st.warning(f"⚠️ **Risk:** Your portfolio ({port_vol*100:.2f}% volatility) was **more risky** "
                       f"than {benchmark} ({bench_vol*100:.2f}%).")
        else:
            st.info(f"🛡️ **Risk:** Your portfolio ({port_vol*100:.2f}% volatility) was **less risky** "
                    f"than {benchmark} ({bench_vol*100:.2f}%).")

    # Question 3: Was it efficient (Sharpe Ratio)?
    # Sharpe > 1.0 is generally considered good; > 2.0 is excellent
    if sharpe > 1.5:
        st.success(f"📊 **Efficiency (Sharpe Ratio: {sharpe:.2f}):** Excellent risk-adjusted returns.")
    elif sharpe > 0.5:
        st.info(f"📊 **Efficiency (Sharpe Ratio: {sharpe:.2f}):** Acceptable risk-adjusted returns.")
    else:
        st.warning(f"📊 **Efficiency (Sharpe Ratio: {sharpe:.2f}):** Poor risk-adjusted returns — "
                   f"high risk relative to return.")

    # ── Portfolio Weights Pie Chart ────────────────────────────
    st.markdown("---")
    st.subheader("🥧 Portfolio Allocation")

    fig5, ax5 = plt.subplots(figsize=(6, 6))
    fig5.patch.set_facecolor("#0f1117")
    ax5.set_facecolor("#0f1117")

    colors = ["#4fc3f7", "#ffb300", "#66bb6a", "#ef5350", "#ce93d8"]
    ax5.pie(valid_wts, labels=valid_port, autopct="%1.1f%%",
            colors=colors[:len(valid_port)], textprops={"color": "white"})
    ax5.set_title("Portfolio Weights", color="white", fontsize=13)

    st.pyplot(fig5)
    plt.close(fig5)

    # ── Final Summary Table ────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Full Summary")

    summary2 = pd.DataFrame({
        "Metric": [
            "Portfolio Total Return", f"{benchmark} Total Return",
            "Outperformance", "Portfolio Ann. Volatility",
            f"{benchmark} Ann. Volatility", "Sharpe Ratio (Portfolio)"
        ],
        "Value": [
            f"{port_total_ret*100:.2f}%",
            f"{bench_total_ret*100:.2f}%" if bench_total_ret is not None else "N/A",
            f"{outperf*100:+.2f}%" if outperf is not None else "N/A",
            f"{port_vol*100:.2f}%",
            f"{bench_vol*100:.2f}%" if bench_vol is not None else "N/A",
            f"{sharpe:.2f}"
        ]
    })
    st.dataframe(summary2, use_container_width=True, hide_index=True)

# ── Placeholder when app first loads ──────────────────────────
else:
    st.info("👈 Configure your settings in the sidebar and click **Run Analysis** to begin.")
    st.markdown("""
    ### What this app covers
    **Part 1 — Individual Stock Analysis**
    - Downloads 6 months of daily price data
    - Calculates 20-day and 50-day moving averages for trend detection
    - Computes 14-day RSI for momentum analysis
    - Measures 20-day annualized volatility
    - Generates a final Buy / Hold / Sell recommendation

    **Part 2 — Portfolio Performance Dashboard**
    - Downloads 1 year of data for 5 stocks + a benchmark ETF
    - Calculates daily and total returns
    - Computes portfolio volatility and Sharpe ratio
    - Compares portfolio vs benchmark performance
    - Provides plain-English interpretation of all results
    """)
