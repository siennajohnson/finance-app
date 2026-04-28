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
    .main { background-color: #0f1117; }
    h1, h2, h3 { color: #4fc3f7; }
    [data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 12px;
    }
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
    .badge-ob     { background:#b71c1c; color:#ef9a9a; }
    .badge-os     { background:#1b5e20; color:#a5d6a7; }
    .badge-neutral{ background:#37474f; color:#b0bec5; }
    .badge-high   { background:#b71c1c; color:#ef9a9a; }
    .badge-medium { background:#e65100; color:#ffe0b2; }
    .badge-low    { background:#1b5e20; color:#a5d6a7; }
    hr { border-color: #2e3250; }
</style>
""", unsafe_allow_html=True)


# ── Helper: flatten yfinance MultiIndex columns ───────────────
# yfinance 0.2.x+ returns a MultiIndex DataFrame for ALL downloads
# (even single tickers). The level order depends on how yfinance
# was called and the version installed.
#
# Possible structures:
#   A) level 0 = Price ("Close","Open",...), level 1 = Ticker  → take level 0
#   B) level 0 = Ticker,                    level 1 = Price    → take level 1
#
# We detect which level contains "Close" and keep that one.
PRICE_COLS = {"Close", "Open", "High", "Low", "Volume", "Adj Close"}

def flatten_columns(df):
    """Return df with a flat (non-MultiIndex) column index."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    level0 = set(df.columns.get_level_values(0))
    if PRICE_COLS & level0:
        # Level 0 has price labels → keep level 0
        df.columns = df.columns.get_level_values(0)
    else:
        # Level 0 has ticker labels → keep level 1 (price labels)
        df.columns = df.columns.get_level_values(1)
    return df


# ══════════════════════════════════════════════════════════════
# SIDEBAR — User Inputs
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    # ── Part 1 Inputs ─────────────────────────────────────────
    st.subheader("📊 Part 1: Individual Stock")
    ticker_input = st.text_input(
        "Stock Ticker Symbol", value="AAPL",
        help="Enter any valid Yahoo Finance ticker (e.g. AAPL, MSFT, TSLA)"
    ).upper().strip()

    st.markdown("---")

    # ── Part 2 Inputs ─────────────────────────────────────────
    st.subheader("💼 Part 2: Portfolio")
    st.markdown("**Enter 5 portfolio stocks:**")
    p1 = st.text_input("Stock 1", "AAPL").upper().strip()
    p2 = st.text_input("Stock 2", "MSFT").upper().strip()
    p3 = st.text_input("Stock 3", "GOOGL").upper().strip()
    p4 = st.text_input("Stock 4", "AMZN").upper().strip()
    p5 = st.text_input("Stock 5", "NVDA").upper().strip()

    st.markdown("**Assign Weights (must sum to 1.00):**")
    w1 = st.number_input("Weight 1", 0.0, 1.0, 0.20, 0.05)
    w2 = st.number_input("Weight 2", 0.0, 1.0, 0.20, 0.05)
    w3 = st.number_input("Weight 3", 0.0, 1.0, 0.20, 0.05)
    w4 = st.number_input("Weight 4", 0.0, 1.0, 0.20, 0.05)
    w5 = st.number_input("Weight 5", 0.0, 1.0, 0.20, 0.05)

    total_weight = round(w1 + w2 + w3 + w4 + w5, 2)
    if total_weight != 1.00:
        st.error(f"⚠️ Weights sum to {total_weight:.2f} — must equal 1.00")
    else:
        st.success("✅ Weights sum to 1.00")

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
            # Download 6 months of data
            # auto_adjust=True: prices are split/dividend-adjusted automatically
            # progress=False:   suppresses the download progress bar in console
            raw = yf.download(ticker_input, period="6mo",
                              auto_adjust=True, progress=False)

            if raw.empty:
                st.error(f"No data found for '{ticker_input}'. Check the ticker symbol.")
                st.stop()

            # ── Flatten MultiIndex columns (version-safe fix) ──
            # Newer yfinance always returns MultiIndex; flatten_columns()
            # detects the level order and strips the ticker sub-level so
            # raw["Close"] works regardless of yfinance version.
            raw = flatten_columns(raw)

            if "Close" not in raw.columns:
                st.error(f"'Close' column missing. Got: {list(raw.columns)}")
                st.stop()

            # Extract closing price as a plain 1-D Series
            close = raw["Close"].squeeze().dropna()

            if len(close) < 50:
                st.warning(f"Only {len(close)} data points — MA50 may be unavailable.")

            st.success(f"✅ Downloaded {len(close)} trading days of data.")
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

    ma_20_series = close.rolling(20).mean()
    ma_50_series = close.rolling(50).mean()

    current_price = float(close.iloc[-1])
    ma_20         = float(ma_20_series.iloc[-1])
    ma_50         = float(ma_50_series.iloc[-1])

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("20-Day MA",     f"${ma_20:.2f}", delta=f"{current_price - ma_20:+.2f} vs price")
    col3.metric("50-Day MA",     f"${ma_50:.2f}", delta=f"{current_price - ma_50:+.2f} vs price")

    # ── Trend Signal Logic ────────────────────────────────────
    # Strong Uptrend:   Price > MA20 > MA50
    # Strong Downtrend: Price < MA20 < MA50
    # Mixed:            Anything else
    if current_price > ma_20 and ma_20 > ma_50:
        trend, trend_class = "Strong Uptrend",   "badge-up"
        trend_detail = "Price is above both moving averages — bullish momentum."
    elif current_price < ma_20 and ma_20 < ma_50:
        trend, trend_class = "Strong Downtrend", "badge-down"
        trend_detail = "Price is below both moving averages — bearish momentum."
    else:
        trend, trend_class = "Mixed Trend",      "badge-mixed"
        trend_detail = "Price is between the moving averages — no clear direction."

    st.markdown(f"**Trend Signal:** <span class='badge {trend_class}'>{trend}</span>",
                unsafe_allow_html=True)
    st.caption(trend_detail)

    # ── Moving Average Chart ──────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    fig1.patch.set_facecolor("#0f1117")
    ax1.set_facecolor("#1e2130")
    ax1.plot(close.index,        close.values,        color="#4fc3f7", lw=2,   label="Close Price")
    ax1.plot(ma_20_series.index, ma_20_series.values, color="#ffb300", lw=1.5, label="20-Day MA", ls="--")
    ax1.plot(ma_50_series.index, ma_50_series.values, color="#ef5350", lw=1.5, label="50-Day MA", ls="--")
    ax1.set_title(f"{ticker_input} — Price vs Moving Averages", color="white", fontsize=14)
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#1e2130", labelcolor="white")
    ax1.spines[["top","right","left","bottom"]].set_color("#2e3250")
    st.pyplot(fig1)
    plt.close(fig1)

    st.markdown("---")

    # ── Step 3: Momentum (RSI) ────────────────────────────────
    ## Step 3 — Relative Strength Index (14-day)
    # RSI measures speed and magnitude of price changes.
    # Formula: RSI = 100 - (100 / (1 + RS))  where RS = avg_gain / avg_loss
    st.subheader("⚡ Step 3: Momentum (RSI)")

    delta  = close.diff()
    gains  = delta.clip(lower=0)    # Keep only positive price changes
    losses = -delta.clip(upper=0)   # Flip negatives → positive loss values

    avg_gain = float(gains.rolling(14).mean().iloc[-1])
    avg_loss = float(losses.rolling(14).mean().iloc[-1])

    # Avoid division by zero (pure upward move → RSI = 100)
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs  = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

    # ── RSI Signal ────────────────────────────────────────────
    # Overbought (RSI > 70): possible sell signal
    # Oversold   (RSI < 30): possible buy signal
    # Neutral    (30–70):    no strong signal
    if rsi > 70:
        rsi_signal, rsi_class = "Overbought — Possible Sell Signal", "badge-ob"
        rsi_desc = "RSI above 70 suggests the stock may be overvalued short-term."
    elif rsi < 30:
        rsi_signal, rsi_class = "Oversold — Possible Buy Signal",    "badge-os"
        rsi_desc = "RSI below 30 suggests the stock may be undervalued short-term."
    else:
        rsi_signal, rsi_class = "Neutral",                           "badge-neutral"
        rsi_desc = "RSI between 30–70 — no extreme momentum in either direction."

    st.metric("14-Day RSI", f"{rsi:.2f}")
    st.markdown(f"**RSI Signal:** <span class='badge {rsi_class}'>{rsi_signal}</span>",
                unsafe_allow_html=True)
    st.caption(rsi_desc)

    # RSI chart
    rsi_series = 100 - (100 / (
        1 + gains.rolling(14).mean() / losses.rolling(14).mean().replace(0, np.nan)
    ))
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    fig2.patch.set_facecolor("#0f1117")
    ax2.set_facecolor("#1e2130")
    ax2.plot(rsi_series.index, rsi_series.values, color="#ce93d8", lw=1.8, label="RSI (14)")
    ax2.axhline(70, color="#ef5350", ls="--", lw=1, label="Overbought (70)")
    ax2.axhline(30, color="#66bb6a", ls="--", lw=1, label="Oversold (30)")
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
    ## Step 4 — 20-day Annualized Volatility
    # Daily log returns are used (more statistically stable).
    # Annualize: multiply std dev by √252 (trading days per year).
    st.subheader("🌊 Step 4: Volatility")

    daily_returns   = np.log(close / close.shift(1))
    vol_series      = daily_returns.rolling(20).std() * np.sqrt(252)
    current_vol_pct = float(vol_series.iloc[-1]) * 100

    # ── Volatility Signal ─────────────────────────────────────
    # High:   > 40% — large price swings
    # Medium: 25–40% — moderate risk
    # Low:    < 25%  — stable
    if current_vol_pct > 40:
        vol_signal, vol_class = "High Volatility",   "badge-high"
        vol_detail = "Annualized volatility above 40% — significant price swings."
    elif current_vol_pct >= 25:
        vol_signal, vol_class = "Medium Volatility", "badge-medium"
        vol_detail = "Annualized volatility 25–40% — moderate risk."
    else:
        vol_signal, vol_class = "Low Volatility",    "badge-low"
        vol_detail = "Annualized volatility below 25% — relatively stable."

    st.metric("20-Day Annualized Volatility", f"{current_vol_pct:.2f}%")
    st.markdown(f"**Volatility Level:** <span class='badge {vol_class}'>{vol_signal}</span>",
                unsafe_allow_html=True)
    st.caption(vol_detail)

    fig3, ax3 = plt.subplots(figsize=(12, 3))
    fig3.patch.set_facecolor("#0f1117")
    ax3.set_facecolor("#1e2130")
    ax3.plot(vol_series.index, vol_series.values * 100, color="#ffb300", lw=1.8)
    ax3.axhline(40, color="#ef5350", ls="--", lw=1, label="High (40%)")
    ax3.axhline(25, color="#66bb6a", ls="--", lw=1, label="Low (25%)")
    ax3.set_title(f"{ticker_input} — 20-Day Annualized Volatility (%)", color="white", fontsize=12)
    ax3.tick_params(colors="white")
    ax3.legend(facecolor="#1e2130", labelcolor="white", fontsize=9)
    ax3.spines[["top","right","left","bottom"]].set_color("#2e3250")
    st.pyplot(fig3)
    plt.close(fig3)

    st.markdown("---")

    # ── Step 5: Trading Recommendation ───────────────────────
    ## Step 5 — Combine all signals into one final recommendation
    # BUY:  Uptrend + Oversold RSI + volatility not extreme
    # SELL: Downtrend + Overbought RSI OR extreme volatility
    # HOLD: Mixed or conflicting signals
    st.subheader("🎯 Step 5: Trading Recommendation")

    uptrend   = 1 if "Uptrend"   in trend else (-1 if "Downtrend" in trend else 0)
    rsi_score = 1 if rsi < 30    else (-1 if rsi > 70 else 0)
    vol_ok    = current_vol_pct < 40
    score     = uptrend + rsi_score

    if score >= 1 and vol_ok:
        recommendation, rec_class = "BUY",  "badge-buy"
        reasoning = (f"{ticker_input} shows bullish momentum. Trend is {trend.lower()}, "
                     f"RSI is {rsi:.1f}, volatility is manageable at {current_vol_pct:.1f}%.")
    elif score <= -1 or not vol_ok:
        recommendation, rec_class = "SELL", "badge-sell"
        reasoning = (f"{ticker_input} shows bearish signals. "
                     f"{'Price is below MAs. ' if uptrend == -1 else ''}"
                     f"{'RSI overbought at ' + str(round(rsi,1)) + '. ' if rsi_score == -1 else ''}"
                     f"{'High volatility of ' + str(round(current_vol_pct,1)) + '%. ' if not vol_ok else ''}")
    else:
        recommendation, rec_class = "HOLD", "badge-hold"
        reasoning = (f"{ticker_input} has mixed signals. Trend: {trend.lower()}, "
                     f"RSI: {rsi:.1f}, Volatility: {current_vol_pct:.1f}%.")

    st.markdown(
        f"### Final Recommendation: <span class='badge {rec_class}' "
        f"style='font-size:20px;padding:6px 20px;'>{recommendation}</span>",
        unsafe_allow_html=True
    )
    st.info(f"**Reasoning:** {reasoning}")

    summary_df = pd.DataFrame({
        "Indicator": ["Current Price","20-Day MA","50-Day MA","Trend",
                      "RSI (14)","RSI Signal","Volatility (Ann.)","Vol. Level","Recommendation"],
        "Value":     [f"${current_price:.2f}", f"${ma_20:.2f}", f"${ma_50:.2f}", trend,
                      f"{rsi:.2f}", rsi_signal, f"{current_vol_pct:.2f}%", vol_signal, recommendation]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.markdown("---")


    # ══════════════════════════════════════════════════════════
    # PART 2: PORTFOLIO PERFORMANCE DASHBOARD
    # ══════════════════════════════════════════════════════════
    st.header("Part 2: Portfolio Performance Dashboard")

    if total_weight != 1.00:
        st.error("⚠️ Portfolio weights do not sum to 1.00. Adjust in the sidebar.")
        st.stop()

    portfolio_tickers = [p1, p2, p3, p4, p5]
    weights           = np.array([w1, w2, w3, w4, w5])

    # ── Step 1: Download Portfolio + Benchmark Data ───────────
    ## Step 1 — Download 1 year of price data for all 5 stocks + benchmark
    st.subheader("📥 Step 1: Portfolio Data Collection")

    with st.spinner("Downloading portfolio and benchmark data..."):
        try:
            all_tickers = portfolio_tickers + [benchmark]

            # Download all tickers in one call (faster than separate calls)
            port_full = yf.download(all_tickers, period="1y",
                                    auto_adjust=True, progress=False)

            if port_full.empty:
                st.error("No portfolio data returned. Check your ticker symbols.")
                st.stop()

            # ── Extract Close prices (version-safe) ───────────
            # For multi-ticker downloads, yfinance returns a MultiIndex where
            # the top level is always the price type ("Close", "Open", etc.)
            # and the second level is the ticker symbol.
            # We grab the "Close" slice from the top level directly.
            if isinstance(port_full.columns, pd.MultiIndex):
                level0 = set(port_full.columns.get_level_values(0))
                if PRICE_COLS & level0:
                    # Level 0 = price type → slice "Close" from it
                    port_raw = port_full["Close"]
                else:
                    # Level 0 = ticker → price is in level 1; transpose and slice
                    port_raw = port_full.T.xs("Close", level=1).T
            else:
                # Already flat (older yfinance) — take Close column
                port_raw = port_full[["Close"]] if "Close" in port_full.columns else port_full

            # Ensure we have a DataFrame
            if isinstance(port_raw, pd.Series):
                port_raw = port_raw.to_frame()

            # Clean up: drop all-NaN columns and all-NaN rows
            port_raw.dropna(axis=1, how="all", inplace=True)
            port_raw.dropna(how="all",          inplace=True)

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
    ## Step 2 — Daily returns for each stock, portfolio, and benchmark
    # Daily return = (price_today - price_yesterday) / price_yesterday
    st.subheader("📊 Step 2: Return Calculations")

    daily_ret  = port_raw.pct_change().dropna()
    bench_ret  = daily_ret[benchmark] if benchmark in daily_ret.columns else None

    valid_port = [t for t in portfolio_tickers if t in daily_ret.columns]
    valid_wts  = np.array([weights[portfolio_tickers.index(t)] for t in valid_port])
    if valid_wts.sum() != 1.0:
        valid_wts = valid_wts / valid_wts.sum()   # Re-normalize if tickers dropped

    # Portfolio daily return = weighted sum of individual stock returns
    port_daily_ret = (daily_ret[valid_port] * valid_wts).sum(axis=1)

    with st.expander("📋 View Daily Returns (first 5 rows)"):
        st.dataframe(daily_ret.head(), use_container_width=True)

    st.markdown("---")

    # ── Step 3: Performance Metrics ───────────────────────────
    ## Step 3 — Key metrics: total return, volatility, Sharpe ratio
    # Total Return    = product of (1 + daily_return) over all days − 1
    # Ann. Volatility = std(daily_returns) × √252
    # Sharpe Ratio    = annualized_return / annualized_volatility  (rf = 0)
    st.subheader("📈 Step 3: Performance Metrics")

    port_total_ret  = float((1 + port_daily_ret).prod() - 1)
    bench_total_ret = float((1 + bench_ret).prod() - 1) if bench_ret is not None else None
    outperf         = port_total_ret - bench_total_ret   if bench_total_ret is not None else None
    port_vol        = float(port_daily_ret.std() * np.sqrt(252))
    bench_vol       = float(bench_ret.std() * np.sqrt(252)) if bench_ret is not None else None
    sharpe          = port_total_ret / port_vol if port_vol > 0 else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Portfolio Return",  f"{port_total_ret*100:.2f}%")
    m2.metric("Benchmark Return",  f"{bench_total_ret*100:.2f}%" if bench_total_ret is not None else "N/A",
              delta=f"{outperf*100:+.2f}% vs {benchmark}" if outperf is not None else None)
    m3.metric("Portfolio Vol.",    f"{port_vol*100:.2f}%")
    m4.metric("Benchmark Vol.",    f"{bench_vol*100:.2f}%" if bench_vol is not None else "N/A")
    m5.metric("Sharpe Ratio",      f"{sharpe:.2f}")

    st.markdown("---")

    # Cumulative returns chart
    st.subheader("📉 Cumulative Returns: Portfolio vs Benchmark")
    cum_port  = (1 + port_daily_ret).cumprod()
    cum_bench = (1 + bench_ret).cumprod() if bench_ret is not None else None

    fig4, ax4 = plt.subplots(figsize=(12, 5))
    fig4.patch.set_facecolor("#0f1117")
    ax4.set_facecolor("#1e2130")
    ax4.plot(cum_port.index, cum_port.values, color="#4fc3f7", lw=2, label="Portfolio")
    if cum_bench is not None:
        ax4.plot(cum_bench.index, cum_bench.values, color="#ef5350", lw=2, ls="--", label=benchmark)
    ax4.axhline(1.0, color="#555", ls=":", lw=1)
    ax4.set_title("Cumulative Returns (1 Year)", color="white", fontsize=14)
    ax4.tick_params(colors="white")
    ax4.legend(facecolor="#1e2130", labelcolor="white")
    ax4.spines[["top","right","left","bottom"]].set_color("#2e3250")
    ax4.set_ylabel("Growth of $1", color="white")
    st.pyplot(fig4)
    plt.close(fig4)

    st.markdown("---")

    # ── Step 4: Interpretation ────────────────────────────────
    ## Step 4 — Plain-English answers to the three graded questions
    st.subheader("💬 Step 4: Interpretation")

    # Q1: Did the portfolio outperform the benchmark?
    if outperf is not None:
        if outperf > 0:
            st.success(f"✅ **Outperformance:** Portfolio returned **{port_total_ret*100:.2f}%**, "
                       f"beating {benchmark} by **{outperf*100:.2f} pp**.")
        else:
            st.error(f"❌ **Underperformance:** Portfolio returned **{port_total_ret*100:.2f}%**, "
                     f"lagging {benchmark} by **{abs(outperf)*100:.2f} pp**.")

    # Q2: Was the portfolio more or less risky than the benchmark?
    if bench_vol is not None:
        if port_vol > bench_vol:
            st.warning(f"⚠️ **Risk:** Portfolio vol ({port_vol*100:.2f}%) was **more risky** "
                       f"than {benchmark} ({bench_vol*100:.2f}%).")
        else:
            st.info(f"🛡️ **Risk:** Portfolio vol ({port_vol*100:.2f}%) was **less risky** "
                    f"than {benchmark} ({bench_vol*100:.2f}%).")

    # Q3: Was the portfolio efficient? (Sharpe Ratio > 1.0 = good)
    if sharpe > 1.5:
        st.success(f"📊 **Efficiency (Sharpe {sharpe:.2f}):** Excellent risk-adjusted returns.")
    elif sharpe > 0.5:
        st.info(f"📊 **Efficiency (Sharpe {sharpe:.2f}):** Acceptable risk-adjusted returns.")
    else:
        st.warning(f"📊 **Efficiency (Sharpe {sharpe:.2f}):** Poor risk-adjusted returns — "
                   f"high risk relative to return.")

    # Portfolio pie chart
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

    # Full summary table
    st.markdown("---")
    st.subheader("📋 Full Portfolio Summary")
    summary2 = pd.DataFrame({
        "Metric": ["Portfolio Total Return", f"{benchmark} Total Return",
                   "Outperformance", "Portfolio Ann. Volatility",
                   f"{benchmark} Ann. Volatility", "Sharpe Ratio"],
        "Value":  [f"{port_total_ret*100:.2f}%",
                   f"{bench_total_ret*100:.2f}%" if bench_total_ret is not None else "N/A",
                   f"{outperf*100:+.2f}%"        if outperf is not None else "N/A",
                   f"{port_vol*100:.2f}%",
                   f"{bench_vol*100:.2f}%"        if bench_vol is not None else "N/A",
                   f"{sharpe:.2f}"]
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
