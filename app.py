## ============================================================
## FIN 330 — Final Project: Individual Stock Analysis Dashboard
## ============================================================
## Covers: Trend (Moving Averages), Momentum (RSI),
##         Volatility, and Trading Recommendation
##
## To run locally:   streamlit run fin330_app.py
## To deploy:        Push to GitHub → Streamlit Community Cloud
## ============================================================


# ── Import Libraries ──────────────────────────────────────────
import streamlit as st          # Web app framework
import pandas as pd             # Data manipulation
import numpy as np              # Numerical calculations
import matplotlib.pyplot as plt # Plotting
import yfinance as yf           # Yahoo Finance data
import qrcode                   # QR code generation
import io                       # In-memory byte buffer for QR image


# ── Page Configuration ────────────────────────────────────────
# Must be the FIRST Streamlit command in the script
st.set_page_config(
    page_title="FIN 330 | Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 12px;
    }
    .badge {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 15px;
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
</style>
""", unsafe_allow_html=True)


# ── Helper: flatten yfinance MultiIndex columns ───────────────
# Newer yfinance returns MultiIndex columns like ("Close", "AAPL").
# This function detects the level order and flattens to simple labels
# so that raw["Close"] works regardless of yfinance version.
PRICE_COLS = {"Close", "Open", "High", "Low", "Volume"}

def flatten_columns(df):
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    level0 = set(df.columns.get_level_values(0))
    if PRICE_COLS & level0:
        df.columns = df.columns.get_level_values(0)  # Level 0 = price type
    else:
        df.columns = df.columns.get_level_values(1)  # Level 1 = price type
    return df


# ── QR Code Generator ─────────────────────────────────────────
# Generates a QR code image pointing to the live Streamlit app URL.
# Returns a PNG image as bytes so Streamlit can display it with st.image().
APP_URL = "https://finance-app-o9wwll887loae9vbrotbbh.streamlit.app/"

def make_qr():
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction
        box_size=8,
        border=2,
    )
    qr.add_data(APP_URL)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#4fc3f7", back_color="#0f1117")  # Blue on dark
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════
# SIDEBAR — User Input
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    # User types any valid Yahoo Finance ticker symbol
    ticker_input = st.text_input(
        "Stock Ticker Symbol", value="AAPL",
        help="e.g. AAPL, MSFT, TSLA, GOOGL"
    ).upper().strip()

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True)

    # ── QR Code ───────────────────────────────────────────────
    # Displays a scannable QR code linking to the live app.
    # Useful for sharing during presentations.
    st.markdown("---")
    st.markdown("**📱 Scan to open app:**")
    try:
        st.image(make_qr(), width=200, caption="Scan to visit the live app")
        st.caption(f"[Open link]({APP_URL})")
    except Exception:
        st.markdown(f"[🔗 Open App]({APP_URL})")



# ══════════════════════════════════════════════════════════════
# APP HEADER
# ══════════════════════════════════════════════════════════════
st.title("📈 FIN 330 — Individual Stock Analysis")
st.caption("Real-time financial analysis powered by Yahoo Finance · FIN 330 Final Project")
st.markdown("---")


# ══════════════════════════════════════════════════════════════
# MAIN ANALYSIS — runs when the user clicks Run Analysis
# ══════════════════════════════════════════════════════════════
if run_btn:

    # ── Step 1: Data Collection ───────────────────────────────
    ## Download 6 months of daily closing price data from Yahoo Finance
    st.subheader(f"📥 Step 1: Data Collection — {ticker_input}")

    with st.spinner(f"Downloading data for {ticker_input}..."):
        try:
            # auto_adjust=True: prices are corrected for splits and dividends
            # progress=False:   suppresses the download bar in the console
            raw = yf.download(ticker_input, period="6mo",
                              auto_adjust=True, progress=False)

            if raw.empty:
                st.error(f"No data found for '{ticker_input}'. Please check the ticker symbol.")
                st.stop()

            # Flatten MultiIndex columns so raw["Close"] works correctly
            raw = flatten_columns(raw)

            if "Close" not in raw.columns:
                st.error(f"'Close' column not found. Got: {list(raw.columns)}")
                st.stop()

            # Extract Close price as a clean 1-D Series; remove any NaN rows
            close = raw["Close"].squeeze().dropna()

            if len(close) < 50:
                st.warning(f"Only {len(close)} days of data — MA50 may be incomplete.")

            st.success(f"✅ {len(close)} trading days downloaded.")

            with st.expander("📋 View Raw Data (first 5 rows)"):
                st.dataframe(raw.head(), use_container_width=True)

        except Exception as e:
            st.error(f"Download error: {e}")
            st.stop()

    st.markdown("---")

    # ── Step 2: Moving Averages (Trend Analysis) ──────────────
    ## Calculate the 20-day and 50-day moving averages to identify trend direction.
    # MA20 = average of last 20 closing prices (short-term trend)
    # MA50 = average of last 50 closing prices (long-term trend)
    st.subheader("📉 Step 2: Trend Analysis (Moving Averages)")

    # Rolling mean over the full 6-month series
    ma_20_series = close.rolling(20).mean()
    ma_50_series = close.rolling(50).mean()

    # Most recent values as plain floats for comparison
    current_price = float(close.iloc[-1])
    ma_20         = float(ma_20_series.iloc[-1])
    ma_50         = float(ma_50_series.iloc[-1])

    # Display the three key price values as metric cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Price", f"${current_price:.2f}")
    c2.metric("20-Day MA",     f"${ma_20:.2f}", delta=f"{current_price - ma_20:+.2f} vs price")
    c3.metric("50-Day MA",     f"${ma_50:.2f}", delta=f"{current_price - ma_50:+.2f} vs price")

    # Trend signal logic:
    #   Strong Uptrend   → Price > MA20 > MA50 (all aligned upward)
    #   Strong Downtrend → Price < MA20 < MA50 (all aligned downward)
    #   Mixed            → Everything else
    if current_price > ma_20 and ma_20 > ma_50:
        trend, trend_class = "Strong Uptrend",   "badge-up"
        trend_detail = "Price is above both moving averages — bullish momentum."
    elif current_price < ma_20 and ma_20 < ma_50:
        trend, trend_class = "Strong Downtrend", "badge-down"
        trend_detail = "Price is below both moving averages — bearish momentum."
    else:
        trend, trend_class = "Mixed Trend",      "badge-mixed"
        trend_detail = "Price is between moving averages — no clear direction."

    st.markdown(f"**Trend Signal:** <span class='badge {trend_class}'>{trend}</span>",
                unsafe_allow_html=True)
    st.caption(trend_detail)

    # Plot: closing price vs both moving averages
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
    plt.close(fig1)   # Free memory after rendering

    st.markdown("---")

    # ── Step 3: RSI (Momentum) ────────────────────────────────
    ## Relative Strength Index measures how fast and how much the price is moving.
    # Formula: RSI = 100 - (100 / (1 + RS))
    #   RS = average gain over 14 days / average loss over 14 days
    st.subheader("⚡ Step 3: Momentum (RSI)")

    # Daily price change
    delta = close.diff()

    # Separate gains (positive days) and losses (negative days)
    gains  = delta.clip(lower=0)    # Negative days become 0
    losses = -delta.clip(upper=0)   # Positive days become 0; flip sign so losses are positive

    # 14-day simple average of gains and losses
    avg_gain = float(gains.rolling(14).mean().iloc[-1])
    avg_loss = float(losses.rolling(14).mean().iloc[-1])

    # Calculate RSI; handle edge case where avg_loss = 0 (pure uptrend)
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs  = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

    # RSI signal thresholds:
    #   > 70 → Overbought (price may be too high, possible sell)
    #   < 30 → Oversold   (price may be too low,  possible buy)
    #   30–70 → Neutral   (no extreme momentum)
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

    # Plot the RSI line with overbought/oversold shading
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
    ## 20-day Annualized Volatility measures how much the price fluctuates.
    # Uses log returns (more statistically stable than simple returns).
    # Annualized by multiplying by √252 (trading days per year).
    st.subheader("🌊 Step 4: Volatility")

    # Daily log return: ln(today / yesterday)
    log_returns = np.log(close / close.shift(1))

    # Rolling 20-day std dev, scaled to annual
    vol_series      = log_returns.rolling(20).std() * np.sqrt(252)
    current_vol_pct = float(vol_series.iloc[-1]) * 100

    # Volatility levels:
    #   High   > 40%  — large price swings, higher risk
    #   Medium 25–40% — moderate fluctuation
    #   Low    < 25%  — relatively stable
    if current_vol_pct > 40:
        vol_signal, vol_class = "High Volatility",   "badge-high"
        vol_detail = "Annualized volatility above 40% — significant price swings expected."
    elif current_vol_pct >= 25:
        vol_signal, vol_class = "Medium Volatility", "badge-medium"
        vol_detail = "Annualized volatility 25–40% — moderate risk level."
    else:
        vol_signal, vol_class = "Low Volatility",    "badge-low"
        vol_detail = "Annualized volatility below 25% — relatively stable price movement."

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
    ## Combine trend, RSI, and volatility signals into one final call.
    # Scoring:
    #   uptrend  = +1,  downtrend = -1,  mixed = 0
    #   oversold = +1,  overbought = -1, neutral = 0
    #   score ≥ +1 and volatility not extreme → BUY
    #   score ≤ -1 or extreme volatility      → SELL
    #   everything else                        → HOLD
    st.subheader("🎯 Step 5: Trading Recommendation")

    uptrend   = 1 if "Uptrend"   in trend else (-1 if "Downtrend" in trend else 0)
    rsi_score = 1 if rsi < 30    else (-1 if rsi > 70 else 0)
    vol_ok    = current_vol_pct < 40          # True = volatility is not extreme
    score     = uptrend + rsi_score

    if score >= 1 and vol_ok:
        recommendation, rec_class = "BUY",  "badge-buy"
        reasoning = (f"{ticker_input} shows bullish signals. "
                     f"Trend: {trend}, RSI: {rsi:.1f}, "
                     f"Volatility: {current_vol_pct:.1f}% (manageable).")
    elif score <= -1 or not vol_ok:
        recommendation, rec_class = "SELL", "badge-sell"
        reasoning = (
            f"{ticker_input} shows bearish signals. "
            + (f"Price is below moving averages. " if uptrend == -1 else "")
            + (f"RSI overbought at {rsi:.1f}. "    if rsi_score == -1 else "")
            + (f"High volatility: {current_vol_pct:.1f}%. " if not vol_ok else "")
        )
    else:
        recommendation, rec_class = "HOLD", "badge-hold"
        reasoning = (f"Mixed signals for {ticker_input}. "
                     f"Trend: {trend}, RSI: {rsi:.1f}, "
                     f"Volatility: {current_vol_pct:.1f}%.")

    st.markdown(
        f"### Final Recommendation: "
        f"<span class='badge {rec_class}' style='font-size:20px;padding:6px 22px;'>"
        f"{recommendation}</span>",
        unsafe_allow_html=True
    )
    st.info(f"**Reasoning:** {reasoning}")

    # ── Summary Table ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Full Summary")
    summary = pd.DataFrame({
        "Indicator": ["Current Price", "20-Day MA", "50-Day MA", "Trend",
                      "RSI (14)", "RSI Signal",
                      "Annualized Volatility", "Volatility Level",
                      "Recommendation"],
        "Value":     [f"${current_price:.2f}", f"${ma_20:.2f}", f"${ma_50:.2f}", trend,
                      f"{rsi:.2f}", rsi_signal,
                      f"{current_vol_pct:.2f}%", vol_signal,
                      recommendation]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Placeholder shown before the user clicks Run ──────────────
else:
    st.info("👈 Enter a ticker symbol in the sidebar and click **Run Analysis** to begin.")
    st.markdown("""
    ### What this app analyzes
    - **Step 1** — Downloads 6 months of real stock price data
    - **Step 2** — Calculates 20-day & 50-day Moving Averages for trend direction
    - **Step 3** — Computes 14-day RSI for momentum (overbought / oversold)
    - **Step 4** — Measures 20-day Annualized Volatility (High / Medium / Low)
    - **Step 5** — Generates a final **Buy / Hold / Sell** recommendation
    """)

    # QR code on the welcome screen so it's visible before running analysis
    st.markdown("---")
    col_qr, col_txt = st.columns([1, 2])
    with col_qr:
        try:
            st.image(make_qr(), width=180, caption="Scan to open")
        except Exception:
            pass
    with col_txt:
        st.markdown(f"**Live App URL:**")
        st.code(APP_URL)
