import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="StockIQ", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0f1117; }
section[data-testid="stSidebar"] { background-color: #161b27; }
h1, h2, h3 { color: #e2e8f0 !important; }
[data-testid="metric-container"] {
    background: #1c2333; border: 1px solid #2a3347;
    border-radius: 10px; padding: 16px !important;
}
.stButton > button {
    background: #3b82f6; color: white; border: none;
    border-radius: 8px; font-weight: 600; width: 100%;
}
.rec-box {
    background: #1c2333; border-radius: 10px;
    padding: 20px; border: 1px solid #2a3347; margin-top: 12px;
}
.stTabs [aria-selected="true"] { background: #3b82f6 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

CHART = dict(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
             font=dict(color="#94a3b8"), margin=dict(l=10,r=10,t=40,b=10),
             xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
             hovermode="x unified")

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load(ticker, period="6mo"):
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].squeeze().dropna() if not df.empty else None

def calc_rsi(close, n=14):
    close = pd.Series(close).squeeze()  # ensure 1D
    d = close.diff()
    g = d.clip(lower=0).ewm(com=n-1, min_periods=n).mean()
    l = (-d.clip(upper=0)).ewm(com=n-1, min_periods=n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def badge(text, color):
    colors = {"green":"#10b981","red":"#ef4444","yellow":"#f59e0b","blue":"#3b82f6"}
    c = colors.get(color, "#94a3b8")
    return f'<span style="background:rgba(0,0,0,0.3);border:1px solid {c};color:{c};padding:4px 12px;border-radius:20px;font-size:13px;font-weight:700;margin-right:6px;">{text}</span>'

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 StockIQ")
    page = st.radio("", ["Stock Analysis", "Portfolio"], label_visibility="collapsed")
    st.markdown("---")

    if page == "Stock Analysis":
        ticker = st.text_input("Ticker", "CAT").upper().strip()
        period = st.selectbox("Period", ["3mo","6mo","1y","2y"], index=1)
        go_btn = st.button("Analyze")
    else:
        st.markdown("**5 Stocks + Weights (sum to 100%)**")
        defaults = [("AAPL",25),("MSFT",20),("PLTR",20),("AVGO",15),("MU",20)]
        holdings = []
        for i,(t,w) in enumerate(defaults):
            c1,c2 = st.columns([3,2])
            sym = c1.text_input(f"#{i+1}", t, key=f"t{i}", label_visibility="collapsed").upper()
            wt  = c2.number_input("%", 0, 100, w, key=f"w{i}", label_visibility="collapsed")
            holdings.append((sym, wt))
        bench = st.selectbox("Benchmark", ["SPY","QQQ","DIA"])
        total = sum(w for _,w in holdings)
        if total != 100: st.warning(f"Weights sum to {total}% — need 100%")
        go_btn = st.button("Run Analysis", disabled=(total != 100))

    st.caption("Data via Yahoo Finance · Educational use only")

# ═══════════════════════════════════════════════════════════════════════════════
# STOCK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Stock Analysis":
    st.markdown("## Stock Analysis")
    if go_btn:
        st.session_state.update(run_stock=True, s_tick=ticker, s_per=period)

    if st.session_state.get("run_stock"):
        t, p = st.session_state.s_tick, st.session_state.s_per
        with st.spinner(f"Loading {t}..."):
            close = load(t, p)
        if close is None:
            st.error(f"No data for {t}"); st.stop()

        ma20  = close.rolling(20).mean()
        ma50  = close.rolling(50).mean()
        rsi   = calc_rsi(close)
        price = float(close.iloc[-1])
        m20   = float(ma20.iloc[-1])
        m50   = float(ma50.iloc[-1])
        r     = float(rsi.iloc[-1])
        vol   = float(close.pct_change().dropna().iloc[-20:].std() * np.sqrt(252))
        chg   = float((price - float(close.iloc[-2])) / float(close.iloc[-2]))

        # Signals
        trend = ("Uptrend","green") if price>m20>m50 else ("Downtrend","red") if price<m20<m50 else ("Mixed","yellow")
        rsi_s = ("Overbought","red") if r>70 else ("Oversold","green") if r<30 else ("Neutral RSI","yellow")
        vol_s = ("High Vol","red") if vol>0.4 else ("Low Vol","green") if vol<0.25 else ("Med Vol","yellow")

        # Recommendation
        buys  = sum([price>m20>m50, r<30])
        sells = sum([price<m20<m50, r>70])
        rec   = ("BUY","#10b981") if buys>=2 else ("SELL","#ef4444") if sells>=2 else ("HOLD","#f59e0b")

        # Metrics
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Price", f"${price:.2f}", f"{chg:+.2%}")
        c2.metric("MA 20", f"${m20:.2f}")
        c3.metric("MA 50", f"${m50:.2f}")
        c4.metric("RSI (14)", f"{r:.1f}")
        c5.metric("Volatility", f"{vol:.1%}")

        st.markdown(badge("↑ "+trend[0] if trend[0]=="Uptrend" else "↓ "+trend[0] if trend[0]=="Downtrend" else "↔ "+trend[0], trend[1])
                    + badge(rsi_s[0], rsi_s[1]) + badge(vol_s[0], vol_s[1]), unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📊 Price & MAs", "📋 Recommendation"])

        with tab1:
            fig = go.Figure([
                go.Scatter(x=close.index, y=close, name="Price", line=dict(color="#e2e8f0", width=2)),
                go.Scatter(x=ma20.index,  y=ma20,  name="MA20",  line=dict(color="#3b82f6", width=1.5, dash="dot")),
                go.Scatter(x=ma50.index,  y=ma50,  name="MA50",  line=dict(color="#10b981", width=1.5, dash="dash")),
            ])
            fig.update_layout(**CHART, title=f"{t} — Price & Moving Averages", height=360)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure([go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="#a78bfa", width=2))])
            fig2.add_hline(y=70, line=dict(color="#ef4444", dash="dash"), annotation_text="Overbought")
            fig2.add_hline(y=30, line=dict(color="#10b981", dash="dash"), annotation_text="Oversold")
            fig2.update_layout(**CHART, title="RSI (14-Day)", yaxis=dict(range=[0,100], gridcolor="#1e293b"), height=260)
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            emoji = {"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}[rec[0]]
            st.markdown(f"""<div class="rec-box">
            <h3 style="color:{rec[1]};margin:0 0 8px">{emoji} {rec[0]}</h3>
            <p style="color:#94a3b8;margin:0">
            <b>Trend:</b> {trend[0]} &nbsp;|&nbsp;
            <b>RSI:</b> {rsi_s[0]} ({r:.1f}) &nbsp;|&nbsp;
            <b>Volatility:</b> {vol_s[0]} ({vol:.1%})
            </p></div>""", unsafe_allow_html=True)
    else:
        st.info("👈 Enter a ticker and click **Analyze**")

# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("## Portfolio Dashboard")
    if go_btn:
        st.session_state.update(run_port=True, p_hold=holdings, p_bench=bench)

    if st.session_state.get("run_port"):
        hold  = st.session_state.p_hold
        bench = st.session_state.p_bench
        ticks = [t for t,_ in hold]
        wts   = {t: w/100 for t,w in hold}

        with st.spinner("Loading portfolio data..."):
            frames = {}
            for t in ticks + [bench]:
                c = load(t, "1y")
                if c is not None: frames[t] = c
        prices = pd.DataFrame(frames).dropna()

        miss = [t for t in ticks if t not in prices]
        if miss: st.error(f"No data: {', '.join(miss)}"); st.stop()

        rets   = prices.pct_change().dropna()
        port_d = sum(rets[t] * wts[t] for t in ticks)
        bench_d= rets[bench]
        cum_p  = (1 + port_d).cumprod() - 1
        cum_b  = (1 + bench_d).cumprod() - 1
        pt     = float(cum_p.iloc[-1])
        bt     = float(cum_b.iloc[-1])
        vol    = float(port_d.std() * np.sqrt(252))
        sharpe = (pt - 0.03) / vol if vol > 0 else 0
        alpha  = pt - bt

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Portfolio Return", f"{pt:+.1%}")
        c2.metric(f"{bench} Return",  f"{bt:+.1%}")
        c3.metric("Alpha", f"{alpha:+.1%}")
        c4.metric("Sharpe Ratio", f"{sharpe:.2f}")

        st.markdown(badge("✅ Outperforming" if alpha>0 else "❌ Underperforming", "green" if alpha>0 else "red"), unsafe_allow_html=True)

        fig = go.Figure([
            go.Scatter(x=cum_p.index, y=cum_p*100, name="Portfolio",
                       line=dict(color="#3b82f6", width=2.5),
                       fill="tozeroy", fillcolor="rgba(59,130,246,0.06)"),
            go.Scatter(x=cum_b.index, y=cum_b*100, name=bench,
                       line=dict(color="#64748b", width=1.5, dash="dash")),
        ])
        fig.update_layout(**CHART, title="Cumulative Returns vs Benchmark (1Y)",
                          yaxis=dict(ticksuffix="%", gridcolor="#1e293b"), height=360)
        st.plotly_chart(fig, use_container_width=True)

        ind  = {t: float((1+rets[t]).prod()-1) for t in ticks}
        fig2 = go.Figure(go.Bar(
            x=list(ind.keys()), y=[v*100 for v in ind.values()],
            marker_color=["#10b981" if v>=0 else "#ef4444" for v in ind.values()],
            text=[f"{v:+.1%}" for v in ind.values()], textposition="outside",
        ))
        fig2.update_layout(**CHART, title="Individual Stock Returns (1Y)",
                           yaxis=dict(ticksuffix="%", gridcolor="#1e293b"), showlegend=False, height=300)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("👈 Set weights to 100% and click **Run Analysis**")
