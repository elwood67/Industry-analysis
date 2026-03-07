"""
╔══════════════════════════════════════════════════════════════╗
║          CONGRESSIONAL TRADES INTELLIGENCE DASHBOARD          ║
║              Elwood's Trading Lab  |  March 2026              ║
╚══════════════════════════════════════════════════════════════╝

Reads:  congress_backtest_results.csv        (required)
        congress_backtest_by_politician.csv  (optional, speeds up pol tab)
        congress_backtest_by_ticker.csv      (optional, speeds up ticker tab)

Usage:
    streamlit run congress_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Congressional Trades Intel",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  THEME / STYLE
# ─────────────────────────────────────────────────────────────
DARK_BG    = "#0A0C10"
CARD_BG    = "#111318"
BORDER     = "#1E2330"
DEM_COLOR  = "#3B82F6"   # blue
REP_COLOR  = "#EF4444"   # red
IND_COLOR  = "#A855F7"   # purple
GREEN      = "#22C55E"
RED        = "#EF4444"
GOLD       = "#F59E0B"
TEXT_MAIN  = "#E2E8F0"
TEXT_DIM   = "#64748B"
ACCENT     = "#06B6D4"   # cyan

RETURN_PERIODS = ["5d", "10d", "20d", "30d", "60d", "120d", "252d"]

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {{
    background-color: {DARK_BG};
    color: {TEXT_MAIN};
    font-family: 'DM Sans', sans-serif;
  }}
  .stApp {{ background-color: {DARK_BG}; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background-color: {CARD_BG};
    border-right: 1px solid {BORDER};
  }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    background-color: {CARD_BG};
    border-radius: 10px;
    padding: 4px;
    border: 1px solid {BORDER};
  }}
  .stTabs [data-baseweb="tab"] {{
    color: {TEXT_DIM};
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    border-radius: 8px;
    padding: 8px 16px;
  }}
  .stTabs [aria-selected="true"] {{
    background-color: {ACCENT} !important;
    color: {DARK_BG} !important;
    font-weight: 700;
  }}

  /* Metric cards */
  .metric-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }}
  .metric-label {{
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: {TEXT_DIM};
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
  }}
  .metric-value {{
    font-family: 'Space Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: {TEXT_MAIN};
  }}
  .metric-sub {{
    font-size: 12px;
    color: {TEXT_DIM};
    margin-top: 4px;
  }}
  .positive {{ color: {GREEN}; }}
  .negative {{ color: {RED}; }}
  .neutral  {{ color: {GOLD}; }}

  /* Section headers */
  .section-header {{
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {ACCENT};
    padding: 6px 0;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 16px;
  }}

  /* DataFrames */
  .stDataFrame {{ background-color: {CARD_BG}; }}
  [data-testid="stDataFrameResizable"] {{ background: {CARD_BG}; }}

  /* Selectbox / slider labels */
  label {{ color: {TEXT_DIM} !important; font-size: 12px !important; }}
  .stSelectbox > div > div {{ background-color: {CARD_BG}; border-color: {BORDER}; }}

  /* Plotly charts transparent background */
  .js-plotly-plot .plotly {{ background: transparent !important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def party_color(party):
    p = str(party).upper()
    if "D" in p: return DEM_COLOR
    if "R" in p: return REP_COLOR
    return IND_COLOR

def fmt_pct(v):
    if pd.isna(v): return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"

def color_pct(v):
    if pd.isna(v): return TEXT_DIM
    return GREEN if v > 0 else RED

def metric_card(label, value, sub="", color=TEXT_MAIN):
    color_cls = "positive" if color == GREEN else ("negative" if color == RED else "neutral" if color == GOLD else "")
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {'positive' if color_cls == 'positive' else 'negative' if color_cls == 'negative' else ''}">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>"""

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def plotly_layout(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Mono", size=13, color=TEXT_DIM)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CARD_BG,
        font=dict(family="DM Sans", color=TEXT_MAIN),
        margin=dict(l=16, r=16, t=40, b=16),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    )
    return fig

# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    # Try to find the main results file
    candidates = [
        Path("congress_backtest_results.csv"),
        Path("data/congress_backtest_results.csv"),
        Path("../congress_backtest_results.csv"),
    ]
    path = None
    for c in candidates:
        if c.exists():
            path = c
            break

    if path is None:
        return None, None, None

    df = pd.read_csv(path, low_memory=False)
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')

    # Normalise party
    def norm_party(p):
        p = str(p).strip().upper()
        if p.startswith("D"): return "Democrat"
        if p.startswith("R"): return "Republican"
        return "Independent"
    df['party_clean'] = df['party'].apply(norm_party)

    # Normalise tx_type
    df['is_buy'] = df['tx_type'].str.lower().str.contains("buy", na=False)

    # Cap extreme outliers for visualisation (keep raw for tables)
    for p in RETURN_PERIODS:
        col = f"return_{p}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[f"{col}_capped"] = df[col].clip(-100, 300)

    # Trade year/quarter
    df['year'] = df['trade_date'].dt.year
    df['quarter'] = df['trade_date'].dt.to_period('Q').astype(str)
    df['month'] = df['trade_date'].dt.to_period('M').astype(str)

    # Politician / ticker summaries
    pol_path  = Path("congress_backtest_by_politician.csv")
    tick_path = Path("congress_backtest_by_ticker.csv")
    pol_df   = pd.read_csv(pol_path)  if pol_path.exists()  else None
    tick_df  = pd.read_csv(tick_path) if tick_path.exists() else None

    return df, pol_df, tick_df


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
def sidebar(df):
    st.sidebar.markdown(f"""
    <div style="text-align:center; padding:12px 0 20px 0;">
      <div style="font-family:'Space Mono',monospace; font-size:18px; font-weight:700; color:{ACCENT};">🏛️ CONGRESS</div>
      <div style="font-family:'Space Mono',monospace; font-size:10px; color:{TEXT_DIM}; letter-spacing:0.15em;">TRADE INTELLIGENCE</div>
      <div style="font-family:'Space Mono',monospace; font-size:9px; color:{TEXT_DIM}; margin-top:4px;">ELWOOD'S TRADING LAB</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f'<div style="font-family:Space Mono;font-size:10px;color:{TEXT_DIM};letter-spacing:.1em">FILTERS</div>', unsafe_allow_html=True)

    # Date range
    min_d = df['trade_date'].min().date()
    max_d = df['trade_date'].max().date()
    date_range = st.sidebar.date_input("Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

    # Trade type
    tx_type = st.sidebar.multiselect("Trade Type", ["Buys", "Sells"], default=["Buys", "Sells"])

    # Party
    parties = st.sidebar.multiselect("Party", ["Democrat", "Republican", "Independent"],
                                      default=["Democrat", "Republican", "Independent"])

    # Chamber
    chambers = df['chamber'].dropna().unique().tolist()
    sel_chambers = st.sidebar.multiselect("Chamber", chambers, default=chambers)

    # Return period
    period = st.sidebar.selectbox("Return Period", RETURN_PERIODS, index=3)

    # Min trades filter
    min_trades = st.sidebar.slider("Min Trades (politician/ticker filters)", 1, 50, 10)

    st.sidebar.markdown("---")
    total = len(df)
    st.sidebar.markdown(f'<div style="font-family:Space Mono;font-size:10px;color:{TEXT_DIM}">DATASET: {total:,} trades<br>{df["politician_name"].nunique()} politicians<br>{df["ticker"].nunique()} tickers</div>', unsafe_allow_html=True)

    return date_range, tx_type, parties, sel_chambers, period, min_trades


def apply_filters(df, date_range, tx_type, parties, chambers):
    fdf = df.copy()
    if len(date_range) == 2:
        fdf = fdf[(fdf['trade_date'].dt.date >= date_range[0]) &
                  (fdf['trade_date'].dt.date <= date_range[1])]
    if tx_type and len(tx_type) < 2:
        if "Buys" in tx_type:
            fdf = fdf[fdf['is_buy']]
        else:
            fdf = fdf[~fdf['is_buy']]
    if parties:
        fdf = fdf[fdf['party_clean'].isin(parties)]
    if chambers:
        fdf = fdf[fdf['chamber'].isin(chambers)]
    return fdf


# ─────────────────────────────────────────────────────────────
#  TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────
def tab_overview(df, fdf, period):
    st.markdown(f'<div class="section-header">MARKET OVERVIEW — {len(fdf):,} TRADES IN VIEW</div>', unsafe_allow_html=True)

    col = f"return_{period}_capped"
    ret_col = f"return_{period}"
    buys = fdf[fdf['is_buy']]
    sells = fdf[~fdf['is_buy']]

    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        avg = buys[ret_col].mean()
        st.markdown(metric_card("BUY AVG RETURN", fmt_pct(avg), f"@ {period}", GREEN if avg > 0 else RED), unsafe_allow_html=True)
    with c2:
        wr = (buys[ret_col].dropna() > 0).mean() * 100 if len(buys) else 0
        st.markdown(metric_card("BUY WIN RATE", f"{wr:.1f}%", f"@ {period}", GREEN if wr > 50 else RED), unsafe_allow_html=True)
    with c3:
        avg_s = sells[ret_col].mean()
        st.markdown(metric_card("SELL AVG RETURN", fmt_pct(avg_s), f"@ {period}", GREEN if avg_s > 0 else RED), unsafe_allow_html=True)
    with c4:
        wr_s = (sells[ret_col].dropna() > 0).mean() * 100 if len(sells) else 0
        st.markdown(metric_card("SELL WIN RATE", f"{wr_s:.1f}%", f"@ {period}", GREEN if wr_s > 50 else RED), unsafe_allow_html=True)
    with c5:
        med = buys[ret_col].median()
        st.markdown(metric_card("BUY MEDIAN", fmt_pct(med), f"@ {period}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Trade activity over time
    left, right = st.columns([3, 2])
    with left:
        section("TRADE ACTIVITY OVER TIME")
        monthly = fdf.groupby(['month', 'tx_type']).size().reset_index(name='count')
        monthly['month_dt'] = pd.to_datetime(monthly['month'])
        monthly = monthly.sort_values('month_dt')
        fig = px.bar(monthly, x='month', y='count', color='tx_type',
                     color_discrete_map={"buy": GREEN, "sell": RED,
                                         "buy_full": GREEN, "sell_full": RED},
                     barmode='stack')
        fig.update_traces(marker_line_width=0)
        fig = plotly_layout(fig, height=300)
        fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        section("BUY vs SELL SPLIT")
        split = fdf.groupby('is_buy').size()
        labels = ["Sells", "Buys"]
        vals = [split.get(False, 0), split.get(True, 0)]
        fig = go.Figure(go.Pie(labels=labels, values=vals,
                               marker_colors=[RED, GREEN],
                               hole=0.6, textfont=dict(size=12)))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10,r=10,t=10,b=10), height=300,
                          legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

    # Return distribution heatmap across all periods (buys only)
    section("BUY RETURN ACROSS ALL PERIODS")
    period_data = []
    for p in RETURN_PERIODS:
        c = f"return_{p}"
        if c in buys.columns:
            valid = buys[c].dropna()
            if len(valid):
                period_data.append({
                    "period": p,
                    "avg": valid.mean(),
                    "median": valid.median(),
                    "win_rate": (valid > 0).mean() * 100,
                    "count": len(valid),
                })
    if period_data:
        prd = pd.DataFrame(period_data)
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Avg Return %", "Win Rate %"])
        fig.add_trace(go.Bar(x=prd['period'], y=prd['avg'].clip(-50, 300), name="Avg Return",
                             marker_color=[GREEN if v > 0 else RED for v in prd['avg']]), row=1, col=1)
        fig.add_trace(go.Bar(x=prd['period'], y=prd['win_rate'], name="Win Rate",
                             marker_color=ACCENT), row=1, col=2)
        fig.add_hline(y=50, line_dash="dot", line_color=TEXT_DIM, row=1, col=2)
        fig = plotly_layout(fig, height=320)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Party comparison table
    section("PARTY COMPARISON (BUYS)")
    rows = []
    for party in ["Democrat", "Republican", "Independent"]:
        pb = buys[buys['party_clean'] == party]
        if len(pb) < 5: continue
        row = {"Party": party, "Trades": len(pb)}
        for p in RETURN_PERIODS:
            c = f"return_{p}"
            valid = pb[c].dropna()
            row[f"Avg {p}"] = f"{valid.mean():+.2f}%" if len(valid) else "—"
            row[f"WR {p}"] = f"{(valid>0).mean()*100:.1f}%" if len(valid) else "—"
        rows.append(row)
    if rows:
        ptbl = pd.DataFrame(rows)
        st.dataframe(ptbl.set_index("Party"), use_container_width=True, height=120)


# ─────────────────────────────────────────────────────────────
#  TAB 2 — POLITICIAN ANALYSIS
# ─────────────────────────────────────────────────────────────
def tab_politicians(fdf, period, min_trades):
    ret_col = f"return_{period}"
    buys = fdf[fdf['is_buy']].copy()

    # Build scorecard
    pol = buys.groupby(['politician_name', 'party_clean', 'state', 'chamber']).agg(
        trades=('ticker', 'count'),
        avg_5d=('return_5d', 'mean'),
        avg_30d=('return_30d', 'mean'),
        avg_60d=('return_60d', 'mean'),
        avg_252d=('return_252d', 'mean'),
        wr_30d=('return_30d', lambda x: (x.dropna() > 0).mean() * 100),
        wr_period=(ret_col, lambda x: (x.dropna() > 0).mean() * 100),
        avg_period=(ret_col, 'mean'),
        volume=('est_volume_usd', 'sum'),
        tickers=('ticker', 'nunique'),
    ).reset_index()
    pol = pol[pol['trades'] >= min_trades].sort_values('avg_period', ascending=False)

    section(f"POLITICIAN SCORECARD — {period} BUYS  (min {min_trades} trades)")

    # Top / bottom leaderboard
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div style="color:{GREEN};font-family:Space Mono;font-size:11px;margin-bottom:8px">▲ TOP 15</div>', unsafe_allow_html=True)
        top = pol.head(15)[['politician_name','party_clean','trades','avg_period','wr_period','avg_252d']]
        top.columns = ['Politician','Party','Trades',f'Avg {period}','WR%','Avg 1yr']
        top[f'Avg {period}'] = top[f'Avg {period}'].apply(fmt_pct)
        top['WR%'] = top['WR%'].apply(lambda x: f"{x:.1f}%")
        top['Avg 1yr'] = top['Avg 1yr'].apply(fmt_pct)
        st.dataframe(top, use_container_width=True, height=480, hide_index=True)
    with c2:
        st.markdown(f'<div style="color:{RED};font-family:Space Mono;font-size:11px;margin-bottom:8px">▼ BOTTOM 15</div>', unsafe_allow_html=True)
        bot = pol.tail(15).iloc[::-1][['politician_name','party_clean','trades','avg_period','wr_period','avg_252d']]
        bot.columns = ['Politician','Party','Trades',f'Avg {period}','WR%','Avg 1yr']
        bot[f'Avg {period}'] = bot[f'Avg {period}'].apply(fmt_pct)
        bot['WR%'] = bot['WR%'].apply(lambda x: f"{x:.1f}%")
        bot['Avg 1yr'] = bot['Avg 1yr'].apply(fmt_pct)
        st.dataframe(bot, use_container_width=True, height=480, hide_index=True)

    # Scatter: trades vs return
    section("TRADES vs RETURN SCATTER (bubble = volume)")
    pol_plot = pol.copy()
    pol_plot['vol_safe'] = pol_plot['volume'].fillna(0).clip(0, 1e9)
    fig = px.scatter(
        pol_plot,
        x='trades', y='avg_period',
        size='vol_safe', size_max=40,
        color='party_clean',
        color_discrete_map={"Democrat": DEM_COLOR, "Republican": REP_COLOR, "Independent": IND_COLOR},
        hover_name='politician_name',
        hover_data={'trades': True, 'avg_period': ':.2f', 'wr_period': ':.1f', 'vol_safe': False},
        labels={'avg_period': f'Avg {period} Return %', 'trades': 'Number of Trades'},
    )
    fig.add_hline(y=0, line_dash="dot", line_color=TEXT_DIM)
    fig = plotly_layout(fig, height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Win rate bar chart
    section("WIN RATE BY POLITICIAN (sorted)")
    pol_wr = pol.sort_values('wr_period', ascending=False).head(40)
    fig = px.bar(pol_wr, x='politician_name', y='wr_period',
                 color='party_clean',
                 color_discrete_map={"Democrat": DEM_COLOR, "Republican": REP_COLOR, "Independent": IND_COLOR},
                 labels={'wr_period': f'Win Rate {period} %', 'politician_name': ''})
    fig.add_hline(y=50, line_dash="dot", line_color=TEXT_DIM)
    fig = plotly_layout(fig, height=380)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
    st.plotly_chart(fig, use_container_width=True)

    # Drill-down: single politician
    st.markdown("---")
    section("SINGLE POLITICIAN DRILL-DOWN")
    pol_names = sorted(fdf['politician_name'].dropna().unique())
    sel_pol = st.selectbox("Select Politician", pol_names)
    pol_trades = fdf[fdf['politician_name'] == sel_pol].sort_values('trade_date', ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Trades", f"{len(pol_trades):,}")
    with c2:
        pb = pol_trades[pol_trades['is_buy']]
        avg_r = pb[ret_col].mean()
        st.metric(f"Buy Avg {period}", fmt_pct(avg_r))
    with c3:
        wr = (pb[ret_col].dropna() > 0).mean() * 100 if len(pb) else 0
        st.metric(f"Win Rate {period}", f"{wr:.1f}%")
    with c4:
        st.metric("Unique Tickers", f"{pol_trades['ticker'].nunique():,}")

    # Return curve across periods for this politician
    curve_data = []
    for p in RETURN_PERIODS:
        rc = f"return_{p}"
        valid = pb[rc].dropna()
        if len(valid):
            curve_data.append({"Period": p, "Avg Return": valid.mean(), "Win Rate": (valid>0).mean()*100})
    if curve_data:
        cd = pd.DataFrame(curve_data)
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Avg Return % by Period", "Win Rate % by Period"])
        fig.add_trace(go.Scatter(x=cd['Period'], y=cd['Avg Return'], mode='lines+markers',
                                 line=dict(color=ACCENT, width=2), marker=dict(size=8)), row=1, col=1)
        fig.add_trace(go.Scatter(x=cd['Period'], y=cd['Win Rate'], mode='lines+markers',
                                 line=dict(color=GOLD, width=2), marker=dict(size=8)), row=1, col=2)
        fig.add_hline(y=0, line_dash="dot", line_color=TEXT_DIM, row=1, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color=TEXT_DIM, row=1, col=2)
        fig = plotly_layout(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Their top tickers
    top_ticks = (pol_trades[pol_trades['is_buy']].groupby('ticker')[ret_col]
                 .agg(['mean','count']).reset_index()
                 .rename(columns={'mean': 'Avg Return', 'count': 'Trades'})
                 .query('Trades >= 2')
                 .sort_values('Avg Return', ascending=False).head(20))
    if len(top_ticks):
        section(f"TOP TICKERS FOR {sel_pol.upper()}")
        top_ticks['Avg Return'] = top_ticks['Avg Return'].apply(fmt_pct)
        st.dataframe(top_ticks, use_container_width=True, hide_index=True)

    st.dataframe(pol_trades[['trade_date','ticker','tx_type','size_range','est_volume_usd',
                              'return_5d','return_30d','return_60d','return_252d']].head(100),
                 use_container_width=True, height=350)


# ─────────────────────────────────────────────────────────────
#  TAB 3 — TICKER ANALYSIS
# ─────────────────────────────────────────────────────────────
def tab_tickers(fdf, period, min_trades):
    ret_col = f"return_{period}"
    buys = fdf[fdf['is_buy']].copy()

    tick = buys.groupby('ticker').agg(
        trades=('ticker', 'count'),
        avg_period=(ret_col, 'mean'),
        med_period=(ret_col, 'median'),
        wr_period=(ret_col, lambda x: (x.dropna() > 0).mean() * 100),
        avg_5d=('return_5d', 'mean'),
        avg_30d=('return_30d', 'mean'),
        avg_252d=('return_252d', 'mean'),
        volume=('est_volume_usd', 'sum'),
        politicians=('politician_name', 'nunique'),
    ).reset_index()
    tick = tick[tick['trades'] >= min_trades].sort_values('avg_period', ascending=False)

    section(f"TOP CONGRESS BUYS BY {period} RETURN  (min {min_trades} trades)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div style="color:{GREEN};font-family:Space Mono;font-size:11px;margin-bottom:8px">▲ BEST PERFORMERS</div>', unsafe_allow_html=True)
        top = tick.head(20).copy()
        top['avg_period'] = top['avg_period'].apply(fmt_pct)
        top['wr_period']  = top['wr_period'].apply(lambda x: f"{x:.1f}%")
        top['avg_252d']   = top['avg_252d'].apply(fmt_pct)
        top['volume']     = top['volume'].apply(lambda x: f"${x:,.0f}")
        top = top[['ticker','trades','avg_period','wr_period','avg_252d','politicians','volume']]
        top.columns = ['Ticker','Trades',f'Avg {period}','WR%','Avg 1yr','# Pols','Volume']
        st.dataframe(top, use_container_width=True, height=650, hide_index=True)
    with c2:
        st.markdown(f'<div style="color:{RED};font-family:Space Mono;font-size:11px;margin-bottom:8px">▼ WORST PERFORMERS</div>', unsafe_allow_html=True)
        bot = tick.tail(20).iloc[::-1].copy()
        bot['avg_period'] = bot['avg_period'].apply(fmt_pct)
        bot['wr_period']  = bot['wr_period'].apply(lambda x: f"{x:.1f}%")
        bot['avg_252d']   = bot['avg_252d'].apply(fmt_pct)
        bot['volume']     = bot['volume'].apply(lambda x: f"${x:,.0f}")
        bot = bot[['ticker','trades','avg_period','wr_period','avg_252d','politicians','volume']]
        bot.columns = ['Ticker','Trades',f'Avg {period}','WR%','Avg 1yr','# Pols','Volume']
        st.dataframe(bot, use_container_width=True, height=650, hide_index=True)

    # Bubble chart: volume vs return
    section("TICKER BUBBLE CHART — SIZE = TRADE VOLUME")
    tick_plot = tick.head(60).copy()
    tick_plot['vol_safe'] = tick_plot['volume'].fillna(0).clip(0, 1e9)
    fig = px.scatter(tick_plot, x='trades', y='avg_period',
                     size='vol_safe', size_max=50,
                     color='wr_period', color_continuous_scale='RdYlGn',
                     hover_name='ticker',
                     hover_data={'trades': True, 'avg_period': ':.2f', 'wr_period': ':.1f'},
                     labels={'avg_period': f'Avg {period} Return %', 'trades': 'Buy Trade Count', 'wr_period': 'Win Rate %'})
    fig.add_hline(y=0, line_dash="dot", line_color=TEXT_DIM)
    fig = plotly_layout(fig, height=480)
    st.plotly_chart(fig, use_container_width=True)

    # Single ticker drill-down
    st.markdown("---")
    section("SINGLE TICKER DRILL-DOWN")
    all_tickers = sorted(fdf['ticker'].dropna().unique())
    sel_tick = st.selectbox("Select Ticker", all_tickers)
    tick_trades = fdf[fdf['ticker'] == sel_tick].sort_values('trade_date')

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Trades", f"{len(tick_trades):,}")
    with c2: st.metric("Unique Buyers", f"{tick_trades[tick_trades['is_buy']]['politician_name'].nunique():,}")
    with c3:
        avg_r = tick_trades[tick_trades['is_buy']][ret_col].mean()
        st.metric(f"Buy Avg {period}", fmt_pct(avg_r))
    with c4:
        wr = (tick_trades[tick_trades['is_buy']][ret_col].dropna() > 0).mean() * 100
        st.metric("Win Rate", f"{wr:.1f}%")

    # Buyers of this ticker
    buyers = (tick_trades[tick_trades['is_buy']]
              .groupby(['politician_name','party_clean'])
              .agg(trades=('ticker','count'), avg_return=(ret_col,'mean'))
              .reset_index().sort_values('avg_return', ascending=False))
    if len(buyers):
        st.markdown(f"**Politicians buying {sel_tick}:**")
        buyers['avg_return'] = buyers['avg_return'].apply(fmt_pct)
        st.dataframe(buyers, use_container_width=True, height=260, hide_index=True)


# ─────────────────────────────────────────────────────────────
#  TAB 4 — TRADE SIZE ANALYSIS
# ─────────────────────────────────────────────────────────────
def tab_size(fdf, period):
    ret_col = f"return_{period}"
    buys = fdf[fdf['is_buy']].copy()

    section("TRADE SIZE vs RETURN  (BUYS)")

    size_order = ['< 1K','1K–15K','15K–50K','50K–100K','100K–250K','250K–500K','500K–1M','1M–5M','5M–25M']
    size_data = (buys.groupby('size_range').agg(
        trades=('ticker','count'),
        avg_return=(ret_col,'mean'),
        med_return=(ret_col,'median'),
        win_rate=(ret_col, lambda x: (x.dropna()>0).mean()*100),
    ).reset_index())

    size_data['size_range'] = pd.Categorical(size_data['size_range'],
                                              categories=[s for s in size_order if s in size_data['size_range'].values],
                                              ordered=True)
    size_data = size_data.sort_values('size_range')

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(size_data, x='size_range', y='avg_return',
                     color='avg_return', color_continuous_scale='RdYlGn',
                     labels={'avg_return': f'Avg {period} Return %', 'size_range': 'Trade Size'},
                     text='avg_return')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig = plotly_layout(fig, "Avg Return by Trade Size", height=380)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(size_data, x='size_range', y='win_rate',
                     color='win_rate', color_continuous_scale='Blues',
                     labels={'win_rate': f'Win Rate % @ {period}', 'size_range': 'Trade Size'},
                     text='win_rate')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.add_hline(y=50, line_dash="dot", line_color=TEXT_DIM)
        fig = plotly_layout(fig, "Win Rate by Trade Size", height=380)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Table
    size_data_tbl = size_data.copy()
    size_data_tbl['avg_return'] = size_data_tbl['avg_return'].apply(fmt_pct)
    size_data_tbl['med_return'] = size_data_tbl['med_return'].apply(fmt_pct)
    size_data_tbl['win_rate']   = size_data_tbl['win_rate'].apply(lambda x: f"{x:.1f}%")
    st.dataframe(size_data_tbl, use_container_width=True, hide_index=True)

    # Cross-tab: size x period
    section("SIZE x PERIOD HEATMAP (avg return)")
    heat_rows = []
    for size in size_order:
        sb = buys[buys['size_range'] == size]
        if len(sb) < 3: continue
        row = {'Size': size}
        for p in RETURN_PERIODS:
            c = f"return_{p}"
            row[p] = sb[c].dropna().mean() if c in sb.columns else np.nan
        heat_rows.append(row)
    if heat_rows:
        heat_df = pd.DataFrame(heat_rows).set_index('Size')
        # Cap for display
        heat_display = heat_df.clip(-50, 100)
        fig = px.imshow(heat_display, color_continuous_scale='RdYlGn',
                        labels=dict(color="Avg Return %"), aspect='auto',
                        text_auto='.1f')
        fig = plotly_layout(fig, "Avg Return Heatmap: Trade Size × Period", height=350)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  TAB 5 — SCORING SYSTEM COMBO
# ─────────────────────────────────────────────────────────────
def tab_scoring(fdf, period):
    if 'score_net' not in fdf.columns or fdf['score_net'].notna().sum() < 100:
        st.info("🔍 No scoring data found. Run the backtester with `--scores-dir` to enable this tab.")
        return

    ret_col = f"return_{period}"
    buys = fdf[fdf['is_buy'] & fdf['score_net'].notna()].copy()

    section("YOUR SCORING SYSTEM × CONGRESS TRADES")

    c1, c2, c3 = st.columns(3)
    groups = {
        "Congress + Bullish (net>0)":      buys[buys['score_net'] > 0],
        "Congress + Bearish (net<=0)":     buys[buys['score_net'] <= 0],
        "Congress + Strong Bull (net>30)": buys[buys['score_net'] > 30],
    }
    colors = [GREEN, RED, GOLD]
    for (label, subset), color, col in zip(groups.items(), colors, [c1, c2, c3]):
        avg = subset[ret_col].mean()
        wr  = (subset[ret_col].dropna() > 0).mean() * 100 if len(subset) else 0
        with col:
            st.markdown(metric_card(label.upper(), fmt_pct(avg), f"{len(subset):,} trades  |  WR: {wr:.1f}%"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Multi-period comparison
    section("COMBO PERFORMANCE ACROSS ALL PERIODS")
    combo_rows = []
    for label, subset in groups.items():
        for p in RETURN_PERIODS:
            c = f"return_{p}"
            valid = subset[c].dropna()
            if len(valid):
                combo_rows.append({
                    "Group": label, "Period": p,
                    "Avg Return": valid.mean(),
                    "Win Rate": (valid>0).mean()*100,
                    "Trades": len(valid),
                })
    if combo_rows:
        cdf = pd.DataFrame(combo_rows)
        fig = px.line(cdf, x='Period', y='Avg Return', color='Group',
                      color_discrete_sequence=[GREEN, RED, GOLD],
                      markers=True)
        fig.add_hline(y=0, line_dash="dot", line_color=TEXT_DIM)
        fig = plotly_layout(fig, "Avg Return by Period", height=380)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(cdf, x='Period', y='Win Rate', color='Group',
                       color_discrete_sequence=[GREEN, RED, GOLD],
                       markers=True)
        fig2.add_hline(y=50, line_dash="dot", line_color=TEXT_DIM)
        fig2 = plotly_layout(fig2, "Win Rate by Period", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter: score_net vs return
    section("SCORE NET vs FORWARD RETURN (scatter)")
    scatter_data = buys[[ret_col, 'score_net', 'ticker', 'politician_name']].dropna()
    scatter_data = scatter_data[scatter_data[ret_col].between(-100, 200)]  # cap for viz
    fig = px.scatter(scatter_data, x='score_net', y=ret_col,
                     opacity=0.4,
                     color=ret_col, color_continuous_scale='RdYlGn',
                     hover_data=['ticker','politician_name'],
                     labels={'score_net': 'Score Net (Bullish - Bearish)', ret_col: f'Return {period} %'})
    fig.add_hline(y=0, line_dash="dot", line_color=TEXT_DIM)
    fig.add_vline(x=0, line_dash="dot", line_color=TEXT_DIM)
    fig = plotly_layout(fig, "Does Score Net Predict Forward Returns?", height=450)
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  TAB 6 — RAW DATA EXPLORER
# ─────────────────────────────────────────────────────────────
def tab_raw(fdf, period):
    ret_col = f"return_{period}"
    section("RAW TRADE EXPLORER")

    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        search_pol = st.text_input("Filter by Politician", "")
    with c2:
        search_tick = st.text_input("Filter by Ticker", "")
    with c3:
        min_ret = st.number_input("Min Return %", value=-999.0)

    data = fdf.copy()
    if search_pol:
        data = data[data['politician_name'].str.contains(search_pol, case=False, na=False)]
    if search_tick:
        data = data[data['ticker'].str.contains(search_tick, case=False, na=False)]
    if min_ret > -999:
        data = data[data[ret_col] >= min_ret]

    cols = ['trade_date','ticker','politician_name','party_clean','state','chamber',
            'tx_type','size_range','est_volume_usd'] + [f'return_{p}' for p in RETURN_PERIODS if f'return_{p}' in data.columns]
    if 'score_net' in data.columns:
        cols.append('score_net')

    display_cols = [c for c in cols if c in data.columns]
    st.markdown(f'<div style="font-family:Space Mono;font-size:11px;color:{TEXT_DIM};margin-bottom:8px">{len(data):,} TRADES SHOWN</div>', unsafe_allow_html=True)
    st.dataframe(data[display_cols].sort_values('trade_date', ascending=False).head(2000),
                 use_container_width=True, height=600)

    # Download
    csv_data = data[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered CSV", csv_data, "congress_filtered.csv", "text/csv")


# ─────────────────────────────────────────────────────────────
#  TAB 7 — SECTOR / STATE ANALYSIS
# ─────────────────────────────────────────────────────────────
def tab_geo_sector(fdf, period):
    ret_col = f"return_{period}"
    buys = fdf[fdf['is_buy']].copy()

    # State analysis
    section("STATE ANALYSIS (BUYS)")
    state_data = buys.groupby('state').agg(
        trades=('ticker','count'),
        avg_return=(ret_col,'mean'),
        win_rate=(ret_col, lambda x: (x.dropna()>0).mean()*100),
        politicians=('politician_name','nunique'),
    ).reset_index().dropna(subset=['avg_return'])

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(state_data.sort_values('avg_return', ascending=False).head(20),
                     x='state', y='avg_return', color='avg_return',
                     color_continuous_scale='RdYlGn',
                     labels={'avg_return': f'Avg {period} Return %', 'state': 'State'})
        fig.update_coloraxes(showscale=False)
        fig = plotly_layout(fig, "Top 20 States by Avg Return", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(state_data, x='trades', y='avg_return',
                         size='politicians', hover_name='state',
                         color='win_rate', color_continuous_scale='RdYlGn',
                         labels={'avg_return': f'Avg {period}%', 'trades': 'Total Buy Trades'})
        fig.add_hline(y=0, line_dash="dot", line_color=TEXT_DIM)
        fig = plotly_layout(fig, "State: Trades vs Return (bubble=# politicians)", height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Chamber comparison
    section("CHAMBER COMPARISON (BUYS)")
    chamber_data = buys.groupby('chamber').agg(
        trades=('ticker','count'),
        avg_return=(ret_col,'mean'),
        win_rate=(ret_col, lambda x: (x.dropna()>0).mean()*100),
    ).reset_index()

    chamber_rows = []
    for _, row in chamber_data.iterrows():
        chamber_rows.append({
            "Chamber": row['chamber'],
            "Trades": row['trades'],
            f"Avg {period}": fmt_pct(row['avg_return']),
            "Win Rate": f"{row['win_rate']:.1f}%",
        })
    st.dataframe(pd.DataFrame(chamber_rows), use_container_width=True, hide_index=True)

    # Volume over time by party
    section("TRADE VOLUME OVER TIME BY PARTY")
    vol_time = fdf.groupby(['quarter','party_clean'])['est_volume_usd'].sum().reset_index()
    vol_time = vol_time.sort_values('quarter')
    fig = px.bar(vol_time, x='quarter', y='est_volume_usd', color='party_clean',
                 color_discrete_map={"Democrat": DEM_COLOR, "Republican": REP_COLOR, "Independent": IND_COLOR},
                 barmode='group',
                 labels={'est_volume_usd': 'Estimated Volume ($)', 'quarter': 'Quarter'})
    fig = plotly_layout(fig, height=380)
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    # Load
    with st.spinner("Loading congressional trade data..."):
        df, pol_df, tick_df = load_data()

    if df is None:
        st.error("""
        ⚠️ **No data file found!**

        Place `congress_backtest_results.csv` in the same directory as this script, then refresh.

        Expected location:  `congress_backtest_results.csv`
        """)
        st.stop()

    # Sidebar
    date_range, tx_type, parties, chambers, period, min_trades = sidebar(df)
    fdf = apply_filters(df, date_range, tx_type, parties, chambers)

    if len(fdf) == 0:
        st.warning("No trades match the current filters.")
        st.stop()

    # Page header
    st.markdown(f"""
    <div style="display:flex; align-items:baseline; gap:16px; margin-bottom:24px;">
      <div style="font-family:'Space Mono',monospace; font-size:24px; font-weight:700; color:{ACCENT};">🏛️ CONGRESSIONAL TRADES INTELLIGENCE</div>
      <div style="font-family:'Space Mono',monospace; font-size:11px; color:{TEXT_DIM}; letter-spacing:0.1em;">ELWOOD'S TRADING LAB</div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tabs = st.tabs([
        "📊 Overview",
        "👤 Politicians",
        "📈 Tickers",
        "💰 Trade Size",
        "🔬 Scoring Combo",
        "🗺️ State / Chamber",
        "🔍 Raw Explorer",
    ])

    with tabs[0]: tab_overview(df, fdf, period)
    with tabs[1]: tab_politicians(fdf, period, min_trades)
    with tabs[2]: tab_tickers(fdf, period, min_trades)
    with tabs[3]: tab_size(fdf, period)
    with tabs[4]: tab_scoring(fdf, period)
    with tabs[5]: tab_geo_sector(fdf, period)
    with tabs[6]: tab_raw(fdf, period)


if __name__ == "__main__":
    main()
