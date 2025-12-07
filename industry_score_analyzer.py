"""
Industry Score Analysis Suite
=============================
Comprehensive industry-level analysis of bullish/bearish scores with:
- Trend scores and momentum
- Relative strength
- Breadth analysis
- Mean reversion signals
- Rotation analysis
- Volatility regimes
- And more...

Based on the market cap barometer design.

Author: Dave's Trading Analysis Suite
Created: December 2025
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    layout="wide",
    page_title="Elwood's Industry Score Analysis Suite",
    page_icon="📊"
)

st.title("📊 Elwood's Industry Score Analysis Suite")
st.markdown("*Comprehensive analysis of bullish/bearish scores at the industry level*")

# Configuration - Uses relative path to work with GitHub repo structure
# When running from the repo root, data is in Data/stock_scores/
# The script auto-detects if running locally or from repo
def find_data_directory():
    """Find the data directory - works for both local dev and GitHub repo structure."""
    possible_paths = [
        Path("Data/stock_scores"),                    # Running from repo root
        Path("../Data/stock_scores"),                 # Running from a subdirectory
        Path("../../Data/stock_scores"),              # Running from deeper subdirectory
        Path(__file__).parent / "Data" / "stock_scores",  # Relative to script location
        Path(__file__).parent.parent / "Data" / "stock_scores",
        Path(r"C:\Users\davet\Documents\GitHub\Industry-analysis\Data\stock_scores"),  # Dave's local
        Path(r"C:\Users\davet\Documents\new_dev\Industry-analysis\score_analysis\data"),  # Dave's alt local
    ]
    
    for path in possible_paths:
        try:
            if path.exists() and (path / "historical_data.parquet.gzip").exists():
                return path
        except:
            continue
    
    # Default fallback - will show error if not found
    return Path("Data/stock_scores")

DATA_DIR = find_data_directory()

# Color schemes
BULLISH_COLOR = '#00cc66'
BEARISH_COLOR = '#ff4444'
NEUTRAL_COLOR = '#ffaa00'
POSITIVE_COLOR = '#00ff00'
NEGATIVE_COLOR = '#ff0000'

# ------------------------------
# 1. Data Loading Functions
# ------------------------------

@st.cache_data
def load_historical_scores(data_dir):
    """Load the historical score data from parquet file."""
    try:
        parquet_path = Path(data_dir) / 'historical_data.parquet.gzip'
        
        if not parquet_path.exists():
            st.error(f"❌ Historical data file not found!")
            st.error(f"Looking for: `{parquet_path}`")
            st.info("""
            **To fix this:**
            1. Make sure you're running from the repository root directory
            2. Ensure the file `Data/stock_scores/historical_data.parquet.gzip` exists
            3. Or update the `find_data_directory()` function with your local path
            """)
            st.stop()
        
        df = pd.read_parquet(parquet_path)
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        
        # Calculate net score
        df['net_score'] = df['bullish_score'] - df['bearish_score']
        
        # Summary stats
        unique_dates = df['date'].nunique()
        unique_industries = df['industry'].nunique()
        unique_sectors = df['sector'].nunique()
        date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
        
        st.sidebar.success(f"✅ Loaded {len(df):,} records")
        st.sidebar.info(f"📅 {unique_dates} trading days ({date_range})")
        st.sidebar.info(f"🏭 {unique_industries} industries, {unique_sectors} sectors")
        st.sidebar.info(f"📂 Data: {data_dir}")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()


@st.cache_data
def load_sector_mapping(data_dir):
    """Load sector/industry mapping file."""
    try:
        sectors_path = Path(data_dir) / 'stock_sectors.xlsx'
        if sectors_path.exists():
            df = pd.read_excel(sectors_path)
            return df
    except:
        pass
    return None


# ------------------------------
# 2. Data Processing Functions
# ------------------------------

@st.cache_data
def calculate_industry_aggregates(df, lookback_days=None):
    """Calculate daily aggregates at the industry level."""
    
    if lookback_days:
        unique_dates = sorted(df['date'].unique())
        if len(unique_dates) > lookback_days:
            start_date = unique_dates[-lookback_days]
            df = df[df['date'] >= start_date]
    
    # First, get the most common sector for each industry (to handle any inconsistencies)
    industry_sector_map = df.groupby('industry')['sector'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]).reset_index()
    industry_sector_map.columns = ['industry', 'sector_mapped']
    
    # Aggregate by date and industry only (not sector to avoid duplicates)
    industry_agg = df.groupby(['date', 'industry']).agg({
        'bullish_score': ['mean', 'median', 'std', 'count'],
        'bearish_score': ['mean', 'median', 'std'],
        'net_score': ['mean', 'median'],
        'data_completeness': 'mean',
        'symbol': 'count'
    }).reset_index()
    
    # Flatten column names
    industry_agg.columns = [
        'date', 'industry',
        'bull_mean', 'bull_median', 'bull_std', 'bull_count',
        'bear_mean', 'bear_median', 'bear_std',
        'net_mean', 'net_median',
        'data_completeness', 'stock_count'
    ]
    
    # Add sector back via the mapping
    industry_agg = industry_agg.merge(industry_sector_map, on='industry', how='left')
    industry_agg = industry_agg.rename(columns={'sector_mapped': 'sector'})
    
    # Calculate breadth (% of stocks with bullish > bearish)
    breadth = df.groupby(['date', 'industry']).apply(
        lambda x: (x['bullish_score'] > x['bearish_score']).mean() * 100
    ).reset_index()
    breadth.columns = ['date', 'industry', 'breadth']
    
    industry_agg = industry_agg.merge(breadth, on=['date', 'industry'], how='left')
    
    # Calculate extreme readings
    extreme_bull = df.groupby(['date', 'industry']).apply(
        lambda x: (x['bullish_score'] > 70).mean() * 100
    ).reset_index()
    extreme_bull.columns = ['date', 'industry', 'pct_extreme_bull']
    
    extreme_bear = df.groupby(['date', 'industry']).apply(
        lambda x: (x['bearish_score'] > 70).mean() * 100
    ).reset_index()
    extreme_bear.columns = ['date', 'industry', 'pct_extreme_bear']
    
    industry_agg = industry_agg.merge(extreme_bull, on=['date', 'industry'], how='left')
    industry_agg = industry_agg.merge(extreme_bear, on=['date', 'industry'], how='left')
    
    # Ensure no duplicates - take mean if somehow there are still any
    industry_agg = industry_agg.groupby(['date', 'industry'], as_index=False).first()
    
    return industry_agg.sort_values(['date', 'industry'])


@st.cache_data
def calculate_market_aggregates(df, lookback_days=None):
    """Calculate overall market aggregates for comparison."""
    
    if lookback_days:
        unique_dates = sorted(df['date'].unique())
        if len(unique_dates) > lookback_days:
            start_date = unique_dates[-lookback_days]
            df = df[df['date'] >= start_date]
    
    market_agg = df.groupby('date').agg({
        'bullish_score': 'mean',
        'bearish_score': 'mean',
        'net_score': 'mean',
        'symbol': 'count'
    }).reset_index()
    
    market_agg.columns = ['date', 'market_bull', 'market_bear', 'market_net', 'total_stocks']
    
    # Market breadth
    market_breadth = df.groupby('date').apply(
        lambda x: (x['bullish_score'] > x['bearish_score']).mean() * 100
    ).reset_index()
    market_breadth.columns = ['date', 'market_breadth']
    
    market_agg = market_agg.merge(market_breadth, on='date', how='left')
    
    return market_agg.sort_values('date')


@st.cache_data
def calculate_score_momentum(industry_agg, periods=[1, 3, 5, 10, 20]):
    """Calculate rate of change in scores for multiple periods."""
    
    industry_agg = industry_agg.sort_values(['industry', 'date'])
    
    for period in periods:
        # Net score momentum
        industry_agg[f'net_roc_{period}d'] = industry_agg.groupby('industry')['net_mean'].diff(period)
        # Breadth momentum
        industry_agg[f'breadth_roc_{period}d'] = industry_agg.groupby('industry')['breadth'].diff(period)
        # Bull score momentum
        industry_agg[f'bull_roc_{period}d'] = industry_agg.groupby('industry')['bull_mean'].diff(period)
    
    return industry_agg


@st.cache_data
def calculate_relative_strength(industry_agg, market_agg, lookback_days=None):
    """Calculate relative strength vs overall market."""
    
    # Merge industry with market data
    merged = industry_agg.merge(
        market_agg[['date', 'market_net', 'market_breadth']],
        on='date',
        how='left'
    )
    
    # Relative strength = industry net score - market net score
    merged['relative_net'] = merged['net_mean'] - merged['market_net']
    merged['relative_breadth'] = merged['breadth'] - merged['market_breadth']
    
    # Cumulative relative strength
    merged = merged.sort_values(['industry', 'date'])
    merged['cumulative_rs'] = merged.groupby('industry')['relative_net'].cumsum()
    
    return merged


@st.cache_data
def calculate_trend_scores(industry_agg, start_date):
    """Calculate cumulative trend scores from a starting date."""
    
    industry_agg = industry_agg.sort_values(['industry', 'date'])
    
    # Direction based on net score change
    industry_agg['direction'] = np.where(
        industry_agg.groupby('industry')['net_mean'].diff() > 0, 1,
        np.where(industry_agg.groupby('industry')['net_mean'].diff() < 0, -1, 0)
    )
    
    # Filter from start date
    filtered = industry_agg[industry_agg['date'] >= pd.to_datetime(start_date)].copy()
    
    # Cumulative score
    filtered['trend_score'] = filtered.groupby('industry')['direction'].cumsum()
    
    return filtered


@st.cache_data
def calculate_mean_reversion_signals(industry_agg, zscore_window=20):
    """Calculate mean reversion signals using z-scores."""
    
    industry_agg = industry_agg.sort_values(['industry', 'date'])
    
    # Calculate rolling stats for z-score
    industry_agg['rolling_mean'] = industry_agg.groupby('industry')['net_mean'].transform(
        lambda x: x.rolling(zscore_window, min_periods=5).mean()
    )
    industry_agg['rolling_std'] = industry_agg.groupby('industry')['net_mean'].transform(
        lambda x: x.rolling(zscore_window, min_periods=5).std()
    )
    
    # Z-score
    industry_agg['zscore'] = (industry_agg['net_mean'] - industry_agg['rolling_mean']) / industry_agg['rolling_std'].replace(0, np.nan)
    
    # Classify signals
    industry_agg['signal'] = 'Neutral'
    industry_agg.loc[industry_agg['zscore'] > 2, 'signal'] = 'Overbought'
    industry_agg.loc[industry_agg['zscore'] < -2, 'signal'] = 'Oversold'
    industry_agg.loc[(industry_agg['zscore'] > 1) & (industry_agg['zscore'] <= 2), 'signal'] = 'Extended Up'
    industry_agg.loc[(industry_agg['zscore'] < -1) & (industry_agg['zscore'] >= -2), 'signal'] = 'Extended Down'
    
    return industry_agg


@st.cache_data
def calculate_rotation_stages(industry_agg, momentum_col='net_roc_5d'):
    """Classify industries into rotation quadrants."""
    
    # Get latest data
    latest_date = industry_agg['date'].max()
    latest = industry_agg[industry_agg['date'] == latest_date].copy()
    
    if momentum_col not in latest.columns or 'cumulative_rs' not in latest.columns:
        return pd.DataFrame()
    
    # Calculate medians for classification
    median_momentum = latest[momentum_col].median()
    median_rs = latest['cumulative_rs'].median()
    
    def classify(row):
        mom = row[momentum_col]
        rs = row['cumulative_rs']
        
        if pd.isna(mom) or pd.isna(rs):
            return 'Unknown'
        elif mom > median_momentum and rs > median_rs:
            return 'Leading'
        elif mom <= median_momentum and rs > median_rs:
            return 'Weakening'
        elif mom <= median_momentum and rs <= median_rs:
            return 'Lagging'
        else:
            return 'Improving'
    
    latest['rotation_stage'] = latest.apply(classify, axis=1)
    
    return latest


@st.cache_data
def calculate_streaks(industry_agg):
    """Calculate win/loss streaks for each industry."""
    
    industry_agg = industry_agg.sort_values(['industry', 'date'])
    
    # Direction based on net score change
    industry_agg['direction'] = np.where(
        industry_agg.groupby('industry')['net_mean'].diff() > 0, 1,
        np.where(industry_agg.groupby('industry')['net_mean'].diff() < 0, -1, 0)
    )
    
    results = []
    
    for industry in industry_agg['industry'].unique():
        ind_data = industry_agg[industry_agg['industry'] == industry].copy()
        
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        temp_streak = 0
        prev_dir = 0
        up_days = 0
        down_days = 0
        
        for direction in ind_data['direction']:
            if direction == 1:
                up_days += 1
            elif direction == -1:
                down_days += 1
            
            if direction == 0:
                temp_streak = 0
            elif direction == prev_dir and prev_dir != 0:
                temp_streak += 1
            else:
                temp_streak = 1
            
            if direction != 0:
                if direction == 1:
                    max_win_streak = max(max_win_streak, temp_streak)
                else:
                    max_loss_streak = max(max_loss_streak, temp_streak)
                prev_dir = direction
        
        # Current streak
        last_dir = ind_data['direction'].iloc[-1]
        if last_dir == 1:
            current_streak = temp_streak
            streak_type = 'Win'
        elif last_dir == -1:
            current_streak = -temp_streak
            streak_type = 'Loss'
        else:
            current_streak = 0
            streak_type = 'Flat'
        
        total_days = up_days + down_days
        win_rate = (up_days / total_days * 100) if total_days > 0 else 50
        
        results.append({
            'industry': industry,
            'sector': ind_data['sector'].iloc[0],
            'current_streak': current_streak,
            'streak_type': streak_type,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'up_days': up_days,
            'down_days': down_days,
            'win_rate': win_rate
        })
    
    return pd.DataFrame(results)


@st.cache_data
def calculate_volatility_metrics(industry_agg, window=20):
    """Calculate score volatility metrics."""
    
    industry_agg = industry_agg.sort_values(['industry', 'date'])
    
    # Rolling volatility of net score
    industry_agg['net_volatility'] = industry_agg.groupby('industry')['net_mean'].transform(
        lambda x: x.rolling(window, min_periods=5).std()
    )
    
    # Rolling volatility of breadth
    industry_agg['breadth_volatility'] = industry_agg.groupby('industry')['breadth'].transform(
        lambda x: x.rolling(window, min_periods=5).std()
    )
    
    # Drawdown from peak net score
    industry_agg['net_peak'] = industry_agg.groupby('industry')['net_mean'].transform(
        lambda x: x.expanding().max()
    )
    industry_agg['net_drawdown'] = industry_agg['net_mean'] - industry_agg['net_peak']
    
    return industry_agg


@st.cache_data  
def get_current_summary(industry_agg, market_agg):
    """Get current state summary for all industries."""
    
    latest_date = industry_agg['date'].max()
    current = industry_agg[industry_agg['date'] == latest_date].copy()
    market_current = market_agg[market_agg['date'] == latest_date].iloc[0]
    
    # Add market context
    current['vs_market_net'] = current['net_mean'] - market_current['market_net']
    current['vs_market_breadth'] = current['breadth'] - market_current['market_breadth']
    
    return current, market_current


# ------------------------------
# 3. Visualization Functions
# ------------------------------

def create_industry_heatmap(industry_agg, metric='net_mean', last_n_days=30):
    """Create heatmap of industry scores over time."""
    
    # Get last N days
    unique_dates = sorted(industry_agg['date'].unique())
    if len(unique_dates) > last_n_days:
        start_date = unique_dates[-last_n_days]
        industry_agg = industry_agg[industry_agg['date'] >= start_date]
    
    # Use pivot_table with mean aggregation to handle any duplicates
    pivot = industry_agg.pivot_table(
        index='industry', 
        columns='date', 
        values=metric,
        aggfunc='mean'  # Average if there are duplicates
    )
    
    # Sort by average value
    pivot['avg'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('avg', ascending=False)
    pivot = pivot.drop('avg', axis=1)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[d.strftime('%m/%d') for d in pivot.columns],
        y=pivot.index,
        colorscale='RdYlGn',
        zmid=0 if 'net' in metric else 50,
        hoverongaps=False,
        hovertemplate='%{y}<br>%{x}<br>Value: %{z:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        height=max(400, len(pivot) * 20),
        xaxis_title="Date",
        yaxis_title="Industry",
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_score_distribution_chart(current_data, metric='net_mean'):
    """Create bar chart of current industry scores."""
    
    sorted_data = current_data.sort_values(metric, ascending=True)
    
    colors = [BULLISH_COLOR if x > 0 else BEARISH_COLOR for x in sorted_data[metric]]
    
    fig = go.Figure(go.Bar(
        x=sorted_data[metric],
        y=sorted_data['industry'],
        orientation='h',
        marker_color=colors,
        text=[f"{x:.1f}" for x in sorted_data[metric]],
        textposition='outside'
    ))
    
    fig.add_vline(x=0, line_dash="solid", line_color="white", line_width=2)
    
    fig.update_layout(
        height=max(500, len(sorted_data) * 22),
        xaxis_title=metric.replace('_', ' ').title(),
        yaxis_title="",
        showlegend=False
    )
    
    return fig


def create_momentum_scatter(current_data, x_col='net_roc_5d', y_col='cumulative_rs'):
    """Create momentum vs relative strength scatter plot."""
    
    fig = px.scatter(
        current_data,
        x=x_col,
        y=y_col,
        color='sector',
        hover_name='industry',
        size='stock_count',
        size_max=30
    )
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    # Add quadrant labels
    fig.add_annotation(x=0.95, y=0.95, xref="paper", yref="paper",
                       text="Leading", showarrow=False, font=dict(size=14, color="green"))
    fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper",
                       text="Weakening", showarrow=False, font=dict(size=14, color="orange"))
    fig.add_annotation(x=0.05, y=0.05, xref="paper", yref="paper",
                       text="Lagging", showarrow=False, font=dict(size=14, color="red"))
    fig.add_annotation(x=0.95, y=0.05, xref="paper", yref="paper",
                       text="Improving", showarrow=False, font=dict(size=14, color="blue"))
    
    fig.update_layout(
        height=600,
        xaxis_title="5-Day Net Score Change (Momentum)",
        yaxis_title="Cumulative Relative Strength"
    )
    
    return fig


def create_breadth_chart(industry_agg, top_n=20):
    """Create breadth over time chart."""
    
    # Get top/bottom industries by current breadth
    latest = industry_agg[industry_agg['date'] == industry_agg['date'].max()]
    top_industries = latest.nlargest(top_n // 2, 'breadth')['industry'].tolist()
    bottom_industries = latest.nsmallest(top_n // 2, 'breadth')['industry'].tolist()
    selected = top_industries + bottom_industries
    
    filtered = industry_agg[industry_agg['industry'].isin(selected)]
    
    fig = px.line(
        filtered,
        x='date',
        y='breadth',
        color='industry',
        hover_data=['sector', 'net_mean']
    )
    
    fig.add_hline(y=50, line_dash="dash", line_color="white")
    fig.add_hline(y=30, line_dash="dot", line_color="red")
    fig.add_hline(y=70, line_dash="dot", line_color="green")
    
    fig.update_layout(
        height=500,
        xaxis_title="Date",
        yaxis_title="Breadth (%)",
        legend=dict(orientation="h", y=-0.2)
    )
    
    return fig


def create_zscore_chart(current_data):
    """Create z-score distribution chart."""
    
    sorted_data = current_data.sort_values('zscore', ascending=True)
    
    def get_color(z):
        if z > 2:
            return '#ff0000'  # Overbought
        elif z > 1:
            return '#ffaa00'  # Extended up
        elif z < -2:
            return '#00ff00'  # Oversold
        elif z < -1:
            return '#88ff88'  # Extended down
        else:
            return '#888888'  # Neutral
    
    colors = [get_color(z) for z in sorted_data['zscore']]
    
    fig = go.Figure(go.Bar(
        x=sorted_data['zscore'],
        y=sorted_data['industry'],
        orientation='h',
        marker_color=colors,
        text=[f"{x:.2f}" for x in sorted_data['zscore']],
        textposition='outside'
    ))
    
    fig.add_vline(x=-2, line_dash="dot", line_color="green")
    fig.add_vline(x=-1, line_dash="dot", line_color="lightgreen")
    fig.add_vline(x=0, line_dash="solid", line_color="white")
    fig.add_vline(x=1, line_dash="dot", line_color="orange")
    fig.add_vline(x=2, line_dash="dot", line_color="red")
    
    fig.update_layout(
        height=max(500, len(sorted_data) * 22),
        xaxis_title="Z-Score",
        yaxis_title="",
        title="Mean Reversion Signals (Z-Score)"
    )
    
    return fig


def create_streaks_chart(streaks_data):
    """Create current streaks visualization."""
    
    sorted_data = streaks_data.sort_values('current_streak', ascending=True)
    
    colors = [BULLISH_COLOR if x > 0 else BEARISH_COLOR if x < 0 else NEUTRAL_COLOR 
              for x in sorted_data['current_streak']]
    
    fig = go.Figure(go.Bar(
        x=sorted_data['current_streak'],
        y=sorted_data['industry'],
        orientation='h',
        marker_color=colors,
        text=[f"{x:+d}" for x in sorted_data['current_streak']],
        textposition='outside'
    ))
    
    fig.add_vline(x=0, line_dash="solid", line_color="white")
    
    fig.update_layout(
        height=max(500, len(sorted_data) * 20),
        xaxis_title="Current Streak (days)",
        yaxis_title=""
    )
    
    return fig


def create_rotation_matrix(rotation_data):
    """Create rotation quadrant scatter plot."""
    
    if rotation_data.empty or 'net_roc_5d' not in rotation_data.columns:
        return None
    
    stage_colors = {
        'Leading': 'green',
        'Weakening': 'orange',
        'Lagging': 'red',
        'Improving': 'blue',
        'Unknown': 'gray'
    }
    
    rotation_data['color'] = rotation_data['rotation_stage'].map(stage_colors)
    
    fig = go.Figure()
    
    for stage in ['Leading', 'Weakening', 'Lagging', 'Improving']:
        stage_data = rotation_data[rotation_data['rotation_stage'] == stage]
        if not stage_data.empty:
            fig.add_trace(go.Scatter(
                x=stage_data['net_roc_5d'],
                y=stage_data['cumulative_rs'],
                mode='markers+text',
                name=stage,
                text=stage_data['industry'].apply(lambda x: x[:15] + '...' if len(x) > 15 else x),
                textposition='top center',
                marker=dict(size=10, color=stage_colors[stage]),
                hovertemplate='%{text}<br>Momentum: %{x:.1f}<br>RS: %{y:.1f}<extra></extra>'
            ))
    
    # Quadrant lines
    fig.add_hline(y=rotation_data['cumulative_rs'].median(), line_dash="dash", line_color="gray")
    fig.add_vline(x=rotation_data['net_roc_5d'].median(), line_dash="dash", line_color="gray")
    
    fig.update_layout(
        height=700,
        xaxis_title="5-Day Momentum (Net Score Change)",
        yaxis_title="Cumulative Relative Strength",
        showlegend=True,
        legend=dict(orientation="h", y=-0.1)
    )
    
    return fig


def create_sector_summary_chart(industry_agg):
    """Create sector-level summary chart."""
    
    latest = industry_agg[industry_agg['date'] == industry_agg['date'].max()]
    
    sector_agg = latest.groupby('sector').agg({
        'net_mean': 'mean',
        'breadth': 'mean',
        'stock_count': 'sum',
        'industry': 'count'
    }).reset_index()
    
    sector_agg.columns = ['sector', 'avg_net', 'avg_breadth', 'total_stocks', 'num_industries']
    sector_agg = sector_agg.sort_values('avg_net', ascending=True)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Avg Net Score', 'Avg Breadth'))
    
    colors = [BULLISH_COLOR if x > 0 else BEARISH_COLOR for x in sector_agg['avg_net']]
    
    fig.add_trace(
        go.Bar(x=sector_agg['avg_net'], y=sector_agg['sector'], orientation='h',
               marker_color=colors, name='Net Score'),
        row=1, col=1
    )
    
    breadth_colors = [BULLISH_COLOR if x > 50 else BEARISH_COLOR for x in sector_agg['avg_breadth']]
    
    fig.add_trace(
        go.Bar(x=sector_agg['avg_breadth'], y=sector_agg['sector'], orientation='h',
               marker_color=breadth_colors, name='Breadth'),
        row=1, col=2
    )
    
    fig.add_vline(x=0, line_dash="solid", line_color="white", row=1, col=1)
    fig.add_vline(x=50, line_dash="solid", line_color="white", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig


# ------------------------------
# 4. Main Application
# ------------------------------

def main():
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Lookback period
    lookback_options = {
        "30 Days": 30,
        "60 Days": 60,
        "90 Days": 90,
        "120 Days": 120,
        "All Data": None
    }
    lookback_choice = st.sidebar.selectbox("Lookback Period", list(lookback_options.keys()), index=1)
    lookback_days = lookback_options[lookback_choice]
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    enable_momentum = st.sidebar.checkbox("Enable Momentum Analysis", value=True)
    enable_relative_strength = st.sidebar.checkbox("Enable Relative Strength", value=True)
    enable_mean_reversion = st.sidebar.checkbox("Enable Mean Reversion", value=True)
    enable_rotation = st.sidebar.checkbox("Enable Rotation Analysis", value=True)
    enable_streaks = st.sidebar.checkbox("Enable Streak Analysis", value=True)
    
    # Z-score window
    zscore_window = st.sidebar.slider("Z-Score Window", 10, 40, 20)
    
    # Load data
    with st.spinner("Loading data..."):
        raw_data = load_historical_scores(DATA_DIR)
        sector_mapping = load_sector_mapping(DATA_DIR)
    
    # Calculate aggregates
    with st.spinner("Calculating industry aggregates..."):
        industry_agg = calculate_industry_aggregates(raw_data, lookback_days)
        market_agg = calculate_market_aggregates(raw_data, lookback_days)
    
    # Calculate additional metrics
    if enable_momentum:
        industry_agg = calculate_score_momentum(industry_agg)
    
    if enable_relative_strength:
        industry_agg = calculate_relative_strength(industry_agg, market_agg)
    
    if enable_mean_reversion:
        industry_agg = calculate_mean_reversion_signals(industry_agg, zscore_window)
    
    industry_agg = calculate_volatility_metrics(industry_agg)
    
    # Get current summary
    current_data, market_current = get_current_summary(industry_agg, market_agg)
    
    # Calculate additional analysis
    if enable_rotation and 'cumulative_rs' in industry_agg.columns:
        rotation_data = calculate_rotation_stages(industry_agg)
    else:
        rotation_data = pd.DataFrame()
    
    if enable_streaks:
        streaks_data = calculate_streaks(industry_agg)
    else:
        streaks_data = pd.DataFrame()
    
    # ==================== MAIN CONTENT ====================
    
    # Summary metrics
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Market Net Score", f"{market_current['market_net']:.1f}",
                  delta=f"{market_current['market_net'] - 3.86:.1f} vs avg")
    with col2:
        st.metric("Market Breadth", f"{market_current['market_breadth']:.1f}%",
                  delta=f"{market_current['market_breadth'] - 50.37:.1f}")
    with col3:
        bullish_industries = len(current_data[current_data['net_mean'] > 0])
        st.metric("Bullish Industries", f"{bullish_industries}/{len(current_data)}")
    with col4:
        avg_breadth = current_data['breadth'].mean()
        st.metric("Avg Industry Breadth", f"{avg_breadth:.1f}%")
    with col5:
        st.metric("Trading Days", f"{industry_agg['date'].nunique()}")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Overview",
        "🎯 Current Scores",
        "📈 Momentum",
        "💪 Relative Strength",
        "🔄 Rotation",
        "📉 Mean Reversion",
        "🔥 Streaks",
        "🌡️ Heatmaps",
        "📋 Data Export"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("📊 Market Overview")
        
        st.info("""
        **What This Shows:** A bird's-eye view of the entire market's technical health.
        
        - **Net Score** = Bullish Score minus Bearish Score. Positive means more bullish signals, negative means more bearish.
        - **Breadth** = Percentage of stocks where bullish score > bearish score. Above 50% means more stocks are bullish than bearish.
        - **Top/Bottom Industries** = Which industries have the strongest/weakest technical setups right now.
        
        💡 *Use this tab to quickly gauge overall market sentiment and identify sector leaders/laggards.*
        """)
        
        # Sector summary
        st.subheader("Sector Summary")
        fig_sector = create_sector_summary_chart(industry_agg)
        st.plotly_chart(fig_sector, use_container_width=True)
        
        # Quick stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌟 Top 10 Industries (Net Score)")
            top_10 = current_data.nlargest(10, 'net_mean')[['industry', 'sector', 'net_mean', 'breadth', 'stock_count']]
            top_10.columns = ['Industry', 'Sector', 'Net Score', 'Breadth %', 'Stocks']
            st.dataframe(top_10.style.format({'Net Score': '{:.1f}', 'Breadth %': '{:.1f}'}), 
                        use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("⚠️ Bottom 10 Industries (Net Score)")
            bottom_10 = current_data.nsmallest(10, 'net_mean')[['industry', 'sector', 'net_mean', 'breadth', 'stock_count']]
            bottom_10.columns = ['Industry', 'Sector', 'Net Score', 'Breadth %', 'Stocks']
            st.dataframe(bottom_10.style.format({'Net Score': '{:.1f}', 'Breadth %': '{:.1f}'}),
                        use_container_width=True, hide_index=True)
    
    # Tab 2: Current Scores
    with tab2:
        st.header("🎯 Current Industry Scores")
        
        st.info("""
        **What This Shows:** Today's snapshot of all industries ranked by their technical scores.
        
        - **Net Score** = How bullish vs bearish the industry is overall. Green bars = bullish, Red bars = bearish.
        - **Breadth** = What % of stocks in that industry are bullish. 100% = every stock is bullish.
        - **Bull/Bear Mean** = Average bullish or bearish score across all stocks in the industry.
        
        💡 *Look for industries with high net scores AND high breadth - these have the strongest technical setups. 
        Industries with very low scores may be oversold and due for a bounce.*
        """)
        
        metric_choice = st.selectbox("Select Metric", ['net_mean', 'breadth', 'bull_mean', 'bear_mean'])
        
        fig = create_score_distribution_chart(current_data, metric_choice)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        if st.checkbox("Show detailed data"):
            display_cols = ['industry', 'sector', 'net_mean', 'bull_mean', 'bear_mean', 
                          'breadth', 'stock_count', 'data_completeness']
            st.dataframe(
                current_data[display_cols].sort_values('net_mean', ascending=False).style.format({
                    'net_mean': '{:.1f}',
                    'bull_mean': '{:.1f}',
                    'bear_mean': '{:.1f}',
                    'breadth': '{:.1f}%',
                    'data_completeness': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True
            )
    
    # Tab 3: Momentum
    with tab3:
        st.header("📈 Score Momentum Analysis")
        
        st.info("""
        **What This Shows:** How quickly industry scores are CHANGING, not just where they are now.
        
        - **5d Change** = How much the net score changed over the last 5 trading days.
        - **Positive momentum** = Scores are improving (getting more bullish).
        - **Negative momentum** = Scores are deteriorating (getting more bearish).
        
        💡 *Momentum often leads price! Industries with improving momentum may be early in a move up.
        Industries with rapidly declining momentum may be warning of trouble ahead, even if current scores look okay.*
        
        🎯 *Key insight from backtesting: When bullish scores drop more than 10 points in 5 days, 
        markets often bounce (85% win rate historically).*
        """)
        
        if enable_momentum:
            # Momentum metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                improving = len(current_data[current_data.get('net_roc_5d', pd.Series([0])) > 0])
                st.metric("Industries Improving (5d)", f"{improving}/{len(current_data)}")
            
            with col2:
                if 'net_roc_5d' in current_data.columns:
                    avg_mom = current_data['net_roc_5d'].mean()
                    st.metric("Avg 5d Momentum", f"{avg_mom:.2f}")
            
            with col3:
                if 'breadth_roc_5d' in current_data.columns:
                    avg_breadth_mom = current_data['breadth_roc_5d'].mean()
                    st.metric("Avg Breadth Momentum", f"{avg_breadth_mom:.2f}")
            
            # Momentum rankings
            if 'net_roc_5d' in current_data.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🚀 Fastest Improving")
                    top_mom = current_data.nlargest(10, 'net_roc_5d')[['industry', 'sector', 'net_roc_5d', 'net_mean']]
                    top_mom.columns = ['Industry', 'Sector', '5d Change', 'Current Net']
                    st.dataframe(top_mom.style.format({'5d Change': '{:+.2f}', 'Current Net': '{:.1f}'}),
                               use_container_width=True, hide_index=True)
                
                with col2:
                    st.subheader("📉 Fastest Declining")
                    bot_mom = current_data.nsmallest(10, 'net_roc_5d')[['industry', 'sector', 'net_roc_5d', 'net_mean']]
                    bot_mom.columns = ['Industry', 'Sector', '5d Change', 'Current Net']
                    st.dataframe(bot_mom.style.format({'5d Change': '{:+.2f}', 'Current Net': '{:.1f}'}),
                               use_container_width=True, hide_index=True)
        else:
            st.warning("Momentum analysis is disabled. Enable it in the sidebar.")
    
    # Tab 4: Relative Strength
    with tab4:
        st.header("💪 Relative Strength Analysis")
        
        st.info("""
        **What This Shows:** How each industry is performing COMPARED TO the overall market.
        
        - **Cumulative RS** = Running total of how much better/worse this industry has scored vs the market average.
        - **Positive RS** = This industry has been stronger than the market.
        - **Negative RS** = This industry has been weaker than the market.
        
        💡 *Relative strength helps identify true leaders vs laggards. An industry can have a positive 
        net score but still be underperforming the market (negative RS). True leaders have BOTH 
        positive scores AND positive relative strength.*
        
        🎯 *The scatter plot shows momentum (x-axis) vs relative strength (y-axis). 
        Industries in the upper-right quadrant are the strongest overall.*
        """)
        
        if enable_relative_strength and 'cumulative_rs' in current_data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Strongest vs Market")
                strongest = current_data.nlargest(10, 'cumulative_rs')[['industry', 'sector', 'cumulative_rs', 'net_mean']]
                strongest.columns = ['Industry', 'Sector', 'Cum RS', 'Net Score']
                st.dataframe(strongest.style.format({'Cum RS': '{:+.1f}', 'Net Score': '{:.1f}'}),
                           use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("Weakest vs Market")
                weakest = current_data.nsmallest(10, 'cumulative_rs')[['industry', 'sector', 'cumulative_rs', 'net_mean']]
                weakest.columns = ['Industry', 'Sector', 'Cum RS', 'Net Score']
                st.dataframe(weakest.style.format({'Cum RS': '{:+.1f}', 'Net Score': '{:.1f}'}),
                           use_container_width=True, hide_index=True)
            
            # Scatter plot
            if 'net_roc_5d' in current_data.columns:
                st.subheader("Momentum vs Relative Strength")
                fig = create_momentum_scatter(current_data)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Relative strength analysis is disabled or data not available.")
    
    # Tab 5: Rotation
    with tab5:
        st.header("🔄 Rotation Analysis")
        
        st.info("""
        **What This Shows:** Where each industry sits in the "rotation cycle" - a framework used by 
        institutional investors to identify market leadership changes.
        
        **The Four Quadrants:**
        - 🟢 **Leading** = Strong momentum + Strong relative strength. These are the current market leaders.
        - 🟠 **Weakening** = Slowing momentum + Still strong RS. Leaders that may be topping out.
        - 🔴 **Lagging** = Weak momentum + Weak RS. These are the market's losers.
        - 🔵 **Improving** = Gaining momentum + Still weak RS. Potential NEW leaders emerging!
        
        💡 *The classic rotation pattern is: Improving → Leading → Weakening → Lagging → Improving...*
        
        🎯 *Watch the "Improving" quadrant closely - these industries are gaining strength and 
        may become the next leaders. "Weakening" industries may be good candidates to take profits on.*
        """)
        
        if enable_rotation and not rotation_data.empty:
            # Stage counts
            stage_counts = rotation_data['rotation_stage'].value_counts()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🟢 Leading", stage_counts.get('Leading', 0))
            with col2:
                st.metric("🟠 Weakening", stage_counts.get('Weakening', 0))
            with col3:
                st.metric("🔴 Lagging", stage_counts.get('Lagging', 0))
            with col4:
                st.metric("🔵 Improving", stage_counts.get('Improving', 0))
            
            # Rotation matrix
            fig = create_rotation_matrix(rotation_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Stage tables
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌟 Leaders & Improvers")
                leaders = rotation_data[rotation_data['rotation_stage'].isin(['Leading', 'Improving'])]
                if not leaders.empty:
                    display = leaders[['industry', 'sector', 'rotation_stage', 'net_mean', 'net_roc_5d']].sort_values('net_roc_5d', ascending=False)
                    display.columns = ['Industry', 'Sector', 'Stage', 'Net Score', 'Momentum']
                    st.dataframe(display.style.format({'Net Score': '{:.1f}', 'Momentum': '{:+.2f}'}),
                               use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("⚠️ Weakening & Lagging")
                laggards = rotation_data[rotation_data['rotation_stage'].isin(['Weakening', 'Lagging'])]
                if not laggards.empty:
                    display = laggards[['industry', 'sector', 'rotation_stage', 'net_mean', 'net_roc_5d']].sort_values('net_roc_5d', ascending=True)
                    display.columns = ['Industry', 'Sector', 'Stage', 'Net Score', 'Momentum']
                    st.dataframe(display.style.format({'Net Score': '{:.1f}', 'Momentum': '{:+.2f}'}),
                               use_container_width=True, hide_index=True)
        else:
            st.warning("Rotation analysis requires momentum and relative strength data.")
    
    # Tab 6: Mean Reversion
    with tab6:
        st.header("📉 Mean Reversion Signals")
        
        st.info("""
        **What This Shows:** Which industries are stretched too far from their normal levels and may "snap back."
        
        - **Z-Score** = How many standard deviations away from the recent average. 
          - Z > +2 = Overbought (unusually bullish, may pull back)
          - Z < -2 = Oversold (unusually bearish, may bounce)
          - Z between -1 and +1 = Normal range
        
        💡 *Mean reversion is the tendency for extreme readings to return to normal levels.
        Markets are like rubber bands - the more stretched they get, the harder they tend to snap back.*
        
        🎯 *Key insight from backtesting: Industries with extremely bearish readings (top 10% of bearish scores)
        have historically seen positive 10-day returns 83% of the time. Oversold = opportunity!*
        """)
        
        if enable_mean_reversion and 'zscore' in current_data.columns:
            # Signal counts
            signal_counts = current_data['signal'].value_counts()
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("🔴 Overbought", signal_counts.get('Overbought', 0))
            with col2:
                st.metric("🟠 Extended Up", signal_counts.get('Extended Up', 0))
            with col3:
                st.metric("⚪ Neutral", signal_counts.get('Neutral', 0))
            with col4:
                st.metric("🟢 Extended Down", signal_counts.get('Extended Down', 0))
            with col5:
                st.metric("💚 Oversold", signal_counts.get('Oversold', 0))
            
            # Z-score chart
            fig = create_zscore_chart(current_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Oversold opportunities
            st.subheader("🎯 Oversold Opportunities (Z < -1)")
            oversold = current_data[current_data['zscore'] < -1].sort_values('zscore')
            if not oversold.empty:
                display = oversold[['industry', 'sector', 'zscore', 'net_mean', 'breadth', 'signal']]
                display.columns = ['Industry', 'Sector', 'Z-Score', 'Net Score', 'Breadth', 'Signal']
                st.dataframe(display.style.format({'Z-Score': '{:.2f}', 'Net Score': '{:.1f}', 'Breadth': '{:.1f}%'}),
                           use_container_width=True, hide_index=True)
            else:
                st.info("No oversold industries currently")
        else:
            st.warning("Mean reversion analysis is disabled.")
    
    # Tab 7: Streaks
    with tab7:
        st.header("🔥 Streak Analysis")
        
        st.info("""
        **What This Shows:** Consecutive days of improvement or deterioration for each industry.
        
        - **Current Streak** = How many days in a row scores have been improving (+) or declining (-).
        - **Win Rate** = What percentage of days showed improvement over the lookback period.
        - **Max Win/Loss Streak** = The longest consecutive run of up or down days.
        
        💡 *Long winning streaks indicate strong momentum and trend persistence.
        Long losing streaks may indicate capitulation and potential bottoming.*
        
        🎯 *Watch for streak reversals! When a long losing streak finally turns positive,
        it can signal the start of a new uptrend. Conversely, when a long winning streak breaks,
        it may be time to take profits.*
        """)
        
        if enable_streaks and not streaks_data.empty:
            # Summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                winning = len(streaks_data[streaks_data['current_streak'] > 0])
                st.metric("Currently Winning", f"{winning}/{len(streaks_data)}")
            with col2:
                max_win = streaks_data['max_win_streak'].max()
                st.metric("Longest Win Streak", f"{max_win} days")
            with col3:
                max_loss = streaks_data['max_loss_streak'].max()
                st.metric("Longest Loss Streak", f"{max_loss} days")
            with col4:
                avg_wr = streaks_data['win_rate'].mean()
                st.metric("Avg Win Rate", f"{avg_wr:.1f}%")
            
            # Streaks chart
            fig = create_streaks_chart(streaks_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            if st.checkbox("Show streak details"):
                display = streaks_data.sort_values('current_streak', ascending=False)
                st.dataframe(display.style.format({'win_rate': '{:.1f}%'}),
                           use_container_width=True, hide_index=True)
        else:
            st.warning("Streak analysis is disabled.")
    
    # Tab 8: Heatmaps
    with tab8:
        st.header("🌡️ Historical Heatmaps")
        
        st.info("""
        **What This Shows:** A visual history of how each industry's scores have evolved over time.
        
        - **Colors:** Green = Bullish/High, Red = Bearish/Low, Yellow = Neutral
        - **Rows:** Each industry (sorted by average value)
        - **Columns:** Each trading day
        
        💡 *Heatmaps help you spot patterns that might not be obvious in tables:*
        - *Clusters of green = Sustained strength (trend)*
        - *Sudden color shifts = Potential reversals*
        - *Horizontal patterns = Industry-specific moves*
        - *Vertical patterns = Market-wide moves (all industries moving together)*
        
        🎯 *The Breadth Trends chart below shows how the most bullish and bearish industries 
        have tracked over time. Watch for divergences and crossovers!*
        """)
        
        heatmap_metric = st.selectbox("Select Metric", ['net_mean', 'breadth', 'bull_mean', 'bear_mean'], key='heatmap')
        heatmap_days = st.slider("Days to Display", 10, 90, 30)
        
        fig = create_industry_heatmap(industry_agg, heatmap_metric, heatmap_days)
        st.plotly_chart(fig, use_container_width=True)
        
        # Breadth over time
        st.subheader("Breadth Trends (Top/Bottom Industries)")
        fig_breadth = create_breadth_chart(industry_agg)
        st.plotly_chart(fig_breadth, use_container_width=True)
    
    # Tab 9: Data Export
    with tab9:
        st.header("📋 Data Export")
        
        st.info("""
        **What This Shows:** Download all the analysis data for your own use.
        
        **Exported sheets include:**
        - **Current Summary** = Today's snapshot of all industry metrics
        - **Historical Data** = Daily data for all industries over the lookback period
        - **Market Aggregates** = Daily market-wide averages
        - **Rotation Stages** = Current rotation quadrant for each industry
        - **Streaks** = Win/loss streak data for each industry
        
        💡 *Use this data to build your own analysis, create custom charts, 
        or integrate with other tools like Excel or Python.*
        """)
        
        if st.button("💾 Export All Analysis"):
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                current_data.to_excel(writer, sheet_name='Current_Summary', index=False)
                industry_agg.to_excel(writer, sheet_name='Historical_Data', index=False)
                market_agg.to_excel(writer, sheet_name='Market_Aggregates', index=False)
                if not rotation_data.empty:
                    rotation_data.to_excel(writer, sheet_name='Rotation_Stages', index=False)
                if not streaks_data.empty:
                    streaks_data.to_excel(writer, sheet_name='Streaks', index=False)
            
            output.seek(0)
            
            st.download_button(
                label="📥 Download Excel",
                data=output,
                file_name=f"industry_score_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Preview data
        st.subheader("Data Preview")
        preview_choice = st.selectbox("Select Dataset", 
                                      ['Current Summary', 'Historical Data', 'Market Aggregates'])
        
        if preview_choice == 'Current Summary':
            st.dataframe(current_data, use_container_width=True)
        elif preview_choice == 'Historical Data':
            st.dataframe(industry_agg.tail(100), use_container_width=True)
        else:
            st.dataframe(market_agg, use_container_width=True)


if __name__ == "__main__":
    main()