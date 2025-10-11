import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO
import glob
from collections import defaultdict

# Set page configuration to wide layout
st.set_page_config(layout="wide", page_title="Market Analysis Suite", page_icon="📊")
st.title("📊 Complete Market Valuation Analysis Suite")

# ------------------------------
# 1. Data Loading Functions
# ------------------------------
@st.cache_data
def load_sectors_file(file_path):
    """Load the sectors file from the data directory."""
    try:
        df = pd.read_excel(file_path)
        st.sidebar.success(f"✅ Loaded {len(df):,} sector mappings")
        return df
    except Exception as e:
        st.error(f"❌ Error loading stock_sectors.xlsx: {str(e)}")
        st.stop()

@st.cache_data
def load_market_caps_file(file_path):
    """Load and clean the market caps file."""
    try:
        # Load the file
        df = pd.read_excel(file_path)
        initial_rows = len(df)
        st.sidebar.info(f"📂 Loaded {initial_rows:,} initial records")
        
        # Validate required columns
        required_cols = ['symbol', 'market_cap', 'fetch_date']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.stop()
        
        # Clean and convert dates
        df['fetch_date'] = pd.to_datetime(df['fetch_date'], errors='coerce')
        
        # Remove rows with invalid dates or market caps
        df = df.dropna(subset=['fetch_date', 'market_cap'])
        df = df[df['market_cap'] > 0]  # Remove zero or negative market caps
        
        final_rows = len(df)
        if initial_rows != final_rows:
            st.sidebar.warning(f"🧹 Cleaned data: removed {initial_rows - final_rows:,} invalid rows")
        
        if len(df) == 0:
            st.error("❌ No valid data remaining after cleaning")
            st.stop()
        
        # Show data summary
        unique_dates = df['fetch_date'].dt.date.unique()
        unique_symbols = df['symbol'].nunique()
        date_range = f"{min(unique_dates)} to {max(unique_dates)}"
        
        st.sidebar.success(f"✅ Clean data: {final_rows:,} records")
        st.sidebar.info(f"📅 {len(unique_dates)} dates ({date_range})")
        st.sidebar.info(f"🏷️ {unique_symbols:,} unique symbols")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading market_caps.xlsx: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        st.stop()

def find_data_files():
    """Find the required data files in various possible locations."""
    possible_paths = [
        ".",
        "Data", 
        "data",
        "stock_scores",
        os.path.join("Data", "stock_scores"),
        os.path.join("data", "stock_scores"),
        "..",
        os.path.join("..", "Data")
    ]
    
    files_found = {}
    required_files = ["stock_sectors.xlsx", "market_caps.xlsx"]
    
    for filename in required_files:
        for path in possible_paths:
            test_path = os.path.join(path, filename) if path != "." else filename
            if os.path.exists(test_path):
                files_found[filename] = test_path
                break
    
    return files_found

# ------------------------------
# 2. Data Processing Functions
# ------------------------------
@st.cache_data
def calculate_breadth_indicators(market_caps_df, sectors_df, group_by, lookback_days):
    """Calculate breadth indicators: % stocks up, advance/decline ratio, participation."""
    
    # Merge market caps with sector data
    merged_df = pd.merge(
        market_caps_df,
        sectors_df[['symbol', 'sector', 'industry']],
        on='symbol',
        how='inner'
    )
    
    # Sort by symbol and date
    merged_df = merged_df.sort_values(['symbol', 'fetch_date'])
    
    # Calculate daily change for each stock
    merged_df['prev_market_cap'] = merged_df.groupby('symbol')['market_cap'].shift(1)
    merged_df['daily_change'] = merged_df['market_cap'] - merged_df['prev_market_cap']
    merged_df['direction'] = np.where(
        merged_df['daily_change'] > 0, 1,
        np.where(merged_df['daily_change'] < 0, -1, 0)
    )
    
    # Get unique dates
    unique_dates = sorted(merged_df['fetch_date'].unique())
    
    # Filter to lookback period
    if len(unique_dates) > lookback_days:
        start_date = unique_dates[-lookback_days]
        merged_df = merged_df[merged_df['fetch_date'] >= start_date]
    
    # Calculate breadth by group and date
    breadth_data = merged_df.groupby(['fetch_date', group_by]).agg({
        'symbol': 'count',  # Total stocks
        'direction': lambda x: (x == 1).sum()  # Stocks up
    }).reset_index()
    
    breadth_data.columns = ['fetch_date', group_by, 'total_stocks', 'stocks_up']
    breadth_data['stocks_down'] = breadth_data['total_stocks'] - breadth_data['stocks_up']
    breadth_data['pct_stocks_up'] = (breadth_data['stocks_up'] / breadth_data['total_stocks']) * 100
    breadth_data['advance_decline_ratio'] = breadth_data['stocks_up'] / breadth_data['stocks_down'].replace(0, 1)
    
    # Calculate breadth momentum (change in % stocks up)
    breadth_data = breadth_data.sort_values(['fetch_date', group_by])
    breadth_data['breadth_momentum'] = breadth_data.groupby(group_by)['pct_stocks_up'].diff()
    
    return breadth_data

@st.cache_data
def calculate_relative_strength(daily_data, group_by, lookback_days):
    """Calculate relative strength vs overall market."""
    
    # Sort by date
    daily_data = daily_data.sort_values('fetch_date')
    unique_dates = sorted(daily_data['fetch_date'].unique())
    
    # Filter to lookback period
    if len(unique_dates) > lookback_days:
        start_date = unique_dates[-lookback_days]
        daily_data = daily_data[daily_data['fetch_date'] >= start_date]
        unique_dates = sorted(daily_data['fetch_date'].unique())
    
    # Calculate overall market performance (sum of all industry market caps)
    market_totals = daily_data.groupby('fetch_date')['market_cap'].sum().reset_index()
    market_totals.columns = ['fetch_date', 'market_total']
    
    # Calculate market returns
    market_totals = market_totals.sort_values('fetch_date')
    market_totals['market_return'] = market_totals['market_total'].pct_change() * 100
    
    # Calculate industry returns
    industry_data = daily_data.sort_values(['fetch_date', group_by])
    industry_returns = []
    
    for group in daily_data[group_by].unique():
        group_data = daily_data[daily_data[group_by] == group].copy()
        group_data = group_data.sort_values('fetch_date')
        group_data['industry_return'] = group_data['market_cap'].pct_change() * 100
        industry_returns.append(group_data[['fetch_date', group_by, 'industry_return', 'market_cap']])
    
    industry_returns_df = pd.concat(industry_returns, ignore_index=True)
    
    # Merge with market returns
    relative_strength = pd.merge(
        industry_returns_df,
        market_totals[['fetch_date', 'market_return']],
        on='fetch_date',
        how='left'
    )
    
    # Calculate relative strength (industry return - market return)
    relative_strength['relative_return'] = relative_strength['industry_return'] - relative_strength['market_return']
    
    # Calculate cumulative relative strength
    relative_strength = relative_strength.sort_values(['fetch_date', group_by])
    relative_strength['cumulative_relative_strength'] = relative_strength.groupby(group_by)['relative_return'].cumsum()
    
    return relative_strength

@st.cache_data
def calculate_volatility_drawdown(daily_data, group_by, lookback_days, volatility_window=20):
    """Calculate volatility and drawdown metrics."""
    
    daily_data = daily_data.sort_values('fetch_date')
    unique_dates = sorted(daily_data['fetch_date'].unique())
    
    # Filter to lookback period
    if len(unique_dates) > lookback_days:
        start_date = unique_dates[-lookback_days]
        daily_data = daily_data[daily_data['fetch_date'] >= start_date]
    
    results = []
    
    for group in daily_data[group_by].unique():
        group_data = daily_data[daily_data[group_by] == group].copy()
        group_data = group_data.sort_values('fetch_date')
        
        # Calculate returns
        group_data['return'] = group_data['market_cap'].pct_change() * 100
        
        # Calculate rolling volatility (std dev of returns)
        group_data['volatility'] = group_data['return'].rolling(window=volatility_window, min_periods=1).std()
        
        # Calculate cumulative returns for drawdown calculation
        group_data['cumulative_return'] = (1 + group_data['return'] / 100).cumprod()
        
        # Calculate running maximum (peak)
        group_data['running_max'] = group_data['cumulative_return'].expanding().max()
        
        # Calculate drawdown
        group_data['drawdown'] = ((group_data['cumulative_return'] - group_data['running_max']) / group_data['running_max']) * 100
        
        # Calculate days underwater (days since peak)
        group_data['is_at_peak'] = group_data['cumulative_return'] == group_data['running_max']
        group_data['days_underwater'] = 0
        
        # Calculate consecutive days underwater
        days_count = 0
        for idx in range(len(group_data)):
            if group_data.iloc[idx]['is_at_peak']:
                days_count = 0
            else:
                days_count += 1
            group_data.iloc[idx, group_data.columns.get_loc('days_underwater')] = days_count
        
        group_data[group_by] = group
        results.append(group_data[['fetch_date', group_by, 'volatility', 'drawdown', 'days_underwater', 'return', 'market_cap']])
    
    volatility_drawdown_df = pd.concat(results, ignore_index=True)
    
    # Calculate max drawdown for each group
    max_drawdown = volatility_drawdown_df.groupby(group_by)['drawdown'].min().reset_index()
    max_drawdown.columns = [group_by, 'max_drawdown']
    
    # Get current metrics (latest date)
    latest_date = volatility_drawdown_df['fetch_date'].max()
    current_metrics = volatility_drawdown_df[volatility_drawdown_df['fetch_date'] == latest_date].copy()
    
    # Merge max drawdown
    current_metrics = pd.merge(current_metrics, max_drawdown, on=group_by, how='left')
    
    return volatility_drawdown_df, current_metrics

@st.cache_data
def calculate_momentum(daily_data, periods, group_by):
    """Calculate momentum (rate of change) for multiple periods."""
    
    if 'market_cap' not in daily_data.columns:
        return pd.DataFrame()
    
    # Sort by date
    daily_data = daily_data.sort_values('fetch_date')
    
    # Get unique dates and groups
    unique_dates = sorted(daily_data['fetch_date'].unique())
    all_groups = daily_data[group_by].unique()
    
    results = []
    
    for group in all_groups:
        group_data = daily_data[daily_data[group_by] == group].copy()
        group_data = group_data.sort_values('fetch_date')
        
        # Get the latest date data
        if group_data.empty:
            continue
            
        latest_row = group_data.iloc[-1]
        latest_date = latest_row['fetch_date']
        latest_market_cap = latest_row['market_cap']
        
        momentum_data = {
            group_by: group,
            'date': latest_date,
            'current_market_cap': latest_market_cap
        }
        
        # Calculate momentum for each period
        for period in periods:
            # Find the market cap N days ago (using trading days, not calendar days)
            if len(group_data) > period:
                past_row = group_data.iloc[-(period + 1)]
                past_market_cap = past_row['market_cap']
                
                if past_market_cap > 0:
                    # Rate of Change (ROC) as percentage
                    roc = ((latest_market_cap - past_market_cap) / past_market_cap) * 100
                    momentum_data[f'momentum_{period}d'] = roc
                    momentum_data[f'momentum_{period}d_abs'] = latest_market_cap - past_market_cap
                else:
                    momentum_data[f'momentum_{period}d'] = 0
                    momentum_data[f'momentum_{period}d_abs'] = 0
            else:
                momentum_data[f'momentum_{period}d'] = None
                momentum_data[f'momentum_{period}d_abs'] = None
        
        results.append(momentum_data)
    
    momentum_df = pd.DataFrame(results)
    
    # Calculate momentum strength (average of available periods)
    momentum_cols = [col for col in momentum_df.columns if col.startswith('momentum_') and col.endswith('d')]
    if momentum_cols:
        momentum_df['avg_momentum'] = momentum_df[momentum_cols].mean(axis=1, skipna=True)
        momentum_df['momentum_consistency'] = momentum_df[momentum_cols].std(axis=1, skipna=True)
    
    return momentum_df

@st.cache_data
def calculate_momentum_trends(daily_data, momentum_period, group_by, lookback_days):
    """Calculate momentum trends over time for visualization."""
    
    if 'market_cap' not in daily_data.columns:
        return pd.DataFrame()
    
    daily_data = daily_data.sort_values('fetch_date')
    unique_dates = sorted(daily_data['fetch_date'].unique())
    
    # Only use the most recent lookback_days
    if len(unique_dates) > lookback_days:
        start_date = unique_dates[-lookback_days]
        daily_data = daily_data[daily_data['fetch_date'] >= start_date]
        unique_dates = sorted(daily_data['fetch_date'].unique())
    
    all_groups = daily_data[group_by].unique()
    results = []
    
    for group in all_groups:
        group_data = daily_data[daily_data[group_by] == group].copy()
        group_data = group_data.sort_values('fetch_date')
        
        for i in range(len(group_data)):
            current_row = group_data.iloc[i]
            
            # Need enough historical data to calculate momentum
            if i >= momentum_period:
                past_row = group_data.iloc[i - momentum_period]
                current_cap = current_row['market_cap']
                past_cap = past_row['market_cap']
                
                if past_cap > 0:
                    momentum = ((current_cap - past_cap) / past_cap) * 100
                    
                    results.append({
                        'date': current_row['fetch_date'],
                        group_by: group,
                        'momentum': momentum,
                        'market_cap': current_cap
                    })
    
    return pd.DataFrame(results)

@st.cache_data
def calculate_mean_reversion(daily_data, group_by, lookback_days, zscore_window=20):
    """Calculate mean reversion signals: Z-scores, Bollinger Bands, overbought/oversold."""
    
    if 'market_cap' not in daily_data.columns:
        return pd.DataFrame(), pd.DataFrame()
    
    daily_data = daily_data.sort_values('fetch_date')
    unique_dates = sorted(daily_data['fetch_date'].unique())
    
    # Filter to lookback period
    if len(unique_dates) > lookback_days:
        start_date = unique_dates[-lookback_days]
        daily_data = daily_data[daily_data['fetch_date'] >= start_date]
    
    results = []
    
    for group in daily_data[group_by].unique():
        group_data = daily_data[daily_data[group_by] == group].copy()
        group_data = group_data.sort_values('fetch_date')
        
        # Calculate returns
        group_data['return'] = group_data['market_cap'].pct_change() * 100
        
        # Calculate rolling mean and std for Z-score
        group_data['rolling_mean'] = group_data['return'].rolling(window=zscore_window, min_periods=1).mean()
        group_data['rolling_std'] = group_data['return'].rolling(window=zscore_window, min_periods=1).std()
        
        # Calculate Z-score
        group_data['zscore'] = (group_data['return'] - group_data['rolling_mean']) / group_data['rolling_std'].replace(0, np.nan)
        
        # Calculate Bollinger Bands (on cumulative returns for better visualization)
        group_data['cumulative_return'] = (1 + group_data['return'] / 100).cumprod() - 1
        group_data['bb_middle'] = group_data['cumulative_return'].rolling(window=zscore_window, min_periods=1).mean()
        group_data['bb_std'] = group_data['cumulative_return'].rolling(window=zscore_window, min_periods=1).std()
        group_data['bb_upper'] = group_data['bb_middle'] + (2 * group_data['bb_std'])
        group_data['bb_lower'] = group_data['bb_middle'] - (2 * group_data['bb_std'])
        
        # Calculate Bollinger Band position (0-100 scale)
        band_width = group_data['bb_upper'] - group_data['bb_lower']
        group_data['bb_position'] = ((group_data['cumulative_return'] - group_data['bb_lower']) / band_width.replace(0, np.nan)) * 100
        
        # Classify signals
        group_data['signal'] = 'Neutral'
        group_data.loc[group_data['zscore'] > 2, 'signal'] = 'Overbought'
        group_data.loc[group_data['zscore'] < -2, 'signal'] = 'Oversold'
        group_data.loc[(group_data['zscore'] > 1) & (group_data['zscore'] <= 2), 'signal'] = 'Extended Up'
        group_data.loc[(group_data['zscore'] < -1) & (group_data['zscore'] >= -2), 'signal'] = 'Extended Down'
        
        group_data[group_by] = group
        results.append(group_data[[
            'fetch_date', group_by, 'return', 'zscore', 'cumulative_return',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'signal', 'market_cap'
        ]])
    
    all_data = pd.concat(results, ignore_index=True)
    
    # Get current signals (latest date)
    latest_date = all_data['fetch_date'].max()
    current_signals = all_data[all_data['fetch_date'] == latest_date].copy()
    
    return all_data, current_signals

@st.cache_data
def calculate_concentration(market_caps_df, sectors_df, group_by):
    """Calculate market cap concentration metrics within each group."""
    
    # Merge with sector data
    merged_df = pd.merge(
        market_caps_df,
        sectors_df[['symbol', 'sector', 'industry']],
        on='symbol',
        how='inner'
    )
    
    # Get latest date
    latest_date = merged_df['fetch_date'].max()
    latest_data = merged_df[merged_df['fetch_date'] == latest_date].copy()
    
    concentration_results = []
    
    for group in latest_data[group_by].unique():
        group_data = latest_data[latest_data[group_by] == group].copy()
        group_data = group_data.sort_values('market_cap', ascending=False)
        
        total_market_cap = group_data['market_cap'].sum()
        num_stocks = len(group_data)
        
        # Top 10% contribution
        top_10pct_count = max(1, int(num_stocks * 0.1))
        top_10pct_cap = group_data.head(top_10pct_count)['market_cap'].sum()
        top_10pct_contribution = (top_10pct_cap / total_market_cap * 100) if total_market_cap > 0 else 0
        
        # Top stock dominance
        top_stock_cap = group_data.iloc[0]['market_cap'] if len(group_data) > 0 else 0
        top_stock_pct = (top_stock_cap / total_market_cap * 100) if total_market_cap > 0 else 0
        top_stock_name = group_data.iloc[0]['symbol'] if len(group_data) > 0 else 'N/A'
        
        # Herfindahl Index (concentration measure)
        group_data['market_share'] = group_data['market_cap'] / total_market_cap
        herfindahl_index = (group_data['market_share'] ** 2).sum() * 10000  # Scaled to 0-10000
        
        # Small vs Large cap split (bottom 50% vs top 50% by market cap)
        median_cap = group_data['market_cap'].median()
        large_cap_data = group_data[group_data['market_cap'] >= median_cap]
        small_cap_data = group_data[group_data['market_cap'] < median_cap]
        
        large_cap_total = large_cap_data['market_cap'].sum()
        small_cap_total = small_cap_data['market_cap'].sum()
        large_cap_contribution = (large_cap_total / total_market_cap * 100) if total_market_cap > 0 else 0
        
        concentration_results.append({
            group_by: group,
            'total_stocks': num_stocks,
            'total_market_cap': total_market_cap,
            'top_10pct_contribution': top_10pct_contribution,
            'top_stock_name': top_stock_name,
            'top_stock_pct': top_stock_pct,
            'herfindahl_index': herfindahl_index,
            'large_cap_contribution': large_cap_contribution,
            'small_cap_contribution': 100 - large_cap_contribution,
            'concentration_level': 'High' if herfindahl_index > 1800 else 'Medium' if herfindahl_index > 1000 else 'Low'
        })
    
    return pd.DataFrame(concentration_results)

@st.cache_data
def calculate_rotation_stages(momentum_data, relative_strength_data, group_by):
    """Classify industries into rotation stages based on momentum and relative strength."""
    
    if momentum_data.empty or relative_strength_data.empty:
        return pd.DataFrame()
    
    # Get latest relative strength
    latest_rs_date = relative_strength_data['fetch_date'].max()
    latest_rs = relative_strength_data[relative_strength_data['fetch_date'] == latest_rs_date][[
        group_by, 'cumulative_relative_strength'
    ]].copy()
    
    # Merge momentum with relative strength
    rotation_data = pd.merge(
        momentum_data[[group_by, 'avg_momentum']],
        latest_rs,
        on=group_by,
        how='inner'
    )
    
    # Classify into quadrants
    median_momentum = rotation_data['avg_momentum'].median()
    median_rs = rotation_data['cumulative_relative_strength'].median()
    
    def classify_stage(row):
        momentum = row['avg_momentum']
        rs = row['cumulative_relative_strength']
        
        if momentum > median_momentum and rs > median_rs:
            return 'Leading (Growth)'
        elif momentum <= median_momentum and rs > median_rs:
            return 'Weakening (Mature)'
        elif momentum <= median_momentum and rs <= median_rs:
            return 'Lagging (Decline)'
        else:  # momentum > median_momentum and rs <= median_rs
            return 'Improving (Early)'
    
    rotation_data['stage'] = rotation_data.apply(classify_stage, axis=1)
    
    # Classify as defensive vs cyclical based on characteristics
    # Simple heuristic: negative momentum = defensive, positive = cyclical
    rotation_data['type'] = rotation_data['avg_momentum'].apply(
        lambda x: 'Cyclical' if x > 0 else 'Defensive'
    )
    
    return rotation_data

@st.cache_data
def calculate_streaks(daily_changes, group_by, lookback_days):
    """Calculate win/loss streaks and pattern metrics."""
    
    daily_changes = daily_changes.sort_values('fetch_date')
    unique_dates = sorted(daily_changes['fetch_date'].unique())
    
    # Filter to lookback period
    if len(unique_dates) > lookback_days:
        start_date = unique_dates[-lookback_days]
        daily_changes = daily_changes[daily_changes['fetch_date'] >= start_date]
    
    streak_results = []
    
    for group in daily_changes[group_by].unique():
        group_data = daily_changes[daily_changes[group_by] == group].copy()
        group_data = group_data.sort_values('fetch_date')
        
        # Calculate current streak
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        temp_streak = 0
        prev_direction = 0
        
        up_days = 0
        down_days = 0
        
        for direction in group_data['direction']:
            if direction == 1:
                up_days += 1
            elif direction == -1:
                down_days += 1
            
            if direction == 0:
                temp_streak = 0
            elif direction == prev_direction and prev_direction != 0:
                temp_streak += 1
            else:
                temp_streak = 1
            
            if direction != 0:
                current_streak = temp_streak if direction == group_data['direction'].iloc[-1] else 0
                
                if direction == 1:
                    max_win_streak = max(max_win_streak, temp_streak)
                elif direction == -1:
                    max_loss_streak = max(max_loss_streak, temp_streak)
                
                prev_direction = direction
        
        # Get current direction
        current_direction = group_data['direction'].iloc[-1]
        if current_direction == 1:
            current_streak_type = 'Win'
        elif current_direction == -1:
            current_streak_type = 'Loss'
            current_streak = -current_streak
        else:
            current_streak_type = 'Neutral'
            current_streak = 0
        
        # Win rate
        total_days = up_days + down_days
        win_rate = (up_days / total_days * 100) if total_days > 0 else 0
        
        streak_results.append({
            group_by: group,
            'current_streak': current_streak,
            'current_streak_type': current_streak_type,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'up_days': up_days,
            'down_days': down_days,
            'win_rate': win_rate,
            'total_days': total_days
        })
    
    return pd.DataFrame(streak_results)

@st.cache_data
def calculate_daily_changes(market_caps_df, sectors_df, group_by):
    """Calculate daily market cap changes by sector/industry."""
    
    # Filter market caps to only include symbols we have sector data for
    all_symbols = set(sectors_df['symbol'].unique())
    market_caps_filtered = market_caps_df[market_caps_df['symbol'].isin(all_symbols)].copy()
    
    # Merge with sector data
    merged_df = pd.merge(
        market_caps_filtered, 
        sectors_df[['symbol', 'sector', 'industry']], 
        on='symbol', 
        how='inner'
    )
    
    # Ensure market cap is numeric
    merged_df['market_cap'] = pd.to_numeric(merged_df['market_cap'], errors='coerce')
    merged_df = merged_df.dropna(subset=['market_cap'])
    
    # Sort by symbol and date for proper change calculation
    merged_df = merged_df.sort_values(['symbol', 'fetch_date'])
    
    # Calculate daily changes per symbol
    merged_df['prev_market_cap'] = merged_df.groupby('symbol')['market_cap'].shift(1)
    merged_df['daily_change'] = merged_df['market_cap'] - merged_df['prev_market_cap']
    
    # Aggregate by date and group
    daily_aggregated = merged_df.groupby(['fetch_date', group_by]).agg({
        'daily_change': 'sum',
        'market_cap': 'sum'
    }).reset_index()
    
    # Calculate direction (1 for up, -1 for down, 0 for no change)
    daily_aggregated['direction'] = np.where(
        daily_aggregated['daily_change'] > 0, 1,
        np.where(daily_aggregated['daily_change'] < 0, -1, 0)
    )
    
    # Set first day changes to 0 (no previous day to compare)
    first_date = daily_aggregated['fetch_date'].min()
    first_day_mask = daily_aggregated['fetch_date'] == first_date
    daily_aggregated.loc[first_day_mask, ['daily_change', 'direction']] = 0
    
    return daily_aggregated

@st.cache_data
def calculate_trend_scores(daily_changes, score_start_date, group_by):
    """Calculate cumulative trend scores starting from a specific date."""
    
    # Sort by date
    daily_changes = daily_changes.sort_values('fetch_date')
    
    # Get all unique dates and groups
    all_dates = sorted(daily_changes['fetch_date'].unique())
    all_groups = daily_changes[group_by].unique()
    
    results = []
    
    # Process each group separately
    for group in all_groups:
        group_data = daily_changes[daily_changes[group_by] == group].copy()
        
        # Create complete date series for this group
        date_series = pd.DataFrame({'fetch_date': all_dates})
        group_complete = pd.merge(date_series, group_data, on='fetch_date', how='left')
        group_complete[group_by] = group
        group_complete = group_complete.fillna({
            'direction': 0, 
            'daily_change': 0, 
            'market_cap': 0
        })
        
        # Initialize scoring variables
        cumulative_score = 0
        current_streak = 0
        prev_direction = 0
        
        # Process each date
        for _, row in group_complete.iterrows():
            current_date = row['fetch_date'].date()
            direction = row['direction']
            
            # Reset score at start date
            if current_date == score_start_date:
                cumulative_score = 0
                current_streak = 0
                prev_direction = 0
            
            # Only process dates from start date forward
            if current_date >= score_start_date:
                # Update score
                cumulative_score += direction
                
                # Update streak
                if direction == 0:
                    # No change in streak for flat days
                    pass
                elif direction == prev_direction and prev_direction != 0:
                    # Continuing streak
                    current_streak += direction
                else:
                    # New streak starts
                    current_streak = direction
                
                if direction != 0:
                    prev_direction = direction
                
                # Store result
                results.append({
                    'date': row['fetch_date'],
                    group_by: group,
                    'score': cumulative_score,
                    'streak': current_streak,
                    'direction': direction,
                    'market_cap': row['market_cap'],
                    'daily_change': row['daily_change']
                })
    
    return pd.DataFrame(results)

@st.cache_data
def calculate_percent_changes(daily_data, lookback_days, group_by):
    """Calculate percent changes for current and previous periods for each group."""
    
    if 'market_cap' not in daily_data.columns:
        return pd.DataFrame()
    
    # Sort by date
    daily_data = daily_data.sort_values('fetch_date')
    
    # Get unique dates sorted (these are actual trading days only)
    unique_dates = sorted(daily_data['fetch_date'].unique())
    all_groups = daily_data[group_by].unique()
    
    # We need at least 2 * lookback_days dates for both current and previous periods
    min_required_dates = 2 * lookback_days
    if len(unique_dates) < min_required_dates:
        # Adjust lookback if we don't have enough data
        max_possible_lookback = len(unique_dates) // 2
        if max_possible_lookback < 1:
            st.warning("Not enough trading days for percent change calculation with previous period")
            return pd.DataFrame()
        st.warning(f"Adjusted lookback period from {lookback_days} to {max_possible_lookback} days due to data availability")
        lookback_days = max_possible_lookback
    
    # Calculate the date ranges using actual trading days (not calendar days)
    latest_date = unique_dates[-1]
    current_start_idx = -lookback_days if lookback_days <= len(unique_dates) else 0
    current_start_date = unique_dates[current_start_idx]
    previous_end_idx = current_start_idx - 1 if current_start_idx - 1 >= -len(unique_dates) else 0
    previous_end_date = unique_dates[previous_end_idx]
    previous_start_idx = previous_end_idx - lookback_days + 1 if previous_end_idx - lookback_days + 1 >= -len(unique_dates) else 0
    previous_start_date = unique_dates[previous_start_idx]
    
    # Filter data for the relevant dates
    latest_data = daily_data[daily_data['fetch_date'] == latest_date]
    current_start_data = daily_data[daily_data['fetch_date'] == current_start_date]
    previous_end_data = daily_data[daily_data['fetch_date'] == previous_end_date]
    previous_start_data = daily_data[daily_data['fetch_date'] == previous_start_date]
    
    # Prepare the result dataframe
    percent_changes = []
    
    # Calculate percent change for ALL groups
    unique_groups = daily_data[group_by].unique()
    
    for group in unique_groups:
        # Current period data
        latest_group = latest_data[latest_data[group_by] == group]
        current_start_group = current_start_data[current_start_data[group_by] == group]
        
        # Previous period data
        previous_end_group = previous_end_data[previous_end_data[group_by] == group]
        previous_start_group = previous_start_data[previous_start_data[group_by] == group]
        
        current_pct_change = 0
        previous_pct_change = 0
        
        # Calculate current period percent change
        if not latest_group.empty and not current_start_group.empty:
            latest_value = latest_group['market_cap'].iloc[0]
            current_start_value = current_start_group['market_cap'].iloc[0]
            
            if current_start_value > 0:
                current_pct_change = ((latest_value - current_start_value) / current_start_value) * 100
        
        # Calculate previous period percent change
        if not previous_end_group.empty and not previous_start_group.empty:
            previous_end_value = previous_end_group['market_cap'].iloc[0]
            previous_start_value = previous_start_group['market_cap'].iloc[0]
            
            if previous_start_value > 0:
                previous_pct_change = ((previous_end_value - previous_start_value) / previous_start_value) * 100
        
        percent_changes.append({
            group_by: group,
            'current_percent_change': current_pct_change,
            'previous_percent_change': previous_pct_change,
            'latest_date': latest_date,
            'current_start_date': current_start_date,
            'previous_end_date': previous_end_date,
            'previous_start_date': previous_start_date
        })
    
    return pd.DataFrame(percent_changes)

# ------------------------------
# 3. Visualization Functions
# ------------------------------
def create_breadth_chart(breadth_data, group_by):
    """Create line chart showing breadth (% stocks up) over time."""
    
    if breadth_data.empty:
        return None
    
    unique_groups = sorted(breadth_data[group_by].unique())
    num_groups = len(unique_groups)
    
    fig = go.Figure()
    
    # If many groups (>20), show aggregate statistics instead of individual lines
    if num_groups > 20:
        # Calculate aggregate statistics by date
        agg_stats = breadth_data.groupby('fetch_date')['pct_stocks_up'].agg([
            ('mean', 'mean'),
            ('median', 'median'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('min', 'min'),
            ('max', 'max')
        ]).reset_index()
        
        # Add max/min range (very light)
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['max'],
            mode='lines',
            name='Max',
            line=dict(color='rgba(150, 150, 150, 0.3)', width=1),
            showlegend=True,
            hovertemplate='Max: %{y:.1f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['min'],
            mode='lines',
            name='Min',
            line=dict(color='rgba(150, 150, 150, 0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(150, 150, 150, 0.1)',
            showlegend=True,
            hovertemplate='Min: %{y:.1f}%<extra></extra>'
        ))
        
        # Add quartile range (shaded)
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['q75'],
            mode='lines',
            name='75th Percentile',
            line=dict(color='rgba(99, 110, 250, 0.4)', width=1),
            showlegend=True,
            hovertemplate='75th: %{y:.1f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['q25'],
            mode='lines',
            name='25th Percentile',
            line=dict(color='rgba(99, 110, 250, 0.4)', width=1),
            fill='tonexty',
            fillcolor='rgba(99, 110, 250, 0.2)',
            showlegend=True,
            hovertemplate='25th: %{y:.1f}%<extra></extra>'
        ))
        
        # Add median line (prominent)
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['median'],
            mode='lines',
            name='Median',
            line=dict(color='rgb(255, 127, 14)', width=3),
            showlegend=True,
            hovertemplate='Median: %{y:.1f}%<extra></extra>'
        ))
        
        # Add mean line (prominent)
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['mean'],
            mode='lines',
            name='Average',
            line=dict(color='rgb(99, 110, 250)', width=3),
            showlegend=True,
            hovertemplate='Average: %{y:.1f}%<extra></extra>'
        ))
        
        title_text = f"Market Breadth: Aggregate Statistics Across {num_groups} {group_by.title()}s"
        
    else:
        # For few groups, show individual lines
        colors = px.colors.qualitative.Plotly
        color_dict = {group: colors[i % len(colors)] for i, group in enumerate(unique_groups)}
        
        for group in unique_groups:
            group_data = breadth_data[breadth_data[group_by] == group]
            
            fig.add_trace(go.Scatter(
                x=group_data['fetch_date'],
                y=group_data['pct_stocks_up'],
                mode='lines',
                name=group,
                line=dict(color=color_dict[group], width=2),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Stocks Up: %{y:.1f}%<extra></extra>'
            ))
        
        title_text = f"Market Breadth: % of Stocks Up by {group_by.title()}"
    
    # Add 50% reference line
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% Line")
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="% of Stocks Up",
        height=700,
        hovermode='x unified',
        showlegend=True if num_groups > 20 else False,
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray', range=[0, 100]),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="black",
            borderwidth=2,
            font=dict(size=12, color="black")
        )
    )
    
    return fig

def create_zscore_chart(mean_reversion_data, group_by):
    """Create chart showing Z-scores over time."""
    
    if mean_reversion_data.empty:
        return None
    
    unique_groups = sorted(mean_reversion_data[group_by].unique())
    num_groups = len(unique_groups)
    
    fig = go.Figure()
    
    if num_groups > 20:
        # Aggregate statistics
        agg_stats = mean_reversion_data.groupby('fetch_date')['zscore'].agg([
            ('mean', 'mean'),
            ('median', 'median'),
            ('q75', lambda x: x.quantile(0.75)),
            ('q25', lambda x: x.quantile(0.25))
        ]).reset_index()
        
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'], y=agg_stats['q75'], mode='lines',
            name='75th Percentile', line=dict(color='rgba(99, 110, 250, 0.4)', width=1),
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'], y=agg_stats['q25'], mode='lines',
            name='25th Percentile', line=dict(color='rgba(99, 110, 250, 0.4)', width=1),
            fill='tonexty', fillcolor='rgba(99, 110, 250, 0.2)', showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'], y=agg_stats['median'], mode='lines',
            name='Median Z-Score', line=dict(color='rgb(99, 110, 250)', width=3), showlegend=True
        ))
    else:
        colors = px.colors.qualitative.Plotly
        for i, group in enumerate(unique_groups):
            group_data = mean_reversion_data[mean_reversion_data[group_by] == group]
            fig.add_trace(go.Scatter(
                x=group_data['fetch_date'], y=group_data['zscore'],
                mode='lines', name=group, line=dict(color=colors[i % len(colors)], width=2)
            ))
    
    # Add reference lines
    fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="Overbought (+2σ)")
    fig.add_hline(y=-2, line_dash="dash", line_color="green", annotation_text="Oversold (-2σ)")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    
    fig.update_layout(
        title="Z-Score Analysis (Mean Reversion Signals)",
        xaxis_title="Date", yaxis_title="Z-Score (Standard Deviations)",
        height=600, hovermode='x unified',
        showlegend=num_groups > 20,
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="black", borderwidth=2,
            font=dict(size=12, color="black")
        ) if num_groups > 20 else None
    )
    
    return fig

def create_concentration_chart(concentration_data, group_by):
    """Create chart showing market cap concentration."""
    
    if concentration_data.empty:
        return None
    
    # Sort by concentration
    sorted_data = concentration_data.sort_values('herfindahl_index', ascending=True)
    
    # Color by concentration level
    color_map = {'High': 'rgb(239, 85, 59)', 'Medium': 'rgb(255, 193, 7)', 'Low': 'rgb(0, 204, 150)'}
    colors = [color_map[level] for level in sorted_data['concentration_level']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=sorted_data[group_by],
        x=sorted_data['top_10pct_contribution'],
        orientation='h',
        name='Top 10% Contribution',
        marker=dict(color=colors, line=dict(width=1, color='rgb(50, 50, 50)')),
        text=sorted_data['top_10pct_contribution'].apply(lambda x: f"{x:.1f}%"),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Top 10% Contribution: %{x:.1f}%<br>Concentration: %{customdata}<extra></extra>',
        customdata=sorted_data['concentration_level']
    ))
    
    fig.update_layout(
        title=f"Market Cap Concentration by {group_by.title()}",
        xaxis_title="% of Market Cap from Top 10% of Stocks",
        yaxis_title=group_by.title(),
        height=max(600, len(sorted_data) * 20),
        margin=dict(l=200, r=100)
    )
    
    return fig

def create_rotation_matrix(rotation_data, group_by):
    """Create rotation stage matrix visualization."""
    
    if rotation_data.empty:
        return None
    
    fig = go.Figure()
    
    # Define stage colors
    stage_colors = {
        'Leading (Growth)': 'rgb(0, 204, 150)',
        'Weakening (Mature)': 'rgb(255, 193, 7)',
        'Lagging (Decline)': 'rgb(239, 85, 59)',
        'Improving (Early)': 'rgb(99, 110, 250)'
    }
    
    # Create scatter plot
    for stage in rotation_data['stage'].unique():
        stage_data = rotation_data[rotation_data['stage'] == stage]
        
        fig.add_trace(go.Scatter(
            x=stage_data['cumulative_relative_strength'],
            y=stage_data['avg_momentum'],
            mode='markers+text',
            name=stage,
            marker=dict(
                size=15,
                color=stage_colors.get(stage, 'gray'),
                line=dict(width=2, color='white')
            ),
            text=stage_data[group_by] if len(rotation_data) <= 30 else '',
            textposition='top center',
            textfont=dict(size=9),
            hovertemplate='<b>%{customdata}</b><br>Stage: ' + stage + '<br>Rel Strength: %{x:.2f}%<br>Momentum: %{y:.2f}%<extra></extra>',
            customdata=stage_data[group_by]
        ))
    
    # Add quadrant lines
    median_momentum = rotation_data['avg_momentum'].median()
    median_rs = rotation_data['cumulative_relative_strength'].median()
    
    fig.add_hline(y=median_momentum, line_dash="dash", line_color="gray")
    fig.add_vline(x=median_rs, line_dash="dash", line_color="gray")
    
    # Add quadrant labels
    fig.add_annotation(x=rotation_data['cumulative_relative_strength'].max() * 0.8, 
                      y=rotation_data['avg_momentum'].max() * 0.9,
                      text="<b>LEADING</b><br>(Growth)", showarrow=False, 
                      font=dict(size=14, color="green"))
    fig.add_annotation(x=rotation_data['cumulative_relative_strength'].max() * 0.8,
                      y=rotation_data['avg_momentum'].min() * 0.9,
                      text="<b>WEAKENING</b><br>(Mature)", showarrow=False,
                      font=dict(size=14, color="orange"))
    fig.add_annotation(x=rotation_data['cumulative_relative_strength'].min() * 0.8,
                      y=rotation_data['avg_momentum'].min() * 0.9,
                      text="<b>LAGGING</b><br>(Decline)", showarrow=False,
                      font=dict(size=14, color="red"))
    fig.add_annotation(x=rotation_data['cumulative_relative_strength'].min() * 0.8,
                      y=rotation_data['avg_momentum'].max() * 0.9,
                      text="<b>IMPROVING</b><br>(Early)", showarrow=False,
                      font=dict(size=14, color="blue"))
    
    fig.update_layout(
        title="Sector Rotation Matrix",
        xaxis_title="Relative Strength vs Market (%)",
        yaxis_title="Average Momentum (%)",
        height=700,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="black",
            borderwidth=2,
            font=dict(size=11, color="black")
        )
    )
    
    return fig

def create_streaks_chart(streaks_data, group_by):
    """Create chart showing current streaks."""
    
    if streaks_data.empty:
        return None
    
    # Sort by current streak
    sorted_data = streaks_data.sort_values('current_streak', ascending=True)
    
    # Color by streak type
    colors = ['rgb(0, 204, 150)' if x > 0 else 'rgb(239, 85, 59)' if x < 0 else 'gray' 
              for x in sorted_data['current_streak']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=sorted_data[group_by],
        x=sorted_data['current_streak'],
        orientation='h',
        marker=dict(color=colors, line=dict(width=1, color='rgb(50, 50, 50)')),
        text=sorted_data.apply(lambda row: f"{abs(row['current_streak'])} {row['current_streak_type']}", axis=1),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Current Streak: %{x}<br>Win Rate: %{customdata:.1f}%<extra></extra>',
        customdata=sorted_data['win_rate']
    ))
    
    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)
    
    fig.update_layout(
        title=f"Current Streaks by {group_by.title()}",
        xaxis_title="Current Streak (Consecutive Days)",
        yaxis_title=group_by.title(),
        height=max(600, len(sorted_data) * 20),
        margin=dict(l=200, r=100)
    )
    
    return fig

def create_breadth_distribution(breadth_data, group_by):
    """Create histogram showing distribution of breadth across groups over time."""
    
    if breadth_data.empty:
        return None
    
    # Get latest date
    latest_date = breadth_data['fetch_date'].max()
    latest_breadth = breadth_data[breadth_data['fetch_date'] == latest_date]
    
    # Create histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=latest_breadth['pct_stocks_up'],
        nbinsx=20,
        name='Distribution',
        marker=dict(
            color='rgb(99, 110, 250)',
            line=dict(color='rgb(255, 255, 255)', width=1)
        ),
        hovertemplate='Breadth Range: %{x}%<br>Count: %{y}<extra></extra>'
    ))
    
    # Add vertical lines for reference
    mean_breadth = latest_breadth['pct_stocks_up'].mean()
    median_breadth = latest_breadth['pct_stocks_up'].median()
    
    fig.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="50%")
    fig.add_vline(x=mean_breadth, line_dash="dot", line_color="blue", 
                  annotation_text=f"Avg: {mean_breadth:.1f}%")
    
    fig.update_layout(
        title=f"Breadth Distribution Across {group_by.title()}s (Latest: {latest_date.strftime('%Y-%m-%d')})",
        xaxis_title="% of Stocks Up",
        yaxis_title=f"Number of {group_by.title()}s",
        height=500,
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='lightgray', range=[0, 100]),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    
    return fig

def create_breadth_top_bottom(breadth_data, group_by, n=10):
    """Create bar chart showing top and bottom N groups by breadth."""
    
    if breadth_data.empty:
        return None
    
    # Get latest date
    latest_date = breadth_data['fetch_date'].max()
    latest_breadth = breadth_data[breadth_data['fetch_date'] == latest_date].copy()
    
    # Sort and get top/bottom N
    latest_breadth = latest_breadth.sort_values('pct_stocks_up', ascending=False)
    
    if len(latest_breadth) > n * 2:
        # Get top N and bottom N
        top_n = latest_breadth.head(n)
        bottom_n = latest_breadth.tail(n)
        display_data = pd.concat([top_n, bottom_n])
    else:
        display_data = latest_breadth
    
    # Sort for display
    display_data = display_data.sort_values('pct_stocks_up', ascending=True)
    
    # Color based on value
    colors = ['rgb(239, 85, 59)' if x < 40 else 'rgb(255, 193, 7)' if x < 60 else 'rgb(0, 204, 150)' 
              for x in display_data['pct_stocks_up']]
    
    fig = go.Figure(go.Bar(
        y=display_data[group_by],
        x=display_data['pct_stocks_up'],
        orientation='h',
        marker=dict(color=colors, line=dict(width=1, color='rgb(50, 50, 50)')),
        text=display_data['pct_stocks_up'].apply(lambda x: f"{x:.1f}%"),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Breadth: %{x:.1f}%<br>Stocks Up: %{customdata[0]:.0f}/%{customdata[1]:.0f}<extra></extra>',
        customdata=display_data[['stocks_up', 'total_stocks']].values
    ))
    
    # Add reference lines
    fig.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="50%")
    fig.add_vline(x=60, line_dash="dot", line_color="green", opacity=0.3)
    fig.add_vline(x=40, line_dash="dot", line_color="red", opacity=0.3)
    
    fig.update_layout(
        title=f"Breadth Leaders & Laggards by {group_by.title()} (Top & Bottom {n})",
        xaxis_title="% of Stocks Up",
        yaxis_title=group_by.title(),
        height=max(600, len(display_data) * 25),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            range=[0, 100]
        ),
        margin=dict(l=200, r=100)
    )
    
    return fig

def create_breadth_heatmap(breadth_data, group_by):
    """Create heatmap of breadth over time."""
    
    if breadth_data.empty:
        return None
    
    # Create pivot table
    breadth_data['date_str'] = breadth_data['fetch_date'].dt.strftime('%Y-%m-%d')
    pivot_data = breadth_data.pivot(
        index='date_str',
        columns=group_by,
        values='pct_stocks_up'
    )
    
    # Sort dates (newest first)
    dates = sorted(pivot_data.index, reverse=True)
    pivot_data = pivot_data.loc[dates]
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlGn',
        zmid=50,
        zmin=0,
        zmax=100,
        hovertemplate='<b>%{y}</b><br>%{x}<br>% Up: %{z:.1f}%<extra></extra>',
        colorbar=dict(title="% Stocks Up")
    ))
    
    fig.update_layout(
        title=f"Breadth Heatmap by {group_by.title()}",
        xaxis_title=group_by.title(),
        yaxis_title="Date",
        height=max(400, len(dates) * 20),
        xaxis=dict(
            side='top',
            tickangle=45 if len(pivot_data.columns) > 15 else 0
        ),
        yaxis=dict(type='category')
    )
    
    return fig

def create_relative_strength_chart(relative_strength_data, group_by):
    """Create line chart showing cumulative relative strength."""
    
    if relative_strength_data.empty:
        return None
    
    unique_groups = sorted(relative_strength_data[group_by].unique())
    num_groups = len(unique_groups)
    
    fig = go.Figure()
    
    # If many groups (>20), show aggregate statistics instead of individual lines
    if num_groups > 20:
        # Calculate aggregate statistics by date
        agg_stats = relative_strength_data.groupby('fetch_date')['cumulative_relative_strength'].agg([
            ('mean', 'mean'),
            ('median', 'median'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('min', 'min'),
            ('max', 'max')
        ]).reset_index()
        
        # Add max/min range (very light)
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['max'],
            mode='lines',
            name='Max',
            line=dict(color='rgba(0, 204, 150, 0.3)', width=1),
            showlegend=True,
            hovertemplate='Max: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['min'],
            mode='lines',
            name='Min',
            line=dict(color='rgba(239, 85, 59, 0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(150, 150, 150, 0.1)',
            showlegend=True,
            hovertemplate='Min: %{y:.2f}%<extra></extra>'
        ))
        
        # Add quartile range (shaded)
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['q75'],
            mode='lines',
            name='75th Percentile',
            line=dict(color='rgba(99, 110, 250, 0.4)', width=1),
            showlegend=True,
            hovertemplate='75th: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['q25'],
            mode='lines',
            name='25th Percentile',
            line=dict(color='rgba(99, 110, 250, 0.4)', width=1),
            fill='tonexty',
            fillcolor='rgba(99, 110, 250, 0.2)',
            showlegend=True,
            hovertemplate='25th: %{y:.2f}%<extra></extra>'
        ))
        
        # Add median line (prominent)
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['median'],
            mode='lines',
            name='Median',
            line=dict(color='rgb(255, 127, 14)', width=3),
            showlegend=True,
            hovertemplate='Median: %{y:.2f}%<extra></extra>'
        ))
        
        # Add mean line (prominent)
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['mean'],
            mode='lines',
            name='Average',
            line=dict(color='rgb(99, 110, 250)', width=3),
            showlegend=True,
            hovertemplate='Average: %{y:.2f}%<extra></extra>'
        ))
        
        title_text = f"Relative Strength: Aggregate Statistics Across {num_groups} {group_by.title()}s"
        
    else:
        # For few groups, show individual lines
        colors = px.colors.qualitative.Plotly
        color_dict = {group: colors[i % len(colors)] for i, group in enumerate(unique_groups)}
        
        for group in unique_groups:
            group_data = relative_strength_data[relative_strength_data[group_by] == group]
            
            fig.add_trace(go.Scatter(
                x=group_data['fetch_date'],
                y=group_data['cumulative_relative_strength'],
                mode='lines',
                name=group,
                line=dict(color=color_dict[group], width=2),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Rel Strength: %{y:.2f}%<extra></extra>'
            ))
        
        title_text = f"Relative Strength vs Market by {group_by.title()}"
    
    # Add zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Market Baseline")
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Cumulative Relative Return (%)",
        height=700,
        hovermode='x unified',
        showlegend=True if num_groups > 20 else False,
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    return fig

def create_relative_strength_distribution(relative_strength_data, group_by):
    """Create histogram showing distribution of relative strength."""
    
    if relative_strength_data.empty:
        return None
    
    # Get latest date
    latest_date = relative_strength_data['fetch_date'].max()
    latest_rs = relative_strength_data[relative_strength_data['fetch_date'] == latest_date]
    
    # Create histogram
    fig = go.Figure()
    
    # Separate positive and negative for coloring
    positive_rs = latest_rs[latest_rs['cumulative_relative_strength'] >= 0]
    negative_rs = latest_rs[latest_rs['cumulative_relative_strength'] < 0]
    
    fig.add_trace(go.Histogram(
        x=positive_rs['cumulative_relative_strength'],
        nbinsx=15,
        name='Outperformers',
        marker=dict(
            color='rgb(0, 204, 150)',
            line=dict(color='rgb(255, 255, 255)', width=1)
        ),
        hovertemplate='RS Range: %{x:.1f}%<br>Count: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Histogram(
        x=negative_rs['cumulative_relative_strength'],
        nbinsx=15,
        name='Underperformers',
        marker=dict(
            color='rgb(239, 85, 59)',
            line=dict(color='rgb(255, 255, 255)', width=1)
        ),
        hovertemplate='RS Range: %{x:.1f}%<br>Count: %{y}<extra></extra>'
    ))
    
    # Add vertical lines for reference
    mean_rs = latest_rs['cumulative_relative_strength'].mean()
    
    fig.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Market")
    fig.add_vline(x=mean_rs, line_dash="dot", line_color="blue", 
                  annotation_text=f"Avg: {mean_rs:.1f}%")
    
    fig.update_layout(
        title=f"Relative Strength Distribution (Latest: {latest_date.strftime('%Y-%m-%d')})",
        xaxis_title="Cumulative Relative Return (%)",
        yaxis_title=f"Number of {group_by.title()}s",
        height=500,
        barmode='overlay',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    
    return fig

def create_relative_strength_top_bottom(relative_strength_data, group_by, n=15):
    """Create bar chart showing top and bottom N groups by relative strength."""
    
    if relative_strength_data.empty:
        return None
    
    # Get latest date
    latest_date = relative_strength_data['fetch_date'].max()
    latest_rs = relative_strength_data[relative_strength_data['fetch_date'] == latest_date].copy()
    
    # Sort and get top/bottom N
    latest_rs = latest_rs.sort_values('cumulative_relative_strength', ascending=False)
    
    if len(latest_rs) > n * 2:
        top_n = latest_rs.head(n)
        bottom_n = latest_rs.tail(n)
        display_data = pd.concat([top_n, bottom_n])
    else:
        display_data = latest_rs
    
    # Sort for display
    display_data = display_data.sort_values('cumulative_relative_strength', ascending=True)
    
    # Color based on value
    colors = ['rgb(0, 204, 150)' if x > 0 else 'rgb(239, 85, 59)' 
              for x in display_data['cumulative_relative_strength']]
    
    fig = go.Figure(go.Bar(
        y=display_data[group_by],
        x=display_data['cumulative_relative_strength'],
        orientation='h',
        marker=dict(color=colors, line=dict(width=1, color='rgb(50, 50, 50)')),
        text=display_data['cumulative_relative_strength'].apply(lambda x: f"{x:+.1f}%"),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Rel Strength: %{x:.2f}%<extra></extra>'
    ))
    
    # Add reference line
    fig.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Market")
    
    fig.update_layout(
        title=f"Relative Strength Leaders & Laggards (Top & Bottom {n})",
        xaxis_title="Cumulative Relative Return (%)",
        yaxis_title=group_by.title(),
        height=max(600, len(display_data) * 25),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2
        ),
        margin=dict(l=200, r=100)
    )
    
    return fig

def create_relative_strength_ranking(relative_strength_data, group_by):
    """Create bar chart of current relative strength rankings."""
    
    if relative_strength_data.empty:
        return None
    
    # Get latest date
    latest_date = relative_strength_data['fetch_date'].max()
    latest_data = relative_strength_data[relative_strength_data['fetch_date'] == latest_date].copy()
    
    # Sort by relative strength
    latest_data = latest_data.sort_values('cumulative_relative_strength', ascending=True)
    
    # Create color based on positive/negative
    colors = ['rgb(0, 204, 150)' if x > 0 else 'rgb(239, 85, 59)' 
              for x in latest_data['cumulative_relative_strength']]
    
    fig = go.Figure(go.Bar(
        y=latest_data[group_by],
        x=latest_data['cumulative_relative_strength'],
        orientation='h',
        marker=dict(color=colors, line=dict(width=1, color='rgb(50, 50, 50)')),
        text=latest_data['cumulative_relative_strength'].apply(lambda x: f"{x:.1f}%"),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Relative Strength: %{x:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Relative Strength Rankings by {group_by.title()} (vs Market)",
        xaxis_title="Cumulative Relative Return (%)",
        yaxis_title=group_by.title(),
        height=max(600, len(latest_data) * 20),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2
        ),
        margin=dict(l=200, r=100)
    )
    
    return fig

def create_volatility_chart(volatility_data, group_by):
    """Create line chart showing volatility over time."""
    
    if volatility_data.empty:
        return None
    
    unique_groups = sorted(volatility_data[group_by].unique())
    num_groups = len(unique_groups)
    
    fig = go.Figure()
    
    # If many groups (>20), show aggregate statistics
    if num_groups > 20:
        # Calculate aggregate statistics by date
        agg_stats = volatility_data.groupby('fetch_date')['volatility'].agg([
            ('mean', 'mean'),
            ('median', 'median'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('min', 'min'),
            ('max', 'max')
        ]).reset_index()
        
        # Add max/min range
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['max'],
            mode='lines',
            name='Max',
            line=dict(color='rgba(239, 85, 59, 0.3)', width=1),
            showlegend=True,
            hovertemplate='Max: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['min'],
            mode='lines',
            name='Min',
            line=dict(color='rgba(239, 85, 59, 0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(239, 85, 59, 0.1)',
            showlegend=True,
            hovertemplate='Min: %{y:.2f}%<extra></extra>'
        ))
        
        # Add quartile range
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['q75'],
            mode='lines',
            name='75th Percentile',
            line=dict(color='rgba(255, 127, 14, 0.4)', width=1),
            showlegend=True,
            hovertemplate='75th: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['q25'],
            mode='lines',
            name='25th Percentile',
            line=dict(color='rgba(255, 127, 14, 0.4)', width=1),
            fill='tonexty',
            fillcolor='rgba(255, 127, 14, 0.2)',
            showlegend=True,
            hovertemplate='25th: %{y:.2f}%<extra></extra>'
        ))
        
        # Add median and mean
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['median'],
            mode='lines',
            name='Median',
            line=dict(color='rgb(255, 127, 14)', width=3),
            showlegend=True,
            hovertemplate='Median: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['mean'],
            mode='lines',
            name='Average',
            line=dict(color='rgb(239, 85, 59)', width=3),
            showlegend=True,
            hovertemplate='Average: %{y:.2f}%<extra></extra>'
        ))
        
        title_text = f"Rolling Volatility: Aggregate Statistics Across {num_groups} {group_by.title()}s"
        
    else:
        # For few groups, show individual lines
        colors = px.colors.qualitative.Plotly
        color_dict = {group: colors[i % len(colors)] for i, group in enumerate(unique_groups)}
        
        for group in unique_groups:
            group_data = volatility_data[volatility_data[group_by] == group]
            
            fig.add_trace(go.Scatter(
                x=group_data['fetch_date'],
                y=group_data['volatility'],
                mode='lines',
                name=group,
                line=dict(color=color_dict[group], width=2),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Volatility: %{y:.2f}%<extra></extra>'
            ))
        
        title_text = f"Rolling Volatility by {group_by.title()}"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Volatility (Std Dev of Returns %)",
        height=700,
        hovermode='x unified',
        showlegend=True if num_groups > 20 else False,
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    return fig

def create_drawdown_chart(volatility_data, group_by):
    """Create line chart showing drawdowns over time."""
    
    if volatility_data.empty:
        return None
    
    unique_groups = sorted(volatility_data[group_by].unique())
    num_groups = len(unique_groups)
    
    fig = go.Figure()
    
    # If many groups (>20), show aggregate statistics
    if num_groups > 20:
        # Calculate aggregate statistics by date
        agg_stats = volatility_data.groupby('fetch_date')['drawdown'].agg([
            ('mean', 'mean'),
            ('median', 'median'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('min', 'min'),  # Most negative (worst)
            ('max', 'max')   # Least negative (best)
        ]).reset_index()
        
        # Add worst/best range
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['max'],  # Best (least negative)
            mode='lines',
            name='Best',
            line=dict(color='rgba(0, 204, 150, 0.3)', width=1),
            showlegend=True,
            hovertemplate='Best: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['min'],  # Worst (most negative)
            mode='lines',
            name='Worst',
            line=dict(color='rgba(239, 85, 59, 0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(239, 85, 59, 0.1)',
            showlegend=True,
            hovertemplate='Worst: %{y:.2f}%<extra></extra>'
        ))
        
        # Add quartile range
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['q75'],
            mode='lines',
            name='75th Percentile',
            line=dict(color='rgba(255, 127, 14, 0.4)', width=1),
            showlegend=True,
            hovertemplate='75th: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['q25'],
            mode='lines',
            name='25th Percentile',
            line=dict(color='rgba(255, 127, 14, 0.4)', width=1),
            fill='tonexty',
            fillcolor='rgba(255, 127, 14, 0.2)',
            showlegend=True,
            hovertemplate='25th: %{y:.2f}%<extra></extra>'
        ))
        
        # Add median and mean
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['median'],
            mode='lines',
            name='Median',
            line=dict(color='rgb(255, 127, 14)', width=3),
            showlegend=True,
            hovertemplate='Median: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=agg_stats['fetch_date'],
            y=agg_stats['mean'],
            mode='lines',
            name='Average',
            line=dict(color='rgb(239, 85, 59)', width=3),
            showlegend=True,
            hovertemplate='Average: %{y:.2f}%<extra></extra>'
        ))
        
        title_text = f"Drawdowns from Peak: Aggregate Statistics Across {num_groups} {group_by.title()}s"
        
    else:
        # For few groups, show individual lines
        colors = px.colors.qualitative.Plotly
        color_dict = {group: colors[i % len(colors)] for i, group in enumerate(unique_groups)}
        
        for group in unique_groups:
            group_data = volatility_data[volatility_data[group_by] == group]
            
            fig.add_trace(go.Scatter(
                x=group_data['fetch_date'],
                y=group_data['drawdown'],
                mode='lines',
                name=group,
                line=dict(color=color_dict[group], width=2),
                fill='tozeroy',
                fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(color_dict[group])) + [0.2])}',
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            ))
        
        title_text = f"Drawdowns from Peak by {group_by.title()}"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=700,
        hovermode='x unified',
        showlegend=True if num_groups > 20 else False,
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    return fig

def create_risk_metrics_table(current_metrics, group_by):
    """Create a table showing current risk metrics."""
    
    if current_metrics.empty:
        return None
    
    # Prepare display data
    display_df = current_metrics[[
        group_by, 'volatility', 'drawdown', 'max_drawdown', 'days_underwater'
    ]].copy()
    
    # Sort by volatility
    display_df = display_df.sort_values('volatility', ascending=False)
    
    # Rename columns
    display_df.columns = [
        group_by.title(),
        'Current Volatility (%)',
        'Current Drawdown (%)',
        'Max Drawdown (%)',
        'Days Underwater'
    ]
    
    # Format numbers
    display_df['Current Volatility (%)'] = display_df['Current Volatility (%)'].round(2)
    display_df['Current Drawdown (%)'] = display_df['Current Drawdown (%)'].round(2)
    display_df['Max Drawdown (%)'] = display_df['Max Drawdown (%)'].round(2)
    display_df['Days Underwater'] = display_df['Days Underwater'].astype(int)
    
    return display_df

def create_momentum_bar_chart(momentum_df, periods, group_by):
    """Create grouped bar chart showing momentum across different periods."""
    
    if momentum_df.empty:
        return None
    
    # Sort by average momentum
    momentum_sorted = momentum_df.sort_values('avg_momentum', ascending=True)
    
    fig = go.Figure()
    
    # Define colors for different periods
    colors = {
        3: 'rgb(255, 127, 14)',
        5: 'rgb(99, 110, 250)',
        10: 'rgb(239, 85, 59)',
        20: 'rgb(0, 204, 150)',
        30: 'rgb(171, 99, 250)',
    }
    
    # Add a bar for each period
    for period in sorted(periods):
        col_name = f'momentum_{period}d'
        if col_name in momentum_sorted.columns:
            fig.add_trace(go.Bar(
                name=f'{period}-Day',
                y=momentum_sorted[group_by],
                x=momentum_sorted[col_name],
                orientation='h',
                marker=dict(
                    color=colors.get(period, 'rgb(150, 150, 150)'),
                    line=dict(width=1, color='rgb(50, 50, 50)')
                ),
                text=momentum_sorted[col_name].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>' + f'{period}-Day Momentum: %{{x:.2f}}%<extra></extra>'
            ))
    
    fig.update_layout(
        title=f"Multi-Period Momentum by {group_by.title()}",
        xaxis_title="Rate of Change (%)",
        yaxis_title=group_by.title(),
        barmode='group',
        height=max(800, len(momentum_sorted) * 25),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=200, r=100, t=80, b=50),
        hovermode='closest'
    )
    
    return fig

def create_momentum_trend_lines(momentum_trends, group_by, momentum_period):
    """Create line chart showing momentum trends over time."""
    
    if momentum_trends.empty:
        return None
    
    unique_groups = sorted(momentum_trends[group_by].unique())
    num_groups = len(unique_groups)
    
    fig = go.Figure()
    
    # If many groups (>20), show aggregate statistics
    if num_groups > 20:
        # Calculate aggregate statistics by date
        agg_stats = momentum_trends.groupby('date')['momentum'].agg([
            ('mean', 'mean'),
            ('median', 'median'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('min', 'min'),
            ('max', 'max')
        ]).reset_index()
        
        # Add max/min range
        fig.add_trace(go.Scatter(
            x=agg_stats['date'],
            y=agg_stats['max'],
            mode='lines',
            name='Max',
            line=dict(color='rgba(0, 204, 150, 0.3)', width=1),
            showlegend=True,
            hovertemplate='Max: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=agg_stats['date'],
            y=agg_stats['min'],
            mode='lines',
            name='Min',
            line=dict(color='rgba(239, 85, 59, 0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(150, 150, 150, 0.1)',
            showlegend=True,
            hovertemplate='Min: %{y:.2f}%<extra></extra>'
        ))
        
        # Add quartile range
        fig.add_trace(go.Scatter(
            x=agg_stats['date'],
            y=agg_stats['q75'],
            mode='lines',
            name='75th Percentile',
            line=dict(color='rgba(99, 110, 250, 0.4)', width=1),
            showlegend=True,
            hovertemplate='75th: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=agg_stats['date'],
            y=agg_stats['q25'],
            mode='lines',
            name='25th Percentile',
            line=dict(color='rgba(99, 110, 250, 0.4)', width=1),
            fill='tonexty',
            fillcolor='rgba(99, 110, 250, 0.2)',
            showlegend=True,
            hovertemplate='25th: %{y:.2f}%<extra></extra>'
        ))
        
        # Add median and mean
        fig.add_trace(go.Scatter(
            x=agg_stats['date'],
            y=agg_stats['median'],
            mode='lines',
            name='Median',
            line=dict(color='rgb(255, 127, 14)', width=3),
            showlegend=True,
            hovertemplate='Median: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=agg_stats['date'],
            y=agg_stats['mean'],
            mode='lines',
            name='Average',
            line=dict(color='rgb(99, 110, 250)', width=3),
            showlegend=True,
            hovertemplate='Average: %{y:.2f}%<extra></extra>'
        ))
        
        title_text = f"{momentum_period}-Day Momentum: Aggregate Statistics Across {num_groups} {group_by.title()}s"
        
    else:
        # For few groups, show individual lines
        colors = px.colors.qualitative.Plotly
        color_dict = {group: colors[i % len(colors)] for i, group in enumerate(unique_groups)}
        
        for group in unique_groups:
            group_data = momentum_trends[momentum_trends[group_by] == group]
            
            fig.add_trace(go.Scatter(
                x=group_data['date'],
                y=group_data['momentum'],
                mode='lines',
                name=group,
                line=dict(color=color_dict[group], width=2),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Momentum: %{y:.2f}%<extra></extra>'
            ))
        
        title_text = f"{momentum_period}-Day Momentum Trends by {group_by.title()}"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title=f"{momentum_period}-Day Rate of Change (%)",
        height=700,
        hovermode='x unified',
        showlegend=True if num_groups > 20 else False,
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    return fig

def create_momentum_heatmap(momentum_trends, group_by, momentum_period):
    """Create heatmap of momentum over time."""
    
    if momentum_trends.empty:
        return None
    
    # Create pivot table
    momentum_trends['date_str'] = momentum_trends['date'].dt.strftime('%Y-%m-%d')
    pivot_data = momentum_trends.pivot(
        index='date_str',
        columns=group_by,
        values='momentum'
    )
    
    # Sort dates (newest first)
    dates = sorted(pivot_data.index, reverse=True)
    pivot_data = pivot_data.loc[dates]
    
    # Normalize for better color scale (cap at +/- 20% for visualization)
    vmax = min(20, pivot_data.abs().max().max())
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlGn',
        zmid=0,
        zmin=-vmax,
        zmax=vmax,
        hovertemplate='<b>%{y}</b><br>%{x}<br>Momentum: %{z:.2f}%<extra></extra>',
        colorbar=dict(title="Momentum (%)")
    ))
    
    fig.update_layout(
        title=f"{momentum_period}-Day Momentum Heatmap by {group_by.title()}",
        xaxis_title=group_by.title(),
        yaxis_title="Date",
        height=max(400, len(dates) * 20),
        xaxis=dict(
            side='top',
            tickangle=45 if len(pivot_data.columns) > 15 else 0
        ),
        yaxis=dict(type='category')
    )
    
    return fig

def create_trend_line_chart(filtered_scores, group_by, score_start_date):
    """Create the main trend line chart."""
    
    if filtered_scores.empty:
        st.warning("No data available for visualization")
        return None
    
    unique_groups = sorted(filtered_scores[group_by].unique())
    num_groups = len(unique_groups)
    
    # Create hover data
    hover_data = defaultdict(list)
    for _, row in filtered_scores.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        score = row['score']
        group = row[group_by]
        key = (date_str, score)
        hover_data[key].append(group)
    
    fig = go.Figure()
    
    # If many groups (>20), show aggregate statistics
    if num_groups > 20:
        # Calculate aggregate statistics by date
        agg_stats = filtered_scores.groupby('date')['score'].agg([
            ('mean', 'mean'),
            ('median', 'median'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('min', 'min'),
            ('max', 'max')
        ]).reset_index()
        
        # Add max/min range
        fig.add_trace(go.Scatter(
            x=agg_stats['date'],
            y=agg_stats['max'],
            mode='lines',
            name='Max Score',
            line=dict(color='rgba(0, 204, 150, 0.3)', width=1),
            showlegend=True,
            hovertemplate='Max: %{y:.0f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=agg_stats['date'],
            y=agg_stats['min'],
            mode='lines',
            name='Min Score',
            line=dict(color='rgba(239, 85, 59, 0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(150, 150, 150, 0.1)',
            showlegend=True,
            hovertemplate='Min: %{y:.0f}<extra></extra>'
        ))
        
        # Add quartile range
        fig.add_trace(go.Scatter(
            x=agg_stats['date'],
            y=agg_stats['q75'],
            mode='lines',
            name='75th Percentile',
            line=dict(color='rgba(99, 110, 250, 0.4)', width=1),
            showlegend=True,
            hovertemplate='75th: %{y:.0f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=agg_stats['date'],
            y=agg_stats['q25'],
            mode='lines',
            name='25th Percentile',
            line=dict(color='rgba(99, 110, 250, 0.4)', width=1),
            fill='tonexty',
            fillcolor='rgba(99, 110, 250, 0.2)',
            showlegend=True,
            hovertemplate='25th: %{y:.0f}<extra></extra>'
        ))
        
        # Add median and mean
        fig.add_trace(go.Scatter(
            x=agg_stats['date'],
            y=agg_stats['median'],
            mode='lines',
            name='Median Score',
            line=dict(color='rgb(255, 127, 14)', width=3),
            showlegend=True,
            hovertemplate='Median: %{y:.0f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=agg_stats['date'],
            y=agg_stats['mean'],
            mode='lines',
            name='Average Score',
            line=dict(color='rgb(99, 110, 250)', width=3),
            showlegend=True,
            hovertemplate='Average: %{y:.0f}<extra></extra>'
        ))
        
        title_text = f"Trend Scores: Aggregate Statistics Across {num_groups} {group_by.title()}s (Reset at {score_start_date})"
        show_legend = True
        
    else:
        # For few groups, show individual lines
        colors = px.colors.qualitative.Plotly
        color_dict = {group: colors[i % len(colors)] for i, group in enumerate(unique_groups)}
        
        # Add traces for each group
        for group in unique_groups:
            group_data = filtered_scores[filtered_scores[group_by] == group]
            
            # Create hover text
            hover_texts = []
            for _, row in group_data.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d')
                score = row['score']
                
                # Find other groups with same score on same date
                same_score_groups = hover_data.get((date_str, score), [])
                other_groups = [g for g in same_score_groups if g != group]
                
                if other_groups:
                    hover_text = f"<b>{group}</b><br>Date: {date_str}<br>Score: {score}<br><br>"
                    hover_text += f"Also at score {score}:<br>" + "<br>".join(other_groups)
                    hover_text += f"<br><br>Total: {len(same_score_groups)} groups"
                else:
                    hover_text = f"<b>{group}</b><br>Date: {date_str}<br>Score: {score}"
                
                hover_texts.append(hover_text)
            
            # Add line trace
            fig.add_trace(go.Scatter(
                x=group_data['date'],
                y=group_data['score'],
                mode='lines',
                name=group,
                line=dict(color=color_dict[group], width=3),
                hoverinfo='text',
                hovertext=hover_texts,
                hoverlabel=dict(
                    bgcolor='rgba(0, 0, 0, 0.8)',
                    font=dict(color='white', size=12),
                    bordercolor='white'
                )
            ))
        
        title_text = f"Trend Scores by {group_by.title()} (Reset at {score_start_date})"
        show_legend = False
    
    # Update layout
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Cumulative Score",
        height=800,
        hovermode='closest',
        showlegend=show_legend,
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='black'),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="black",
            borderwidth=2,
            font=dict(size=12, color="black")
        ) if show_legend else None
    )
    
    return fig

def create_current_scores_chart(current_data, group_by):
    """Create bar chart of current scores."""
    
    if current_data.empty:
        return None
    
    # Filter non-zero scores
    nonzero_data = current_data[current_data['score'] != 0].copy()
    
    if nonzero_data.empty:
        st.info("All groups currently have a score of 0")
        return None
    
    # Sort by absolute score
    nonzero_data = nonzero_data.reindex(
        nonzero_data['score'].abs().sort_values(ascending=False).index
    )
    
    # Create color scale
    colorscale = [[0, 'red'], [0.5, 'white'], [1, 'blue']]
    
    fig = px.bar(
        nonzero_data,
        x=group_by,
        y='score',
        color='score',
        color_continuous_scale=colorscale,
        title=f"Current Scores by {group_by.title()}",
        height=600
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=group_by.title(),
        yaxis_title="Score (+ Up, - Down)",
        xaxis=dict(tickangle=45 if len(nonzero_data) > 15 else 0),
        yaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=2)
    )
    
    # Add score labels on bars
    for i, row in nonzero_data.iterrows():
        fig.add_annotation(
            x=row[group_by],
            y=row['score'],
            text=str(int(row['score'])),
            showarrow=False,
            font=dict(color="black", size=10),
            bgcolor="white",
            bordercolor="black"
        )
    
    return fig

def create_heatmap(daily_data, group_by, display_start_date):
    """Create daily direction heatmap."""
    
    # Filter data for display period
    display_data = daily_data[daily_data['fetch_date'] >= pd.Timestamp(display_start_date)].copy()
    
    if display_data.empty:
        return None
    
    # Create date strings and pivot
    display_data['date_str'] = display_data['fetch_date'].dt.strftime('%Y-%m-%d')
    pivot_data = display_data.pivot(
        index='date_str', 
        columns=group_by, 
        values='direction'
    ).fillna(0)
    
    # Sort dates (newest first)
    trading_days = sorted(pivot_data.index, reverse=True)
    pivot_data = pivot_data.loc[trading_days]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale=[[0, 'red'], [0.5, 'white'], [1, 'green']],
        zmin=-1, zmax=1,
        hovertemplate='<b>%{y}</b><br>%{x}: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Daily Direction Heatmap by {group_by.title()}",
        xaxis_title=group_by.title(),
        yaxis_title="Date",
        height=max(400, len(trading_days) * 20),
        xaxis=dict(side='top', tickangle=45 if len(pivot_data.columns) > 15 else 0),
        yaxis=dict(type='category')
    )
    
    return fig

# ------------------------------
# 4. Main Application
# ------------------------------
def main():
    # Find and load data files
    files_found = find_data_files()
    
    missing_files = [f for f in ["stock_sectors.xlsx", "market_caps.xlsx"] if f not in files_found]
    if missing_files:
        st.error(f"❌ Missing required files: {missing_files}")
        st.info("🔍 Current directory contents:")
        for item in os.listdir('.'):
            if item.endswith('.xlsx'):
                st.info(f"  Found: {item}")
        st.stop()
    
    # Load data
    with st.sidebar:
        st.header("📊 Data Loading")
        
        with st.spinner("Loading sectors data..."):
            sectors_df = load_sectors_file(files_found["stock_sectors.xlsx"])
        
        with st.spinner("Loading market caps data..."):
            market_caps_df = load_market_caps_file(files_found["market_caps.xlsx"])
    
    # Sidebar controls
    st.sidebar.header("⚙️ Analysis Settings")
    
    # Group selection
    group_by = st.sidebar.selectbox(
        "Group By", 
        options=["sector", "industry"],
        help="Choose how to group the analysis"
    )
    
    # Date controls
    unique_dates = sorted(market_caps_df['fetch_date'].dt.date.unique())
    min_date, max_date = min(unique_dates), max(unique_dates)
    
    st.sidebar.subheader("📅 Date Settings")
    
    # Display period
    max_days = len(unique_dates)
    days_to_display = st.sidebar.slider(
        "Days to display", 
        min_value=1, 
        max_value=max_days, 
        value=min(max_days, 30),
        help="Number of recent days to show in charts"
    )
    
    display_start_date = unique_dates[-days_to_display] if days_to_display < max_days else min_date
    
    # Score calculation start date
    score_start_date = st.sidebar.date_input(
        "Score calculation start date",
        value=display_start_date,
        min_value=min_date,
        max_value=max_date,
        help="Scores will reset to 0 starting from this date"
    )
    
    # Percent change lookback
    percent_change_days = st.sidebar.slider(
        "Percent change lookback (days)",
        min_value=1,
        max_value=min(max_days, 30),
        value=5,
        help="Number of days to look back for percent change calculation"
    )
    
    # Momentum settings
    st.sidebar.subheader("🚀 Momentum Settings")
    momentum_periods = st.sidebar.multiselect(
        "Momentum periods (days)",
        options=[3, 5, 10, 20, 30],
        default=[5, 10, 20],
        help="Select multiple periods to calculate momentum rate of change"
    )
    
    momentum_trend_period = st.sidebar.selectbox(
        "Momentum trend period",
        options=[5, 10, 20],
        index=0,
        help="Period for momentum trend visualization"
    )
    
    momentum_lookback = st.sidebar.slider(
        "Momentum trend lookback (days)",
        min_value=10,
        max_value=min(max_days, 60),
        value=30,
        help="How many days to show in momentum trends"
    )
    
    # Advanced analysis settings
    st.sidebar.subheader("📈 Advanced Analysis")
    
    enable_mean_reversion = st.sidebar.checkbox(
        "Enable Mean Reversion Analysis",
        value=True,
        help="Calculate Z-scores and Bollinger Bands for mean reversion signals"
    )
    
    zscore_window = st.sidebar.slider(
        "Z-score window (days)",
        min_value=10,
        max_value=30,
        value=20,
        help="Rolling window for Z-score calculation"
    ) if enable_mean_reversion else 20
    
    enable_concentration = st.sidebar.checkbox(
        "Enable Concentration Analysis",
        value=True,
        help="Analyze market cap concentration within industries"
    )
    
    enable_rotation = st.sidebar.checkbox(
        "Enable Rotation Analysis",
        value=True,
        help="Classify industries into lifecycle stages"
    )
    
    enable_streaks = st.sidebar.checkbox(
        "Enable Streak Analysis",
        value=True,
        help="Track win/loss streaks and patterns"
    )
    
    streaks_lookback = st.sidebar.slider(
        "Streak lookback (days)",
        min_value=20,
        max_value=90,
        value=60,
        help="How many days to analyze for streak patterns"
    )
    
    breadth_lookback = st.sidebar.slider(
        "Breadth lookback (days)",
        min_value=10,
        max_value=min(max_days, 60),
        value=30,
        help="How many days to show in breadth analysis"
    )
    
    relative_strength_lookback = st.sidebar.slider(
        "Relative strength lookback (days)",
        min_value=10,
        max_value=min(max_days, 60),
        value=30,
        help="How many days to show in relative strength analysis"
    )
    
    volatility_lookback = st.sidebar.slider(
        "Volatility/Drawdown lookback (days)",
        min_value=20,
        max_value=min(max_days, 90),
        value=60,
        help="How many days to show in volatility and drawdown analysis"
    )
    
    volatility_window = st.sidebar.slider(
        "Volatility window (days)",
        min_value=5,
        max_value=30,
        value=20,
        help="Rolling window for volatility calculation"
    )
    
    # Process data
    with st.spinner("Processing data..."):
        try:
            # Calculate daily changes
            daily_changes = calculate_daily_changes(market_caps_df, sectors_df, group_by)
            
            # Calculate trend scores
            trend_scores = calculate_trend_scores(daily_changes, score_start_date, group_by)
            
            # Calculate percent changes
            percent_changes = calculate_percent_changes(daily_changes, percent_change_days, group_by)
            
            # Calculate momentum if periods are selected
            momentum_data = pd.DataFrame()
            momentum_trends_data = pd.DataFrame()
            if momentum_periods:
                momentum_data = calculate_momentum(daily_changes, momentum_periods, group_by)
                if momentum_trend_period:
                    momentum_trends_data = calculate_momentum_trends(
                        daily_changes, 
                        momentum_trend_period, 
                        group_by, 
                        momentum_lookback
                    )
            
            # Calculate breadth indicators
            breadth_data = calculate_breadth_indicators(
                market_caps_df,
                sectors_df,
                group_by,
                breadth_lookback
            )
            
            # Calculate relative strength
            relative_strength_data = calculate_relative_strength(
                daily_changes,
                group_by,
                relative_strength_lookback
            )
            
            # Calculate volatility and drawdown
            volatility_data, current_risk_metrics = calculate_volatility_drawdown(
                daily_changes,
                group_by,
                volatility_lookback,
                volatility_window
            )
            
            # Calculate mean reversion signals
            mean_reversion_data = pd.DataFrame()
            current_signals = pd.DataFrame()
            if enable_mean_reversion:
                mean_reversion_data, current_signals = calculate_mean_reversion(
                    daily_changes,
                    group_by,
                    breadth_lookback,
                    zscore_window
                )
            
            # Calculate concentration
            concentration_data = pd.DataFrame()
            if enable_concentration:
                concentration_data = calculate_concentration(
                    market_caps_df,
                    sectors_df,
                    group_by
                )
            
            # Calculate rotation stages
            rotation_data = pd.DataFrame()
            if enable_rotation and not momentum_data.empty and not relative_strength_data.empty:
                rotation_data = calculate_rotation_stages(
                    momentum_data,
                    relative_strength_data,
                    group_by
                )
            
            # Calculate streaks
            streaks_data = pd.DataFrame()
            if enable_streaks:
                streaks_data = calculate_streaks(
                    daily_changes,
                    group_by,
                    streaks_lookback
                )
            
            # Filter for display period
            filtered_scores = trend_scores[
                trend_scores['date'] >= pd.Timestamp(display_start_date)
            ].sort_values(['date', group_by])
            
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            st.stop()
    
    # Show summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(market_caps_df):,}")
    with col2:
        st.metric("Unique Symbols", f"{market_caps_df['symbol'].nunique():,}")
    with col3:
        st.metric("Date Range", f"{len(unique_dates)} days")
    with col4:
        st.metric(f"Total {group_by.title()}s", f"{len(sectors_df[group_by].unique())}")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📈 Trend Scores",
        "🚀 Momentum",
        "📊 Breadth", 
        "🎯 Relative Strength",
        "⚠️ Risk",
        "🎲 Mean Reversion",
        "🏢 Concentration",
        "🔄 Rotation",
        "🔥 Streaks",
        "🗺️ Heatmaps"
    ])
    
    # Tab 1: Trend Scores
    with tab1:
        st.header(f"📈 Trend Analysis by {group_by.title()}")
        st.info(f"Score calculation: +1 for up days, -1 for down days. Scores reset to 0 starting {score_start_date}")
        
        # Trend line chart
        if not filtered_scores.empty:
            num_groups = len(filtered_scores[group_by].unique())
            if num_groups > 20:
                st.caption(f"Showing aggregate statistics across {num_groups} {group_by}s")
            
            fig_trends = create_trend_line_chart(filtered_scores, group_by, score_start_date)
            if fig_trends:
                st.plotly_chart(fig_trends, use_container_width=True)
            
            # Group selector for detailed view
            if num_groups > 20:
                st.subheader("🔍 Compare Specific Groups")
                selected_trend_groups = st.multiselect(
                    f"Select {group_by}s to compare (max 10)",
                    options=sorted(filtered_scores[group_by].unique()),
                    max_selections=10,
                    help=f"Select up to 10 {group_by}s to view their individual trend scores",
                    key="trend_selector"
                )
                
                if selected_trend_groups:
                    filtered_trends = filtered_scores[filtered_scores[group_by].isin(selected_trend_groups)]
                    
                    # Create hover data
                    hover_data = defaultdict(list)
                    for _, row in filtered_trends.iterrows():
                        date_str = row['date'].strftime('%Y-%m-%d')
                        score = row['score']
                        group = row[group_by]
                        key = (date_str, score)
                        hover_data[key].append(group)
                    
                    fig_selected_trends = go.Figure()
                    colors = px.colors.qualitative.Plotly
                    color_dict = {group: colors[i % len(colors)] for i, group in enumerate(selected_trend_groups)}
                    
                    for group in selected_trend_groups:
                        group_data = filtered_trends[filtered_trends[group_by] == group]
                        
                        # Create hover text
                        hover_texts = []
                        for _, row in group_data.iterrows():
                            date_str = row['date'].strftime('%Y-%m-%d')
                            score = row['score']
                            
                            same_score_groups = hover_data.get((date_str, score), [])
                            other_groups = [g for g in same_score_groups if g != group]
                            
                            if other_groups:
                                hover_text = f"<b>{group}</b><br>Date: {date_str}<br>Score: {score}<br><br>"
                                hover_text += f"Also at score {score}:<br>" + "<br>".join(other_groups)
                                hover_text += f"<br><br>Total: {len(same_score_groups)} groups"
                            else:
                                hover_text = f"<b>{group}</b><br>Date: {date_str}<br>Score: {score}"
                            
                            hover_texts.append(hover_text)
                        
                        fig_selected_trends.add_trace(go.Scatter(
                            x=group_data['date'],
                            y=group_data['score'],
                            mode='lines',
                            name=group,
                            line=dict(color=color_dict[group], width=3),
                            hoverinfo='text',
                            hovertext=hover_texts,
                            hoverlabel=dict(
                                bgcolor='rgba(0, 0, 0, 0.8)',
                                font=dict(color='white', size=12),
                                bordercolor='white'
                            )
                        ))
                    
                    fig_selected_trends.update_layout(
                        title=f"Selected {group_by.title()}s Trend Score Comparison",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Score",
                        height=700,
                        hovermode='closest',
                        xaxis=dict(showgrid=True, gridcolor='lightgray'),
                        yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='black'),
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01,
                            bgcolor="rgba(255, 255, 255, 0.95)",
                            bordercolor="black",
                            borderwidth=2,
                            font=dict(size=11, color="black")
                        )
                    )
                    
                    st.plotly_chart(fig_selected_trends, use_container_width=True)
        
        # Current scores
        st.subheader("📊 Current Scores")
        if not filtered_scores.empty:
            latest_date = filtered_scores['date'].max()
            current_data = filtered_scores[filtered_scores['date'] == latest_date]
            
            fig_current = create_current_scores_chart(current_data, group_by)
            if fig_current:
                st.plotly_chart(fig_current, use_container_width=True)
        
        # Percent changes
        if not percent_changes.empty:
            st.subheader(f"📊 Percent Change: Current vs Previous {percent_change_days} {'Day' if percent_change_days == 1 else 'Days'}")
            
            # Get the date ranges
            latest_date = percent_changes['latest_date'].iloc[0].strftime('%Y-%m-%d')
            current_start_date = percent_changes['current_start_date'].iloc[0].strftime('%Y-%m-%d')
            previous_end_date = percent_changes['previous_end_date'].iloc[0].strftime('%Y-%m-%d')
            previous_start_date = percent_changes['previous_start_date'].iloc[0].strftime('%Y-%m-%d')
            
            # Display date ranges
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Current Period:** {current_start_date} to {latest_date}")
            with col2:
                st.info(f"**Previous Period:** {previous_start_date} to {previous_end_date}")
            
            # Sort by current percent change
            pct_sorted = percent_changes.sort_values('current_percent_change', ascending=True)
            
            # Create the grouped bar chart
            fig_pct = go.Figure()
            
            # Add bars for current period
            fig_pct.add_trace(go.Bar(
                name='Current Period',
                y=pct_sorted[group_by],
                x=pct_sorted['current_percent_change'],
                orientation='h',
                marker=dict(
                    color='rgb(99, 110, 250)',
                    line=dict(color='rgb(69, 80, 220)', width=1)
                ),
                text=pct_sorted['current_percent_change'].apply(lambda x: f"{x:.1f}%"),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Current Period: %{x:.2f}%<extra></extra>'
            ))
            
            # Add bars for previous period
            fig_pct.add_trace(go.Bar(
                name='Previous Period',
                y=pct_sorted[group_by],
                x=pct_sorted['previous_percent_change'],
                orientation='h',
                marker=dict(
                    color='rgba(150, 150, 150, 0.6)',
                    line=dict(color='rgba(100, 100, 100, 0.8)', width=1)
                ),
                text=pct_sorted['previous_percent_change'].apply(lambda x: f"{x:.1f}%"),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Previous Period: %{x:.2f}%<extra></extra>'
            ))
            
            # Update layout
            fig_pct.update_layout(
                title=f"Valuation Percent Change by {group_by.title()} - Comparing {percent_change_days}-Day Periods",
                xaxis_title="Percent Change (%)",
                yaxis_title=group_by.title(),
                barmode='group',
                bargap=0.15,
                bargroupgap=0.1,
                height=max(800, len(pct_sorted) * 20),
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=2
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray'
                ),
                margin=dict(l=200, r=100, t=80, b=50),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode='closest'
            )
            
            st.plotly_chart(fig_pct, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_current = pct_sorted['current_percent_change'].mean()
                st.metric("Avg Current Period", f"{avg_current:.2f}%")
            
            with col2:
                avg_previous = pct_sorted['previous_percent_change'].mean()
                st.metric("Avg Previous Period", f"{avg_previous:.2f}%")
            
            with col3:
                improved = (pct_sorted['current_percent_change'] > 
                           pct_sorted['previous_percent_change']).sum()
                total = len(pct_sorted)
                improvement_pct = (improved / total * 100) if total > 0 else 0
                st.metric("Improved vs Previous", f"{improved}/{total}", f"{improvement_pct:.1f}%")
    
    # Tab 2: Momentum Analysis
    with tab2:
        if not momentum_data.empty and momentum_periods:
            st.header("🚀 Momentum Analysis")
            st.info("""
            **Momentum** measures the rate of change in market capitalization over different time periods. 
            Positive momentum indicates accelerating growth, while negative momentum indicates accelerating decline.
            """)
            
            # Multi-period momentum comparison
            st.subheader(f"📊 Multi-Period Momentum Comparison")
            fig_momentum = create_momentum_bar_chart(momentum_data, momentum_periods, group_by)
            if fig_momentum:
                st.plotly_chart(fig_momentum, use_container_width=True)
            
            # Momentum statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_momentum = momentum_data['avg_momentum'].mean()
                st.metric("Average Momentum", f"{avg_momentum:.2f}%")
            
            with col2:
                positive_count = (momentum_data['avg_momentum'] > 0).sum()
                total_count = len(momentum_data)
                st.metric("Positive Momentum", f"{positive_count}/{total_count}", 
                         f"{(positive_count/total_count*100):.1f}%")
            
            with col3:
                if not momentum_data.empty:
                    strongest_idx = momentum_data['avg_momentum'].abs().idxmax()
                    strongest_group = momentum_data.loc[strongest_idx, group_by]
                    strongest_value = momentum_data.loc[strongest_idx, 'avg_momentum']
                    st.metric("Strongest", strongest_group, f"{strongest_value:.2f}%")
            
            with col4:
                avg_consistency = momentum_data['momentum_consistency'].mean()
                st.metric("Avg Consistency (σ)", f"{avg_consistency:.2f}%")
            
            # Momentum trends over time
            if not momentum_trends_data.empty:
                st.subheader(f"📈 {momentum_trend_period}-Day Momentum Trends")
                
                num_groups = len(momentum_trends_data[group_by].unique())
                if num_groups > 20:
                    st.caption(f"Showing aggregate statistics across {num_groups} {group_by}s")
                
                fig_momentum_trends = create_momentum_trend_lines(
                    momentum_trends_data, 
                    group_by, 
                    momentum_trend_period
                )
                if fig_momentum_trends:
                    st.plotly_chart(fig_momentum_trends, use_container_width=True)
                
                # Group selector for detailed view
                if num_groups > 20:
                    st.subheader("🔍 Compare Specific Groups")
                    selected_mom_groups = st.multiselect(
                        f"Select {group_by}s to compare (max 10)",
                        options=sorted(momentum_trends_data[group_by].unique()),
                        max_selections=10,
                        help=f"Select up to 10 {group_by}s to view their individual momentum trends",
                        key="momentum_selector"
                    )
                    
                    if selected_mom_groups:
                        filtered_mom = momentum_trends_data[momentum_trends_data[group_by].isin(selected_mom_groups)]
                        
                        fig_selected_mom = go.Figure()
                        colors = px.colors.qualitative.Plotly
                        
                        for i, group in enumerate(selected_mom_groups):
                            group_data = filtered_mom[filtered_mom[group_by] == group]
                            fig_selected_mom.add_trace(go.Scatter(
                                x=group_data['date'],
                                y=group_data['momentum'],
                                mode='lines',
                                name=group,
                                line=dict(color=colors[i % len(colors)], width=2),
                                hovertemplate=f'<b>{group}</b><br>Date: %{{x}}<br>Momentum: %{{y:.2f}}%<extra></extra>'
                            ))
                        
                        fig_selected_mom.add_hline(y=0, line_dash="dash", line_color="black")
                        
                        fig_selected_mom.update_layout(
                            title=f"Selected {group_by.title()}s {momentum_trend_period}-Day Momentum Comparison",
                            xaxis_title="Date",
                            yaxis_title=f"{momentum_trend_period}-Day Momentum (%)",
                            height=600,
                            hovermode='x unified',
                            xaxis=dict(showgrid=True, gridcolor='lightgray'),
                            yaxis=dict(
                                showgrid=True,
                                gridcolor='lightgray',
                                zeroline=True,
                                zerolinecolor='black',
                                zerolinewidth=2
                            ),
                            legend=dict(
                                orientation="v",
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01,
                                bgcolor="rgba(255, 255, 255, 0.95)",
                                bordercolor="black",
                                borderwidth=2,
                                font=dict(size=11, color="black")
                            )
                        )
                        
                        st.plotly_chart(fig_selected_mom, use_container_width=True)
                
                st.subheader(f"🔥 {momentum_trend_period}-Day Momentum Heatmap")
                fig_momentum_heatmap = create_momentum_heatmap(
                    momentum_trends_data,
                    group_by,
                    momentum_trend_period
                )
                if fig_momentum_heatmap:
                    st.plotly_chart(fig_momentum_heatmap, use_container_width=True)
            
            # Momentum rankings table
            if st.checkbox("Show momentum rankings"):
                st.subheader("🏆 Momentum Rankings")
                
                display_cols = [group_by] + [f'momentum_{p}d' for p in sorted(momentum_periods)] + ['avg_momentum', 'momentum_consistency']
                momentum_display = momentum_data[display_cols].copy()
                
                rename_dict = {f'momentum_{p}d': f'{p}d %' for p in momentum_periods}
                rename_dict['avg_momentum'] = 'Avg %'
                rename_dict['momentum_consistency'] = 'Consistency (σ)'
                momentum_display = momentum_display.rename(columns=rename_dict)
                
                momentum_display = momentum_display.sort_values('Avg %', ascending=False)
                
                pct_cols = [col for col in momentum_display.columns if col.endswith('%') or col == 'Consistency (σ)']
                for col in pct_cols:
                    momentum_display[col] = momentum_display[col].round(2)
                
                st.dataframe(
                    momentum_display.style.format({
                        col: '{:.2f}%' for col in pct_cols
                    }).background_gradient(subset=['Avg %'], cmap='RdYlGn'),
                    use_container_width=True
                )
    
    # Tab 3: Breadth Analysis
    with tab3:
        st.header("📊 Breadth Analysis")
        st.info("""
        **Breadth** measures the participation rate in market moves. High breadth (most stocks up) = strong, healthy trend.
        Low breadth (few stocks up) = weak trend that may reverse.
        """)
        
        if not breadth_data.empty:
            # Breadth statistics
            latest_breadth = breadth_data[breadth_data['fetch_date'] == breadth_data['fetch_date'].max()]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_breadth = latest_breadth['pct_stocks_up'].mean()
                st.metric("Avg % Stocks Up", f"{avg_breadth:.1f}%")
            
            with col2:
                strong_breadth = (latest_breadth['pct_stocks_up'] > 60).sum()
                total_groups = len(latest_breadth)
                st.metric("Strong Breadth (>60%)", f"{strong_breadth}/{total_groups}")
            
            with col3:
                weak_breadth = (latest_breadth['pct_stocks_up'] < 40).sum()
                st.metric("Weak Breadth (<40%)", f"{weak_breadth}/{total_groups}")
            
            with col4:
                best_breadth_idx = latest_breadth['pct_stocks_up'].idxmax()
                best_breadth_group = latest_breadth.loc[best_breadth_idx, group_by]
                best_breadth_value = latest_breadth.loc[best_breadth_idx, 'pct_stocks_up']
                st.metric("Best Breadth", best_breadth_group, f"{best_breadth_value:.1f}%")
            
            # Breadth distribution
            num_groups = len(breadth_data[group_by].unique())
            if num_groups > 20:
                st.subheader("📊 Breadth Distribution")
                st.caption(f"Showing distribution across {num_groups} {group_by}s - too many to display individually")
                fig_breadth_dist = create_breadth_distribution(breadth_data, group_by)
                if fig_breadth_dist:
                    st.plotly_chart(fig_breadth_dist, use_container_width=True)
            
            # Breadth trends (aggregate for many groups, individual for few)
            st.subheader("📈 Breadth Trends Over Time")
            if num_groups > 20:
                st.caption(f"Showing aggregate statistics (mean, median, quartiles) across {num_groups} {group_by}s")
            fig_breadth = create_breadth_chart(breadth_data, group_by)
            if fig_breadth:
                st.plotly_chart(fig_breadth, use_container_width=True)
            
            # Top/Bottom performers
            if num_groups > 20:
                st.subheader("🏆 Breadth Leaders & Laggards")
                st.caption("Top and bottom performers by breadth")
                fig_breadth_top_bottom = create_breadth_top_bottom(breadth_data, group_by, n=15)
                if fig_breadth_top_bottom:
                    st.plotly_chart(fig_breadth_top_bottom, use_container_width=True)
            
            # Industry selector for detailed view
            if num_groups > 20:
                st.subheader("🔍 Compare Specific Groups")
                selected_groups = st.multiselect(
                    f"Select {group_by}s to compare (max 10)",
                    options=sorted(breadth_data[group_by].unique()),
                    max_selections=10,
                    help=f"Select up to 10 {group_by}s to view their individual breadth trends"
                )
                
                if selected_groups:
                    filtered_breadth = breadth_data[breadth_data[group_by].isin(selected_groups)]
                    
                    fig_selected = go.Figure()
                    colors = px.colors.qualitative.Plotly
                    
                    for i, group in enumerate(selected_groups):
                        group_data = filtered_breadth[filtered_breadth[group_by] == group]
                        fig_selected.add_trace(go.Scatter(
                            x=group_data['fetch_date'],
                            y=group_data['pct_stocks_up'],
                            mode='lines',
                            name=group,
                            line=dict(color=colors[i % len(colors)], width=2),
                            hovertemplate=f'<b>{group}</b><br>Date: %{{x}}<br>Breadth: %{{y:.1f}}%<extra></extra>'
                        ))
                    
                    fig_selected.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%")
                    
                    fig_selected.update_layout(
                        title=f"Selected {group_by.title()}s Breadth Comparison",
                        xaxis_title="Date",
                        yaxis_title="% of Stocks Up",
                        height=600,
                        hovermode='x unified',
                        xaxis=dict(showgrid=True, gridcolor='lightgray'),
                        yaxis=dict(showgrid=True, gridcolor='lightgray', range=[0, 100]),
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    
                    st.plotly_chart(fig_selected, use_container_width=True)
            
            # Breadth heatmap
            st.subheader("🔥 Breadth Heatmap")
            if num_groups > 30:
                st.caption(f"Heatmap across all {num_groups} {group_by}s")
            fig_breadth_heatmap = create_breadth_heatmap(breadth_data, group_by)
            if fig_breadth_heatmap:
                st.plotly_chart(fig_breadth_heatmap, use_container_width=True)
    
    # Tab 4: Relative Strength
    with tab4:
        st.header("🎯 Relative Strength Analysis")
        st.info("""
        **Relative Strength** shows performance vs the overall market. Positive = outperforming, Negative = underperforming.
        Leaders have sustained positive relative strength. Use this to identify where to allocate capital.
        """)
        
        if not relative_strength_data.empty:
            # Get latest relative strength
            latest_rs = relative_strength_data[relative_strength_data['fetch_date'] == relative_strength_data['fetch_date'].max()]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                outperformers = (latest_rs['cumulative_relative_strength'] > 0).sum()
                total_groups = len(latest_rs)
                st.metric("Outperforming Market", f"{outperformers}/{total_groups}")
            
            with col2:
                strongest_idx = latest_rs['cumulative_relative_strength'].idxmax()
                strongest_group = latest_rs.loc[strongest_idx, group_by]
                strongest_value = latest_rs.loc[strongest_idx, 'cumulative_relative_strength']
                st.metric("Strongest", strongest_group, f"+{strongest_value:.1f}%")
            
            with col3:
                weakest_idx = latest_rs['cumulative_relative_strength'].idxmin()
                weakest_group = latest_rs.loc[weakest_idx, group_by]
                weakest_value = latest_rs.loc[weakest_idx, 'cumulative_relative_strength']
                st.metric("Weakest", weakest_group, f"{weakest_value:.1f}%")
            
            with col4:
                avg_rs = latest_rs['cumulative_relative_strength'].mean()
                st.metric("Avg Rel Strength", f"{avg_rs:.2f}%")
            
            # Get number of groups for smart display
            num_groups = len(relative_strength_data[group_by].unique())
            
            # Distribution
            if num_groups > 20:
                st.subheader("📊 Relative Strength Distribution")
                st.caption(f"Showing distribution across {num_groups} {group_by}s")
                fig_rs_dist = create_relative_strength_distribution(relative_strength_data, group_by)
                if fig_rs_dist:
                    st.plotly_chart(fig_rs_dist, use_container_width=True)
            
            # Top/Bottom performers (always useful, even for few groups)
            st.subheader("🏆 Relative Strength Leaders & Laggards")
            if num_groups > 30:
                st.caption("Top and bottom performers vs market")
                fig_rs_top_bottom = create_relative_strength_top_bottom(relative_strength_data, group_by, n=15)
            else:
                # Show all if there aren't too many
                fig_rs_top_bottom = create_relative_strength_ranking(relative_strength_data, group_by)
            
            if fig_rs_top_bottom:
                st.plotly_chart(fig_rs_top_bottom, use_container_width=True)
            
            # Relative strength trends (aggregate for many groups, individual for few)
            st.subheader("📈 Relative Strength Trends Over Time")
            if num_groups > 20:
                st.caption(f"Showing aggregate statistics across {num_groups} {group_by}s")
            fig_rs_trends = create_relative_strength_chart(relative_strength_data, group_by)
            if fig_rs_trends:
                st.plotly_chart(fig_rs_trends, use_container_width=True)
            
            # Group selector for detailed view
            if num_groups > 20:
                st.subheader("🔍 Compare Specific Groups")
                selected_rs_groups = st.multiselect(
                    f"Select {group_by}s to compare (max 10)",
                    options=sorted(relative_strength_data[group_by].unique()),
                    max_selections=10,
                    help=f"Select up to 10 {group_by}s to view their individual relative strength trends",
                    key="rs_selector"
                )
                
                if selected_rs_groups:
                    filtered_rs = relative_strength_data[relative_strength_data[group_by].isin(selected_rs_groups)]
                    
                    fig_selected_rs = go.Figure()
                    colors = px.colors.qualitative.Plotly
                    
                    for i, group in enumerate(selected_rs_groups):
                        group_data = filtered_rs[filtered_rs[group_by] == group]
                        fig_selected_rs.add_trace(go.Scatter(
                            x=group_data['fetch_date'],
                            y=group_data['cumulative_relative_strength'],
                            mode='lines',
                            name=group,
                            line=dict(color=colors[i % len(colors)], width=2),
                            hovertemplate=f'<b>{group}</b><br>Date: %{{x}}<br>Rel Strength: %{{y:.2f}}%<extra></extra>'
                        ))
                    
                    fig_selected_rs.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Market")
                    
                    fig_selected_rs.update_layout(
                        title=f"Selected {group_by.title()}s Relative Strength Comparison",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Relative Return (%)",
                        height=600,
                        hovermode='x unified',
                        xaxis=dict(showgrid=True, gridcolor='lightgray'),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='lightgray',
                            zeroline=True,
                            zerolinecolor='black',
                            zerolinewidth=2
                        ),
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01,
                            bgcolor="rgba(255, 255, 255, 0.95)",
                            bordercolor="black",
                            borderwidth=2,
                            font=dict(size=11, color="black")
                        )
                    )
                    
                    st.plotly_chart(fig_selected_rs, use_container_width=True)
    
    # Tab 5: Risk Analysis
    with tab5:
        st.header("⚠️ Risk Analysis: Volatility & Drawdowns")
        st.info("""
        **Volatility** = Risk/choppiness. **Drawdown** = % decline from peak. 
        High volatility + deep drawdowns = risky. Low volatility + shallow drawdowns = stable.
        """)
        
        if not current_risk_metrics.empty:
            # Risk metrics summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_vol = current_risk_metrics['volatility'].mean()
                st.metric("Avg Volatility", f"{avg_vol:.2f}%")
            
            with col2:
                in_drawdown = (current_risk_metrics['drawdown'] < 0).sum()
                total_groups = len(current_risk_metrics)
                st.metric("In Drawdown", f"{in_drawdown}/{total_groups}")
            
            with col3:
                worst_drawdown = current_risk_metrics['drawdown'].min()
                worst_dd_idx = current_risk_metrics['drawdown'].idxmin()
                worst_dd_group = current_risk_metrics.loc[worst_dd_idx, group_by]
                st.metric("Worst Current DD", f"{worst_drawdown:.1f}%", worst_dd_group)
            
            with col4:
                max_days_underwater = current_risk_metrics['days_underwater'].max()
                st.metric("Max Days Underwater", int(max_days_underwater))
            
            # Get number of groups
            num_groups = len(current_risk_metrics)
            
            # Risk metrics table
            st.subheader("📋 Current Risk Metrics")
            if num_groups > 30:
                st.caption(f"Showing top 20 highest risk and 20 lowest risk {group_by}s")
                # Sort by volatility and show top/bottom
                sorted_by_vol = current_risk_metrics.sort_values('volatility', ascending=False)
                display_risk = pd.concat([sorted_by_vol.head(20), sorted_by_vol.tail(20)])
            else:
                display_risk = current_risk_metrics
            
            risk_table = create_risk_metrics_table(display_risk, group_by)
            if risk_table is not None:
                # Use container to limit height visually
                st.dataframe(
                    risk_table.style.background_gradient(
                        subset=['Current Volatility (%)'], cmap='YlOrRd'
                    ),
                    use_container_width=True
                )
        
        if not volatility_data.empty:
            num_groups = len(volatility_data[group_by].unique())
            
            # Volatility chart
            st.subheader("📉 Rolling Volatility")
            if num_groups > 20:
                st.caption(f"Showing aggregate statistics across {num_groups} {group_by}s")
            fig_vol = create_volatility_chart(volatility_data, group_by)
            if fig_vol:
                st.plotly_chart(fig_vol, use_container_width=True)
            
            # Drawdown chart
            st.subheader("📊 Drawdowns from Peak")
            if num_groups > 20:
                st.caption(f"Showing aggregate statistics across {num_groups} {group_by}s")
            fig_dd = create_drawdown_chart(volatility_data, group_by)
            if fig_dd:
                st.plotly_chart(fig_dd, use_container_width=True)
            
            # Group selector for detailed view
            if num_groups > 20:
                st.subheader("🔍 Compare Specific Groups")
                selected_risk_groups = st.multiselect(
                    f"Select {group_by}s to compare (max 10)",
                    options=sorted(volatility_data[group_by].unique()),
                    max_selections=10,
                    help=f"Select up to 10 {group_by}s to view their individual volatility and drawdown",
                    key="risk_selector"
                )
                
                if selected_risk_groups:
                    filtered_vol = volatility_data[volatility_data[group_by].isin(selected_risk_groups)]
                    
                    # Volatility comparison
                    fig_selected_vol = go.Figure()
                    colors = px.colors.qualitative.Plotly
                    
                    for i, group in enumerate(selected_risk_groups):
                        group_data = filtered_vol[filtered_vol[group_by] == group]
                        fig_selected_vol.add_trace(go.Scatter(
                            x=group_data['fetch_date'],
                            y=group_data['volatility'],
                            mode='lines',
                            name=group,
                            line=dict(color=colors[i % len(colors)], width=2),
                            hovertemplate=f'<b>{group}</b><br>Date: %{{x}}<br>Volatility: %{{y:.2f}}%<extra></extra>'
                        ))
                    
                    fig_selected_vol.update_layout(
                        title=f"Selected {group_by.title()}s Volatility Comparison",
                        xaxis_title="Date",
                        yaxis_title="Volatility (%)",
                        height=500,
                        hovermode='x unified',
                        xaxis=dict(showgrid=True, gridcolor='lightgray'),
                        yaxis=dict(showgrid=True, gridcolor='lightgray')
                    )
                    
                    st.plotly_chart(fig_selected_vol, use_container_width=True)
                    
                    # Drawdown comparison
                    fig_selected_dd = go.Figure()
                    
                    for i, group in enumerate(selected_risk_groups):
                        group_data = filtered_vol[filtered_vol[group_by] == group]
                        fig_selected_dd.add_trace(go.Scatter(
                            x=group_data['fetch_date'],
                            y=group_data['drawdown'],
                            mode='lines',
                            name=group,
                            line=dict(color=colors[i % len(colors)], width=2),
                            fill='tozeroy',
                            fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(colors[i % len(colors)])) + [0.2])}',
                            hovertemplate=f'<b>{group}</b><br>Date: %{{x}}<br>Drawdown: %{{y:.2f}}%<extra></extra>'
                        ))
                    
                    fig_selected_dd.update_layout(
                        title=f"Selected {group_by.title()}s Drawdown Comparison",
                        xaxis_title="Date",
                        yaxis_title="Drawdown (%)",
                        height=500,
                        hovermode='x unified',
                        xaxis=dict(showgrid=True, gridcolor='lightgray'),
                        yaxis=dict(showgrid=True, gridcolor='lightgray')
                    )
                    
                    st.plotly_chart(fig_selected_dd, use_container_width=True)
    
    # Tab 6: Mean Reversion
    with tab6:
        st.header("🎲 Mean Reversion Analysis")
        st.info("""
        **Mean Reversion** identifies industries that have moved too far from their average, presenting potential reversal opportunities.
        Z-scores > 2 = Overbought, Z-scores < -2 = Oversold.
        """)
        
        if not current_signals.empty and enable_mean_reversion:
            # Signal summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                overbought = (current_signals['signal'] == 'Overbought').sum()
                st.metric("Overbought", overbought, help="Z-score > 2")
            
            with col2:
                oversold = (current_signals['signal'] == 'Oversold').sum()
                st.metric("Oversold", oversold, help="Z-score < -2")
            
            with col3:
                extreme_zscore = current_signals['zscore'].abs().idxmax()
                extreme_group = current_signals.loc[extreme_zscore, group_by]
                extreme_value = current_signals.loc[extreme_zscore, 'zscore']
                st.metric("Most Extreme", extreme_group, f"{extreme_value:.2f}σ")
            
            with col4:
                avg_abs_zscore = current_signals['zscore'].abs().mean()
                st.metric("Avg |Z-Score|", f"{avg_abs_zscore:.2f}σ")
            
            # Current signals table
            st.subheader("📋 Current Mean Reversion Signals")
            
            # Filter to show only interesting signals
            interesting_signals = current_signals[
                (current_signals['zscore'].abs() > 1.5) | 
                (current_signals['signal'] != 'Neutral')
            ].copy()
            
            if not interesting_signals.empty:
                display_signals = interesting_signals[[
                    group_by, 'zscore', 'bb_position', 'signal'
                ]].sort_values('zscore', ascending=False)
                
                display_signals.columns = [
                    group_by.title(),
                    'Z-Score',
                    'BB Position (%)',
                    'Signal'
                ]
                
                st.dataframe(
                    display_signals[[group_by.title(), 'Z-Score', 'Signal']].style.format({
                        'Z-Score': '{:.2f}'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No significant mean reversion signals currently")
            
            # Z-score trends
            if not mean_reversion_data.empty:
                st.subheader("📈 Z-Score Trends")
                num_groups = len(mean_reversion_data[group_by].unique())
                if num_groups > 20:
                    st.caption(f"Showing aggregate statistics across {num_groups} {group_by}s")
                
                fig_zscore = create_zscore_chart(mean_reversion_data, group_by)
                if fig_zscore:
                    st.plotly_chart(fig_zscore, use_container_width=True)
        elif not enable_mean_reversion:
            st.warning("Mean reversion analysis is disabled. Enable it in the sidebar.")
    
    # Tab 7: Concentration
    with tab7:
        st.header("🏢 Market Cap Concentration Analysis")
        st.info("""
        **Concentration** shows how market cap is distributed within each industry. 
        High concentration = dominated by few stocks. Low concentration = more evenly distributed.
        """)
        
        if not concentration_data.empty and enable_concentration:
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_top10 = concentration_data['top_10pct_contribution'].mean()
                st.metric("Avg Top 10% Contribution", f"{avg_top10:.1f}%")
            
            with col2:
                high_concentration = (concentration_data['concentration_level'] == 'High').sum()
                total = len(concentration_data)
                st.metric("High Concentration", f"{high_concentration}/{total}")
            
            with col3:
                most_concentrated_idx = concentration_data['herfindahl_index'].idxmax()
                most_concentrated = concentration_data.loc[most_concentrated_idx, group_by]
                most_concentrated_hhi = concentration_data.loc[most_concentrated_idx, 'herfindahl_index']
                st.metric("Most Concentrated", most_concentrated, f"HHI: {most_concentrated_hhi:.0f}")
            
            with col4:
                avg_large_cap = concentration_data['large_cap_contribution'].mean()
                st.metric("Avg Large Cap %", f"{avg_large_cap:.1f}%")
            
            # Concentration chart
            st.subheader("📊 Top 10% Market Cap Contribution")
            fig_concentration = create_concentration_chart(concentration_data, group_by)
            if fig_concentration:
                st.plotly_chart(fig_concentration, use_container_width=True)
            
            # Detailed table
            if st.checkbox("Show detailed concentration data"):
                st.subheader("📋 Concentration Metrics")
                
                display_conc = concentration_data[[
                    group_by, 'total_stocks', 'top_10pct_contribution',
                    'top_stock_pct', 'herfindahl_index', 'concentration_level',
                    'large_cap_contribution', 'small_cap_contribution'
                ]].sort_values('herfindahl_index', ascending=False)
                
                display_conc.columns = [
                    group_by.title(), '# Stocks', 'Top 10% %', 'Top Stock %',
                    'HHI', 'Level', 'Large Cap %', 'Small Cap %'
                ]
                
                st.dataframe(
                    display_conc.style.format({
                        'Top 10% %': '{:.1f}%',
                        'Top Stock %': '{:.1f}%',
                        'HHI': '{:.0f}',
                        'Large Cap %': '{:.1f}%',
                        'Small Cap %': '{:.1f}%'
                    }),
                    use_container_width=True
                )
        elif not enable_concentration:
            st.warning("Concentration analysis is disabled. Enable it in the sidebar.")
    
    # Tab 8: Rotation
    with tab8:
        st.header("🔄 Sector Rotation Analysis")
        st.info("""
        **Rotation Analysis** classifies industries into lifecycle stages based on momentum and relative strength:
        - **Leading (Growth)**: High momentum + Outperforming → Strong uptrend
        - **Weakening (Mature)**: Low momentum + Outperforming → Topping out
        - **Lagging (Decline)**: Low momentum + Underperforming → Downtrend
        - **Improving (Early)**: High momentum + Underperforming → Early reversal
        """)
        
        if not rotation_data.empty and enable_rotation:
            # Stage distribution
            stage_counts = rotation_data['stage'].value_counts()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                leading = stage_counts.get('Leading (Growth)', 0)
                st.metric("Leading (Growth)", leading)
            
            with col2:
                weakening = stage_counts.get('Weakening (Mature)', 0)
                st.metric("Weakening (Mature)", weakening)
            
            with col3:
                lagging = stage_counts.get('Lagging (Decline)', 0)
                st.metric("Lagging (Decline)", lagging)
            
            with col4:
                improving = stage_counts.get('Improving (Early)', 0)
                st.metric("Improving (Early)", improving)
            
            # Rotation matrix
            st.subheader("📊 Rotation Matrix")
            fig_rotation = create_rotation_matrix(rotation_data, group_by)
            if fig_rotation:
                st.plotly_chart(fig_rotation, use_container_width=True)
            
            # Stage breakdown tables
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌟 Leaders & Improvers")
                leaders = rotation_data[rotation_data['stage'].isin(['Leading (Growth)', 'Improving (Early)'])]
                if not leaders.empty:
                    leaders_display = leaders[[group_by, 'stage', 'avg_momentum', 'cumulative_relative_strength']].sort_values(
                        'cumulative_relative_strength', ascending=False
                    )
                    leaders_display.columns = [group_by.title(), 'Stage', 'Momentum %', 'Rel Strength %']
                    st.dataframe(
                        leaders_display.style.format({
                            'Momentum %': '{:.2f}%',
                            'Rel Strength %': '{:.2f}%'
                        }),
                        use_container_width=True
                    )
            
            with col2:
                st.subheader("⚠️ Weakening & Laggards")
                laggards = rotation_data[rotation_data['stage'].isin(['Weakening (Mature)', 'Lagging (Decline)'])]
                if not laggards.empty:
                    laggards_display = laggards[[group_by, 'stage', 'avg_momentum', 'cumulative_relative_strength']].sort_values(
                        'cumulative_relative_strength', ascending=True
                    )
                    laggards_display.columns = [group_by.title(), 'Stage', 'Momentum %', 'Rel Strength %']
                    st.dataframe(
                        laggards_display.style.format({
                            'Momentum %': '{:.2f}%',
                            'Rel Strength %': '{:.2f}%'
                        }),
                        use_container_width=True
                    )
        elif not enable_rotation:
            st.warning("Rotation analysis is disabled. Enable it in the sidebar.")
        else:
            st.warning("Rotation analysis requires both momentum and relative strength data.")
    
    # Tab 9: Streaks
    with tab9:
        st.header("🔥 Streak & Pattern Analysis")
        st.info("""
        **Streaks** track consecutive winning or losing days. 
        Long win streaks may indicate strong trends. Long loss streaks may indicate oversold conditions.
        """)
        
        if not streaks_data.empty and enable_streaks:
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                win_streaks = (streaks_data['current_streak'] > 0).sum()
                total = len(streaks_data)
                st.metric("Currently Winning", f"{win_streaks}/{total}")
            
            with col2:
                longest_win_idx = streaks_data['max_win_streak'].idxmax()
                longest_win_group = streaks_data.loc[longest_win_idx, group_by]
                longest_win = streaks_data.loc[longest_win_idx, 'max_win_streak']
                st.metric("Longest Win Streak", f"{longest_win} days", longest_win_group)
            
            with col3:
                longest_loss_idx = streaks_data['max_loss_streak'].idxmax()
                longest_loss_group = streaks_data.loc[longest_loss_idx, group_by]
                longest_loss = streaks_data.loc[longest_loss_idx, 'max_loss_streak']
                st.metric("Longest Loss Streak", f"{longest_loss} days", longest_loss_group)
            
            with col4:
                avg_win_rate = streaks_data['win_rate'].mean()
                st.metric("Avg Win Rate", f"{avg_win_rate:.1f}%")
            
            # Current streaks chart
            st.subheader("📊 Current Streaks")
            fig_streaks = create_streaks_chart(streaks_data, group_by)
            if fig_streaks:
                st.plotly_chart(fig_streaks, use_container_width=True)
            
            # Detailed streaks table
            if st.checkbox("Show detailed streak data"):
                st.subheader("📋 Streak Details")
                
                display_streaks = streaks_data[[
                    group_by, 'current_streak', 'current_streak_type',
                    'max_win_streak', 'max_loss_streak', 'win_rate'
                ]].sort_values('current_streak', ascending=False)
                
                display_streaks.columns = [
                    group_by.title(), 'Current Streak', 'Type',
                    'Max Win', 'Max Loss', 'Win Rate %'
                ]
                
                st.dataframe(
                    display_streaks.style.format({
                        'Win Rate %': '{:.1f}%'
                    }),
                    use_container_width=True
                )
        elif not enable_streaks:
            st.warning("Streak analysis is disabled. Enable it in the sidebar.")
    
    # Tab 10: Heatmaps (formerly Tab 6)
    with tab10:
        st.header("🔥 Daily Direction Heatmap")
        fig_heatmap = create_heatmap(daily_changes, group_by, display_start_date)
        if fig_heatmap:
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Optional detailed data
    if st.checkbox("Show detailed data"):
        st.subheader("📋 Detailed Trend Data")
        st.dataframe(
            filtered_scores.sort_values(['date', group_by]),
            use_container_width=True
        )
    
    # Export functionality
    if st.button("💾 Export All Data"):
        export_date = datetime.now().strftime("%Y%m%d_%H%M")
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            filtered_scores.to_excel(writer, sheet_name='Trend_Scores', index=False)
            if not percent_changes.empty:
                percent_changes.to_excel(writer, sheet_name='Percent_Changes', index=False)
            if not momentum_data.empty:
                momentum_data.to_excel(writer, sheet_name='Momentum', index=False)
            if not momentum_trends_data.empty:
                momentum_trends_data.to_excel(writer, sheet_name='Momentum_Trends', index=False)
            if not breadth_data.empty:
                breadth_data.to_excel(writer, sheet_name='Breadth', index=False)
            if not relative_strength_data.empty:
                relative_strength_data.to_excel(writer, sheet_name='Relative_Strength', index=False)
            if not volatility_data.empty:
                volatility_data.to_excel(writer, sheet_name='Volatility_Drawdown', index=False)
            if not current_risk_metrics.empty:
                current_risk_metrics.to_excel(writer, sheet_name='Current_Risk_Metrics', index=False)
            if not current_signals.empty:
                current_signals.to_excel(writer, sheet_name='Mean_Reversion_Signals', index=False)
            if not concentration_data.empty:
                concentration_data.to_excel(writer, sheet_name='Concentration', index=False)
            if not rotation_data.empty:
                rotation_data.to_excel(writer, sheet_name='Rotation_Stages', index=False)
            if not streaks_data.empty:
                streaks_data.to_excel(writer, sheet_name='Streaks', index=False)
            daily_changes.to_excel(writer, sheet_name='Daily_Changes', index=False)
        
        output.seek(0)
        st.download_button(
            label="📥 Download Complete Analysis (Excel)",
            data=output,
            file_name=f"complete_valuation_analysis_{export_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()