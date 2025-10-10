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
st.set_page_config(layout="wide", page_title="Valuation Analysis", page_icon="📊")
st.title("📊 Valuation Analysis")

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
    # Current period: the most recent lookback_days of trading
    # For 3 days with data [Mon, Tue, Wed, Thu, Fri], current would be [Wed, Thu, Fri]
    latest_date = unique_dates[-1]  # Most recent trading day
    
    # Current period starts lookback_days ago (counting only trading days)
    # Index -lookback_days gives us the start of the current period
    # For 3 days: unique_dates[-3] is 3 trading days before the end
    current_start_idx = -lookback_days if lookback_days <= len(unique_dates) else 0
    current_start_date = unique_dates[current_start_idx]
    
    # Previous period ends exactly where current period starts
    # This is the trading day immediately before the current period
    previous_end_idx = current_start_idx - 1 if current_start_idx - 1 >= -len(unique_dates) else 0
    previous_end_date = unique_dates[previous_end_idx]
    
    # Previous period starts lookback_days trading days before it ends
    previous_start_idx = previous_end_idx - lookback_days + 1 if previous_end_idx - lookback_days + 1 >= -len(unique_dates) else 0
    previous_start_date = unique_dates[previous_start_idx]
    
    # Filter data for the relevant dates
    latest_data = daily_data[daily_data['fetch_date'] == latest_date]
    current_start_data = daily_data[daily_data['fetch_date'] == current_start_date]
    previous_end_data = daily_data[daily_data['fetch_date'] == previous_end_date]
    previous_start_data = daily_data[daily_data['fetch_date'] == previous_start_date]
    
    # Prepare the result dataframe
    percent_changes = []
    
    # Calculate percent change for ALL groups - ensure we process every single one
    unique_groups = daily_data[group_by].unique()
    
    # Calculate percent change for each group
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
def create_momentum_bar_chart(momentum_df, periods, group_by):
    """Create grouped bar chart showing momentum across different periods."""
    
    if momentum_df.empty:
        return None
    
    # Sort by average momentum
    momentum_sorted = momentum_df.sort_values('avg_momentum', ascending=True)
    
    fig = go.Figure()
    
    # Define colors for different periods
    colors = {
        3: 'rgb(255, 127, 14)',     # Orange
        5: 'rgb(99, 110, 250)',      # Blue
        10: 'rgb(239, 85, 59)',      # Red-Orange
        20: 'rgb(0, 204, 150)',      # Green
        30: 'rgb(171, 99, 250)',     # Purple
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
    
    fig = go.Figure()
    
    unique_groups = sorted(momentum_trends[group_by].unique())
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
    
    fig.update_layout(
        title=f"{momentum_period}-Day Momentum Trends by {group_by.title()}",
        xaxis_title="Date",
        yaxis_title=f"{momentum_period}-Day Rate of Change (%)",
        height=700,
        hovermode='x unified',
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2
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
    
    # Create hover data
    hover_data = defaultdict(list)
    for _, row in filtered_scores.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        score = row['score']
        group = row[group_by]
        key = (date_str, score)
        hover_data[key].append(group)
    
    # Create figure
    fig = go.Figure()
    
    # Get unique groups and colors
    unique_groups = sorted(filtered_scores[group_by].unique())
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
    
    # Update layout
    fig.update_layout(
        title=f"Trend Scores by {group_by.title()} (Reset at {score_start_date})",
        xaxis_title="Date",
        yaxis_title="Cumulative Score",
        height=800,
        hovermode='closest',
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='black')
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
            
            # Filter for display period
            filtered_scores = trend_scores[
                trend_scores['date'] >= pd.Timestamp(display_start_date)
            ].sort_values(['date', group_by])
            
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
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
    
    # Main visualizations
    st.header(f"📈 Trend Analysis by {group_by.title()}")
    st.info(f"Score calculation: +1 for up days, -1 for down days. Scores reset to 0 starting {score_start_date}")
    
    # Trend line chart
    if not filtered_scores.empty:
        fig_trends = create_trend_line_chart(filtered_scores, group_by, score_start_date)
        if fig_trends:
            st.plotly_chart(fig_trends, use_container_width=True)
    
    # Current scores
    st.header("📊 Current Scores")
    if not filtered_scores.empty:
        latest_date = filtered_scores['date'].max()
        current_data = filtered_scores[filtered_scores['date'] == latest_date]
        
        fig_current = create_current_scores_chart(current_data, group_by)
        if fig_current:
            st.plotly_chart(fig_current, use_container_width=True)
    
    # Percent changes
    if not percent_changes.empty:
        st.header(f"📊 Percent Change: Current vs Previous {percent_change_days} {'Day' if percent_change_days == 1 else 'Days'}")
        
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
        
        # Sort by current percent change (ascending for horizontal bar chart)
        # IMPORTANT: Don't filter or limit the data - show ALL groups
        pct_sorted = percent_changes.sort_values('current_percent_change', ascending=True)
        
        # Create the grouped bar chart using plotly graph objects
        fig_pct = go.Figure()
        
        # Add bars for current period - using a consistent blue color
        fig_pct.add_trace(go.Bar(
            name='Current Period',
            y=pct_sorted[group_by],
            x=pct_sorted['current_percent_change'],
            orientation='h',
            marker=dict(
                color='rgb(99, 110, 250)',  # Consistent blue color for all current period bars
                line=dict(color='rgb(69, 80, 220)', width=1)  # Darker blue border
            ),
            text=pct_sorted['current_percent_change'].apply(lambda x: f"{x:.1f}%"),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Current Period: %{x:.2f}%<extra></extra>'
        ))
        
        # Add bars for previous period - using a consistent gray color
        fig_pct.add_trace(go.Bar(
            name='Previous Period',
            y=pct_sorted[group_by],
            x=pct_sorted['previous_percent_change'],
            orientation='h',
            marker=dict(
                color='rgba(150, 150, 150, 0.6)',  # Consistent gray color with transparency
                line=dict(color='rgba(100, 100, 100, 0.8)', width=1)  # Darker gray border
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
            height=max(800, len(pct_sorted) * 20),  # Increased height to accommodate all groups
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
            # Count improvements
            improved = (pct_sorted['current_percent_change'] > 
                       pct_sorted['previous_percent_change']).sum()
            total = len(pct_sorted)
            improvement_pct = (improved / total * 100) if total > 0 else 0
            st.metric("Improved vs Previous", f"{improved}/{total}", f"{improvement_pct:.1f}%")
        
        # Show data table with both periods
        if st.checkbox("Show percent change data"):
            display_df = pct_sorted[[
                group_by, 
                'current_percent_change', 
                'previous_percent_change'
            ]].copy()
            
            # Calculate difference
            display_df['change_vs_previous'] = (
                display_df['current_percent_change'] - 
                display_df['previous_percent_change']
            )
            
            # Rename columns for clarity
            display_df.columns = [
                group_by.title(), 
                f'Current {percent_change_days}d %', 
                f'Previous {percent_change_days}d %',
                'Change vs Previous'
            ]
            
            # Sort by current period for better readability
            display_df = display_df.sort_values(f'Current {percent_change_days}d %', ascending=False)
            
            # Format as percentages
            for col in display_df.columns[1:]:
                display_df[col] = display_df[col].round(2)
            
            st.dataframe(
                display_df.style.format({
                    f'Current {percent_change_days}d %': '{:.2f}%',
                    f'Previous {percent_change_days}d %': '{:.2f}%',
                    'Change vs Previous': '{:+.2f}%'
                }).background_gradient(subset=[f'Current {percent_change_days}d %'], cmap='RdYlGn'),
                use_container_width=True
            )
    
    # Momentum Analysis Section
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
            # Find strongest momentum
            if not momentum_data.empty:
                strongest_idx = momentum_data['avg_momentum'].abs().idxmax()
                strongest_group = momentum_data.loc[strongest_idx, group_by]
                strongest_value = momentum_data.loc[strongest_idx, 'avg_momentum']
                st.metric("Strongest", strongest_group, f"{strongest_value:.2f}%")
        
        with col4:
            # Momentum consistency (lower is more consistent across periods)
            avg_consistency = momentum_data['momentum_consistency'].mean()
            st.metric("Avg Consistency (σ)", f"{avg_consistency:.2f}%")
        
        # Momentum trends over time
        if not momentum_trends_data.empty:
            st.subheader(f"📈 {momentum_trend_period}-Day Momentum Trends")
            
            # Line chart
            fig_momentum_trends = create_momentum_trend_lines(
                momentum_trends_data, 
                group_by, 
                momentum_trend_period
            )
            if fig_momentum_trends:
                st.plotly_chart(fig_momentum_trends, use_container_width=True)
            
            # Heatmap
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
            
            # Prepare display dataframe
            display_cols = [group_by] + [f'momentum_{p}d' for p in sorted(momentum_periods)] + ['avg_momentum', 'momentum_consistency']
            momentum_display = momentum_data[display_cols].copy()
            
            # Rename columns
            rename_dict = {f'momentum_{p}d': f'{p}d %' for p in momentum_periods}
            rename_dict['avg_momentum'] = 'Avg %'
            rename_dict['momentum_consistency'] = 'Consistency (σ)'
            momentum_display = momentum_display.rename(columns=rename_dict)
            
            # Sort by average momentum
            momentum_display = momentum_display.sort_values('Avg %', ascending=False)
            
            # Format percentages
            pct_cols = [col for col in momentum_display.columns if col.endswith('%') or col == 'Consistency (σ)']
            for col in pct_cols:
                momentum_display[col] = momentum_display[col].round(2)
            
            # Display with styling
            st.dataframe(
                momentum_display.style.format({
                    col: '{:.2f}%' for col in pct_cols
                }).background_gradient(subset=['Avg %'], cmap='RdYlGn'),
                use_container_width=True
            )
    
    # Heatmap
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
    if st.button("💾 Export Data"):
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
            daily_changes.to_excel(writer, sheet_name='Daily_Changes', index=False)
        
        output.seek(0)
        st.download_button(
            label="📥 Download Excel File",
            data=output,
            file_name=f"valuation_analysis_{export_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()