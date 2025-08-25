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
    
    # Get unique dates sorted
    unique_dates = sorted(daily_data['fetch_date'].unique())
    all_groups = daily_data[group_by].unique()
    
    # We need at least 2 * lookback_days + 1 dates for both current and previous periods
    min_required_dates = 2 * lookback_days + 1
    if len(unique_dates) < min_required_dates:
        # Adjust lookback if we don't have enough data
        max_possible_lookback = (len(unique_dates) - 1) // 2
        if max_possible_lookback < 1:
            st.warning("Not enough dates for percent change calculation with previous period")
            return pd.DataFrame()
        st.warning(f"Adjusted lookback period from {lookback_days} to {max_possible_lookback} days due to data availability")
        lookback_days = max_possible_lookback
    
    # Calculate the date indices - FIXED: Correct indexing for 5-day periods
    # For 5 days: we want days at indices -5, -4, -3, -2, -1 (last 5 days)
    # Previous 5 days: indices -10, -9, -8, -7, -6
    
    latest_date = unique_dates[-1]  # This should be Aug 22
    
    # Current period: last lookback_days of data
    # For 5 days, this gets index -6 (which is 5 days before the last day, so Aug 18)
    current_start_date = unique_dates[-lookback_days] if lookback_days <= len(unique_dates) else unique_dates[0]
    
    # Previous period ends one day before current period starts
    # For 5 days, this gets index -6 (which would be Aug 15)
    previous_end_date = unique_dates[-(lookback_days + 1)] if lookback_days + 1 <= len(unique_dates) else unique_dates[0]
    
    # Previous period starts lookback_days before it ends
    # For 5 days, this gets index -10 (which would be Aug 11)
    previous_start_date = unique_dates[-(2 * lookback_days)] if 2 * lookback_days <= len(unique_dates) else unique_dates[0]
    
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
        st.info("📁 Current directory contents:")
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
    
    # Process data
    with st.spinner("Processing data..."):
        try:
            # Calculate daily changes
            daily_changes = calculate_daily_changes(market_caps_df, sectors_df, group_by)
            
            # Calculate trend scores
            trend_scores = calculate_trend_scores(daily_changes, score_start_date, group_by)
            
            # Calculate percent changes
            percent_changes = calculate_percent_changes(daily_changes, percent_change_days, group_by)
            
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
            title=f"Percent Change by {group_by.title()} - Comparing {percent_change_days}-Day Periods",
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