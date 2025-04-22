import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO
import glob  # For finding files

# Set page configuration to wide layout
st.set_page_config(layout="wide")  # Use wide layout for better use of screen space
st.title("Valuation Analysis")

# ------------------------------
# 1. Data Loading Functions
# ------------------------------
@st.cache_data
def load_sectors_file(file_path):
    """Load the sectors file from the data directory."""
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        st.error(f"Error loading stock_sectors.xlsx: {str(e)}")
        st.stop()

@st.cache_data
def load_market_caps_file(file_path):
    """Load the market caps file from the data directory."""
    try:
        df = pd.read_excel(file_path)
        
        # Check if 'fetch_date' column exists
        if 'fetch_date' not in df.columns:
            st.error("market_caps.xlsx does not have a 'fetch_date' column.")
            st.stop()
        
        # Handle multiple date formats by converting to datetime with flexible parsing
        df['fetch_date'] = pd.to_datetime(df['fetch_date'], errors='coerce', infer_datetime_format=True)
        
        # Check if all fetch_date values are invalid
        if df['fetch_date'].isna().all():
            st.error("All 'fetch_date' values in market_caps.xlsx are invalid.")
            st.stop()
        
        # List unique dates found in the data
        unique_dates = df['fetch_date'].dt.date.unique()
        unique_dates_sorted = sorted(unique_dates)
        
        # Display date range info
        st.sidebar.write("Date range found in data:", 
                        min(unique_dates_sorted).strftime('%Y-%m-%d') if len(unique_dates_sorted) > 0 else "No valid dates", 
                        "to", 
                        max(unique_dates_sorted).strftime('%Y-%m-%d') if len(unique_dates_sorted) > 0 else "No valid dates")
        
        # Count unique dates
        st.sidebar.write(f"Number of unique dates found: {len(unique_dates_sorted)}")
        
        # Display the list of dates found
        st.sidebar.write("Dates found:", ", ".join([d.strftime('%Y-%m-%d') for d in unique_dates_sorted]))
        
        return df
    except Exception as e:
        st.error(f"Error loading market_caps.xlsx: {str(e)}")
        st.stop()

# ------------------------------
# 2. Load Data
# ------------------------------
# Define file paths for GitHub repository structure
# From the screenshot, I can see the data files are in the /Data folder
# and there's also a nested data folder under stock_scores
# Let's try multiple possible paths

# Try different possible file paths based on the GitHub structure
possible_data_paths = [
    "Data",                      # Root Data folder
    "data",                      # Lowercase variant 
    "stock_scores",              # From the folder structure in your screenshot
    os.path.join("Data", "stock_scores"),  # Nested path
    os.path.join("data", "stock_scores"),  # Lowercase nested path
    "."                          # Current directory
]

# Initialize with default paths
sectors_file_path = "stock_sectors.xlsx"
market_caps_file_path = "market_caps.xlsx"

# Try to find the data files
found_sectors = False
found_market_caps = False

# Function to check for file existence in different folders
def find_file(filename):
    # First try direct path
    if os.path.exists(filename):
        return filename
    
    # Try in each possible data directory
    for data_path in possible_data_paths:
        test_path = os.path.join(data_path, filename)
        if os.path.exists(test_path):
            return test_path
    
    # Try one level up (parent directory)
    parent_path = os.path.join("..", filename)
    if os.path.exists(parent_path):
        return parent_path
    
    # Try in Data folder one level up
    parent_data_path = os.path.join("..", "Data", filename)
    if os.path.exists(parent_data_path):
        return parent_data_path
    
    return None

# Try to find the files
sectors_file_path = find_file("stock_sectors.xlsx")
market_caps_file_path = find_file("market_caps.xlsx")

# Check if files exist
if not sectors_file_path:
    st.error("Could not find stock_sectors.xlsx in any of the expected locations")
    st.error(f"Current working directory: {os.getcwd()}")
    st.error(f"Files in current directory: {os.listdir('.')}")
    # Try to find any xlsx files as a last resort
    xlsx_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.xlsx'):
                xlsx_files.append(os.path.join(root, file))
    
    if xlsx_files:
        st.info(f"Found these Excel files in the repository: {xlsx_files}")
    
    st.info("For Streamlit Cloud deployment, please make sure the Excel files are committed to your repository.")
    st.stop()

if not market_caps_file_path:
    st.error("Could not find market_caps.xlsx in any of the expected locations")
    st.error(f"Current working directory: {os.getcwd()}")
    st.error(f"Files in current directory: {os.listdir('.')}")
    st.info("For Streamlit Cloud deployment, please make sure the Excel files are committed to your repository.")
    st.stop()

# Success messages to help with troubleshooting
st.sidebar.info(f"Found sectors file at: {sectors_file_path}")
st.sidebar.info(f"Found market caps file at: {market_caps_file_path}")

# Load data from files
st.sidebar.info("Loading data from files...")
sectors_df = load_sectors_file(sectors_file_path)
market_caps_df = load_market_caps_file(market_caps_file_path)
st.sidebar.success("Data loaded successfully!")

# ------------------------------
# 3. User Inputs
# ------------------------------
st.sidebar.header("Analysis Settings")

# Grouping selection: sector or industry
group_by = st.sidebar.selectbox("Group By", options=["sector", "industry"])

# No need for selection - we'll show all
unique_groups = sectors_df[group_by].unique()
st.sidebar.write(f"Displaying all {len(unique_groups)} {group_by}s")
selected_group = unique_groups  # Use all groups by default

# Display date range
unique_dates = sorted(market_caps_df['fetch_date'].dt.date.unique())
if len(unique_dates) > 0:
    min_date = min(unique_dates)
    max_date = max(unique_dates)
    
    # How many days to display
    max_days = len(unique_dates)
    days_to_display = st.sidebar.slider(
        "Days to display", 
        min_value=1, 
        max_value=max_days, 
        value=min(max_days, 20)
    )
    
    # Calculate display start date based on number of days
    if days_to_display < max_days:
        display_start_date = unique_dates[-days_to_display]
    else:
        display_start_date = min_date
    
    # New: Add Date Picker for Score Calculation Start Date
    st.sidebar.subheader("Score Calculation Settings")
    score_start_date = st.sidebar.date_input(
        "Score Calculation Start Date",
        value=display_start_date,  # Default to display start date
        min_value=min_date,
        max_value=max_date
    )
    
    # New: Add description about score reset
    st.sidebar.info("Selecting a start date will reset the score calculation to zero as of that date. This helps analyze recent performance trends without historical bias.")
    
    # New: Percent change lookback period
    percent_change_days = st.sidebar.slider(
        "Percent change lookback period (days)",
        min_value=1,
        max_value=max_days,
        value=min(max_days, 5)
    )
else:
    st.error("No valid dates found in the data.")
    st.stop()

# ------------------------------
# 4. Process Data
# ------------------------------
@st.cache_data
def calculate_daily_changes(market_caps_df, sectors_df, group_by):
    """Calculate daily changes in market cap by sector/industry"""
    # Debugging: Show unique dates before processing
    unique_dates_before = sorted(market_caps_df['fetch_date'].dt.date.unique())
    st.sidebar.write(f"Processing {len(unique_dates_before)} dates: {', '.join([d.strftime('%Y-%m-%d') for d in unique_dates_before[:5]]) + ('...' if len(unique_dates_before) > 5 else '')}")
    
    # Get all symbols from the sectors dataframe
    all_symbols = sectors_df['symbol'].unique()
    
    # Filter market caps to only include symbols in the sectors dataframe
    market_caps_filtered = market_caps_df[market_caps_df['symbol'].isin(all_symbols)]
    
    # Create a cross-join of all symbols with all dates to ensure complete data
    all_dates = market_caps_filtered['fetch_date'].dt.date.unique()
    
    # Merge with sectors data
    merged_df = pd.merge(market_caps_filtered, sectors_df[['symbol', 'sector', 'industry']], on='symbol', how='inner')
    
    # Display counts per date after merging
    date_counts = merged_df.groupby(merged_df['fetch_date'].dt.date).size()
    for date, count in list(date_counts.items())[:5]:  # Show only first 5 for brevity
        st.sidebar.write(f"After merging - {date}: {count} entries")
    if len(date_counts) > 5:
        st.sidebar.write("...")
    
    # Convert market cap to numeric, force type to float to handle scientific notation
    merged_df['market_cap'] = pd.to_numeric(merged_df['market_cap'], errors='coerce')
    
    # Sort by date and symbol
    merged_df = merged_df.sort_values(['symbol', 'fetch_date'])
    
    # Calculate day-to-day market cap change for each symbol
    merged_df['prev_market_cap'] = merged_df.groupby('symbol')['market_cap'].shift(1)
    merged_df['daily_change'] = merged_df['market_cap'] - merged_df['prev_market_cap']
    
    # For the first date, we need to set a baseline (no previous data to compare to)
    first_date = merged_df['fetch_date'].min().date()
    
    # Group by date and sector/industry - using sum() for aggregation
    daily_group = merged_df.groupby(['fetch_date', group_by])['daily_change'].sum().reset_index()
    
    # Calculate total market cap per group per date for percentage calculations
    total_market_cap = merged_df.groupby(['fetch_date', group_by])['market_cap'].sum().reset_index()
    daily_group = pd.merge(daily_group, total_market_cap, on=['fetch_date', group_by], how='left')
    
    # Determine direction (1 for up, -1 for down, 0 for no change)
    daily_group['direction'] = daily_group['daily_change'].apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )
    
    # Set a default direction (0) for the first date since we don't have prior data
    # This is important to handle the first day correctly in trend calculations
    daily_group.loc[daily_group['fetch_date'].dt.date == first_date, 'direction'] = 0
    daily_group.loc[daily_group['fetch_date'].dt.date == first_date, 'daily_change'] = 0
    
    # Debugging: Show unique dates after processing
    unique_dates_after = sorted(daily_group['fetch_date'].dt.date.unique())
    st.sidebar.write(f"After processing: {len(unique_dates_after)} dates: {', '.join([d.strftime('%Y-%m-%d') for d in unique_dates_after[:5]]) + ('...' if len(unique_dates_after) > 5 else '')}")
    
    return daily_group

@st.cache_data
def calculate_trend_score(daily_changes, score_start_date):
    """Calculate trend scores with simple cumulative scoring - NO PENALTY SYSTEM"""
    # Sort daily changes by date to ensure chronological processing
    daily_changes = daily_changes.sort_values('fetch_date')
    
    # Get unique dates and groups
    all_dates = sorted(daily_changes['fetch_date'].unique())
    groups = daily_changes[group_by].unique()
    
    # Initialize result structure
    results = []
    
    # Process each group
    for group in groups:
        # Get data for this group and sort by date
        group_data = daily_changes[daily_changes[group_by] == group].sort_values('fetch_date')
        
        # Create a dataframe with all dates for this group
        date_df = pd.DataFrame({'fetch_date': all_dates})
        group_df = pd.merge(date_df, group_data, on='fetch_date', how='left')
        group_df = group_df.fillna({group_by: group, 'direction': 0, 'daily_change': 0})
        group_df = group_df.sort_values('fetch_date')
        
        # Initialize variables for this group - reset at score_start_date
        score = 0
        streak = 0  # Positive for up streak, negative for down streak
        prev_direction = 0
        
        # Process each day for this group
        for _, row in group_df.iterrows():
            date = row['fetch_date']
            direction = row['direction']
            
            # Reset score and streak if we're at or past the score_start_date
            if date.date() == score_start_date:
                score = 0
                streak = 0
                prev_direction = 0
            
            # Skip dates before score_start_date
            if date.date() < score_start_date:
                continue
                
            # Update streak count (for display purposes only - no penalties applied)
            if direction == 0:
                # No change in direction means no change in streak
                pass
            elif direction == prev_direction:
                # Continuing in same direction
                streak += direction  # Will be positive for up streak, negative for down streak
            else:
                # Direction changed or starting from zero
                streak = direction
            
            # Simple cumulative score - just add the direction value
            if direction != 0:
                score += direction
                prev_direction = direction  # Only update prev_direction if we have a non-zero direction
            
            # Save the current state
            results.append({
                'date': date,
                group_by: group,
                'score': score,
                'streak': streak,
                'direction': direction,
                'market_cap': row.get('market_cap', 0)  # Add market cap for percentage calculations
            })
    
    return pd.DataFrame(results)

@st.cache_data
def calculate_percent_changes(daily_group_data, lookback_days):
    """Calculate percent changes for each group over specified lookback period"""
    # Make sure we have the market_cap column
    if 'market_cap' not in daily_group_data.columns:
        st.error("Market cap data not available for percent change calculation")
        return pd.DataFrame()
    
    # Sort by date
    daily_group_data = daily_group_data.sort_values('fetch_date')
    
    # Get unique dates and groups
    all_dates = sorted(daily_group_data['fetch_date'].unique())
    groups = daily_group_data[group_by].unique()
    
    # If we don't have enough dates for the lookback, adjust
    if len(all_dates) <= lookback_days:
        lookback_days = len(all_dates) - 1
        if lookback_days < 1:
            st.warning("Not enough dates for percent change calculation")
            return pd.DataFrame()
    
    # Get the latest date and the reference date (lookback days ago)
    latest_date = all_dates[-1]
    reference_date = all_dates[-lookback_days-1] if lookback_days < len(all_dates) else all_dates[0]
    
    # Filter data for the two dates
    latest_data = daily_group_data[daily_group_data['fetch_date'] == latest_date]
    reference_data = daily_group_data[daily_group_data['fetch_date'] == reference_date]
    
    # Prepare the result dataframe
    percent_changes = []
    
    # Calculate percent change for each group
    for group in groups:
        latest_group = latest_data[latest_data[group_by] == group]
        reference_group = reference_data[reference_data[group_by] == group]
        
        if not latest_group.empty and not reference_group.empty:
            latest_value = latest_group['market_cap'].iloc[0]
            reference_value = reference_group['market_cap'].iloc[0]
            
            if reference_value > 0:  # Avoid division by zero
                percent_change = ((latest_value - reference_value) / reference_value) * 100
            else:
                percent_change = 0  # Default if reference is zero
                
            percent_changes.append({
                group_by: group,
                'percent_change': percent_change,
                'latest_value': latest_value,
                'reference_value': reference_value,
                'latest_date': latest_date,
                'reference_date': reference_date
            })
    
    return pd.DataFrame(percent_changes)

# Calculate
st.sidebar.write("Starting data processing...")
daily_changes = calculate_daily_changes(market_caps_df, sectors_df, group_by)
trend_scores = calculate_trend_score(daily_changes, score_start_date)  # Pass score_start_date to the function
st.sidebar.write("Data processing complete.")

# Calculate percent changes
percent_changes = calculate_percent_changes(daily_changes, percent_change_days)

# Filter for time range (all groups are included by default)
filtered_scores = trend_scores[
    trend_scores['date'] >= pd.Timestamp(display_start_date)
].sort_values(['date', group_by])

# ------------------------------
# 5. Visualize Results
# ------------------------------
st.header(f"Valuation Trend Scores by {group_by.capitalize()}")
st.write(f"For each day the combined market cap goes up a +1 is given.  Each day the combined market cap goes down a score of -1 is given. Score calculation starts from: **{score_start_date.strftime('%Y-%m-%d')}**")

# Create a line chart with improved hover showing all industries at the same data point
import plotly.graph_objects as go
from collections import defaultdict

# Helper function to create hover text showing all industries with the same score on a date
def create_grouped_hover_data(data):
    # Group by date and score
    grouped_data = defaultdict(list)
    for _, row in data.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        score = row['score']
        group = row[group_by]
        key = (date_str, score)
        grouped_data[key].append(group)
    
    # Create a lookup dictionary for hover text
    hover_lookup = {}
    for (date_str, score), groups in grouped_data.items():
        for group in groups:
            hover_lookup[(date_str, score, group)] = {
                'date': date_str,
                'score': score,
                'group': group,
                'all_groups': groups,
                'count': len(groups)
            }
    
    return hover_lookup

# Create the hover lookup data
hover_data = create_grouped_hover_data(filtered_scores)

# Create a figure with go.Scatter for more control
fig = go.Figure()

# Create a sorted list of unique groups for consistent colors
unique_sorted_groups = sorted(filtered_scores[group_by].unique())

# Create a colormap 
colormap = px.colors.qualitative.Plotly  # Using Plotly's default color scheme
color_dict = {group: colormap[i % len(colormap)] for i, group in enumerate(unique_sorted_groups)}

# Add each group as a separate trace
for group in unique_sorted_groups:
    group_data = filtered_scores[filtered_scores[group_by] == group]
    
    # Custom hover text showing all industries with same score on same date
    hover_texts = []
    for _, row in group_data.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        score = row['score']
        key = (date_str, score, group)
        
        if key in hover_data:
            info = hover_data[key]
            # Create hover text with all groups that have the same score on this date
            if info['count'] > 1:
                other_groups = [g for g in info['all_groups'] if g != group]
                hover_text = f"<b>{group}</b><br>Date: {date_str}<br>Score: {score}<br><br>Also at this score:<br>"
                hover_text += "<br>".join(other_groups)
                hover_text += f"<br><br>Total: {info['count']} industries"
            else:
                hover_text = f"<b>{group}</b><br>Date: {date_str}<br>Score: {score}<br><br>No other industries at this score"
            
            hover_texts.append(hover_text)
    
    # Add the trace for this group
    fig.add_trace(go.Scatter(
        x=group_data['date'],
        y=group_data['score'],
        mode='lines',
        name=group,
        line=dict(color=color_dict[group], width=4),
        hoverinfo='text',
        hovertext=hover_texts,
        hoverlabel=dict(
            bgcolor='rgba(0, 0, 0, 0.8)',  # Dark background with opacity
            font=dict(color='white', size=14),  # White text
            bordercolor='white',  # White border
            align='left'  # Left-aligned text
        ),
    ))

# Update layout
fig.update_layout(
    title=f"Valuation Trend Barometer Scores Over Time (Reset to 0 at {score_start_date.strftime('%Y-%m-%d')})",
    xaxis_title="Date",
    yaxis_title="Cumulative Score",
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
    hovermode='closest',
    showlegend=False,
    height=800
)

# Add hover points to increase hover reliability
for group in unique_sorted_groups:
    group_data = filtered_scores[filtered_scores[group_by] == group]
    
    # Add invisible markers at each data point to improve hover detection
    fig.add_trace(go.Scatter(
        x=group_data['date'],
        y=group_data['score'],
        mode='markers',
        marker=dict(
            size=10,
            opacity=0,  # Invisible markers
            color=color_dict[group]
        ),
        hoverinfo='skip',  # Skip hover for these points
        showlegend=False,
        name=f"{group} (markers)"
    ))

st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Current Scores Visualization
# ------------------------------
st.header(f"Trend-O-Meter")

# Get the latest date
latest_date = filtered_scores['date'].max()
st.write(f"Latest date for data: {latest_date.strftime('%Y-%m-%d')}")

# Get current scores
current_data = filtered_scores[filtered_scores['date'] == latest_date].copy()
st.write(f"Number of groups with data: {len(current_data)}")

# Filter out zero scores since we don't want to display them
current_data_nonzero = current_data[current_data['score'] != 0].copy()
st.write(f"Number of groups with non-zero scores: {len(current_data_nonzero)}")

# Sort by absolute score value
current_data_nonzero = current_data_nonzero.sort_values('score', key=abs, ascending=False)

# Create colorscale that works well with positive/negative scores
colorscale = [
    [0, 'red'],      # Negative scores
    [0.4, 'lightcoral'],
    [0.5, 'white'],  # Zero (neutral - though we don't display these)
    [0.6, 'lightblue'],
    [1, 'blue']      # Positive scores
]

# Only create the chart if we have non-zero scores
if len(current_data_nonzero) > 0:
    # Create the bar chart using the actual score data
    fig2 = px.bar(
        current_data_nonzero, 
        x=group_by, 
        y='score',
        color='score',
        color_continuous_scale=colorscale,
        title=f"Current Scores by {group_by.capitalize()} (Since {score_start_date.strftime('%Y-%m-%d')})",
        labels={'score': 'Score'},
        height=600
    )

    # Add grid for better readability, rotate labels if many industries
    fig2.update_layout(
        xaxis_title=group_by.capitalize(), 
        yaxis_title="Score (+ Up, - Down)",
        xaxis=dict(
            tickangle=90 if len(unique_groups) > 20 else 0,
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray'
        ),
        yaxis=dict(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray',
            zeroline=True,  # Add zero line
            zerolinecolor='black',
            zerolinewidth=2
        ),
        margin=dict(b=150 if len(unique_groups) > 20 else 80),  # More bottom margin for rotated labels
    )

    # Label each bar with its score value
    for i, row in current_data_nonzero.iterrows():
        fig2.add_annotation(
            x=row[group_by],
            y=row['score'],
            text=str(int(row['score'])),
            showarrow=False,
            font=dict(color="black", size=10),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )

    st.plotly_chart(fig2, use_container_width=True)
else:
    st.write("No non-zero scores to display.")

# ------------------------------
# NEW: Percent Change Visualization
# ------------------------------
st.header(f"Percent Change over the Last {percent_change_days} {'Day' if percent_change_days == 1 else 'Days'}")

if not percent_changes.empty:
    # Get the latest and reference dates
    latest_date = percent_changes['latest_date'].iloc[0].strftime('%Y-%m-%d')
    reference_date = percent_changes['reference_date'].iloc[0].strftime('%Y-%m-%d')
    
    st.write(f"Change from {reference_date} to {latest_date}")
    
    # Sort by percent change value (largest positive to largest negative)
    percent_changes_sorted = percent_changes.sort_values('percent_change', ascending=False)
    
    # Define color scale for percent changes
    percent_colorscale = [
        [0, 'red'],      # Negative changes
        [0.4, 'lightcoral'],
        [0.5, 'white'],  # Zero (neutral)
        [0.6, 'lightblue'],
        [1, 'green']     # Positive changes
    ]
    
    # Create the bar chart
    fig_pct = px.bar(
        percent_changes_sorted, 
        x=group_by, 
        y='percent_change',
        color='percent_change',
        color_continuous_scale=percent_colorscale,
        title=f"Percent Change by {group_by.capitalize()} ({reference_date} to {latest_date})",
        labels={'percent_change': '% Change'},
        height=600
    )
    
    # Add grid for better readability, rotate labels if many industries
    fig_pct.update_layout(
        xaxis_title=group_by.capitalize(), 
        yaxis_title="Percent Change",
        xaxis=dict(
            tickangle=90 if len(unique_groups) > 20 else 0,
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray'
        ),
        yaxis=dict(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray',
            zeroline=True,  # Add zero line
            zerolinecolor='black',
            zerolinewidth=2
        ),
        margin=dict(b=150 if len(unique_groups) > 20 else 80),  # More bottom margin for rotated labels
    )
    
    # We're removing the labels above the bars by commenting out this code
    # for i, row in percent_changes_sorted.iterrows():
    #     fig_pct.add_annotation(
    #         x=row[group_by],
    #         y=row['percent_change'],
    #         text=f"{row['percent_change']:.1f}%",
    #         showarrow=False,
    #         font=dict(color="black", size=10),
    #         bgcolor="white",
    #         bordercolor="black",
    #         borderwidth=1
    #     )
    
    st.plotly_chart(fig_pct, use_container_width=True)
    
    # Optional: Show percent change data table
    if st.checkbox("Show Percent Change Data Table"):
        st.dataframe(
            percent_changes_sorted[[group_by, 'percent_change', 'latest_value', 'reference_value']]
            .sort_values('percent_change', ascending=False)
        )
else:
    st.write("Not enough data for percent change calculation.")

# ------------------------------
# Current Streaks Visualization
# ------------------------------
st.header(f"Current Streaks")

# Filter out zero streaks since we don't want to display them
current_streak_nonzero = current_data[current_data['streak'] != 0].copy()
st.write(f"Number of groups with non-zero streaks: {len(current_streak_nonzero)}")

# Sort by absolute streak value
current_streak_nonzero = current_streak_nonzero.sort_values('streak', key=abs, ascending=False)

# Only create the chart if we have non-zero streaks
if len(current_streak_nonzero) > 0:
    # Create the bar chart using the actual streak data
    fig3 = px.bar(
        current_streak_nonzero, 
        x=group_by, 
        y='streak',
        color='streak',
        color_continuous_scale=colorscale,
        title=f"Current Streak Length by {group_by.capitalize()}",
        labels={'streak': 'Streak Length'},
        height=600
    )

    # Add grid for better readability, rotate labels if many industries
    fig3.update_layout(
        xaxis_title=group_by.capitalize(), 
        yaxis_title="Streak Length (+ Up, - Down)",
        xaxis=dict(
            tickangle=90 if len(unique_groups) > 20 else 0,
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray'
        ),
        yaxis=dict(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray',
            zeroline=True,  # Add zero line
            zerolinecolor='black',
            zerolinewidth=2
        ),
        margin=dict(b=150 if len(unique_groups) > 20 else 80),  # More bottom margin for rotated labels
    )

    # Label each bar with its streak value
    for i, row in current_streak_nonzero.iterrows():
        fig3.add_annotation(
            x=row[group_by],
            y=row['streak'],
            text=str(int(row['streak'])),
            showarrow=False,
            font=dict(color="black", size=10),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )

    st.plotly_chart(fig3, use_container_width=True)
else:
    st.write("No non-zero streaks to display.")

# ------------------------------
# Daily Direction Heatmap with Explicit Date Control
# ------------------------------
st.header("Daily Direction Heatmap")

# Get data for the heatmap, filtered for display start date
display_data = daily_changes[daily_changes['fetch_date'] >= pd.Timestamp(display_start_date)].copy()

# Create string date for easier handling
display_data['date_str'] = display_data['fetch_date'].dt.strftime('%Y-%m-%d')

# Get unique trading days - these are the only days we want to show
trading_days = sorted(display_data['date_str'].unique(), reverse=True)  # Newest first

# Pivot the data using the string date
pivot_data = display_data.pivot(index='date_str', columns=group_by, values='direction').fillna(0)

# Make sure we only include the dates we have data for
pivot_data = pivot_data.loc[trading_days]

# Create the heatmap with explicit control over the y-axis categories
fig4 = go.Figure(data=go.Heatmap(
    z=pivot_data.values,
    x=pivot_data.columns,
    y=pivot_data.index,  # These are our trading days only
    colorscale=[[0, 'red'], [0.5, 'white'], [1, 'green']],
    zmin=-1, zmax=1
))

# The critical part: explicitly tell Plotly to use category mode for y-axis
# and provide the exact list of categories (our trading days)
fig4.update_layout(
    title=f"Daily Direction by {group_by.capitalize()} (Green=Up, Red=Down, White=No Change)",
    xaxis_title=group_by.capitalize(),
    yaxis_title="Date",
    height=max(600, len(pivot_data.index) * 25),  # Dynamic height based on number of dates
    xaxis=dict(
        tickangle=90 if len(unique_groups) > 20 else 0,
        side='top'  # Move labels to top for better visibility
    ),
    # Explicitly set up the y-axis as categories with our trading days
    yaxis=dict(
        type='category',  # Force category mode
        categoryorder='array',  # Use explicit ordering
        categoryarray=trading_days,  # Our explicit list of trading days
        autorange=True  # Let Plotly handle the range automatically
    ),
    margin=dict(b=20, t=100 if len(unique_groups) > 20 else 80)  # Adjust margins
)

st.plotly_chart(fig4, use_container_width=True)

# ------------------------------
# Overall Market Cap Visualization
# ------------------------------
st.header("Overall Market Performance")

# Function to calculate overall market metrics
@st.cache_data
def calculate_market_metrics(market_caps_df, sectors_df):
    """Calculate daily overall market cap metrics"""
    # Get all symbols from the sectors dataframe (these are the ones we want to track)
    all_symbols = sectors_df['symbol'].unique()
    
    # Filter market caps to only include symbols in the sectors dataframe
    market_caps_filtered = market_caps_df[market_caps_df['symbol'].isin(all_symbols)]
    
    # Convert market cap to numeric
    market_caps_filtered['market_cap'] = pd.to_numeric(market_caps_filtered['market_cap'], errors='coerce')
    
    # Group by date and calculate total market cap and counts
    daily_totals = market_caps_filtered.groupby('fetch_date').agg(
        total_market_cap=('market_cap', 'sum'),
        company_count=('symbol', 'count')
    ).reset_index()
    
    # Sort by date
    daily_totals = daily_totals.sort_values('fetch_date')
    
    # Calculate daily percentage changes
    daily_totals['pct_change'] = daily_totals['total_market_cap'].pct_change() * 100
    
    # Calculate cumulative percentage change (indexed to first day = 100)
    first_value = daily_totals['total_market_cap'].iloc[0]
    daily_totals['cumulative_index'] = daily_totals['total_market_cap'] / first_value * 100
    
    # Calculate moving averages
    daily_totals['ma_5d'] = daily_totals['total_market_cap'].rolling(window=5, min_periods=1).mean()
    daily_totals['ma_10d'] = daily_totals['total_market_cap'].rolling(window=10, min_periods=1).mean()
    
    # Calculate daily direction (up/down)
    daily_totals['direction'] = daily_totals['pct_change'].apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )
    
    # Mark days with significant moves (e.g., >1% change)
    daily_totals['significant_move'] = abs(daily_totals['pct_change']) > 1
    
    return daily_totals

# Filter for the display period
market_metrics = calculate_market_metrics(market_caps_df, sectors_df)
filtered_metrics = market_metrics[market_metrics['fetch_date'] >= pd.Timestamp(display_start_date)]

# Create a two-column layout
col1, col2 = st.columns(2)

# Column 1: Summary metrics
with col1:
    st.subheader("Market Summary")
    
    # Only proceed if we have data
    if not filtered_metrics.empty:
        # Get latest values
        latest = filtered_metrics.iloc[-1]
        first = filtered_metrics.iloc[0]
        
        # Calculate metrics
        total_change = ((latest['total_market_cap'] / first['total_market_cap']) - 1) * 100
        avg_daily_change = filtered_metrics['pct_change'].mean()
        up_days = (filtered_metrics['direction'] > 0).sum()
        down_days = (filtered_metrics['direction'] < 0).sum()
        flat_days = (filtered_metrics['direction'] == 0).sum()
        total_days = len(filtered_metrics)
        
        # Format values for display
        total_market_cap = f"${latest['total_market_cap'] / 1_000_000_000_000:.2f}T"
        
        # Create metrics
        st.metric("Current Market Cap", total_market_cap, f"{latest['pct_change']:.2f}% today")
        st.metric("Period Change", f"{total_change:.2f}%", f"Over {total_days} trading days")
        st.metric("Avg. Daily Change", f"{avg_daily_change:.2f}%")
        
        # Create a gauge-style chart for up/down day ratio
        if up_days + down_days > 0:  # Avoid division by zero
            up_ratio = up_days / (up_days + down_days)
            
            # Use a horizontal bar chart as a simple gauge
            fig_gauge = go.Figure()
            
            # Add the gauge bar
            fig_gauge.add_trace(go.Bar(
                x=[up_ratio * 100, (1-up_ratio) * 100],
                y=["Up/Down Ratio"],
                orientation='h',
                marker=dict(
                    color=['rgba(0, 200, 0, 0.8)', 'rgba(255, 0, 0, 0.8)']
                ),
                text=[f"{up_days} days up", f"{down_days} days down"],
                textposition='inside',
                insidetextanchor='middle',
                name=''
            ))
            
            fig_gauge.update_layout(
                title="Trading Days: Up vs Down",
                barmode='stack',
                height=150,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False
                )
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.write("No data available for the selected period.")

# Column 2: Distribution of daily changes
with col2:
    st.subheader("Daily Change Distribution")
    
    if not filtered_metrics.empty and len(filtered_metrics) > 1:  # Need at least 2 days for changes
        # Create histogram of daily changes
        fig_hist = px.histogram(
            filtered_metrics, 
            x='pct_change',
            nbins=20,
            color_discrete_sequence=['rgba(55, 126, 184, 0.7)'],
            labels={'pct_change': 'Daily % Change'}
        )
        
        fig_hist.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(title='Daily % Change'),
            yaxis=dict(title='Frequency')
        )
        
        # Add a vertical line at 0%
        fig_hist.add_shape(
            type='line',
            x0=0, y0=0,
            x1=0, y1=1,
            yref='paper',
            line=dict(color='red', width=2, dash='dash')
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.write("Not enough data for histogram.")

# Main chart: Market Cap Trends section - MODIFIED to remove Combined View
st.subheader("Market Cap Trends")

# Create tabs for different visualizations - Removed Combined View
tab1, tab2 = st.tabs(["Total Market Cap", "Daily % Change"])

with tab1:
    if not filtered_metrics.empty:
        # Create line chart of total market cap
        fig_total = px.line(
            filtered_metrics, 
            x='fetch_date', 
            y=['total_market_cap', 'ma_5d', 'ma_10d'],
            labels={
                'fetch_date': 'Date',
                'total_market_cap': 'Total Market Cap',
                'ma_5d': '5-Day MA',
                'ma_10d': '10-Day MA',
                'value': 'Market Cap ($)'
            },
            title="Total Market Capitalization Over Time"
        )
        
        # Format y-axis to show in trillions
        fig_total.update_layout(
            yaxis=dict(
                tickformat='.2f',
                title='Market Cap (Trillions $)'
            ),
            hovermode='x unified'
        )
        
        # Scale y-values to trillions for better readability
        fig_total.update_traces(y=filtered_metrics['total_market_cap'] / 1_000_000_000_000, selector=dict(name='total_market_cap'))
        fig_total.update_traces(y=filtered_metrics['ma_5d'] / 1_000_000_000_000, selector=dict(name='ma_5d'))
        fig_total.update_traces(y=filtered_metrics['ma_10d'] / 1_000_000_000_000, selector=dict(name='ma_10d'))
        
        # Update line colors and names
        fig_total.for_each_trace(
            lambda trace: trace.update(
                name='Total Market Cap' if trace.name == 'total_market_cap' else 
                     '5-Day Moving Avg' if trace.name == 'ma_5d' else
                     '10-Day Moving Avg',
                line=dict(
                    width=3 if trace.name == 'total_market_cap' else 2,
                    dash=None if trace.name == 'total_market_cap' else 'dash'
                )
            )
        )
        
        st.plotly_chart(fig_total, use_container_width=True)
    else:
        st.write("No data available for the selected period.")

with tab2:
    if not filtered_metrics.empty and len(filtered_metrics) > 1:
        # Create bar chart of daily percentage changes
        fig_pct = px.bar(
            filtered_metrics, 
            x='fetch_date', 
            y='pct_change',
            color='direction',
            color_discrete_map={1: 'green', -1: 'red', 0: 'gray'},
            labels={
                'fetch_date': 'Date',
                'pct_change': 'Daily % Change'
            },
            title="Daily Market Cap Percentage Change"
        )
        
        # Add a horizontal line at 0%
        fig_pct.add_shape(
            type='line',
            x0=filtered_metrics['fetch_date'].min(),
            y0=0,
            x1=filtered_metrics['fetch_date'].max(),
            y1=0,
            line=dict(color='black', width=1)
        )
        
        # Hide the legend (not needed)
        fig_pct.update_layout(
            showlegend=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_pct, use_container_width=True)
    else:
        st.write("Not enough data for percentage change chart.")

# ------------------------------
# Detailed Data and Export
# ------------------------------
# Optional: Show detailed data
if st.checkbox("Show Detailed Data"):
    st.subheader("Detailed Trend Score Data")
    st.dataframe(filtered_scores.sort_values(['date', group_by]))

# Add export functionality
if st.button("Export Current Data"):
    # Get the latest data for all groups
    export_date = datetime.now().strftime("%Y%m%dT%H%M")
    export_data = filtered_scores.sort_values(['date', group_by])
    
    # Create Excel buffer
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_data.to_excel(writer, index=False, sheet_name='Trend_Data')
        # Add percent change data to a separate sheet
        if not percent_changes.empty:
            percent_changes.to_excel(writer, index=False, sheet_name='Percent_Changes')
    output.seek(0)
    
    # Create download button for Excel
    st.download_button(
        label="Download Excel",
        data=output,
        file_name=f"{export_date}_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# End of app