import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO

# Set page configuration to wide layout
st.set_page_config(layout="wide")
st.title("Market Cap ROC and Relative Strength Analysis")

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
        if 'fetch_date' not in df.columns:
            st.error("market_caps.xlsx does not have a 'fetch_date' column.")
            st.stop()
        df['fetch_date'] = pd.to_datetime(df['fetch_date'], errors='coerce')
        if df['fetch_date'].isna().all():
            st.error("All 'fetch_date' values in market_caps.xlsx are invalid.")
            st.stop()
        return df
    except Exception as e:
        st.error(f"Error loading market_caps.xlsx: {str(e)}")
        st.stop()

# Define file paths
possible_data_paths = [
    "Data", "data", "stock_scores",
    os.path.join("Data", "stock_scores"),
    os.path.join("data", "stock_scores"),
    "."
]

def find_file(filename):
    if os.path.exists(filename):
        return filename
    for data_path in possible_data_paths:
        test_path = os.path.join(data_path, filename)
        if os.path.exists(test_path):
            return test_path
    parent_path = os.path.join("..", filename)
    if os.path.exists(parent_path):
        return parent_path
    parent_data_path = os.path.join("..", "Data", filename)
    if os.path.exists(parent_data_path):
        return parent_data_path
    return None

sectors_file_path = find_file("stock_sectors.xlsx")
market_caps_file_path = find_file("market_caps.xlsx")

if not sectors_file_path or not market_caps_file_path:
    st.error("Could not find required Excel files.")
    st.stop()

# Load data
sectors_df = load_sectors_file(sectors_file_path)
market_caps_df = load_market_caps_file(market_caps_file_path)
st.sidebar.success("Data loaded successfully!")

# ------------------------------
# 2. User Inputs
# ------------------------------
st.sidebar.header("Analysis Settings")

# Grouping selection (fixed to industry for this app)
group_by = "industry"

# Date range for display
unique_dates = sorted(market_caps_df['fetch_date'].dt.date.unique())
min_date = min(unique_dates)
max_date = max(unique_dates)
max_days = len(unique_dates)

days_to_display = st.sidebar.slider(
    "Days to display",
    min_value=1,
    max_value=max_days,
    value=min(max_days, 21)
)

display_start_date = unique_dates[-days_to_display] if days_to_display < max_days else min_date

# ROC lookback periods
roc_periods = st.sidebar.multiselect(
    "ROC Lookback Periods (days)",
    options=[5, 10, 21],
    default=[5, 10, 21]
)

# ------------------------------
# 3. Process Data
# ------------------------------
@st.cache_data
def calculate_roc_and_relative_strength(market_caps_df, sectors_df, roc_periods):
    """Calculate ROC and Relative Strength for industries."""
    # Filter market caps to only include symbols in sectors_df
    all_symbols = sectors_df['symbol'].unique()
    market_caps_filtered = market_caps_df[market_caps_df['symbol'].isin(all_symbols)]

    # Merge with sectors data
    merged_df = pd.merge(
        market_caps_filtered,
        sectors_df[['symbol', 'sector', 'industry']],
        on='symbol',
        how='inner'
    )

    # Convert market cap to numeric
    merged_df['market_cap'] = pd.to_numeric(merged_df['market_cap'], errors='coerce')

    # Calculate total market cap per date
    total_market_cap = merged_df.groupby('fetch_date')['market_cap'].sum().reset_index()
    total_market_cap = total_market_cap.sort_values('fetch_date')

    # Calculate market cap by industry and sector
    industry_caps = merged_df.groupby(['fetch_date', 'industry'])['market_cap'].sum().reset_index()
    sector_caps = merged_df.groupby(['fetch_date', 'sector'])['market_cap'].sum().reset_index()

    # Sort by date
    industry_caps = industry_caps.sort_values(['industry', 'fetch_date'])
    sector_caps = sector_caps.sort_values(['sector', 'fetch_date'])

    # Initialize dictionaries to store results
    roc_results = {}
    rs_industry_vs_sector = {}
    rs_industry_vs_total = {}

    # Calculate ROC for each industry
    for period in roc_periods:
        roc_results[period] = industry_caps.copy()
        roc_results[period]['prev_market_cap'] = roc_results[period].groupby('industry')['market_cap'].shift(period)
        roc_results[period]['roc'] = (
            (roc_results[period]['market_cap'] - roc_results[period]['prev_market_cap']) /
            roc_results[period]['prev_market_cap'] * 100
        ).fillna(0)

    # Calculate ROC for sectors and total market
    sector_roc = {}
    for period in roc_periods:
        sector_roc[period] = sector_caps.copy()
        sector_roc[period]['prev_market_cap'] = sector_roc[period].groupby('sector')['market_cap'].shift(period)
        sector_roc[period]['roc'] = (
            (sector_roc[period]['market_cap'] - sector_roc[period]['prev_market_cap']) /
            sector_roc[period]['prev_market_cap'] * 100
        ).fillna(0)

    total_roc = {}
    for period in roc_periods:
        total_roc[period] = total_market_cap.copy()
        total_roc[period]['prev_market_cap'] = total_roc[period]['market_cap'].shift(period)
        total_roc[period]['roc'] = (
            (total_roc[period]['market_cap'] - total_roc[period]['prev_market_cap']) /
            total_roc[period]['prev_market_cap'] * 100
        ).fillna(0)

    # Calculate Relative Strength
    industry_to_sector = sectors_df[['industry', 'sector']].drop_duplicates().set_index('industry')['sector'].to_dict()

    for period in roc_periods:
        # Industry vs Sector
        rs_industry_vs_sector[period] = roc_results[period].copy()
        rs_industry_vs_sector[period]['sector'] = rs_industry_vs_sector[period]['industry'].map(industry_to_sector)
        rs_industry_vs_sector[period] = pd.merge(
            rs_industry_vs_sector[period],
            sector_roc[period][['fetch_date', 'sector', 'roc']],
            on=['fetch_date', 'sector'],
            how='left',
            suffixes=('', '_sector')
        )
        rs_industry_vs_sector[period]['relative_strength'] = (
            rs_industry_vs_sector[period]['roc'] - rs_industry_vs_sector[period]['roc_sector']
        )

        # Industry vs Total Market
        rs_industry_vs_total[period] = roc_results[period].copy()
        rs_industry_vs_total[period] = pd.merge(
            rs_industry_vs_total[period],
            total_roc[period][['fetch_date', 'roc']],
            on='fetch_date',
            how='left',
            suffixes=('', '_total')
        )
        rs_industry_vs_total[period]['relative_strength'] = (
            rs_industry_vs_total[period]['roc'] - rs_industry_vs_total[period]['roc_total']
        )

    return roc_results, rs_industry_vs_sector, rs_industry_vs_total

# Calculate ROC and Relative Strength
roc_data, rs_vs_sector, rs_vs_total = calculate_roc_and_relative_strength(market_caps_df, sectors_df, roc_periods)

# Filter for display period
for period in roc_periods:
    roc_data[period] = roc_data[period][roc_data[period]['fetch_date'] >= pd.Timestamp(display_start_date)]
    rs_vs_sector[period] = rs_vs_sector[period][rs_vs_sector[period]['fetch_date'] >= pd.Timestamp(display_start_date)]
    rs_vs_total[period] = rs_vs_total[period][rs_vs_total[period]['fetch_date'] >= pd.Timestamp(display_start_date)]

# ------------------------------
# 4. Visualize Results
# ------------------------------
st.header("Rate of Change (ROC) Analysis")

# ROC Heatmaps for each period
for period in roc_periods:
    st.subheader(f"ROC over {period} Days")
    display_data = roc_data[period].copy()
    display_data['date_str'] = display_data['fetch_date'].dt.strftime('%Y-%m-%d')
    trading_days = sorted(display_data['date_str'].unique(), reverse=True)
    pivot_data = display_data.pivot(index='date_str', columns='industry', values='roc').fillna(0)
    pivot_data = pivot_data.loc[trading_days]

    fig_roc = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale=[[0, 'red'], [0.5, 'white'], [1, 'green']],
        zmin=-10, zmax=10
    ))

    fig_roc.update_layout(
        title=f"ROC by Industry over {period} Days (Green=Positive, Red=Negative)",
        xaxis_title="Industry",
        yaxis_title="Date",
        height=max(600, len(pivot_data.index) * 25),
        xaxis=dict(tickangle=90, side='top'),
        yaxis=dict(type='category', categoryorder='array', categoryarray=trading_days),
        margin=dict(b=20, t=100)
    )

    st.plotly_chart(fig_roc, use_container_width=True)

# Current ROC Bar Chart (latest date)
st.subheader("Current ROC by Industry")
latest_date = max(unique_dates)
for period in roc_periods:
    current_roc = roc_data[period][roc_data[period]['fetch_date'].dt.date == latest_date].copy()
    current_roc = current_roc.sort_values('roc', ascending=False)

    if not current_roc.empty:
        fig_roc_bar = px.bar(
            current_roc,
            x='industry',
            y='roc',
            color='roc',
            color_continuous_scale=[[0, 'red'], [0.5, 'white'], [1, 'green']],
            title=f"Current ROC over {period} Days ({latest_date})",
            labels={'roc': 'ROC (%)'},
            height=600
        )

        fig_roc_bar.update_layout(
            xaxis_title="Industry",
            yaxis_title="ROC (%)",
            xaxis=dict(tickangle=90),
            yaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=2),
            margin=dict(b=150)
        )

        st.plotly_chart(fig_roc_bar, use_container_width=True)

# ------------------------------
# Relative Strength Analysis
# ------------------------------
st.header("Relative Strength Analysis")

# Relative Strength vs Sector
st.subheader("Relative Strength vs Sector")
for period in roc_periods:
    st.write(f"Over {period} Days")
    display_data = rs_vs_sector[period].copy()
    display_data['date_str'] = display_data['fetch_date'].dt.strftime('%Y-%m-%d')
    trading_days = sorted(display_data['date_str'].unique(), reverse=True)
    pivot_data = display_data.pivot(index='date_str', columns='industry', values='relative_strength').fillna(0)
    pivot_data = pivot_data.loc[trading_days]

    fig_rs_sector = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale=[[0, 'red'], [0.5, 'white'], [1, 'blue']],
        zmin=-10, zmax=10
    ))

    fig_rs_sector.update_layout(
        title=f"Relative Strength vs Sector over {period} Days (Blue=Outperforming, Red=Underperforming)",
        xaxis_title="Industry",
        yaxis_title="Date",
        height=max(600, len(pivot_data.index) * 25),
        xaxis=dict(tickangle=90, side='top'),
        yaxis=dict(type='category', categoryorder='array', categoryarray=trading_days),
        margin=dict(b=20, t=100)
    )

    st.plotly_chart(fig_rs_sector, use_container_width=True)

# Current Relative Strength vs Sector Bar Chart
st.subheader("Current Relative Strength vs Sector")
for period in roc_periods:
    current_rs_sector = rs_vs_sector[period][rs_vs_sector[period]['fetch_date'].dt.date == latest_date].copy()
    current_rs_sector = current_rs_sector.sort_values('relative_strength', ascending=False)

    if not current_rs_sector.empty:
        fig_rs_sector_bar = px.bar(
            current_rs_sector,
            x='industry',
            y='relative_strength',
            color='relative_strength',
            color_continuous_scale=[[0, 'red'], [0.5, 'white'], [1, 'blue']],
            title=f"Current Relative Strength vs Sector over {period} Days ({latest_date})",
            labels={'relative_strength': 'Relative Strength (%)'},
            height=600
        )

        fig_rs_sector_bar.update_layout(
            xaxis_title="Industry",
            yaxis_title="Relative Strength (%)",
            xaxis=dict(tickangle=90),
            yaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=2),
            margin=dict(b=150)
        )

        st.plotly_chart(fig_rs_sector_bar, use_container_width=True)

# Relative Strength vs Total Market Cap
st.subheader("Relative Strength vs Total Market Cap")
for period in roc_periods:
    st.write(f"Over {period} Days")
    display_data = rs_vs_total[period].copy()
    display_data['date_str'] = display_data['fetch_date'].dt.strftime('%Y-%m-%d')
    trading_days = sorted(display_data['date_str'].unique(), reverse=True)
    pivot_data = display_data.pivot(index='date_str', columns='industry', values='relative_strength').fillna(0)
    pivot_data = pivot_data.loc[trading_days]

    fig_rs_total = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale=[[0, 'red'], [0.5, 'white'], [1, 'purple']],
        zmin=-10, zmax=10
    ))

    fig_rs_total.update_layout(
        title=f"Relative Strength vs Total Market over {period} Days (Purple=Outperforming, Red=Underperforming)",
        xaxis_title="Industry",
        yaxis_title="Date",
        height=max(600, len(pivot_data.index) * 25),
        xaxis=dict(tickangle=90, side='top'),
        yaxis=dict(type='category', categoryorder='array', categoryarray=trading_days),
        margin=dict(b=20, t=100)
    )

    st.plotly_chart(fig_rs_total, use_container_width=True)

# Current Relative Strength vs Total Market Cap Bar Chart
st.subheader("Current Relative Strength vs Total Market Cap")
for period in roc_periods:
    current_rs_total = rs_vs_total[period][rs_vs_total[period]['fetch_date'].dt.date == latest_date].copy()
    current_rs_total = current_rs_total.sort_values('relative_strength', ascending=False)

    if not current_rs_total.empty:
        fig_rs_total_bar = px.bar(
            current_rs_total,
            x='industry',
            y='relative_strength',
            color='relative_strength',
            color_continuous_scale=[[0, 'red'], [0.5, 'white'], [1, 'purple']],
            title=f"Current Relative Strength vs Total Market over {period} Days ({latest_date})",
            labels={'relative_strength': 'Relative Strength (%)'},
            height=600
        )

        fig_rs_total_bar.update_layout(
            xaxis_title="Industry",
            yaxis_title="Relative Strength (%)",
            xaxis=dict(tickangle=90),
            yaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=2),
            margin=dict(b=150)
        )

        st.plotly_chart(fig_rs_total_bar, use_container_width=True)

# ------------------------------
# Combined Analysis for Trade Decisions
# ------------------------------
st.header("Combined ROC and Relative Strength Analysis")

for period in roc_periods:
    st.subheader(f"Over {period} Days")

    # Get latest data
    latest_roc = roc_data[period][roc_data[period]['fetch_date'].dt.date == latest_date][['industry', 'roc']]
    latest_rs_sector = rs_vs_sector[period][rs_vs_sector[period]['fetch_date'].dt.date == latest_date][['industry', 'relative_strength']]
    latest_rs_total = rs_vs_total[period][rs_vs_total[period]['fetch_date'].dt.date == latest_date][['industry', 'relative_strength']]

    # Merge the data
    combined_data = pd.merge(latest_roc, latest_rs_sector, on='industry', suffixes=('_roc', '_rs_sector'))
    combined_data = pd.merge(combined_data, latest_rs_total, on='industry')
    combined_data.columns = ['Industry', 'ROC (%)', 'RS vs Sector (%)', 'RS vs Total Market (%)']

    # Sort by ROC descending
    combined_data = combined_data.sort_values('ROC (%)', ascending=False)

    # Display the table
    st.dataframe(combined_data)

    # Scatter plot
    fig_scatter = px.scatter(
        combined_data,
        x='ROC (%)',
        y='RS vs Sector (%)',
        color='RS vs Total Market (%)',
        hover_data=['Industry'],
        title=f"ROC vs Relative Strength (Sector) over {period} Days",
        labels={'ROC (%)': 'ROC (%)', 'RS vs Sector (%)': 'Relative Strength vs Sector (%)'},
        height=600
    )

    fig_scatter.update_layout(
        xaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=2),
        yaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=2),
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------------
# Export Functionality
# ------------------------------
if st.button("Export Current Data"):
    export_date = datetime.now().strftime("%Y%m%dT%H%M")
    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for period in roc_periods:
            roc_data[period].to_excel(writer, sheet_name=f'ROC_{period}_days', index=False)
            rs_vs_sector[period].to_excel(writer, sheet_name=f'RS_Sector_{period}_days', index=False)
            rs_vs_total[period].to_excel(writer, sheet_name=f'RS_Total_{period}_days', index=False)

    output.seek(0)

    st.download_button(
        label="Download Excel",
        data=output,
        file_name=f"{export_date}_roc_rs_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )