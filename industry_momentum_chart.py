import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import os
import logging
from datetime import datetime
import glob
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_json_files():
    """Load all JSON files from the directory and sort by date."""
    try:
        base_path = "Data/stock_scores"
        json_files = glob.glob(os.path.join(base_path, "market_analysis_*.json"))
        json_files = [f for f in json_files if 'latest' not in f]
        json_files.sort(key=lambda x: os.path.getmtime(x))
        return json_files
    except Exception as e:
        logger.error(f"Error in load_json_files: {str(e)}")
        raise

def load_and_process_file(file_path):
    """Load and process a single JSON file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            df = pd.DataFrame(data['stocks'])
            
            # Group by industry and calculate mean scores
            industry_scores = df.groupby('industry').agg({
                'bullish_score': 'mean',
                'bearish_score': 'mean',
                'symbol': 'count'
            }).round(2)
            
            # Add date from filename
            file_date = datetime.fromtimestamp(os.path.getmtime(file_path))
            industry_scores['date'] = file_date
            
            return industry_scores.reset_index()
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        raise

def calculate_latest_change(dfs, score_type):
    """Calculate change between last two datasets."""
    if len(dfs) < 2:
        return pd.DataFrame()
        
    df1 = dfs[-2]  # Second to last dataset
    df2 = dfs[-1]  # Latest dataset
    
    merged = pd.merge(df1, df2, on='industry', suffixes=('_prev', '_curr'))
    merged[f'{score_type}_change'] = (
        merged[f'{score_type}_score_curr'] - merged[f'{score_type}_score_prev']
    ).round(2)
    
    merged['momentum_type'] = 'Latest Change'
    merged['start_date'] = merged['date_prev']
    merged['end_date'] = merged['date_curr']
    merged['current_score'] = merged[f'{score_type}_score_curr']
    
    return merged

def calculate_rolling_momentum(dfs, score_type, window=3):
    """Calculate rolling average of changes."""
    if len(dfs) < 2:
        return pd.DataFrame()
        
    changes_list = []
    for i in range(len(dfs) - 1):
        df1 = dfs[i]
        df2 = dfs[i + 1]
        
        merged = pd.merge(df1, df2, on='industry', suffixes=('_prev', '_curr'))
        merged['date'] = merged['date_curr']
        merged[f'{score_type}_change'] = (
            merged[f'{score_type}_score_curr'] - merged[f'{score_type}_score_prev']
        )
        changes_list.append(merged)
    
    all_changes = pd.concat(changes_list)
    
    # Calculate rolling average by industry
    rolling = all_changes.groupby('industry').agg({
        f'{score_type}_change': lambda x: x.rolling(window=min(window, len(x))).mean().iloc[-1],
        'date': 'last',
        f'{score_type}_score_curr': 'last',
        'symbol_curr': 'last'
    }).reset_index()
    
    rolling['momentum_type'] = f'{window}-Period Rolling Average'
    rolling['current_score'] = rolling[f'{score_type}_score_curr']
    rolling['start_date'] = all_changes['date'].min()
    rolling['end_date'] = all_changes['date'].max()
    
    return rolling

def calculate_weighted_momentum(dfs, score_type):
    """Calculate weighted average of changes (more weight to recent changes)."""
    if len(dfs) < 2:
        return pd.DataFrame()
        
    changes_list = []
    weights = []
    
    for i in range(len(dfs) - 1):
        df1 = dfs[i]
        df2 = dfs[i + 1]
        
        merged = pd.merge(df1, df2, on='industry', suffixes=('_prev', '_curr'))
        merged[f'{score_type}_change'] = (
            merged[f'{score_type}_score_curr'] - merged[f'{score_type}_score_prev']
        )
        changes_list.append(merged)
        weights.append(2 ** i)  # Exponential weights
    
    # Normalize weights
    weights = [w / sum(weights) for w in weights]
    
    all_changes = pd.concat(changes_list)
    weighted_changes = all_changes.groupby('industry').agg({
        f'{score_type}_change': lambda x: np.average(x, weights=weights[:len(x)]),
        'date': 'last',
        f'{score_type}_score_curr': 'last',
        'symbol_curr': 'last'
    }).reset_index()
    
    weighted_changes['momentum_type'] = 'Weighted Average (Recent Heavy)'
    weighted_changes['current_score'] = weighted_changes[f'{score_type}_score_curr']
    weighted_changes['start_date'] = all_changes['date'].min()
    weighted_changes['end_date'] = all_changes['date'].max()
    
    return weighted_changes

def calculate_rate_of_change(dfs, score_type):
    """Calculate rate of change over all available periods."""
    if len(dfs) < 2:
        return pd.DataFrame()
        
    df_first = dfs[0]
    df_last = dfs[-1]
    
    merged = pd.merge(df_first, df_last, on='industry', suffixes=('_start', '_end'))
    
    # Calculate total change divided by number of periods
    periods = len(dfs) - 1
    merged[f'{score_type}_change'] = (
        (merged[f'{score_type}_score_end'] - merged[f'{score_type}_score_start']) / periods
    ).round(2)
    
    merged['momentum_type'] = 'Rate of Change'
    merged['current_score'] = merged[f'{score_type}_score_end']
    merged['start_date'] = merged['date_start']
    merged['end_date'] = merged['date_end']
    
    return merged

def calculate_momentum(dfs, score_type, momentum_type):
    """Calculate momentum based on selected method."""
    if momentum_type == 'Latest Change':
        return calculate_latest_change(dfs, score_type)
    elif momentum_type == 'Rolling Average':
        return calculate_rolling_momentum(dfs, score_type)
    elif momentum_type == 'Weighted Average':
        return calculate_weighted_momentum(dfs, score_type)
    else:  # Rate of Change
        return calculate_rate_of_change(dfs, score_type)

def create_momentum_chart(changes_df, score_type='bullish', top_n=10, momentum_type='Latest Change'):
    """Create a bar chart showing the industries with highest score increases."""
    try:
        change_col = f'{score_type}_change'
        
        # Sort by positive changes only
        top_changes = changes_df.nlargest(top_n, change_col)
        
        fig = go.Figure()
        
        # Add bars for changes
        fig.add_trace(go.Bar(
            x=top_changes[change_col],
            y=top_changes['industry'],
            orientation='h',
            text=[
                f"+{x:.2f} (Score: {y:.1f})" 
                for x, y in zip(top_changes[change_col], top_changes['current_score'])
            ],
            textposition='auto',
            marker_color='green',
            name='Score Increase'
        ))
        
        # Update layout
        date_range = f"{top_changes['start_date'].min().strftime('%Y-%m-%d')} to {top_changes['end_date'].max().strftime('%Y-%m-%d')}"
        
        fig.update_layout(
            title=f'Top {top_n} Industries Gaining {score_type.title()} Momentum<br>' +
                  f'<sup>Method: {momentum_type} | {date_range}</sup>',
            xaxis_title=f'{score_type.title()} Score Change',
            yaxis_title='Industry',
            height=max(400, len(top_changes) * 30),
            showlegend=False,
            margin=dict(l=20, r=20, t=100, b=20)
        )
        
        # Set x-axis to start at 0 since we're only showing increases
        fig.update_xaxes(range=[0, top_changes[change_col].max() * 1.1])
        
        return fig
    except Exception as e:
        logger.error(f"Error in create_momentum_chart: {str(e)}")
        raise

def main():
    try:
        st.title("Industry Momentum Analysis")
        
        # Load all JSON files
        json_files = load_json_files()
        
        if len(json_files) < 2:
            st.error("Need at least 2 JSON files for momentum analysis!")
            return
        
        # Process all files
        st.text("Processing data files...")
        progress_bar = st.progress(0)
        all_data = []
        
        for idx, file_path in enumerate(json_files):
            processed_data = load_and_process_file(file_path)
            all_data.append(processed_data)
            progress_bar.progress((idx + 1) / len(json_files))
        
        # Sidebar controls
        st.sidebar.header("Analysis Controls")
        
        score_type = st.sidebar.radio(
            "Select Score Type",
            ["bullish", "bearish"],
            help="Show industries with increasing bullish or bearish scores"
        )
        
        momentum_options = [
            'Latest Change',
            'Rolling Average',
            'Weighted Average',
            'Rate of Change'
        ]
        
        momentum_type = st.sidebar.selectbox(
            "Momentum Calculation Method",
            momentum_options,
            help="""
            Latest Change: Compare most recent two datasets
            Rolling Average: Average of changes over available periods
            Weighted Average: Recent changes weighted more heavily
            Rate of Change: Average change per period over all data
            """
        )
        
        top_n = st.sidebar.slider(
            "Number of Industries to Show",
            min_value=5,
            max_value=20,
            value=10
        )
        
        # Calculate momentum based on selected method
        changes_df = calculate_momentum(all_data, score_type, momentum_type)
        
        # Create and display the chart
        fig = create_momentum_chart(changes_df, score_type, top_n, momentum_type)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed changes table
        st.markdown(f"### Detailed Changes ({momentum_type})")
        
        # Prepare display dataframe
        display_df = changes_df.copy()
        display_df['start_date'] = pd.to_datetime(display_df['start_date']).dt.strftime('%Y-%m-%d')
        display_df['end_date'] = pd.to_datetime(display_df['end_date']).dt.strftime('%Y-%m-%d')
        
        # Sort by selected score type's change (positive only)
        change_col = f'{score_type}_change'
        display_df = display_df.nlargest(top_n, change_col)
        
        # Select columns to display
        display_columns = [
            'industry',
            change_col,
            'current_score',
            'symbol_curr',
            'start_date',
            'end_date'
        ]
        
        # Rename columns for display
        column_names = {
            change_col: f'{score_type.title()} Change',
            'current_score': f'Current {score_type.title()} Score',
            'symbol_curr': 'Stock Count'
        }
        
        st.dataframe(
            display_df[display_columns].rename(columns=column_names),
            use_container_width=True
        )
        
        # Add download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Momentum Data",
            data=csv,
            file_name=f"industry_{score_type}_momentum_{momentum_type}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        logger.error(f"Main application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()