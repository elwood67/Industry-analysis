import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
from datetime import datetime
import glob
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Extended color palette with more distinct colors
COLORS = [
    '#1f77b4',  # Blue
    '#d62728',  # Red
    '#2ca02c',  # Green
    '#9467bd',  # Purple
    '#ff7f0e',  # Orange
    '#17becf',  # Cyan
    '#e377c2',  # Pink
    '#bcbd22',  # Yellow-green
    '#7f7f7f',  # Gray
    '#8c564b',  # Brown
    '#3366cc',  # Royal blue
    '#dc3912',  # Bright red
    '#109618',  # Dark green
    '#990099',  # Purple
    '#0099c6',  # Light blue
    '#dd4477',  # Rose
    '#66aa00',  # Lime green
    '#b82e2e',  # Dark red
    '#316395',  # Navy
    '#994499',  # Dark purple
]

def load_historical_data(base_path="Data/stock_scores"):
    """Load and process all historical JSON files."""
    try:
        json_files = glob.glob(os.path.join(base_path, "market_analysis_*.json"))
        json_files = [f for f in json_files if 'latest' not in f]
        json_files.sort(key=lambda x: os.path.getmtime(x))
        
        all_data = []
        for file_path in json_files:
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    df = pd.DataFrame(data['stocks'])
                    
                    # Extract date from filename
                    date_str = os.path.basename(file_path).split('_')[2].split('.')[0]
                    df['date'] = date_str
                    
                    all_data.append(df)
                    logger.debug(f"Successfully processed {file_path}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
        
        return pd.concat(all_data, ignore_index=True)
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        raise

def create_comparison_chart(df, selected_categories, category_type, score_type):
    """Create a comparison chart for selected categories."""
    # Calculate average scores for each category by date
    grouped = df.groupby(['date', category_type])[score_type].mean().reset_index()
    
    # Pivot the data for plotting
    pivot_data = grouped.pivot(index='date', columns=category_type, values=score_type)
    
    # Create figure
    fig = go.Figure()
    
    # Add a line for each selected category
    for i, category in enumerate(selected_categories):
        if category in pivot_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=pivot_data.index,
                    y=pivot_data[category],
                    name=category,
                    line=dict(color=COLORS[i % len(COLORS)], width=2),
                    hovertemplate=f"{category}<br>Date: %{{x}}<br>{score_type}: %{{y:.2f}}<extra></extra>"
                )
            )
    
    # Update layout
    fig.update_layout(
        title=f'{category_type.title()} Comparison - {score_type.replace("_", " ").title()}',
        xaxis_title='Date',
        yaxis_title=score_type.replace('_', ' ').title(),
        template='plotly_dark',
        height=600,
        hovermode='closest',  # Changed to 'closest' for individual line hover
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        xaxis=dict(
            type='category',
            tickangle=-45,
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=True,
            zerolinecolor='rgba(255, 255, 255, 0.2)'
        ),
    )
    
    return fig

def main():
    st.set_page_config(page_title="Industry/Sector Score Comparison", layout="wide")
    st.title("Industry/Sector Score Comparison")
    
    try:
        # Load data
        with st.spinner("Loading historical data..."):
            df = load_historical_data()
        
        # Sidebar controls
        st.sidebar.header("Visualization Controls")
        
        # Score type selection
        score_type = st.sidebar.radio(
            "Select Score Type",
            ["net_score", "bullish_score", "bearish_score"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Category type selection (Industry/Sector)
        category_type = st.sidebar.radio(
            "Select Category Type",
            ["industry", "sector"]
        )
        
        # Get unique categories
        categories = sorted(df[category_type].unique())
        
        # Multi-select for categories
        selected_categories = st.sidebar.multiselect(
            f"Select {category_type.title()}s to Compare",
            options=categories,
            default=categories[:3] if len(categories) >= 3 else categories
        )
        
        if not selected_categories:
            st.warning(f"Please select at least one {category_type} to display.")
            return
            
        # Create and display chart
        fig = create_comparison_chart(df, selected_categories, category_type, score_type)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred during analysis.")
        if st.sidebar.checkbox("Show Error Details"):
            st.error(str(e))

if __name__ == "__main__":
    main()