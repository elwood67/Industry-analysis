import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
from datetime import datetime
import numpy as np

def load_data():
    """Load and merge market data and scores"""
    # Load market data
    market_df = pd.read_csv("./data/industry_classification/stock_sectors.csv")
    
    # Load scores
    with open("./data/stock_scores/market_analysis_latest.json", 'r') as file:
        scores_df = pd.DataFrame(json.load(file)['stocks'])
    
    # Merge data
    merged_df = market_df.merge(
        scores_df[['symbol', 'bullish_score', 'bearish_score', 'net_score']],
        on='symbol',
        how='left'
    )
    
    return merged_df

def create_treemap(df, score_type='net'):
    """Create the treemap visualization"""
    # Calculate aggregated metrics
    agg_data = df.groupby(['sector', 'industry']).agg({
        'market_cap_B': 'sum',
        'bullish_score': 'mean',
        'bearish_score': 'mean',
        'net_score': 'mean',
        'symbol': 'count'
    }).reset_index()
    
    # Add labels
    agg_data['label'] = agg_data.apply(
        lambda row: f"{row['industry']} ({row['symbol']} companies)",
        axis=1
    )
    
    # Set up color settings
    if score_type == 'net':
        color_column = 'net_score'
        max_abs_score = max(abs(agg_data[color_column].min()), abs(agg_data[color_column].max()))
        color_continuous_scale = [
            [0, 'rgb(165,0,38)'],
            [0.4, 'rgb(215,148,148)'],
            [0.5, 'rgb(255,255,255)'],
            [0.6, 'rgb(144,238,144)'],
            [1, 'rgb(0,100,0)']
        ]
        range_color = [-max_abs_score, max_abs_score]
    elif score_type == 'bullish':
        color_column = 'bullish_score'
        color_continuous_scale = 'Greens'
        range_color = [0, 100]
    else:
        color_column = 'bearish_score'
        color_continuous_scale = 'Reds'
        range_color = [0, 100]

    # Create figure
    fig = px.treemap(
        agg_data,
        path=['sector', 'industry'],
        values='market_cap_B',
        color=color_column,
        color_continuous_scale=color_continuous_scale,
        range_color=range_color
    )

    # Update hover template
    hover_template = (
        "<b>%{label}</b><br>" +
        "Market Cap: $%{value:.1f}B<br>" +
        f"{score_type.title()} Score: %{{color:.1f}}" +
        "<extra></extra>"
    )

    fig.update_traces(
        textinfo="label",
        hovertemplate=hover_template
    )

    fig.update_layout(
        title=dict(
            text=f"Market Structure Heat Map ({score_type.title()} Score)",
            x=0.5,
            xanchor='center'
        ),
        height=800
    )

    return fig

def show_industry_details(df, industry):
    """Show details for selected industry"""
    stocks = df[df['industry'] == industry].copy()
    stocks = stocks.sort_values('market_cap_B', ascending=False)
    
    cols = ['symbol', 'market_cap_B', 'bullish_score', 'bearish_score', 'net_score']
    display_df = stocks[cols].copy()
    display_df.columns = ['Symbol', 'Market Cap ($B)', 'Bullish Score', 'Bearish Score', 'Net Score']
    
    st.markdown(f"### {industry}")
    st.dataframe(
        display_df.style.format({
            'Market Cap ($B)': '{:.2f}',
            'Bullish Score': '{:.2f}',
            'Bearish Score': '{:.2f}',
            'Net Score': '{:.2f}'
        }),
        use_container_width=True
    )

def main():
    st.set_page_config(layout="wide")
    
    st.title("Market Structure Score Analysis")
    st.write("Visualize sector and industry scores through a heat map. Click an industry to see its stocks.")
    
    # Score type selection
    score_type = st.radio(
        "Select Score Type for Heat Map",
        options=['net', 'bullish', 'bearish'],
        format_func=lambda x: f"{x.title()} Score",
        horizontal=True,
        key='score_type'
    )
    
    # Load data
    df = load_data()
    
    # Create treemap
    fig = create_treemap(df, score_type)
    
    # Display treemap
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
        'toImageButtonOptions': {'height': None, 'width': None}
    })
    
    # Add industry selector
    industries = sorted(df['industry'].unique())
    selected_industry = st.selectbox(
        "Select an industry to view details",
        options=industries,
        key='industry_selector'
    )
    
    if selected_industry:
        show_industry_details(df, selected_industry)

if __name__ == "__main__":
    main()