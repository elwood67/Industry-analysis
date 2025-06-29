import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load and parse the JSON data file."""
    base_path = "Data/stock_scores"
    latest_file = os.path.join(base_path, "market_analysis_latest.json")
    
    try:
        with open(latest_file, 'r') as file:
            data = json.load(file)
        return pd.DataFrame(data['stocks'])
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def categorize_market_cap(cap):
    """Categorize market cap value into size category."""
    if cap >= 200:
        return "Mega Cap"
    elif cap >= 10:
        return "Large Cap"
    elif cap >= 2:
        return "Mid Cap"
    elif cap >= 0.3:
        return "Small Cap"
    elif cap >= 0.05:
        return "Micro Cap"
    else:
        return "Nano Cap"

def process_data(df, selected_caps, score_type):
    """Process the dataframe based on selected filters."""
    try:
        # Add market cap categories
        df['market_cap_category'] = df['market_cap_B'].apply(categorize_market_cap)
        
        # Filter by market cap
        filtered_df = df[df['market_cap_category'].isin(selected_caps)] if selected_caps else df
        
        # Group by industry
        score_col = f"{score_type.lower()}_score"
        industry_metrics = filtered_df.groupby('industry').agg({
            'symbol': 'count',
            score_col: 'mean'
        }).reset_index()
        
        # Add few stocks flag
        industry_metrics['few_stocks'] = industry_metrics['symbol'] < 5
        
        # Sort by score
        industry_metrics = industry_metrics.sort_values(score_col, ascending=True)
        
        return industry_metrics
    
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

def create_score_chart(industry_metrics, score_type):
    """Create the plotly visualization."""
    try:
        score_col = f"{score_type.lower()}_score"
        
        fig = go.Figure()
        
        # Add bars with fixed hover template
        fig.add_trace(go.Bar(
            x=industry_metrics[score_col],
            y=industry_metrics['industry'],
            orientation='h',
            name=score_type,
            marker_color='rgb(49, 130, 189)',
            text=industry_metrics['symbol'].apply(lambda x: f"{int(x)} stocks"),
            textposition='auto',
            customdata=industry_metrics['symbol'],
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "Score: %{x:.1f}<br>" +
                "Stock count: %{customdata}<br>" +
                "<extra></extra>"
            )
        ))
        
        # Add markers for few stocks
        few_stocks = industry_metrics[industry_metrics['few_stocks']]
        if not few_stocks.empty:
            fig.add_trace(go.Scatter(
                x=[max(industry_metrics[score_col]) * 1.05] * len(few_stocks),
                y=few_stocks['industry'],
                mode='markers',
                name='< 5 stocks',
                marker=dict(color='red', size=10, symbol='triangle-right'),
                hovertext=[f"Only {int(count)} stocks" for count in few_stocks['symbol']]
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Industry {score_type} Score Analysis',
            xaxis_title=f'Average {score_type} Score',
            yaxis_title='Industry',
            height=max(600, len(industry_metrics) * 30),
            showlegend=True,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode='closest',
            hoverlabel=dict(
                bgcolor="#1e1e1e",
                font=dict(color="white")
            )
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}")
        raise

def main():
    st.title("Stock Market Industry Analysis")
    
    try:
        # Load data
        df = load_data()
        
        # Sidebar controls
        st.sidebar.header("Market Cap Filters")
        
        # Market cap categories
        cap_categories = [
            "Mega Cap", "Large Cap", "Mid Cap",
            "Small Cap", "Micro Cap", "Nano Cap"
        ]
        
        # Simple multiselect without default values
        selected_caps = st.sidebar.multiselect(
            "Select Market Cap Categories",
            options=cap_categories
        )
        
        # Score type selection
        score_type = st.sidebar.radio(
            "Select Score Type",
            ["Bullish", "Bearish"]
        )
        
        if not selected_caps:
            st.warning("👈 Click Menu arrow on the left to combine any of the market cap categories and begin your analysis!")
            return
            
        # Process data
        industry_metrics = process_data(df, selected_caps, score_type)
        
        # Create visualization
        fig = create_score_chart(industry_metrics, score_type)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.sidebar.markdown("### Summary Statistics")
        st.sidebar.markdown(f"Total Industries: {len(industry_metrics)}")
        st.sidebar.markdown(f"Total Stocks: {int(industry_metrics['symbol'].sum())}")
        st.sidebar.markdown(f"Industries with <5 stocks: {sum(industry_metrics['few_stocks'])}")
        st.sidebar.markdown("# 📊 Menu")  # Add a clear menu header with emoji

        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()