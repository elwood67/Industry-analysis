import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
import logging
from datetime import datetime, timedelta
import glob

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_historical_data(base_path="Data/stock_scores"):
    """Load all historical JSON files including the latest."""
    try:
        # Get all JSON files
        json_files = glob.glob(os.path.join(base_path, "market_analysis_*.json"))
        
        all_data = []
        for file_path in json_files:
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    df = pd.DataFrame(data['stocks'])
                    
                    # Get date from filename
                    if 'latest' in file_path:
                        # Use file modification time for latest file
                        timestamp = os.path.getmtime(file_path)
                        date_str = datetime.fromtimestamp(timestamp).strftime('%Y%m%d')
                    else:
                        # Extract date from filename
                        date_str = os.path.basename(file_path).split('_')[2].split('.')[0]
                    
                    df['date'] = pd.to_datetime(date_str, format='%Y%m%d')
                    all_data.append(df)
                    logger.debug(f"Successfully processed {file_path}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
        
        return pd.concat(all_data, ignore_index=True)
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        raise

def get_time_period_options():
    """Return available time period options."""
    return {
        "1 Day": 1,
        "1 Week": 5,
        "1 Month": 21,
        "2 Months": 42,
        "3 Months": 63,
        "4 Months": 84,
        "5 Months": 105,
        "6 Months": 126
    }

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

def process_data(df, selected_caps, score_type, time_period_days):
    """Process the dataframe to calculate score changes over time."""
    try:
        # Add market cap categories
        df['market_cap_category'] = df['market_cap_B'].apply(categorize_market_cap)
        
        # Filter by market cap if selected
        filtered_df = df[df['market_cap_category'].isin(selected_caps)] if selected_caps else df
        
        # Sort by date
        filtered_df = filtered_df.sort_values('date')
        
        # Get the latest date and the comparison date
        latest_date = filtered_df['date'].max()
        comparison_date = latest_date - timedelta(days=time_period_days)
        
        # Get scores for both dates
        score_col = f"{score_type.lower()}_score"
        
        latest_scores = filtered_df[filtered_df['date'] == latest_date].groupby('industry').agg({
            score_col: 'mean',
            'symbol': 'count'
        }).reset_index()
        
        comparison_scores = filtered_df[
            (filtered_df['date'] >= comparison_date) & 
            (filtered_df['date'] <= comparison_date + timedelta(days=1))
        ].groupby('industry').agg({
            score_col: 'mean'
        }).reset_index()
        
        # Merge and calculate changes
        merged_df = pd.merge(
            latest_scores, 
            comparison_scores, 
            on='industry', 
            suffixes=('_latest', '_previous')
        )
        
        # Calculate absolute and percentage changes
        merged_df['absolute_change'] = merged_df[f"{score_col}_latest"] - merged_df[f"{score_col}_previous"]
        merged_df['percentage_change'] = (
            (merged_df[f"{score_col}_latest"] - merged_df[f"{score_col}_previous"]) / 
            merged_df[f"{score_col}_previous"] * 100
        )
        
        # Add few stocks flag
        merged_df['few_stocks'] = merged_df['symbol'] < 5
        
        # Sort by absolute change
        merged_df = merged_df.sort_values('absolute_change', ascending=True)
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

def create_change_chart(industry_metrics, score_type, time_period):
    """Create the plotly visualization for score changes."""
    try:
        # Create figure
        fig = go.Figure()
        
        # Split data into positive and negative changes
        positive_data = industry_metrics[industry_metrics['absolute_change'] >= 0].copy()
        negative_data = industry_metrics[industry_metrics['absolute_change'] < 0].copy()
        
        # Function to create hover text
        def create_hover_text(row, score_type):
            return (
                f"<b>{row['industry']}</b><br>"
                f"Change: {row['absolute_change']:.1f}<br>"
                f"Current Score: {row[f'{score_type}_score_latest']:.1f}<br>"
                f"Previous Score: {row[f'{score_type}_score_previous']:.1f}<br>"
                f"% Change: {row['percentage_change']:.1f}%<br>"
                f"Stock Count: {int(row['symbol'])}"
            )
        
        # Add negative changes
        if not negative_data.empty:
            fig.add_trace(go.Bar(
                x=negative_data['absolute_change'],
                y=negative_data['industry'],
                orientation='h',
                name='Negative Change',
                marker_color='rgb(239, 85, 59)',
                text=negative_data.apply(
                    lambda row: f"{row['absolute_change']:.1f}",
                    axis=1
                ),
                textposition='outside',
                hovertext=negative_data.apply(lambda row: create_hover_text(row, score_type.lower()), axis=1),
                hoverinfo='text'
            ))
        
        # Add positive changes
        if not positive_data.empty:
            fig.add_trace(go.Bar(
                x=positive_data['absolute_change'],
                y=positive_data['industry'],
                orientation='h',
                name='Positive Change',
                marker_color='rgb(99, 110, 250)',
                text=positive_data.apply(
                    lambda row: f"{row['absolute_change']:.1f}",
                    axis=1
                ),
                textposition='outside',
                hovertext=positive_data.apply(lambda row: create_hover_text(row, score_type.lower()), axis=1),
                hoverinfo='text'
            ))
        
        # Add few stocks markers
        few_stocks = industry_metrics[industry_metrics['few_stocks']]
        if not few_stocks.empty:
            max_change = industry_metrics['absolute_change'].max()
            fig.add_trace(go.Scatter(
                x=[max_change * 1.05] * len(few_stocks),
                y=few_stocks['industry'],
                mode='markers',
                name='< 5 stocks',
                marker=dict(color='red', size=10, symbol='triangle-right'),
                hovertext=[f"Only {int(count)} stocks" for count in few_stocks['symbol']],
                hoverinfo='text'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Industry {score_type} Score Changes ({time_period})',
            xaxis_title=f'{score_type} Score Change',
            yaxis_title=None,
            height=max(600, len(industry_metrics) * 30),
            showlegend=True,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode='y unified',
            yaxis=dict(
                showticklabels=True,
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinecolor='white',
                zerolinewidth=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}")
        raise

def main():
    st.title("Stock Market Industry Score Changes")
    
    try:
        # Load data
        df = load_historical_data()
        
        # Get available time periods based on data
        max_days = (df['date'].max() - df['date'].min()).days
        time_periods = {k: v for k, v in get_time_period_options().items() 
                       if v <= max_days}
        
        # Sidebar controls
        st.sidebar.header("Analysis Controls")
        
        # Market cap categories
        cap_categories = [
            "Mega Cap", "Large Cap", "Mid Cap",
            "Small Cap", "Micro Cap", "Nano Cap"
        ]
        
        selected_caps = st.sidebar.multiselect(
            "Select Market Cap Categories",
            options=cap_categories
        )
        
        # Score type selection
        score_type = st.sidebar.radio(
            "Select Score Type",
            ["Bullish", "Bearish"]
        )
        
        # Time period selection
        time_period = st.sidebar.selectbox(
            "Select Time Period",
            options=list(time_periods.keys())
        )
        
        if not selected_caps:
            st.warning("👈 Select market cap categories from the sidebar to begin analysis!")
            return
            
        # Process data
        industry_metrics = process_data(
            df, 
            selected_caps, 
            score_type, 
            time_periods[time_period]
        )
        
        # Create visualization
        fig = create_change_chart(industry_metrics, score_type, time_period)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.sidebar.markdown("### Summary Statistics")
        st.sidebar.markdown(f"Total Industries: {len(industry_metrics)}")
        st.sidebar.markdown(f"Total Stocks: {int(industry_metrics['symbol'].sum())}")
        st.sidebar.markdown(f"Industries with <5 stocks: {sum(industry_metrics['few_stocks'])}")
        
        # Download button
        csv = industry_metrics.to_csv(index=False)
        st.sidebar.download_button(
            label="Download Analysis Data",
            data=csv,
            file_name=f"industry_changes_{score_type.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()