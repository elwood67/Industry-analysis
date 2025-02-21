import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data
def load_historical_data(base_path="Data/stock_scores"):
    """Load historical data from the consolidated file."""
    try:
        file_path = Path(base_path) / 'historical_data.parquet.gzip'
        if not file_path.exists():
            raise ValueError("Historical data file not found. Please run the optimizer first.")
            
        df = pd.read_parquet(file_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
        
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        raise

def get_time_period_options():
    """Return available time period options."""
    return {
        "1 Day": 1,
        "1 Week": 7,
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
        # Add market cap categories and filter
        df['market_cap_category'] = df['market_cap_B'].apply(categorize_market_cap)
        filtered_df = df[df['market_cap_category'].isin(selected_caps)] if selected_caps else df
        
        # Get all trading days (excluding weekends)
        trading_days = sorted(filtered_df['date'].unique())
        trading_days = [d for d in trading_days if d.weekday() < 5]
        
        if len(trading_days) < 3:
            raise ValueError("Not enough trading days available")
            
        # Get the most recent trading day
        latest_date = trading_days[-1]
        
        # Calculate previous dates based on trading days
        if time_period_days <= 5:
            # For short periods, use consecutive trading days
            middle_date = trading_days[-2]
            earliest_date = trading_days[-3]
        else:
            # For longer periods, find closest matching days
            middle_idx = max(0, len(trading_days) - time_period_days - 1)
            earliest_idx = max(0, middle_idx - time_period_days)
            middle_date = trading_days[middle_idx]
            earliest_date = trading_days[earliest_idx]
        
        logger.info(f"Processing dates: Latest={latest_date}, Middle={middle_date}, Earliest={earliest_date}")
        
        score_col = f"{score_type.lower()}_score"
        
        # Calculate metrics
        metrics = []
        for date in [latest_date, middle_date, earliest_date]:
            daily_data = filtered_df[filtered_df['date'] == date].groupby('industry', observed=True).agg({
                score_col: 'mean',
                'symbol': 'count'
            }).reset_index()
            metrics.append(daily_data)
        
        # Merge and calculate changes
        merged_df = pd.merge(metrics[0], metrics[1], on='industry', suffixes=('_latest', '_middle'))
        merged_df = pd.merge(merged_df, metrics[2], on='industry', suffixes=('', '_earliest'))
        
        merged_df['current_change'] = merged_df[f"{score_col}_latest"] - merged_df[f"{score_col}_middle"]
        merged_df['previous_change'] = merged_df[f"{score_col}_middle"] - merged_df[f"{score_col}"]
        merged_df['few_stocks'] = merged_df['symbol_latest'] < 5
        
        return merged_df.sort_values('current_change', ascending=True)
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise

def create_change_chart(industry_metrics, score_type, time_period):
    """Create the plotly visualization for score changes."""
    try:
        fig = go.Figure()
        
        # Create hover text
        def create_hover_text(row, period_type):
            change_col = 'current_change' if period_type == 'current' else 'previous_change'
            score_col = f"{score_type.lower()}_score_latest"
            return (
                f"<b>{row['industry']}</b><br>"
                f"Current Score: {row[score_col]:.1f}<br>"
                f"{period_type.title()} Period Change: {row[change_col]:.1f}<br>"
                f"Stock Count: {int(row['symbol_latest'])}"
            )
        
        # Add bars for both periods
        for period, color in [('current', 'rgb(99, 110, 250)'), ('previous', 'rgba(99, 110, 250, 0.5)')]:
            change_col = f'{period}_change'
            fig.add_trace(go.Bar(
                x=industry_metrics[change_col],
                y=industry_metrics['industry'],
                orientation='h',
                name=f'{period.title()} Period',
                marker_color=color,
                text=industry_metrics[change_col].apply(lambda x: f"{x:.1f}"),
                textposition='outside',
                hovertext=[create_hover_text(row, period) for _, row in industry_metrics.iterrows()],
                hoverinfo='text'
            ))
        
        # Add markers for industries with few stocks
        few_stocks = industry_metrics[industry_metrics['few_stocks']]
        if not few_stocks.empty:
            max_change = max(
                industry_metrics['current_change'].max(),
                industry_metrics['previous_change'].max()
            )
            fig.add_trace(go.Scatter(
                x=[max_change * 1.05] * len(few_stocks),
                y=few_stocks['industry'],
                mode='markers',
                name='< 5 stocks',
                marker=dict(color='red', size=10, symbol='triangle-right'),
                hovertext=[f"Only {int(count)} stocks" for count in few_stocks['symbol_latest']],
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title=f'Industry {score_type} Score Changes ({time_period})',
            xaxis_title=f'{score_type} Score Change',
            yaxis_title=None,
            height=max(600, len(industry_metrics) * 30),
            showlegend=True,
            barmode='group',
            bargap=0.1,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                gridwidth=1
            )
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        raise

def main():
    st.title("Stock Market Industry Score Changes")
    
    try:
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
        time_periods = get_time_period_options()
        time_period = st.sidebar.selectbox(
            "Select Time Period",
            options=list(time_periods.keys())
        )
        
        if not selected_caps:
            st.warning("👈 Select market cap categories from the sidebar to begin analysis!")
            return
            
        # Load and process data
        with st.spinner("Loading data..."):
            df = load_historical_data()
            
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
            st.sidebar.markdown(f"Total Stocks: {int(industry_metrics['symbol_latest'].sum())}")
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