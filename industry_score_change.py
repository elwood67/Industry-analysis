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
    """Load all historical JSON files excluding 'latest'."""
    try:
        # Get all JSON files EXCEPT 'latest'
        json_files = glob.glob(os.path.join(base_path, "market_analysis_*.json"))
        json_files = [f for f in json_files if 'latest' not in f]
        
        all_data = []
        for file_path in json_files:
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    df = pd.DataFrame(data['stocks'])
                    
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

def get_last_business_day(date):
    """Get the most recent business day from a given date."""
    while date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
        date = date - timedelta(days=1)
    return date

def process_data(df, selected_caps, score_type, time_period_days):
    """Process the dataframe to calculate score changes over time."""
    try:
        # Add market cap categories and filter
        df['market_cap_category'] = df['market_cap_B'].apply(categorize_market_cap)
        filtered_df = df[df['market_cap_category'].isin(selected_caps)] if selected_caps else df
        filtered_df = filtered_df.sort_values('date')
        
        # Check if we have enough data for comparison
        date_range = (filtered_df['date'].max() - filtered_df['date'].min()).days
        if date_range < (time_period_days * 2):
            raise ValueError(f"Not enough historical data for {time_period_days} day comparison. " 
                           f"Need at least {time_period_days * 2} days, but only have {date_range} days.")
        
        # Get the latest business day from our data
        latest_date = get_last_business_day(filtered_df['date'].max())
        
        # Calculate middle and earliest dates based on available trading days
        available_dates = sorted(filtered_df['date'].unique())
        available_dates = [d for d in available_dates if d.weekday() < 5]  # Remove weekends
        
        # Find index of latest date and calculate other dates
        try:
            latest_idx = available_dates.index(latest_date)
            middle_idx = max(0, latest_idx - time_period_days)
            earliest_idx = max(0, middle_idx - time_period_days)
            
            middle_date = available_dates[middle_idx]
            earliest_date = available_dates[earliest_idx]
        except ValueError:
            raise ValueError(f"Missing data for latest business day: {latest_date}")
        
        score_col = f"{score_type.lower()}_score"
        
        # Get scores for all three dates
        latest_scores = filtered_df[filtered_df['date'] == latest_date].groupby('industry').agg({
            score_col: 'mean',
            'symbol': 'count'
        }).reset_index()
        
        middle_scores = filtered_df[filtered_df['date'] == middle_date].groupby('industry').agg({
            score_col: 'mean'
        }).reset_index()
        
        earliest_scores = filtered_df[filtered_df['date'] == earliest_date].groupby('industry').agg({
            score_col: 'mean'
        }).reset_index()
        
        # Calculate changes
        merged_df = pd.merge(latest_scores, middle_scores, 
                           on='industry', suffixes=('_latest', '_middle'))
        merged_df = pd.merge(merged_df, earliest_scores, 
                           on='industry', suffixes=('', '_earliest'))
        
        # Calculate both periods' changes
        merged_df['current_change'] = merged_df[f"{score_col}_latest"] - merged_df[f"{score_col}_middle"]
        merged_df['previous_change'] = merged_df[f"{score_col}_middle"] - merged_df[f"{score_col}"]
        
        # Add few stocks flag and sort
        merged_df['few_stocks'] = merged_df['symbol'] < 5
        merged_df = merged_df.sort_values('current_change', ascending=True)
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

def create_change_chart(industry_metrics, score_type, time_period):
    """Create the plotly visualization for score changes."""
    try:
        fig = go.Figure()
        
        # Function to create hover text
        def create_hover_text(row, period_type):
            change_col = 'current_change' if period_type == 'current' else 'previous_change'
            return (
                f"<b>{row['industry']}</b><br>"
                f"{period_type.title()} Period Change: {row[change_col]:.1f}<br>"
                f"Stock Count: {int(row['symbol'])}"
            )
        
        # Add current period changes
        fig.add_trace(go.Bar(
            x=industry_metrics['current_change'],
            y=industry_metrics['industry'],
            orientation='h',
            name='Current Period',
            marker_color='rgb(99, 110, 250)',
            offsetgroup=0,
            text=industry_metrics['current_change'].apply(lambda x: f"{x:.1f}"),
            textposition='outside',
            hovertext=industry_metrics.apply(lambda row: create_hover_text(row, 'current'), axis=1),
            hoverinfo='text'
        ))
        
        # Add previous period changes
        fig.add_trace(go.Bar(
            x=industry_metrics['previous_change'],
            y=industry_metrics['industry'],
            orientation='h',
            name='Previous Period',
            marker_color='rgba(99, 110, 250, 0.5)',
            offsetgroup=1,
            text=industry_metrics['previous_change'].apply(lambda x: f"{x:.1f}"),
            textposition='outside',
            hovertext=industry_metrics.apply(lambda row: create_hover_text(row, 'previous'), axis=1),
            hoverinfo='text'
        ))
        
        # Add few stocks markers
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
            barmode='group',
            bargroupgap=0.1,
            bargap=0.05,
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
            
        try:
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
            
        except ValueError as e:
            st.warning(f"⚠️ {str(e)}")
            return
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()