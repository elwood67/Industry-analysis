import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def categorize_market_cap(cap):
    """Categorize market cap value into size category."""
    try:
        if pd.isna(cap):
            return "Unknown"
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
    except Exception as e:
        logger.error(f"Error in market cap categorization: {str(e)}")
        return "Unknown"

@st.cache_data
def load_historical_data(base_path="Data/stock_scores"):
    """Load data from the optimized parquet file."""
    try:
        file_path = Path(base_path) / 'historical_data.parquet.gzip'
        if not file_path.exists():
            raise ValueError("Historical data file not found. Please run the optimizer first.")
            
        df = pd.read_parquet(file_path)
        
        # Ensure date is datetime and add market cap categories
        df['date'] = pd.to_datetime(df['date'])
        df['market_cap_category'] = df['market_cap_B'].apply(categorize_market_cap)
        
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        raise

def calculate_trend_shifts(df, selected_caps, score_type, lookback_days):
    """Calculate recent trend shifts for industries."""
    try:
        # Filter by market cap if selected
        if selected_caps:
            df = df[df['market_cap_category'].isin(selected_caps)]
        
        # Get unique dates and find the lookback period
        all_dates = sorted(df['date'].unique())
        if len(all_dates) < lookback_days + 5:
            raise ValueError(f"Not enough data. Need at least {lookback_days + 5} trading days.")
        
        # Define periods
        latest_date = all_dates[-1]
        recent_start = all_dates[-lookback_days]
        previous_start = all_dates[-2*lookback_days] if len(all_dates) >= 2*lookback_days else all_dates[0]
        previous_end = all_dates[-lookback_days-1]
        
        score_col = f"{score_type.lower()}_score"
        
        # Calculate average scores for each period by industry
        recent_scores = df[
            (df['date'] >= recent_start) & (df['date'] <= latest_date)
        ].groupby('industry')[score_col].mean()
        
        previous_scores = df[
            (df['date'] >= previous_start) & (df['date'] <= previous_end)
        ].groupby('industry')[score_col].mean()
        
        # Calculate stock counts
        stock_counts = df[df['date'] == latest_date].groupby('industry')['symbol'].count()
        
        # Combine data
        trend_data = pd.DataFrame({
            'industry': recent_scores.index,
            'recent_avg': recent_scores.values,
            'previous_avg': previous_scores.reindex(recent_scores.index).values,
            'stock_count': stock_counts.reindex(recent_scores.index).fillna(0).astype(int)
        }).dropna()
        
        # Calculate changes and momentum
        trend_data['score_change'] = trend_data['recent_avg'] - trend_data['previous_avg']
        trend_data['percent_change'] = (trend_data['score_change'] / trend_data['previous_avg'] * 100).round(2)
        
        # Categorize trends
        trend_data['trend_strength'] = pd.cut(
            trend_data['score_change'].abs(),
            bins=[0, 2, 5, 10, float('inf')],
            labels=['Weak', 'Moderate', 'Strong', 'Very Strong']
        )
        
        trend_data['trend_direction'] = np.where(
            trend_data['score_change'] > 1, 'Improving',
            np.where(trend_data['score_change'] < -1, 'Declining', 'Stable')
        )
        
        return trend_data.sort_values('score_change', ascending=False)
        
    except Exception as e:
        logger.error(f"Error calculating trend shifts: {str(e)}")
        raise

def create_trend_shift_chart(trend_data, score_type):
    """Create a comprehensive trend shift visualization."""
    try:
        if trend_data.empty:
            return go.Figure().update_layout(title="No data available")
        
        # Create color mapping based on change
        colors = []
        for change in trend_data['score_change']:
            if change > 5:
                colors.append('#00CC00')  # Bright green for strong positive
            elif change > 2:
                colors.append('#66FF66')  # Light green for moderate positive
            elif change > -2:
                colors.append('#CCCCCC')  # Gray for stable
            elif change > -5:
                colors.append('#FF6666')  # Light red for moderate negative
            else:
                colors.append('#CC0000')  # Bright red for strong negative
        
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            x=trend_data['score_change'],
            y=trend_data['industry'],
            orientation='h',
            marker_color=colors,
            text=[f"{change:+.1f}" for change in trend_data['score_change']],
            textposition='auto',
            hovertemplate=(
                "<b>%{y}</b><br>" +
                f"{score_type} Score Change: %{{x:+.1f}}<br>" +
                "Recent Average: %{customdata[0]:.1f}<br>" +
                "Previous Average: %{customdata[1]:.1f}<br>" +
                "Stock Count: %{customdata[2]}<br>" +
                "Trend: %{customdata[3]}<br>" +
                "<extra></extra>"
            ),
            customdata=np.column_stack((
                trend_data['recent_avg'].round(1),
                trend_data['previous_avg'].round(1),
                trend_data['stock_count'],
                trend_data['trend_direction']
            ))
        ))
        
        # Add reference line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        
        fig.update_layout(
            title=f'Recent {score_type} Score Trend Shifts by Industry',
            xaxis_title=f'{score_type} Score Change',
            yaxis_title='Industry',
            height=max(600, len(trend_data) * 25),
            template='plotly_dark',
            showlegend=False,
            margin=dict(l=150, r=50, t=80, b=50)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating trend chart: {str(e)}")
        return go.Figure().update_layout(title="Error creating chart")

def create_momentum_scatter(trend_data, score_type):
    """Create a scatter plot showing current score vs momentum."""
    try:
        fig = px.scatter(
            trend_data,
            x='recent_avg',
            y='score_change',
            size='stock_count',
            color='trend_direction',
            hover_name='industry',
            color_discrete_map={
                'Improving': '#00CC00',
                'Declining': '#CC0000',
                'Stable': '#CCCCCC'
            },
            title=f'Current {score_type} Score vs Momentum',
            labels={
                'recent_avg': f'Current {score_type} Score',
                'score_change': f'{score_type} Score Change',
                'stock_count': 'Number of Stocks'
            }
        )
        
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.add_vline(x=trend_data['recent_avg'].median(), line_dash="dash", line_color="white", opacity=0.5)
        
        fig.update_layout(
            template='plotly_dark',
            height=500
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating momentum scatter: {str(e)}")
        return go.Figure().update_layout(title="Error creating scatter plot")

def display_trend_summary(trend_data):
    """Display summary statistics about trends."""
    try:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            improving = len(trend_data[trend_data['trend_direction'] == 'Improving'])
            st.metric("Improving Industries", improving)
        
        with col2:
            declining = len(trend_data[trend_data['trend_direction'] == 'Declining'])
            st.metric("Declining Industries", declining)
        
        with col3:
            stable = len(trend_data[trend_data['trend_direction'] == 'Stable'])
            st.metric("Stable Industries", stable)
        
        with col4:
            avg_change = trend_data['score_change'].mean()
            st.metric("Average Change", f"{avg_change:+.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error displaying summary: {str(e)}")
        return False

def main():
    st.set_page_config(page_title="Industry Trend Shifts", layout="wide")
    st.title("🔄 Industry Score Trend Shifts Analysis")
    
    try:
        # Load data
        with st.spinner("Loading historical data..."):
            df = load_historical_data()
        
        # Sidebar controls
        st.sidebar.header("Analysis Settings")
        
        # Market cap filter
        cap_categories = [
            "Mega Cap", "Large Cap", "Mid Cap",
            "Small Cap", "Micro Cap", "Nano Cap"
        ]
        selected_caps = st.sidebar.multiselect(
            "Select Market Cap Categories",
            options=cap_categories,
            default=["Large Cap", "Mid Cap"]  # Default selection
        )
        
        # Score type selection
        score_type = st.sidebar.radio(
            "Select Score Type",
            ["Bullish", "Bearish"]
        )
        
        # Lookback period
        lookback_options = {
            "1 Week (5 days)": 5,
            "2 Weeks (10 days)": 10,
            "3 Weeks (15 days)": 15,
            "1 Month (21 days)": 21,
            "6 Weeks (30 days)": 30
        }
        
        lookback_selection = st.sidebar.selectbox(
            "Trend Analysis Period",
            options=list(lookback_options.keys()),
            index=2  # Default to 3 weeks
        )
        lookback_days = lookback_options[lookback_selection]
        
        # Data info
        st.sidebar.markdown("### Data Information")
        st.sidebar.info(f"""
        **Date Range:** {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}
        
        **Total Trading Days:** {len(df['date'].unique())}
        
        **Analysis Period:** {lookback_selection}
        """)
        
        if not selected_caps:
            st.warning("👈 Please select at least one Market Cap Category to begin analysis!")
            return
        
        # Calculate trend shifts
        with st.spinner("Analyzing trend shifts..."):
            trend_data = calculate_trend_shifts(df, selected_caps, score_type, lookback_days)
        
        if trend_data.empty:
            st.warning("No trend data available for the selected criteria.")
            return
        
        # Display summary metrics
        st.markdown("### 📊 Trend Summary")
        display_trend_summary(trend_data)
        
        # Create two columns for charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Main trend shift chart
            st.markdown("### 📈 Industry Trend Shifts")
            fig1 = create_trend_shift_chart(trend_data, score_type)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Top movers table
            st.markdown("### 🏆 Top Movers")
            
            # Top improving
            st.markdown("**Most Improving:**")
            top_improving = trend_data.head(5)[['industry', 'score_change', 'stock_count']]
            st.dataframe(top_improving, hide_index=True)
            
            # Top declining
            st.markdown("**Most Declining:**")
            top_declining = trend_data.tail(5)[['industry', 'score_change', 'stock_count']]
            st.dataframe(top_declining, hide_index=True)
        
        # Momentum scatter plot
        st.markdown("### 🎯 Score vs Momentum Analysis")
        fig2 = create_momentum_scatter(trend_data, score_type)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed data table
        st.markdown("### 📋 Detailed Analysis")
        
        # Filter options for the table
        trend_filter = st.selectbox(
            "Filter by Trend Direction",
            options=["All", "Improving", "Declining", "Stable"]
        )
        
        if trend_filter != "All":
            filtered_data = trend_data[trend_data['trend_direction'] == trend_filter]
        else:
            filtered_data = trend_data
        
        # Display formatted table
        display_data = filtered_data[['industry', 'recent_avg', 'previous_avg', 'score_change', 'percent_change', 'stock_count', 'trend_direction']].copy()
        display_data.columns = ['Industry', 'Recent Avg', 'Previous Avg', 'Change', '% Change', 'Stocks', 'Trend']
        display_data = display_data.round(2)
        
        st.dataframe(display_data, use_container_width=True, hide_index=True)
        
        # Download button
        csv = trend_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Trend Analysis Data",
            data=csv,
            file_name=f"trend_shifts_{score_type.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred while analyzing the data.")
        if st.checkbox("Show error details"):
            st.exception(e)

if __name__ == "__main__":
    main()