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
    """Calculate recent trend shifts and market phase analysis for industries."""
    try:
        # Filter by market cap if selected
        if selected_caps:
            df = df[df['market_cap_category'].isin(selected_caps)]
        
        # Get unique dates and find the lookback period
        all_dates = sorted(df['date'].unique())
        if len(all_dates) < lookback_days * 3:
            raise ValueError(f"Not enough data. Need at least {lookback_days * 3} trading days for phase analysis.")
        
        # Define three periods for phase analysis
        latest_date = all_dates[-1]
        recent_start = all_dates[-lookback_days]
        middle_start = all_dates[-2*lookback_days]
        middle_end = all_dates[-lookback_days-1]
        early_start = all_dates[-3*lookback_days] if len(all_dates) >= 3*lookback_days else all_dates[0]
        early_end = all_dates[-2*lookback_days-1]
        
        score_col = f"{score_type.lower()}_score"
        
        # Also fix the pandas warning by adding observed=True to groupby operations
        # Calculate average scores for each period by industry
        recent_scores = df[
            (df['date'] >= recent_start) & (df['date'] <= latest_date)
        ].groupby('industry', observed=True)[score_col].mean()
        
        middle_scores = df[
            (df['date'] >= middle_start) & (df['date'] <= middle_end)
        ].groupby('industry', observed=True)[score_col].mean()
        
        early_scores = df[
            (df['date'] >= early_start) & (df['date'] <= early_end)
        ].groupby('industry', observed=True)[score_col].mean()
        
        # Calculate stock counts
        stock_counts = df[df['date'] == latest_date].groupby('industry', observed=True)['symbol'].count()
        
        # Combine data
        trend_data = pd.DataFrame({
            'industry': recent_scores.index,
            'recent_avg': recent_scores.values,
            'middle_avg': middle_scores.reindex(recent_scores.index).values,
            'early_avg': early_scores.reindex(recent_scores.index).values,
            'stock_count': stock_counts.reindex(recent_scores.index).fillna(0).astype(int)
        }).dropna()
        
        # Calculate changes and momentum
        trend_data['recent_change'] = trend_data['recent_avg'] - trend_data['middle_avg']
        trend_data['middle_change'] = trend_data['middle_avg'] - trend_data['early_avg']
        trend_data['total_change'] = trend_data['recent_avg'] - trend_data['early_avg']
        
        # Phase Analysis - Detect bottoming patterns
        trend_data['momentum_shift'] = trend_data['recent_change'] - trend_data['middle_change']
        
        # Identify market phases
        def identify_phase(row):
            recent_chg = row['recent_change']
            middle_chg = row['middle_change']
            momentum_shift = row['momentum_shift']
            recent_score = row['recent_avg']
            
            # Phase 4: Declining (both periods negative)
            if recent_chg < -1 and middle_chg < -1:
                return "Phase 4 - Declining"
            
            # Phase 1: Bottoming (was declining, now stabilizing)
            elif middle_chg < -1 and abs(recent_chg) < 2 and momentum_shift > 1:
                return "Phase 1 - Bottoming"
            
            # Phase 2: Early Recovery (positive momentum after decline)
            elif middle_chg < 0 and recent_chg > 1 and momentum_shift > 2:
                return "Phase 2 - Recovery"
            
            # Phase 3: Advancing (sustained positive momentum)
            elif recent_chg > 1 and middle_chg > 0:
                return "Phase 3 - Advancing"
            
            # Phase 3: Topping (advancing but momentum slowing)
            elif recent_chg < 1 and middle_chg > 1 and momentum_shift < -1:
                return "Phase 3 - Topping"
            
            # Phase 4: Early Decline (turning negative after positive)
            elif recent_chg < -1 and middle_chg > 0:
                return "Phase 4 - Declining"
            
            else:
                return "Stable/Transitioning"
        
        trend_data['market_phase'] = trend_data.apply(identify_phase, axis=1)
        
        # Flag potential bottoming/reversal candidates
        trend_data['bottoming_signal'] = (
            (trend_data['market_phase'].isin(['Phase 1 - Bottoming', 'Phase 2 - Recovery'])) &
            (trend_data['momentum_shift'] > 1) &
            (trend_data['recent_avg'] < 50)  # Only consider if scores are still relatively low
        )
        
        # Legacy columns for compatibility
        trend_data['score_change'] = trend_data['recent_change']
        trend_data['percent_change'] = (trend_data['recent_change'] / trend_data['middle_avg'] * 100).round(2)
        
        # Categorize trends
        trend_data['trend_strength'] = pd.cut(
            trend_data['recent_change'].abs(),
            bins=[0, 2, 5, 10, float('inf')],
            labels=['Weak', 'Moderate', 'Strong', 'Very Strong']
        )
        
        trend_data['trend_direction'] = np.where(
            trend_data['recent_change'] > 1, 'Improving',
            np.where(trend_data['recent_change'] < -1, 'Declining', 'Stable')
        )
        
        return trend_data.sort_values('momentum_shift', ascending=False)
        
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
                "Middle Average: %{customdata[1]:.1f}<br>" +
                "Market Phase: %{customdata[2]}<br>" +
                "Stock Count: %{customdata[3]}<br>" +
                "Momentum Shift: %{customdata[4]:+.1f}<br>" +
                "<extra></extra>"
            ),
            customdata=np.column_stack((
                trend_data['recent_avg'].round(1),
                trend_data['middle_avg'].round(1),
                trend_data['market_phase'],
                trend_data['stock_count'],
                trend_data['momentum_shift'].round(1)
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

def create_industry_trend_chart(df, industry, selected_caps, score_type, lookback_days):
    """Create a detailed trend chart for a specific industry showing phase transitions."""
    try:
        # Filter data for the selected industry and market caps
        if selected_caps:
            industry_df = df[(df['industry'] == industry) & (df['market_cap_category'].isin(selected_caps))]
        else:
            industry_df = df[df['industry'] == industry]
        
        if industry_df.empty:
            return go.Figure().update_layout(title=f"No data available for {industry}")
        
        # Get daily averages for the industry
        score_col = f"{score_type.lower()}_score"
        daily_scores = industry_df.groupby('date')[score_col].mean().reset_index()
        daily_scores = daily_scores.sort_values('date')
        
        # Calculate the three periods we're using for analysis
        all_dates = sorted(daily_scores['date'].unique())
        if len(all_dates) < lookback_days * 3:
            return go.Figure().update_layout(title=f"Insufficient data for {industry}")
        
        # Define period boundaries
        latest_date = all_dates[-1]
        recent_start = all_dates[-lookback_days]
        middle_start = all_dates[-2*lookback_days]
        middle_end = all_dates[-lookback_days-1]
        early_start = all_dates[-3*lookback_days] if len(all_dates) >= 3*lookback_days else all_dates[0]
        early_end = all_dates[-2*lookback_days-1]
        
        # Calculate period averages
        recent_avg = daily_scores[daily_scores['date'] >= recent_start][score_col].mean()
        middle_avg = daily_scores[(daily_scores['date'] >= middle_start) & (daily_scores['date'] <= middle_end)][score_col].mean()
        early_avg = daily_scores[(daily_scores['date'] >= early_start) & (daily_scores['date'] <= early_end)][score_col].mean()
        
        # Calculate changes for phase determination
        recent_change = recent_avg - middle_avg
        middle_change = middle_avg - early_avg
        momentum_shift = recent_change - middle_change
        
        # Determine phase
        def get_phase():
            if recent_change < -1 and middle_change < -1:
                return "Phase 4 - Declining"
            elif middle_change < -1 and abs(recent_change) < 2 and momentum_shift > 1:
                return "Phase 1 - Bottoming"
            elif middle_change < 0 and recent_change > 1 and momentum_shift > 2:
                return "Phase 2 - Recovery"
            elif recent_change > 1 and middle_change > 0:
                return "Phase 3 - Advancing"
            elif recent_change < 1 and middle_change > 1 and momentum_shift < -1:
                return "Phase 3 - Topping"
            elif recent_change < -1 and middle_change > 0:
                return "Phase 4 - Declining"
            else:
                return "Stable/Transitioning"
        
        current_phase = get_phase()
        
        # Create the chart
        fig = go.Figure()
        
        # Add the main score line
        fig.add_trace(go.Scatter(
            x=daily_scores['date'],
            y=daily_scores[score_col],
            mode='lines+markers',
            name=f'{score_type} Score',
            line=dict(color='white', width=2),
            marker=dict(size=4),
            hovertemplate=f"Date: %{{x}}<br>{score_type} Score: %{{y:.1f}}<extra></extra>"
        ))
        
        # Add period average lines
        fig.add_hline(y=recent_avg, line_dash="solid", line_color="green", opacity=0.7,
                     annotation_text=f"Recent Avg: {recent_avg:.1f}")
        fig.add_hline(y=middle_avg, line_dash="dash", line_color="yellow", opacity=0.7,
                     annotation_text=f"Middle Avg: {middle_avg:.1f}")
        fig.add_hline(y=early_avg, line_dash="dot", line_color="red", opacity=0.7,
                     annotation_text=f"Early Avg: {early_avg:.1f}")
        
        # Add vertical lines to separate periods
        fig.add_vline(x=recent_start, line_dash="dash", line_color="white", opacity=0.3)
        fig.add_vline(x=middle_start, line_dash="dash", line_color="white", opacity=0.3)
        if len(all_dates) >= 3*lookback_days:
            fig.add_vline(x=early_start, line_dash="dash", line_color="white", opacity=0.3)
        
        # Add background colors for periods
        fig.add_vrect(
            x0=recent_start, x1=latest_date,
            fillcolor="green", opacity=0.1,
            annotation_text="Recent Period", annotation_position="top left"
        )
        fig.add_vrect(
            x0=middle_start, x1=middle_end,
            fillcolor="yellow", opacity=0.1,
            annotation_text="Middle Period", annotation_position="top left"
        )
        if len(all_dates) >= 3*lookback_days:
            fig.add_vrect(
                x0=early_start, x1=early_end,
                fillcolor="red", opacity=0.1,
                annotation_text="Early Period", annotation_position="top left"
            )
        
        # Update layout
        fig.update_layout(
            title=f'{industry} - {score_type} Score Trend<br><span style="font-size:14px">Current Phase: {current_phase} | Momentum Shift: {momentum_shift:+.1f}</span>',
            xaxis_title='Date',
            yaxis_title=f'{score_type} Score',
            template='plotly_dark',
            height=500,
            hovermode='x unified',
            showlegend=True,
            annotations=[
                dict(
                    text=f"Recent Change: {recent_change:+.1f}<br>Middle Change: {middle_change:+.1f}<br>Momentum Shift: {momentum_shift:+.1f}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="white",
                    borderwidth=1
                )
            ]
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating industry trend chart: {str(e)}")
        return go.Figure().update_layout(title=f"Error creating chart for {industry}")

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
        
        # Time period selection with more options for your extended dataset
        lookback_options = {
            "1 Week (5 days)": 5,
            "2 Weeks (10 days)": 10,
            "3 Weeks (15 days)": 15,
            "1 Month (21 days)": 21,
            "6 Weeks (30 days)": 30,
            "2 Months (42 days)": 42,
            "10 Weeks (50 days)": 50,
            "3 Months (63 days)": 63,
            "4 Months (84 days)": 84,
            "5 Months (105 days)": 105,
            "6 Months (126 days)": 126
        }
        
        lookback_selection = st.sidebar.selectbox(
            "Trend Analysis Period",
            options=list(lookback_options.keys()),
            index=6,  # Default to 10 weeks (50 days) for better phase detection
            help="Longer periods provide more reliable phase detection for major trend shifts"
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
        
        # Top movers and phase analysis
        with col2:
            st.markdown("### 🔄 Phase Analysis")
            
            # Bottoming candidates
            bottoming_candidates = trend_data[trend_data['bottoming_signal']]
            if not bottoming_candidates.empty:
                st.markdown("**🎯 Potential Bottoms:**")
                bottom_display = bottoming_candidates[['industry', 'market_phase', 'momentum_shift']].head(5)
                bottom_display.columns = ['Industry', 'Phase', 'Momentum']
                st.dataframe(bottom_display, hide_index=True)
            else:
                st.info("No clear bottoming patterns detected")
            
            # Phase 2 recoveries
            phase_2 = trend_data[trend_data['market_phase'] == 'Phase 2 - Recovery']
            if not phase_2.empty:
                st.markdown("**📈 Early Recovery:**")
                recovery_display = phase_2[['industry', 'recent_change', 'momentum_shift']].head(3)
                recovery_display.columns = ['Industry', 'Recent Change', 'Momentum']
                st.dataframe(recovery_display, hide_index=True)
        
        # Momentum scatter plot
        st.markdown("### 🎯 Score vs Momentum Analysis")
        fig2 = create_momentum_scatter(trend_data, score_type)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Individual Industry Trend Chart
        st.markdown("### 📈 Individual Industry Trend Analysis")
        
        # Industry selector
        industry_options = sorted(trend_data['industry'].tolist())
        selected_industry = st.selectbox(
            "Select Industry for Detailed Trend View",
            options=industry_options,
            help="View the detailed score trend and phase analysis for a specific industry"
        )
        
        # Create and display the individual industry chart
        if selected_industry:
            with st.spinner(f"Loading trend chart for {selected_industry}..."):
                industry_fig = create_industry_trend_chart(df, selected_industry, selected_caps, score_type, lookback_days)
                st.plotly_chart(industry_fig, use_container_width=True)
            
            # Show the specific analysis for this industry
            industry_data = trend_data[trend_data['industry'] == selected_industry]
            if not industry_data.empty:
                row = industry_data.iloc[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Phase", row['market_phase'])
                with col2:
                    st.metric("Recent Change", f"{row['recent_change']:+.1f}")
                with col3:
                    st.metric("Momentum Shift", f"{row['momentum_shift']:+.1f}")
        
        # Detailed data table with phase information
        st.markdown("### 📋 Detailed Phase Analysis")
        
        # Filter options for the table
        phase_filter = st.selectbox(
            "Filter by Market Phase",
            options=["All", "Phase 1 - Bottoming", "Phase 2 - Recovery", "Phase 3 - Advancing", "Phase 4 - Declining", "Potential Bottoms Only"]
        )
        
        if phase_filter == "Potential Bottoms Only":
            filtered_data = trend_data[trend_data['bottoming_signal']]
        elif phase_filter != "All":
            filtered_data = trend_data[trend_data['market_phase'] == phase_filter]
        else:
            filtered_data = trend_data
        
        # Display formatted table with phase information
        display_data = filtered_data[['industry', 'market_phase', 'recent_avg', 'recent_change', 'momentum_shift', 'stock_count', 'bottoming_signal']].copy()
        display_data.columns = ['Industry', 'Market Phase', 'Current Score', 'Recent Change', 'Momentum Shift', 'Stocks', 'Bottom Signal']
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