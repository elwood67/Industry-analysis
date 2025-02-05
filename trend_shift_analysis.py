import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import logging
from datetime import datetime, timedelta
import glob
import numpy as np
from scipy import stats
import traceback

# Set up logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def safe_rolling_stats(series, window_size):
    """Safely calculate rolling statistics with error handling."""
    try:
        if len(series) < window_size:
            logger.warning(f"Series length ({len(series)}) is less than window size ({window_size})")
            return pd.Series(index=series.index)
        return series.rolling(window=window_size).mean()
    except Exception as e:
        logger.error(f"Error in rolling stats calculation: {str(e)}")
        return pd.Series(index=series.index)

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
        logger.error(f"Error categorizing market cap {cap}: {str(e)}")
        return "Unknown"

def detect_trend_shifts(data, window_size=5, z_score_threshold=2):
    """
    Detect significant trend shifts using rolling statistics and z-scores.
    Includes comprehensive error handling.
    """
    try:
        if data.empty:
            logger.warning("Empty data provided to trend shift detection")
            return pd.Series(False, index=data.index), pd.Series(0, index=data.index)
            
        if len(data) < window_size:
            logger.warning(f"Data length ({len(data)}) less than window size ({window_size})")
            return pd.Series(False, index=data.index), pd.Series(0, index=data.index)
        
        # Calculate rolling mean and standard deviation
        rolling_mean = safe_rolling_stats(data, window_size)
        rolling_std = data.rolling(window=window_size).std()
        
        # Handle zero standard deviation
        rolling_std = rolling_std.replace(0, np.nan)
        
        # Calculate changes and their z-scores
        changes = data.diff()
        changes_mean = changes.rolling(window=window_size).mean()
        changes_std = changes.rolling(window=window_size).std()
        changes_std = changes_std.replace(0, np.nan)  # Avoid division by zero
        
        z_scores = (changes - changes_mean) / changes_std
        
        # Fill NaN z-scores with 0 to avoid false positives
        z_scores = z_scores.fillna(0)
        
        # Identify significant shifts
        significant_shifts = z_scores.abs() > z_score_threshold
        
        return significant_shifts, z_scores
        
    except Exception as e:
        logger.error(f"Error in trend shift detection: {str(e)}")
        return pd.Series(False, index=data.index), pd.Series(0, index=data.index)

def load_and_process_historical_data(base_path="Data/stock_scores"):
    """Load and process all historical JSON files with error handling."""
    try:
        # Get all JSON files
        json_files = glob.glob(os.path.join(base_path, "market_analysis_*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {base_path}")
            
        json_files = [f for f in json_files if 'latest' not in f]
        json_files.sort(key=lambda x: os.path.getmtime(x))
        
        all_data = []
        for file_path in json_files:
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    df = pd.DataFrame(data['stocks'])
                    df['date'] = datetime.fromtimestamp(os.path.getmtime(file_path))
                    df['market_cap_category'] = df['market_cap_B'].apply(categorize_market_cap)
                    all_data.append(df)
                    logger.debug(f"Successfully processed {file_path}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("No valid data files were processed")
            
        return pd.concat(all_data, ignore_index=True)
        
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        raise

def analyze_trends(df, selected_caps, window_size=5, z_score_threshold=2):
    """Analyze trends for filtered data with error handling."""
    try:
        if df.empty:
            logger.warning("Empty DataFrame provided for trend analysis")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Filter by market cap categories
        if selected_caps:
            df = df[df['market_cap_category'].isin(selected_caps)]
            
        if df.empty:
            logger.warning("No data available after market cap filtering")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Group by date and industry
        grouped = df.groupby(['date', 'industry']).agg({
            'bullish_score': 'mean',
            'bearish_score': 'mean',
            'symbol': 'count'
        }).reset_index()
        
        # Pivot data for time series analysis
        bullish_pivot = grouped.pivot(index='date', columns='industry', values='bullish_score')
        bearish_pivot = grouped.pivot(index='date', columns='industry', values='bearish_score')
        
        # Analyze trends for each industry
        trend_results = []
        for industry in bullish_pivot.columns:
            try:
                # Calculate bullish trends
                bullish_shifts, bull_z_scores = detect_trend_shifts(
                    bullish_pivot[industry], 
                    window_size, 
                    z_score_threshold
                )
                
                # Calculate bearish trends
                bearish_shifts, bear_z_scores = detect_trend_shifts(
                    bearish_pivot[industry],
                    window_size,
                    z_score_threshold
                )
                
                # Get the last valid scores
                last_bull_score = bullish_pivot[industry].iloc[-1]
                last_bear_score = bearish_pivot[industry].iloc[-1]
                
                # Calculate trend strengths - using absolute mean of z-scores
                bull_trend = bull_z_scores.abs().mean()
                bear_trend = bear_z_scores.abs().mean()
                
                # Handle NaN values
                bull_trend = 0 if pd.isna(bull_trend) else bull_trend
                bear_trend = 0 if pd.isna(bear_trend) else bear_trend
                
                trend_results.append({
                    'industry': industry,
                    'bullish_shifts': int(bullish_shifts.sum()),
                    'bearish_shifts': int(bearish_shifts.sum()),
                    'bullish_trend_strength': float(bull_trend),
                    'bearish_trend_strength': float(bear_trend),
                    'last_bull_score': float(last_bull_score),
                    'last_bear_score': float(last_bear_score)
                })
            except Exception as e:
                logger.error(f"Error analyzing trends for industry {industry}: {str(e)}")
                continue
        
        if not trend_results:
            logger.warning("No trend results generated")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Create DataFrame and ensure column types
        results_df = pd.DataFrame(trend_results)
        
        # Verify required columns exist
        required_cols = [
            'industry', 
            'bullish_trend_strength', 
            'bearish_trend_strength',
            'bullish_shifts',
            'bearish_shifts',
            'last_bull_score',
            'last_bear_score'
        ]
        
        for col in required_cols:
            if col not in results_df.columns:
                logger.error(f"Missing required column: {col}")
                raise ValueError(f"Missing required column: {col}")
        
        return results_df, bullish_pivot, bearish_pivot
        
    except Exception as e:
        logger.error(f"Error in trend analysis: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def create_trend_visualization(trend_results, score_type='bullish', top_n=10):
    """Create visualization for trend analysis results with error handling."""
    try:
        if trend_results.empty:
            logger.warning("Empty trend results provided for visualization")
            return go.Figure()
            
        strength_col = f'{score_type}_trend_strength'
        shifts_col = f'{score_type}_shifts'
        score_col = f'last_{score_type[0:4]}_score'
        
        # Verify required columns exist
        required_cols = [strength_col, shifts_col, score_col, 'industry']
        if not all(col in trend_results.columns for col in required_cols):
            missing = [col for col in required_cols if col not in trend_results.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        # Sort by trend strength and get top N
        top_trends = trend_results.nlargest(top_n, strength_col)
        
        fig = go.Figure()
        
        # Add bars for trend strength
        fig.add_trace(go.Bar(
            x=top_trends[strength_col].round(2),  # Round for cleaner display
            y=top_trends['industry'],
            orientation='h',
            text=[
                f"Shifts: {int(s)}, Score: {score:.1f}"
                for s, score in zip(top_trends[shifts_col], top_trends[score_col])
            ],
            textposition='auto',
            marker_color='blue',
            name='Trend Strength'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Industries by {score_type.title()} Trend Strength',
            xaxis_title=f'Trend Strength (Average |Z-Score|)',
            yaxis_title='Industry',
            height=max(400, len(top_trends) * 30),
            showlegend=False,
            template='plotly_dark'  # Match dark theme
        )
        
        # Add hover information
        fig.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "Trend Strength: %{x:.2f}<br>" +
                "%{text}<br>" +
                "<extra></extra>"
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating trend visualization: {str(e)}")
        return go.Figure()

def create_timeline_visualization(pivot_data, industry, score_type='bullish'):
    """Create timeline visualization for a specific industry with error handling."""
    try:
        if pivot_data.empty:
            logger.warning("Empty pivot data provided for timeline visualization")
            return go.Figure()
            
        if industry not in pivot_data.columns:
            logger.warning(f"Industry {industry} not found in pivot data")
            return go.Figure()
        
        fig = go.Figure()
        
        # Add line for score
        fig.add_trace(go.Scatter(
            x=pivot_data.index,
            y=pivot_data[industry],
            mode='lines+markers',
            name=f'{score_type.title()} Score'
        ))
        
        # Add trend lines using rolling average
        rolling_avg = safe_rolling_stats(pivot_data[industry], 5)
        fig.add_trace(go.Scatter(
            x=pivot_data.index,
            y=rolling_avg,
            mode='lines',
            line=dict(dash='dash'),
            name='Trend (5-day MA)'
        ))
        
        fig.update_layout(
            title=f'{industry} {score_type.title()} Score Timeline',
            xaxis_title='Date',
            yaxis_title='Score',
            height=400
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating timeline visualization: {str(e)}")
        return go.Figure()

def main():
    st.title("Industry Trend Shift Analysis")
    
    try:
        # Load historical data
        df = load_and_process_historical_data()
        
        # Sidebar controls
        st.sidebar.header("Analysis Controls")
        
        # Market cap filter
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
            ["bullish", "bearish"]
        )
        
        # Analysis parameters
        window_size = st.sidebar.slider(
            "Analysis Window Size",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of periods to consider for trend analysis"
        )
        
        z_score_threshold = st.sidebar.slider(
            "Trend Shift Threshold (Z-Score)",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.1,
            help="Higher values detect more significant trend shifts"
        )
        
        top_n = st.sidebar.slider(
            "Number of Industries to Show",
            min_value=5,
            max_value=20,
            value=10
        )
        
        if not selected_caps:
            st.warning("👈 Please select at least one Market Cap Category to begin analysis!")
            return
            
        # Perform trend analysis only if market caps are selected
        trend_results, bullish_pivot, bearish_pivot = analyze_trends(
            df, selected_caps, window_size, z_score_threshold
        )
        
        if trend_results.empty:
            st.warning("No data available for the selected filters.")
            return
            
        # Show overall trend strength visualization
        fig1 = create_trend_visualization(trend_results, score_type, top_n)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Industry selector for detailed view
        st.markdown("### Detailed Industry Timeline")
        selected_industry = st.selectbox(
            "Select Industry for Detailed View",
            options=trend_results['industry'].tolist()
        )
        
        # Show detailed timeline for selected industry
        pivot_data = bullish_pivot if score_type == 'bullish' else bearish_pivot
        fig2 = create_timeline_visualization(pivot_data, selected_industry, score_type)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Display detailed trend statistics
        st.markdown("### Trend Statistics")
        display_cols = [
            'industry',
            f'{score_type}_shifts',
            f'{score_type}_trend_strength',
            f'last_{score_type[0:4]}_score'
        ]
        
        col_names = {
            f'{score_type}_shifts': 'Significant Shifts',
            f'{score_type}_trend_strength': 'Trend Strength',
            f'last_{score_type[0:4]}_score': 'Current Score'
        }
        
        stats_df = trend_results[display_cols].rename(columns=col_names)
        stats_df = stats_df.sort_values('Trend Strength', ascending=False).head(top_n)
        st.dataframe(stats_df, use_container_width=True)
        
        # Add download button
        csv = stats_df.to_csv(index=False)
        st.download_button(
            label="Download Trend Analysis Data",
            data=csv,
            file_name=f"industry_trends_{score_type}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        logger.error(f"Main application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        if st.sidebar.checkbox("Show detailed error"):
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()