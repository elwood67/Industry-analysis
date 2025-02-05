import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
import logging
from datetime import datetime
import glob
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG)
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

def load_and_process_historical_data(base_path="Data/stock_scores"):
    """Load and process all historical JSON files."""
    try:
        json_files = glob.glob(os.path.join(base_path, "market_analysis_*.json"))
        json_files = [f for f in json_files if 'latest' not in f]
        json_files.sort(key=lambda x: os.path.getmtime(x))
        
        all_data = []
        for file_path in json_files:
            with open(file_path, 'r') as file:
                data = json.load(file)
                df = pd.DataFrame(data['stocks'])
                df['date'] = datetime.fromtimestamp(os.path.getmtime(file_path))
                df['market_cap_category'] = df['market_cap_B'].apply(categorize_market_cap)
                all_data.append(df)
        
        return pd.concat(all_data, ignore_index=True)
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        raise

def detect_score_structure(series, window_size=5):
    """Detect score structure patterns."""
    try:
        # Calculate rolling stats
        rolling_min = series.rolling(window=window_size).min()
        rolling_max = series.rolling(window=window_size).max()
        rolling_mean = series.rolling(window=window_size).mean()
        
        # Calculate momentum
        momentum = series.diff().rolling(window=window_size).mean()
        
        return rolling_min, rolling_max, rolling_mean, momentum
    except Exception as e:
        logger.error(f"Error in structure detection: {str(e)}")
        raise

def analyze_score_structure(df, selected_caps, window_size=5, threshold=0.1):
    """Analyze score structure changes for filtered data."""
    try:
        if selected_caps:
            df = df[df['market_cap_category'].isin(selected_caps)]
        
        # Group by date and industry
        grouped = df.groupby(['date', 'industry']).agg({
            'bullish_score': 'mean',
            'bearish_score': 'mean',
            'symbol': 'count'
        }).reset_index()
        
        # Pivot data
        bullish_pivot = grouped.pivot(index='date', columns='industry', values='bullish_score')
        bearish_pivot = grouped.pivot(index='date', columns='industry', values='bearish_score')
        
        # Analyze structure for each industry
        structure_results = []
        
        for industry in bullish_pivot.columns:
            try:
                bull_scores = bullish_pivot[industry].dropna()
                bear_scores = bearish_pivot[industry].dropna()
                
                # Detect structures
                bull_min, bull_max, bull_mean, bull_momentum = detect_score_structure(bull_scores, window_size)
                bear_min, bear_max, bear_mean, bear_momentum = detect_score_structure(bear_scores, window_size)
                
                # Get latest values
                latest_bull = bull_scores.iloc[-1]
                latest_bear = bear_scores.iloc[-1]
                latest_bull_momentum = bull_momentum.iloc[-1]
                latest_bear_momentum = bear_momentum.iloc[-1]
                
                # Detect structure formation
                forming_bull = (
                    bull_min.diff().rolling(window=window_size).mean().iloc[-1] > 0 and
                    latest_bull_momentum > 0
                )
                
                forming_bear = (
                    bear_max.diff().rolling(window=window_size).mean().iloc[-1] < 0 and
                    latest_bear_momentum < 0
                )
                
                structure_results.append({
                    'industry': industry,
                    'latest_bull_score': latest_bull,
                    'latest_bear_score': latest_bear,
                    'bull_momentum': latest_bull_momentum,
                    'bear_momentum': latest_bear_momentum,
                    'forming_bullish_structure': bool(forming_bull),
                    'forming_bearish_structure': bool(forming_bear),
                    'stock_count': grouped[grouped['industry'] == industry]['symbol'].iloc[-1]
                })
                
            except Exception as e:
                logger.error(f"Error analyzing structure for {industry}: {str(e)}")
                continue
        
        return pd.DataFrame(structure_results), bullish_pivot, bearish_pivot
    
    except Exception as e:
        logger.error(f"Error in structure analysis: {str(e)}")
        raise

def create_structure_visualization(results_df, score_type='bullish', top_n=10):
    """Create visualization for score structure analysis."""
    try:
        if results_df.empty:
            logger.warning("Empty results provided for visualization")
            return go.Figure()
        
        # Filter based on score type and ensure forming structure exists
        forming_col = f'forming_{score_type}_structure'
        momentum_col = f'{score_type[0:4]}_momentum'  # bull_momentum or bear_momentum
        
        # Create a fresh copy of the dataframe
        forming = results_df[results_df[forming_col]].copy()
        
        if forming.empty:
            logger.warning("No forming structures found")
            fig = go.Figure()
            fig.update_layout(
                title=f"No {score_type.title()} Structures Currently Forming",
                template='plotly_dark'
            )
            return fig
        
        # Calculate absolute momentum safely
        forming.loc[:, 'abs_momentum'] = forming[momentum_col].abs()
        
        # Sort and get top N (using actual parameter now)
        top_forming = forming.nlargest(top_n, 'abs_momentum')
        
        fig = go.Figure()
        
        # Add bars for momentum
        fig.add_trace(go.Bar(
            x=top_forming[momentum_col].round(2),
            y=top_forming['industry'],
            orientation='h',
            text=top_forming['stock_count'].astype(str) + ' stocks',
            textposition='auto',
            marker_color=['red' if x < 0 else 'green' for x in top_forming[momentum_col]]
        ))
        
        fig.update_layout(
            title=f'Industries Forming {score_type.title()} Structure',
            xaxis_title=f'{score_type.title()} Score Momentum',
            yaxis_title='Industry',
            height=max(400, len(top_forming) * 30),  # Adjust height based on number of bars
            showlegend=False,
            template='plotly_dark'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating structure visualization: {str(e)}")
        fig = go.Figure()
        fig.update_layout(
            title="Error Creating Visualization",
            annotations=[
                dict(
                    text=str(e),
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(color="red")
                )
            ],
            template='plotly_dark'
        )
        return fig

def create_detail_chart(pivot_data, industry, window_size=5):
    """Create detailed view of industry score structure."""
    try:
        # Ensure index is datetime and sorted
        pivot_data = pivot_data.copy()
        pivot_data.index = pd.to_datetime(pivot_data.index)
        pivot_data = pivot_data.sort_index()
        
        fig = go.Figure()
        
        # Add raw score line
        fig.add_trace(go.Scatter(
            x=pivot_data.index.strftime('%Y-%m-%d'),  # Format dates as strings
            y=pivot_data[industry],
            mode='lines',
            name='Score',
            line=dict(color='white')
        ))
        
        # Calculate and add rolling min/max
        rolling_min = pivot_data[industry].rolling(window=window_size).min()
        rolling_max = pivot_data[industry].rolling(window=window_size).max()
        
        fig.add_trace(go.Scatter(
            x=pivot_data.index.strftime('%Y-%m-%d'),  # Format dates as strings
            y=rolling_min,
            mode='lines',
            name='Rolling Min',
            line=dict(color='red', dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=pivot_data.index.strftime('%Y-%m-%d'),  # Format dates as strings
            y=rolling_max,
            mode='lines',
            name='Rolling Max',
            line=dict(color='green', dash='dot')
        ))
        
        # Get unique dates for tick marks
        unique_dates = pivot_data.index.strftime('%Y-%m-%d').unique()
        
        fig.update_layout(
            title=f'{industry} Score Structure',
            xaxis_title='Date',
            yaxis_title='Score',
            height=400,
            template='plotly_dark',
            hovermode='x unified',
            xaxis=dict(
                ticktext=unique_dates,
                tickvals=unique_dates,
                tickangle=-45,
                tickmode='array',
                autorange='reversed'  # Newest data on right
            )
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating detail chart: {str(e)}")
        raise

def display_crossover_table(structure_results, max_rows=20):
    """Display potential crossover opportunities."""
    try:
        # Create a copy of the dataframe
        results = structure_results.copy()
        
        # Calculate score difference
        results['score_difference'] = results['latest_bull_score'] - results['latest_bear_score']
        
        # Sort by absolute score difference
        results['abs_diff'] = results['score_difference'].abs()
        crossovers = results.nlargest(max_rows, 'abs_diff')  # Show up to max_rows results
        
        # Format display columns
        display_df = pd.DataFrame({
            'Industry': crossovers['industry'],
            'Bullish': crossovers['latest_bull_score'].round(2),
            'Bearish': crossovers['latest_bear_score'].round(2),
            'Stocks': crossovers['stock_count']
        })
        
        return display_df
        
    except Exception as e:
        logger.error(f"Error creating crossover table: {str(e)}")
        return pd.DataFrame()

def main():
    st.title("Industry Score Structure Analysis")
    
    try:
        # Load data
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
        
        # Structure detection parameters
        window_size = st.sidebar.slider(
            "Structure Window Size",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of periods to detect structure formation"
        )
        
        threshold = st.sidebar.slider(
            "Structure Formation Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Percentage of periods needed to confirm structure"
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
        
        # Analyze structure
        structure_results, bullish_pivot, bearish_pivot = analyze_score_structure(
            df, selected_caps, window_size, threshold
        )
        
        if structure_results.empty:
            st.warning("No structure formations detected with current settings.")
            return
        
        # Show main structure visualization
        fig1 = create_structure_visualization(structure_results, score_type, top_n)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Show potential crossovers
        st.markdown("### Potential Score Crossovers")
        crossover_df = display_crossover_table(structure_results)
        if not crossover_df.empty:
            st.dataframe(crossover_df, use_container_width=True)
        else:
            st.info("No potential crossovers detected.")
        
        # Detailed view for selected industry
        st.markdown("### Detailed Structure View")
        selected_industry = st.selectbox(
            "Select Industry for Detailed View",
            options=structure_results['industry'].tolist()
        )
        
        # Show detailed timeline for selected industry
        pivot_data = bullish_pivot if score_type == 'bullish' else bearish_pivot
        fig2 = create_detail_chart(pivot_data, selected_industry, window_size)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Add download button
        csv = structure_results.to_csv(index=False)
        st.download_button(
            label="Download Structure Analysis",
            data=csv,
            file_name=f"score_structure_{score_type}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred while analyzing the data. Check the logs for details.")

if __name__ == "__main__":
    main()