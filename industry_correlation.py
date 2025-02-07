import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import logging
from datetime import datetime
import glob
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_and_process_historical_data(base_path="Data/stock_scores"):
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
                    
                    # Calculate net score
                    df['net_score'] = df['bullish_score'] - df['bearish_score']
                    
                    all_data.append(df)
                    logger.debug(f"Successfully processed {file_path}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
        
        return pd.concat(all_data, ignore_index=True)
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        raise

def calculate_lead_lag_relationships(pivot_data, max_lag=5):
    """Calculate lead-lag relationships using correlation analysis."""
    try:
        industries = pivot_data.columns
        relationships = []
        
        for ind1 in industries:
            for ind2 in industries:
                if ind1 != ind2:
                    # Get clean data for both industries
                    data1 = pivot_data[ind1].dropna()
                    data2 = pivot_data[ind2].dropna()
                    
                    # Ensure both series have same index
                    common_idx = data1.index.intersection(data2.index)
                    if len(common_idx) <= max_lag:
                        continue
                        
                    data1 = data1[common_idx]
                    data2 = data2[common_idx]
                    
                    # Test correlations at different lags
                    best_lag = 0
                    best_corr = 0
                    for lag in range(1, max_lag + 1):
                        # Shift data1 forward and calculate correlation
                        corr = data1.shift(-lag).corr(data2)
                        if pd.notna(corr) and abs(corr) > abs(best_corr):
                            best_corr = corr
                            best_lag = lag
                    
                    if abs(best_corr) > 0:  # Store any non-zero correlation
                        relationships.append({
                            'leader': ind1,
                            'follower': ind2,
                            'lag_days': best_lag,  # Renamed to clarify these are days
                            'correlation': best_corr
                        })
        
        return pd.DataFrame(relationships)
        
    except Exception as e:
        logger.error(f"Error calculating lead-lag relationships: {str(e)}")
        return pd.DataFrame()

def find_leading_industries(relationships_df, min_correlation=0.3):
    """Identify industries that consistently lead others."""
    try:
        # Filter for significant correlations
        significant = relationships_df[
            abs(relationships_df['correlation']) >= min_correlation
        ]
        
        if significant.empty:
            return pd.DataFrame()
        
        # Calculate leadership metrics
        leaders = []
        for industry in significant['leader'].unique():
            leads = significant[significant['leader'] == industry]
            
            if not leads.empty:
                avg_correlation = abs(leads['correlation']).mean()
                avg_lag = leads['lag_days'].mean()
                follower_count = len(leads)
                
                # Calculate leadership score
                # Weight by: number of followers, correlation strength, and inverse of lag
                leadership_score = (
                    follower_count * 
                    avg_correlation * 
                    (1 / avg_lag)
                ) * 1000  # Scale for readability
                
                # Get list of followers
                followers = leads.sort_values('correlation', ascending=False)['follower'].tolist()
                
                leaders.append({
                    'industry': industry,
                    'leadership_score': leadership_score,
                    'follower_count': follower_count,
                    'avg_correlation': avg_correlation,
                    'avg_lag': avg_lag,
                    'followers': ', '.join(followers)
                })
        
        leaders_df = pd.DataFrame(leaders)
        if not leaders_df.empty:
            leaders_df = leaders_df.sort_values('leadership_score', ascending=False)
            
        return leaders_df
        
    except Exception as e:
        logger.error(f"Error finding leading industries: {str(e)}")
        return pd.DataFrame()

def create_correlation_heatmap(pivot_data):
    """Create a heatmap of industry correlations."""
    try:
        # Calculate correlation matrix
        corr_matrix = pivot_data.corr()
        
        # Create heatmap with improved readability
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            hoverongaps=False,
            hovertemplate='%{x}<br>%{y}<br>Correlation: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Industry Correlation Matrix',
            template='plotly_dark',
            height=800,
            width=800,
            xaxis_tickangle=-45,
            xaxis_tickfont={"size": 8},  # Smaller font for better fit
            yaxis_tickfont={"size": 8}
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {str(e)}")
        return None

def create_leader_follower_chart(leader, follower, pivot_data, lag_days):
    """Create a chart showing the lead-lag relationship between two industries."""
    try:
        # Get data for both industries
        leader_data = pivot_data[leader].dropna()
        follower_data = pivot_data[follower].dropna()
        
        # Ensure we have enough data points after considering lag
        if len(leader_data) <= lag_days or len(follower_data) <= lag_days:
            st.warning(f"Not enough data points to show lag of {lag_days} days")
            return None
            
        # Get overlapping dates
        common_dates = sorted(list(set(leader_data.index) & set(follower_data.index)))
        if len(common_dates) <= lag_days:
            st.warning("Not enough overlapping data points")
            return None
            
        # Use only common dates for both series
        leader_data = leader_data[common_dates]
        follower_data = follower_data[common_dates]
        
        # Create shifted leader data
        # We'll shift the leader back by lag_days (this means trimming the end)
        shifted_dates = common_dates[:-lag_days]
        shifted_leader = leader_data[:-lag_days]
        aligned_follower = follower_data[lag_days:]
        
        # Calculate correlation on aligned data
        correlation = shifted_leader.corr(aligned_follower)
        
        # Create the figure
        fig = go.Figure()
        
        # Add leader line
        fig.add_trace(go.Scatter(
            x=common_dates,
            y=leader_data,
            name=f'{leader} (Leader)',
            line=dict(color='green', width=2),
            hovertemplate='Date: %{x}<br>Score: %{y:.2f}<extra></extra>'
        ))
        
        # Add follower line
        fig.add_trace(go.Scatter(
            x=common_dates,
            y=follower_data,
            name=f'{follower} (Follower)',
            line=dict(color='blue', width=2),
            hovertemplate='Date: %{x}<br>Score: %{y:.2f}<extra></extra>'
        ))
        
        # Add aligned leader line
        fig.add_trace(go.Scatter(
            x=common_dates[lag_days:],  # Start from lag_days
            y=shifted_leader,
            name=f'{leader} (Aligned to {follower})',
            line=dict(color='yellow', width=2, dash='dot'),
            hovertemplate='Date: %{x}<br>Score: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Lead-Lag Relationship: {leader} → {follower}<br>' +
                  f'<sup>Correlation: {correlation:.2f} | Lag: {lag_days} days</sup>',
            xaxis_title='Date',
            yaxis_title='Score',
            template='plotly_dark',
            height=500,
            xaxis=dict(
                type='category',
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinecolor='rgba(255, 255, 255, 0.2)'
            ),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating leader-follower chart: {str(e)}")
        st.error(f"Error creating chart: {str(e)}")
        return None

def create_leadership_score_chart(leaders_df):
    """Create a bar chart of leadership scores."""
    try:
        if leaders_df.empty:
            return None
            
        fig = go.Figure()
        
        # Add bars for leadership scores
        fig.add_trace(go.Bar(
            x=leaders_df['industry'],
            y=leaders_df['leadership_score'],
            text=[
                f"Score: {score:.1f}<br>Followers: {count}"
                for score, count in zip(
                    leaders_df['leadership_score'],
                    leaders_df['follower_count']
                )
            ],
            textposition='auto',
            hovertemplate=(
                "Industry: %{x}<br>" +
                "Leadership Score: %{y:.1f}<br>" +
                "%{text}<br>" +
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            title='Industry Leadership Scores',
            xaxis_title='Industry',
            yaxis_title='Leadership Score',
            template='plotly_dark',
            height=500,
            xaxis_tickangle=-45,
            showlegend=False,
            margin=dict(b=100)  # More space for x-axis labels
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating leadership score chart: {str(e)}")
        return None

def main():
    st.title("Industry Lead-Lag Analysis")
    
    try:
        # Load data
        with st.spinner("Loading historical data..."):
            df = load_and_process_historical_data()
        
        # Sidebar controls
        st.sidebar.header("Analysis Controls")
        
        # Score type selection
        score_type = st.sidebar.radio(
            "Select Score Type",
            ["bullish_score", "bearish_score", "net_score"],
            help="Choose which score to analyze for relationships"
        )
        
        # Analysis parameters
        max_lag = st.sidebar.slider(
            "Maximum Lag Days",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum number of days to look back for lead-lag relationships"
        )
        
        min_correlation = st.sidebar.slider(
            "Minimum Correlation",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Minimum correlation strength to consider"
        )
        
        # Process data
        grouped = df.groupby(['date', 'industry']).agg({
            score_type: 'mean',
            'symbol': 'count'
        }).reset_index()
        
        # Create pivot table
        pivot_data = grouped.pivot(index='date', columns='industry', values=score_type)
        pivot_data = pivot_data.sort_index()
        
        # Calculate relationships
        with st.spinner("Analyzing industry relationships..."):
            relationships_df = calculate_lead_lag_relationships(pivot_data, max_lag)
            leaders_df = find_leading_industries(relationships_df, min_correlation)
        
        if not leaders_df.empty:
            # Show leadership scores
            leadership_fig = create_leadership_score_chart(leaders_df)
            if leadership_fig:
                st.plotly_chart(leadership_fig, use_container_width=True)
            
            # Display detailed leaders table
            st.dataframe(
                leaders_df[[
                    'industry', 'leadership_score', 'follower_count',
                    'avg_correlation', 'avg_lag'
                ]].style.format({
                    'leadership_score': '{:.1f}',
                    'avg_correlation': '{:.2f}',
                    'avg_lag': '{:.1f}'
                }),
                hide_index=True,
                use_container_width=True
            )
            
            # Show correlation matrix
            st.markdown("### Industry Correlation Matrix")
            corr_fig = create_correlation_heatmap(pivot_data)
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)
            
            # Relationship explorer
            st.markdown("### Explore Specific Relationships")
            
            # Create two columns for selection
            col1, col2 = st.columns(2)
            
            with col1:
                # Get list of leading industries with significant followers
                valid_leaders = relationships_df[
                    abs(relationships_df['correlation']) >= min_correlation
                ]['leader'].unique()
                leader_options = sorted(valid_leaders)
                
                selected_leader = st.selectbox(
                    "Select Leading Industry",
                    options=leader_options,
                    help="Choose an industry that leads others"
                )
            
            if selected_leader:
                # Get valid followers for this leader
                valid_followers = relationships_df[
                    (relationships_df['leader'] == selected_leader) &
                    (abs(relationships_df['correlation']) >= min_correlation)
                ]
                
                with col2:
                    follower_options = sorted(valid_followers['follower'].unique())
                    if follower_options:
                        selected_follower = st.selectbox(
                            "Select Following Industry",
                            options=follower_options,
                            help="Choose an industry that follows the leader"
                        )
                        
                        if selected_follower:
                            # Get relationship details
                            relationship = valid_followers[
                                valid_followers['follower'] == selected_follower
                            ].iloc[0]
                            
                            lag_days = int(relationship['lag_days'])
                            
                            # Create and show the chart
                            rel_fig = create_leader_follower_chart(
                                selected_leader,
                                selected_follower,
                                pivot_data,
                                lag_days
                            )
                            
                            if rel_fig:
                                st.plotly_chart(rel_fig, use_container_width=True)
                                
                                # Show relationship details
                                st.markdown(f"""
                                #### Relationship Details
                                - Lag Days: {lag_days}
                                - Correlation: {relationship['correlation']:.3f}
                                """)
                    else:
                        st.info(f"No significant followers found for {selected_leader} with current settings.")
            
            # Add download button
            st.markdown("### Download Analysis")
            csv = leaders_df.to_csv(index=False)
            st.download_button(
                label="Download Leader Analysis",
                data=csv,
                file_name=f"industry_leaders_{score_type}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help="Download the complete analysis results"
            )
        else:
            st.warning("No significant leading industries found with current parameters.")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred during analysis.")
        if st.sidebar.checkbox("Show Error Details"):
            st.error(str(e))

if __name__ == "__main__":
    main()