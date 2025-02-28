import streamlit as st
import pandas as pd
import json
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Elwood's Stock Market Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8cff;
        color: white;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .chart-container {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
</style>
""", unsafe_allow_html=True)

# Define functions for loading and processing data
def load_data(file_path):
    """Load data from JSON files"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def find_latest_files(directory):
    """Find the latest JSON and CSV files in the directory"""
    try:
        # Find the latest JSON file
        json_files = list(Path(directory).glob('market_analysis_*_latest.json'))
        if not json_files:
            json_files = sorted(list(Path(directory).glob('market_analysis_*.json')), key=os.path.getmtime, reverse=True)
        
        # Find the latest CSV file
        csv_files = list(Path(directory).glob('market_analysis_*_latest.csv'))
        if not csv_files:
            csv_files = sorted(list(Path(directory).glob('market_analysis_*.csv')), key=os.path.getmtime, reverse=True)
        
        return (json_files[0] if json_files else None, 
                csv_files[0] if csv_files else None)
    except Exception as e:
        st.error(f"Error finding latest files: {e}")
        return None, None

# Dashboard title and description
st.title("🚀 Elwood's Stock Market Dashboard")

# Sidebar for data selection and filters
st.sidebar.header("Data & Filters")

# Path to data directory
default_path = "morning_analysis"
data_path = st.sidebar.text_input(
    "Data Directory Path",
    value=default_path,
    help="Path to directory containing market analysis files"
)

# Find and load the latest data files
json_file, csv_file = find_latest_files(data_path)

if not json_file or not csv_file:
    st.error(f"No data files found in {data_path}. Please check the path.")
    st.stop()

# Load data
data = load_data(json_file)
df = pd.read_csv(csv_file)

if data is None or df.empty:
    st.error("Failed to load data. Please check the files.")
    st.stop()

# Extract timestamp from data
analysis_date = data.get('analysis_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
time_of_day = data.get('analysis_time_of_day', 'morning')

st.sidebar.markdown(f"**Data as of:** {analysis_date}")
st.sidebar.markdown(f"**Time of day:** {time_of_day.capitalize()}")
st.sidebar.markdown(f"**Total stocks analyzed:** {data.get('total_stocks_analyzed', 0)}")

# Filters
st.sidebar.header("Filters")
selected_sectors = st.sidebar.multiselect(
    "Filter by Sector",
    options=sorted(df['sector'].unique()),
    default=[]
)

min_market_cap = st.sidebar.slider(
    "Minimum Market Cap (Billions $)",
    min_value=0.0,
    max_value=float(df['market_cap_B'].max()),
    value=0.0,
    step=0.1
)

min_bullish_score = st.sidebar.slider(
    "Minimum Bullish Score",
    min_value=0,
    max_value=100,
    value=0,
    step=5
)

# Apply filters
filtered_df = df.copy()
if selected_sectors:
    filtered_df = filtered_df[filtered_df['sector'].isin(selected_sectors)]

filtered_df = filtered_df[filtered_df['market_cap_B'] >= min_market_cap]
filtered_df = filtered_df[filtered_df['bullish_score'] >= min_bullish_score]

# Main dashboard content
tabs = st.tabs([
    "📈 Market Overview", 
    "🏭 Industry & Sector Analysis", 
    "🔍 Stock Screening", 
    "📊 Technical Analysis",
    "📋 Data Tables"
])

# Market Overview Tab
with tabs[0]:
    st.header("Market Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(
            "Average Price Change", 
            f"{data['market_summary']['avg_price_change']:.2f}%",
            delta=None
        )
        st.markdown(f"Stocks Up Today: {data['market_summary']['stocks_up_today']} ({data['market_summary']['stocks_up_today']/data['total_stocks_analyzed']*100:.1f}%)")
        st.markdown(f"Stocks Down Today: {data['market_summary']['stocks_down_today']} ({data['market_summary']['stocks_down_today']/data['total_stocks_analyzed']*100:.1f}%)")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(
            "Bullish/Bearish Balance", 
            f"{data['market_summary']['average_net_score']:.2f}",
            delta=None
        )
        st.markdown(f"Strongly Bullish Stocks: {data['market_summary']['strongly_bullish_stocks']} ({data['market_summary']['strongly_bullish_stocks']/data['total_stocks_analyzed']*100:.1f}%)")
        st.markdown(f"Strongly Bearish Stocks: {data['market_summary']['strongly_bearish_stocks']} ({data['market_summary']['strongly_bearish_stocks']/data['total_stocks_analyzed']*100:.1f}%)")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(
            "Volume Activity", 
            f"{data['market_summary']['strong_volume_stocks']}",
            delta=None
        )
        st.markdown(f"Strong Volume Stocks (>2x avg): {data['market_summary']['strong_volume_stocks']} ({data['market_summary']['strong_volume_stocks']/data['total_stocks_analyzed']*100:.1f}%)")
        if 'gapping_up_stocks' in data['market_summary']:
            st.markdown(f"Stocks Gapping Up: {data['market_summary']['gapping_up_stocks']} ({data['market_summary']['gapping_up_stocks']/data['total_stocks_analyzed']*100:.1f}%)")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        if 'near_52w_high_stocks' in data['market_summary']:
            st.metric(
                "Proximity to 52-Week High", 
                f"{data['market_summary']['near_52w_high_stocks']}",
                delta=None
            )
            st.markdown(f"Near 52W High (within 10%): {data['market_summary']['near_52w_high_stocks']} ({data['market_summary']['near_52w_high_stocks']/data['total_stocks_analyzed']*100:.1f}%)")
        if 'high_volatility_stocks' in data['market_summary']:
            st.markdown(f"High Volatility Stocks: {data['market_summary']['high_volatility_stocks']} ({data['market_summary']['high_volatility_stocks']/data['total_stocks_analyzed']*100:.1f}%)")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Market Performance Chart
    st.subheader("Sector Performance")
    
    sector_data = {sector: stats['avg_price_change'] for sector, stats in data['sector_statistics'].items()}
    sector_df = pd.DataFrame([
        {"sector": sector, "change_pct": change, "color": "green" if change >= 0 else "red"} 
        for sector, change in sector_data.items()
    ])
    sector_df = sector_df.sort_values('change_pct', ascending=False)
    
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    fig = px.bar(
        sector_df, 
        x='sector', 
        y='change_pct',
        color='color',
        labels={'change_pct': 'Avg Price Change (%)', 'sector': 'Sector'},
        title="Sector Performance (Average Price Change)",
        color_discrete_map={"green": "#00B5AA", "red": "#FF6B6B"}
    )
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Money Flow Chart
    if 'money_flow' in data:
        st.subheader("Money Flow by Sector")
        
        money_flow_df = pd.DataFrame(data['money_flow'])
        money_flow_df = money_flow_df.sort_values('flow_score', ascending=False)
        money_flow_df['color'] = money_flow_df['flow_score'].apply(
            lambda x: "#00B5AA" if x > 0 else "#FF6B6B"
        )
        money_flow_df['flow_category'] = money_flow_df['flow_score'].apply(
            lambda x: "Strong Inflow" if x > 1 else 
                     "Mild Inflow" if x > 0.2 else 
                     "Neutral" if x > -0.2 else 
                     "Mild Outflow" if x > -1 else 
                     "Strong Outflow"
        )
        
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        fig = px.bar(
            money_flow_df, 
            x='sector', 
            y='flow_score',
            color='flow_category',
            labels={'flow_score': 'Money Flow Score', 'sector': 'Sector'},
            title="Money Flow by Sector (Price Change × Volume)",
            color_discrete_sequence=["#00B5AA", "#87CEEB", "#E0E0E0", "#FFA07A", "#FF6B6B"]
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Top Movers Today
    st.subheader("Top Movers Today")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        gainers = pd.DataFrame(data['top_movers_today'][:10])
        fig = px.bar(
            gainers, 
            x='symbol', 
            y='change_pct',
            color='sector',
            labels={'change_pct': 'Price Change (%)', 'symbol': 'Symbol'},
            title="Top 10 Gainers",
            hover_data=['industry', 'last_price', 'bullish_score']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        losers = pd.DataFrame(sorted(data['top_movers_today'], key=lambda x: x['change_pct'])[:10])
        fig = px.bar(
            losers, 
            x='symbol', 
            y='change_pct',
            color='sector',
            labels={'change_pct': 'Price Change (%)', 'symbol': 'Symbol'},
            title="Top 10 Losers",
            hover_data=['industry', 'last_price', 'bearish_score']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Industry & Sector Analysis Tab
with tabs[1]:
    st.header("Industry & Sector Analysis")
    
    # Top and Bottom Industries
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Performing Industries")
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        top_industries = pd.DataFrame(data['top_industries'])
        
        if not top_industries.empty:
            top_industries = top_industries.sort_values('avg_price_change', ascending=False).head(10)
            fig = px.bar(
                top_industries, 
                x='industry', 
                y='avg_price_change',
                color='avg_price_change',
                labels={'avg_price_change': 'Avg Price Change (%)', 'industry': 'Industry'},
                title="Top 10 Performing Industries",
                text='stock_count',
                color_continuous_scale=['#87CEEB', '#00B5AA']
            )
            fig.update_layout(height=500, coloraxis_showscale=False)
            fig.update_traces(texttemplate='%{text} stocks', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No industry data available")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.subheader("Bottom Performing Industries")
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        bottom_industries = pd.DataFrame(data['bottom_industries'])
        
        if not bottom_industries.empty:
            bottom_industries = bottom_industries.sort_values('avg_price_change').head(10)
            fig = px.bar(
                bottom_industries, 
                x='industry', 
                y='avg_price_change',
                color='avg_price_change',
                labels={'avg_price_change': 'Avg Price Change (%)', 'industry': 'Industry'},
                title="Bottom 10 Performing Industries",
                text='stock_count',
                color_continuous_scale=['#FF6B6B', '#FFA07A']
            )
            fig.update_layout(height=500, coloraxis_showscale=False)
            fig.update_traces(texttemplate='%{text} stocks', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No industry data available")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Most Volatile Industries
    st.subheader("Industry Volatility Analysis")
    
    if 'volatility' in df.columns:
        # Calculate volatility by industry
        industry_vol = df.groupby('industry').agg({
            'volatility': 'mean',
            'symbol': 'count'
        }).reset_index()
        industry_vol = industry_vol[industry_vol['symbol'] >= 3].sort_values('volatility', ascending=False).head(15)
        
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        fig = px.bar(
            industry_vol, 
            x='industry', 
            y='volatility',
            color='volatility',
            labels={'volatility': 'Avg Volatility (%)', 'industry': 'Industry'},
            title="Most Volatile Industries (Min 3 Stocks)",
            text='symbol',
            color_continuous_scale=['#87CEEB', '#FF6B6B']
        )
        fig.update_layout(height=500, coloraxis_showscale=False)
        fig.update_traces(texttemplate='%{text} stocks', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.write("Volatility data not available in this dataset")
    
    # Industry Heatmap
    st.subheader("Industry-Sector Heatmap")
    
    # Prepare data for heatmap
    industry_stats = pd.DataFrame([
        {
            'industry': industry,
            'sector': stats['sector'],
            'avg_price_change': stats['avg_price_change'],
            'avg_bullish_score': stats['avg_bullish_score'],
            'avg_bearish_score': stats['avg_bearish_score'],
            'stock_count': stats['stock_count']
        }
        for industry, stats in data['industry_statistics'].items()
    ])
    
    # Add filter for minimum industry size
    min_stocks = st.slider(
        "Minimum Stocks in Industry",
        min_value=3,
        max_value=int(industry_stats['stock_count'].max()),
        value=5,
        step=1
    )
    
    industry_stats = industry_stats[industry_stats['stock_count'] >= min_stocks]
    
    # Create heatmap
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    fig = px.scatter(
        industry_stats,
        x='avg_bullish_score',
        y='avg_bearish_score',
        size='stock_count',
        color='avg_price_change',
        hover_name='industry',
        color_continuous_scale=px.colors.diverging.RdBu,
        color_continuous_midpoint=0,
        labels={
            'avg_bullish_score': 'Avg Bullish Score',
            'avg_bearish_score': 'Avg Bearish Score',
            'stock_count': 'Number of Stocks',
            'avg_price_change': 'Avg Price Change (%)'
        },
        title=f"Industry Analysis: Bullish vs Bearish Scores (Min {min_stocks} Stocks)",
        hover_data=['sector', 'stock_count']
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Stock Screening Tab
with tabs[2]:
    st.header("Stock Screening")
    
    # Screening filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        view_option = st.selectbox(
            "View Category",
            options=[
                "Top Bullish Stocks", 
                "Top Bearish Stocks", 
                "Top Momentum Stocks",
                "Near 52-Week High",
                "Unusual Volume",
                "High Volatility"
            ],
            index=0
        )
    
    with col2:
        min_price = st.number_input(
            "Minimum Price ($)",
            min_value=0.0,
            value=1.0,
            step=0.1
        )
    
    with col3:
        num_stocks = st.number_input(
            "Number of Stocks to Show",
            min_value=5,
            max_value=100,
            value=20,
            step=5
        )
    
    # Prepare data based on selection
    if view_option == "Top Bullish Stocks":
        screen_df = pd.DataFrame(data['top_bullish_stocks'])
        screen_title = "Top Bullish Stocks"
        color_column = 'bullish_score'
        color_scale = ['#87CEEB', '#00B5AA']
    elif view_option == "Top Bearish Stocks":
        screen_df = pd.DataFrame(data['top_bearish_stocks'])
        screen_title = "Top Bearish Stocks"
        color_column = 'bearish_score'
        color_scale = ['#FFA07A', '#FF6B6B']
    elif view_option == "Top Momentum Stocks":
        screen_df = pd.DataFrame(data['top_movers_today'])
        screen_title = "Top Momentum Stocks (Biggest Price Gainers)"
        color_column = 'change_pct'
        color_scale = ['#87CEEB', '#00B5AA']
    elif view_option == "Near 52-Week High" and 'near_highs' in data:
        screen_df = pd.DataFrame(data['near_highs'])
        screen_title = "Stocks Near 52-Week High"
        color_column = 'pct_from_52w_high'
        color_scale = ['#87CEEB', '#00B5AA']
    elif view_option == "Unusual Volume" and 'unusual_volume' in data:
        screen_df = pd.DataFrame(data['unusual_volume'])
        screen_title = "Stocks with Unusual Volume"
        color_column = 'volume_vs_avg'
        color_scale = ['#87CEEB', '#00B5AA']
    elif view_option == "High Volatility" and 'volatility_leaders' in data:
        screen_df = pd.DataFrame(data['volatility_leaders'])
        screen_title = "High Volatility Stocks"
        color_column = 'volatility'
        color_scale = ['#87CEEB', '#FF6B6B']
    else:
        st.warning(f"Data for {view_option} not available in this dataset")
        screen_df = pd.DataFrame()
    
    # Apply price filter and limit
    if not screen_df.empty:
        screen_df = screen_df[screen_df['last_price'] >= min_price]
        screen_df = screen_df.head(num_stocks)
        
        # Display results
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        
        # Create main figure
        fig = px.bar(
            screen_df,
            x='symbol',
            y=color_column,
            color=color_column,
            labels={color_column: view_option, 'symbol': 'Symbol'},
            title=f"{screen_title} (Min Price: ${min_price})",
            hover_data=['sector', 'industry', 'last_price', 'change_pct'],
            color_continuous_scale=color_scale
        )
        fig.update_layout(height=500, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display sector breakdown
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        sector_counts = screen_df['sector'].value_counts().reset_index()
        sector_counts.columns = ['sector', 'count']
        
        fig = px.pie(
            sector_counts,
            values='count',
            names='sector',
            title=f"Sector Breakdown: {screen_title}",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show the data table
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("Screening Results")
        st.dataframe(screen_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Technical Analysis Tab
with tabs[3]:
    st.header("Technical Analysis")
    
    # Select stock from top performers or by symbol
    st.subheader("Select Stock for Technical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selection_method = st.radio(
            "Selection Method",
            options=["Top Performers", "Enter Symbol"],
            index=0
        )
    
    symbol = None
    
    if selection_method == "Top Performers":
        # Get top 20 performers
        top_performer_options = [
            f"{stock['symbol']} ({stock['change_pct']:.2f}%)" 
            for stock in data['top_movers_today'][:20]
        ]
        if top_performer_options:
            selected_option = st.selectbox(
                "Select from Top Performers",
                options=top_performer_options
            )
            # Extract just the symbol
            symbol = selected_option.split(" ")[0]
    else:
        symbol = st.text_input("Enter Symbol", value="AAPL").upper()
    
    # Show technical indicators for the selected stock
    if symbol:
        # Find the stock data
        stock_data = filtered_df[filtered_df['symbol'] == symbol]
        
        if not stock_data.empty:
            st.subheader(f"Technical Analysis: {symbol}")
            
            # Get stock details
            stock_details = stock_data.iloc[0]
            
            # Display basic stock info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price", 
                    f"${stock_details['last_price']:.2f}",
                    f"{stock_details['change_pct']:.2f}%"
                )
            
            with col2:
                st.metric(
                    "Bullish Score", 
                    f"{stock_details['bullish_score']:.1f}",
                    None
                )
            
            with col3:
                st.metric(
                    "Bearish Score", 
                    f"{stock_details['bearish_score']:.1f}",
                    None
                )
            
            with col4:
                st.metric(
                    "Net Score", 
                    f"{stock_details['net_score']:.1f}",
                    None
                )
            
            # Technical indicators
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Technical Indicators")
            
            # Get bullish and bearish indicators
            bullish_indicators = [col for col in stock_data.columns if col.startswith('bull_') and stock_data[col].iloc[0]]
            bearish_indicators = [col for col in stock_data.columns if col.startswith('bear_') and stock_data[col].iloc[0]]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Bullish Indicators")
                if bullish_indicators:
                    for indicator in bullish_indicators:
                        st.markdown(f"✅ {indicator.replace('bull_', '').replace('_', ' ').title()}")
                else:
                    st.write("No bullish indicators active")
            
            with col2:
                st.markdown("#### Bearish Indicators")
                if bearish_indicators:
                    for indicator in bearish_indicators:
                        st.markdown(f"⚠️ {indicator.replace('bear_', '').replace('_', ' ').title()}")
                else:
                    st.write("No bearish indicators active")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Advanced metrics if available
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Advanced Metrics")
            
            advanced_metrics = []
            
            if 'volatility' in stock_data.columns:
                advanced_metrics.append(("Volatility (ATR%)", f"{stock_data['volatility'].iloc[0]:.2f}%"))
            
            if 'pct_from_52w_high' in stock_data.columns:
                advanced_metrics.append(("Distance from 52W High", f"{stock_data['pct_from_52w_high'].iloc[0]:.2f}%"))
            
            if 'volume_vs_avg' in stock_data.columns:
                advanced_metrics.append(("Volume vs Avg", f"{stock_data['volume_vs_avg'].iloc[0]:.2f}x"))
            
            if 'day_change' in stock_data.columns:
                advanced_metrics.append(("Intraday Change", f"{stock_data['day_change'].iloc[0]:.2f}%"))
            
            # Create metric columns dynamically
            if advanced_metrics:
                cols = st.columns(len(advanced_metrics))
                for i, (label, value) in enumerate(advanced_metrics):
                    with cols[i]:
                        st.metric(label, value)
            else:
                st.write("No advanced metrics available")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Comparison to industry and sector
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("Comparison to Industry & Sector")
            
            industry = stock_details['industry']
            sector = stock_details['sector']
            
            # Get industry and sector averages
            industry_stats = None
            if 'industry_statistics' in data and industry in data['industry_statistics']:
                industry_stats = data['industry_statistics'].get(industry)
            
            sector_stats = None
            if 'sector_statistics' in data and sector in data['sector_statistics']:
                sector_stats = data['sector_statistics'].get(sector)
            
            # Create comparison chart
            comparison_data = []
            metrics = ['bullish_score', 'bearish_score', 'net_score', 'change_pct']
            
            # Add stock data
            for metric in metrics:
                if metric in stock_details:
                    comparison_data.append({
                        'metric': metric.replace('_', ' ').title(),
                        'value': float(stock_details[metric]),
                        'entity': symbol
                    })
            
            # Add industry averages
            if industry_stats:
                for metric in metrics:
                    industry_metric = f'avg_{metric}'
                    if industry_metric in industry_stats:
                        comparison_data.append({
                            'metric': metric.replace('_', ' ').title(),
                            'value': float(industry_stats[industry_metric]),
                            'entity': f"{industry} (Industry)"
                        })
            
            # Add sector averages
            if sector_stats:
                for metric in metrics:
                    sector_metric = f'avg_{metric}'
                    if sector_metric in sector_stats:
                        comparison_data.append({
                            'metric': metric.replace('_', ' ').title(),
                            'value': float(sector_stats[sector_metric]),
                            'entity': f"{sector} (Sector)"
                        })
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame(comparison_data)
            
            if not comparison_df.empty:
                fig = px.bar(
                    comparison_df,
                    x='metric',
                    y='value',
                    color='entity',
                    barmode='group',
                    title=f"{symbol} vs Industry & Sector",
                    labels={'value': 'Value', 'metric': 'Metric'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Comparison data not available")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(f"No data found for symbol {symbol}")

# Data Tables Tab
with tabs[4]:
    st.header("Data Tables")
    
    # Select data table to view
    table_option = st.selectbox(
        "Select Table to View",
        options=[
            "All Stocks", 
            "Top Movers", 
            "Industries Performance", 
            "Sectors Performance"
        ],
        index=0
    )
    
    if table_option == "All Stocks":
        # Display filtered dataframe
        st.subheader(f"All Stocks Data ({len(filtered_df)} stocks)")
        st.dataframe(filtered_df.sort_values('change_pct', ascending=False), use_container_width=True)
        
    elif table_option == "Top Movers":
        st.subheader("Top Movers Today")
        movers_df = pd.DataFrame(data['top_movers_today'])
        st.dataframe(movers_df, use_container_width=True)
        
    elif table_option == "Industries Performance":
        st.subheader("Industries Performance")
        
        # Create industries dataframe
        industries_df = pd.DataFrame([
            {
                'industry': industry,
                'sector': stats['sector'],
                'stock_count': stats['stock_count'],
                'avg_price_change': stats['avg_price_change'],
                'avg_bullish_score': stats['avg_bullish_score'],
                'avg_bearish_score': stats['avg_bearish_score'],
                'avg_net_score': stats['avg_net_score']
            }
            for industry, stats in data['industry_statistics'].items()
        ]).sort_values('avg_price_change', ascending=False)
        
        st.dataframe(industries_df, use_container_width=True)
        
    elif table_option == "Sectors Performance":
        st.subheader("Sectors Performance")
        
        # Create sectors dataframe
        sectors_df = pd.DataFrame([
            {
                'sector': sector,
                'stock_count': stats['stock_count'],
                'avg_price_change': stats['avg_price_change'],
                'avg_bullish_score': stats['avg_bullish_score'],
                'avg_bearish_score': stats['avg_bearish_score'],
                'avg_net_score': stats['avg_net_score']
            }
            for sector, stats in data['sector_statistics'].items()
        ]).sort_values('avg_price_change', ascending=False)
        
        st.dataframe(sectors_df, use_container_width=True)
    
    # Download options
    st.subheader("Download Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv_data,
            file_name=f"market_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
    
    with col2:
        json_data_str = json.dumps(data, indent=2)
        st.download_button(
            label="Download Full JSON Data",
            data=json_data_str,
            file_name=f"market_analysis_{datetime.now().strftime('%Y%m%d')}.json",
            mime='application/json'
        )

# Add footer
st.markdown("""
---
### How to Use This Dashboard


1. **Filters**: Use the sidebar filters to focus on specific sectors, market caps, or bullish scores
2. **Tabs**: Navigate between different views using the tabs at the top
3. **Interactivity**: Most charts support hover interactions and zooming
4. **Data**: Comparing data from most recent update, top left slider menu, to previouse close. 

**Tip**: Click on the top-right menu of any chart to download it as an image.
""")