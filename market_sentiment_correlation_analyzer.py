import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from pathlib import Path
import logging
from datetime import datetime, timedelta
from scipy.stats import pearsonr, spearmanr
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketSentimentAnalyzer:
    def __init__(self):
        self.indices = {
            'SPY': 'S&P 500',
            'IWM': 'Russell 2000', 
            'QQQ': 'NASDAQ 100',
            'DIA': 'Dow Jones',
            'VTI': 'Total Market',
            'XLF': 'Financials',
            'XLK': 'Technology',
            'XLE': 'Energy'
        }
        
    @st.cache_data
    def load_stock_data(_self, base_path="Data/stock_scores"):
        """Load historical stock scoring data."""
        try:
            file_path = Path(base_path) / 'historical_data.parquet.gzip'
            if not file_path.exists():
                raise ValueError("Historical data file not found. Please run the optimizer first.")
                
            df = pd.read_parquet(file_path)
            # Ensure timezone-naive datetime
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            
            # Add market cap categories
            def categorize_market_cap(cap):
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
            
            df['market_cap_category'] = df['market_cap_B'].apply(categorize_market_cap)
            
            logger.info(f"Loaded stock data: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading stock data: {str(e)}")
            raise
    
    @st.cache_data
    def fetch_index_data(_self, symbols, start_date, end_date):
        """Fetch index data from yfinance."""
        try:
            index_data = {}
            
            for symbol, name in symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start_date, end=end_date)
                    
                    if not data.empty:
                        # Ensure timezone-naive datetime and clean column names
                        data = data.reset_index()
                        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
                        
                        # Calculate returns and rolling metrics
                        data['returns'] = data['Close'].pct_change()
                        data['returns_5d'] = data['Close'].pct_change(5)
                        data['returns_20d'] = data['Close'].pct_change(20)
                        data['volatility_20d'] = data['returns'].rolling(20).std() * np.sqrt(252)
                        
                        # Calculate peaks and troughs for market cycle analysis
                        close_prices = data['Close'].values
                        peaks, _ = find_peaks(close_prices, distance=20, prominence=close_prices.std()*0.5)
                        troughs, _ = find_peaks(-close_prices, distance=20, prominence=close_prices.std()*0.5)
                        
                        data['is_peak'] = False
                        data['is_trough'] = False
                        if len(peaks) > 0:
                            data.loc[peaks, 'is_peak'] = True
                        if len(troughs) > 0:
                            data.loc[troughs, 'is_trough'] = True
                        
                        index_data[symbol] = {
                            'name': name,
                            'data': data,
                            'peaks': peaks,
                            'troughs': troughs
                        }
                        
                        logger.info(f"Fetched {name} data: {len(data)} records")
                    
                except Exception as e:
                    logger.warning(f"Could not fetch data for {symbol}: {str(e)}")
                    
            return index_data
            
        except Exception as e:
            logger.error(f"Error fetching index data: {str(e)}")
            return {}
    
    def calculate_market_sentiment_metrics(self, stock_df, selected_caps, bullish_threshold, bearish_threshold):
        """Calculate various market sentiment metrics from stock scoring data."""
        try:
            # Filter by market cap if specified
            if selected_caps:
                filtered_df = stock_df[stock_df['market_cap_category'].isin(selected_caps)]
            else:
                filtered_df = stock_df
            
            # Group by date and calculate sentiment metrics
            daily_metrics = []
            
            for date in sorted(filtered_df['date'].unique()):
                day_data = filtered_df[filtered_df['date'] == date]
                
                total_stocks = len(day_data)
                if total_stocks == 0:
                    continue
                
                # Basic sentiment metrics
                bullish_stocks = len(day_data[day_data['bullish_score'] >= bullish_threshold])
                bearish_stocks = len(day_data[day_data['bearish_score'] >= bearish_threshold])
                
                # Percentage metrics
                bullish_pct = (bullish_stocks / total_stocks) * 100
                bearish_pct = (bearish_stocks / total_stocks) * 100
                
                # Net sentiment
                net_bullish_pct = bullish_pct - bearish_pct
                
                # Advanced metrics
                avg_bullish_score = day_data['bullish_score'].mean()
                avg_bearish_score = day_data['bearish_score'].mean()
                
                # Score distribution metrics
                high_conviction_bullish = len(day_data[day_data['bullish_score'] >= 80])
                high_conviction_bearish = len(day_data[day_data['bearish_score'] >= 80])
                
                # Extreme readings
                extreme_bullish_pct = (high_conviction_bullish / total_stocks) * 100
                extreme_bearish_pct = (high_conviction_bearish / total_stocks) * 100
                
                # Market cap weighted sentiment (if market cap data is available)
                total_market_cap = day_data['market_cap_B'].sum()
                if total_market_cap > 0:
                    bullish_weighted = day_data[day_data['bullish_score'] >= bullish_threshold]['market_cap_B'].sum()
                    bearish_weighted = day_data[day_data['bearish_score'] >= bearish_threshold]['market_cap_B'].sum()
                    
                    bullish_weighted_pct = (bullish_weighted / total_market_cap) * 100
                    bearish_weighted_pct = (bearish_weighted / total_market_cap) * 100
                else:
                    bullish_weighted_pct = bullish_pct
                    bearish_weighted_pct = bearish_pct
                
                # Sector breadth analysis
                sector_bullish = day_data[day_data['bullish_score'] >= bullish_threshold]['sector'].nunique()
                sector_bearish = day_data[day_data['bearish_score'] >= bearish_threshold]['sector'].nunique()
                total_sectors = day_data['sector'].nunique()
                
                sector_bullish_pct = (sector_bullish / total_sectors) * 100 if total_sectors > 0 else 0
                sector_bearish_pct = (sector_bearish / total_sectors) * 100 if total_sectors > 0 else 0
                
                daily_metrics.append({
                    'date': date,
                    'total_stocks': total_stocks,
                    'bullish_stocks': bullish_stocks,
                    'bearish_stocks': bearish_stocks,
                    'bullish_pct': bullish_pct,
                    'bearish_pct': bearish_pct,
                    'net_bullish_pct': net_bullish_pct,
                    'avg_bullish_score': avg_bullish_score,
                    'avg_bearish_score': avg_bearish_score,
                    'extreme_bullish_pct': extreme_bullish_pct,
                    'extreme_bearish_pct': extreme_bearish_pct,
                    'bullish_weighted_pct': bullish_weighted_pct,
                    'bearish_weighted_pct': bearish_weighted_pct,
                    'sector_bullish_pct': sector_bullish_pct,
                    'sector_bearish_pct': sector_bearish_pct,
                    'total_market_cap': total_market_cap
                })
            
            sentiment_df = pd.DataFrame(daily_metrics)
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.tz_localize(None)
            
            # Calculate rolling averages and momentum indicators
            for window in [5, 10, 20, 50]:
                sentiment_df[f'bullish_pct_ma{window}'] = sentiment_df['bullish_pct'].rolling(window).mean()
                sentiment_df[f'bearish_pct_ma{window}'] = sentiment_df['bearish_pct'].rolling(window).mean()
                sentiment_df[f'net_bullish_ma{window}'] = sentiment_df['net_bullish_pct'].rolling(window).mean()
            
            # Calculate sentiment momentum (rate of change)
            sentiment_df['bullish_momentum_5d'] = sentiment_df['bullish_pct'].diff(5)
            sentiment_df['bearish_momentum_5d'] = sentiment_df['bearish_pct'].diff(5)
            sentiment_df['net_momentum_5d'] = sentiment_df['net_bullish_pct'].diff(5)
            
            logger.info(f"Calculated sentiment metrics for {len(sentiment_df)} trading days")
            return sentiment_df.sort_values('date')
            
        except Exception as e:
            logger.error(f"Error calculating sentiment metrics: {str(e)}")
            raise
    
    def calculate_correlations(self, sentiment_df, index_data, lookback_periods=[20, 50, 252]):
        """Calculate correlations between sentiment metrics and index performance."""
        try:
            correlation_results = {}
            
            for symbol, idx_info in index_data.items():
                index_df = idx_info['data'].copy()
                
                # Merge sentiment and index data - both should now be timezone-naive
                merged_df = pd.merge(sentiment_df, index_df, left_on='date', right_on='Date', how='inner')
                
                if len(merged_df) < 50:  # Need sufficient data
                    logger.warning(f"Insufficient data for {symbol}: only {len(merged_df)} overlapping records")
                    continue
                
                logger.info(f"Merged data for {symbol}: {len(merged_df)} records")
                
                index_correlations = {}
                
                # Key sentiment metrics to correlate
                sentiment_metrics = [
                    'bullish_pct', 'bearish_pct', 'net_bullish_pct',
                    'extreme_bullish_pct', 'extreme_bearish_pct',
                    'bullish_weighted_pct', 'bearish_weighted_pct',
                    'sector_bullish_pct', 'sector_bearish_pct'
                ]
                
                # Index metrics to correlate against
                index_metrics = ['returns', 'returns_5d', 'returns_20d']
                
                for lookback in lookback_periods:
                    if len(merged_df) < lookback:
                        continue
                        
                    recent_data = merged_df.tail(lookback)
                    period_correlations = {}
                    
                    for sentiment_metric in sentiment_metrics:
                        for index_metric in index_metrics:
                            if sentiment_metric in recent_data.columns and index_metric in recent_data.columns:
                                # Remove NaN values
                                clean_data = recent_data[[sentiment_metric, index_metric]].dropna()
                                
                                if len(clean_data) >= 20:  # Minimum for meaningful correlation
                                    try:
                                        corr_pearson, p_value = pearsonr(clean_data[sentiment_metric], clean_data[index_metric])
                                        corr_spearman, _ = spearmanr(clean_data[sentiment_metric], clean_data[index_metric])
                                        
                                        period_correlations[f"{sentiment_metric}_vs_{index_metric}"] = {
                                            'pearson': corr_pearson,
                                            'spearman': corr_spearman,
                                            'p_value': p_value,
                                            'significant': p_value < 0.05,
                                            'sample_size': len(clean_data)
                                        }
                                    except Exception as corr_error:
                                        logger.warning(f"Correlation calculation failed for {sentiment_metric} vs {index_metric}: {corr_error}")
                    
                    index_correlations[f"{lookback}d"] = period_correlations
                
                correlation_results[symbol] = {
                    'name': idx_info['name'],
                    'correlations': index_correlations,
                    'merged_data': merged_df
                }
            
            logger.info(f"Calculated correlations for {len(correlation_results)} indices")
            return correlation_results
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {str(e)}")
            return {}
    
    def identify_market_turning_points(self, index_data, sentiment_df, window=20):
        """Identify potential market turning points and analyze sentiment at those times."""
        try:
            turning_points = {}
            
            for symbol, idx_info in index_data.items():
                index_df = idx_info['data'].copy()
                
                # Merge with sentiment data - both timezone-naive
                merged_df = pd.merge(sentiment_df, index_df, left_on='date', right_on='Date', how='inner')
                
                if len(merged_df) < 100:
                    logger.warning(f"Insufficient data for turning points analysis of {symbol}: {len(merged_df)} records")
                    continue
                
                # Find significant peaks and troughs in the index
                close_prices = merged_df['Close'].values
                dates = merged_df['date'].values
                
                # More sophisticated peak/trough detection
                peaks, peak_properties = find_peaks(close_prices, distance=window, prominence=np.std(close_prices)*0.5)
                troughs, trough_properties = find_peaks(-close_prices, distance=window, prominence=np.std(close_prices)*0.5)
                
                # Analyze sentiment at turning points
                peak_analysis = []
                trough_analysis = []
                
                # Analyze peaks (potential market tops)
                for peak_idx in peaks:
                    if peak_idx >= window and peak_idx < len(merged_df) - window:
                        peak_date = dates[peak_idx]
                        peak_price = close_prices[peak_idx]
                        
                        # Get sentiment data around the peak
                        window_data = merged_df.iloc[peak_idx-window:peak_idx+window+1]
                        
                        peak_analysis.append({
                            'date': peak_date,
                            'price': peak_price,
                            'type': 'peak',
                            'bullish_pct_avg': window_data['bullish_pct'].mean(),
                            'bearish_pct_avg': window_data['bearish_pct'].mean(),
                            'net_bullish_avg': window_data['net_bullish_pct'].mean(),
                            'extreme_bullish_avg': window_data['extreme_bullish_pct'].mean(),
                            'extreme_bearish_avg': window_data['extreme_bearish_pct'].mean(),
                            'bullish_pct_at_peak': merged_df.iloc[peak_idx]['bullish_pct'],
                            'bearish_pct_at_peak': merged_df.iloc[peak_idx]['bearish_pct'],
                            'days_before_after': window
                        })
                
                # Analyze troughs (potential market bottoms)
                for trough_idx in troughs:
                    if trough_idx >= window and trough_idx < len(merged_df) - window:
                        trough_date = dates[trough_idx]
                        trough_price = close_prices[trough_idx]
                        
                        # Get sentiment data around the trough
                        window_data = merged_df.iloc[trough_idx-window:trough_idx+window+1]
                        
                        trough_analysis.append({
                            'date': trough_date,
                            'price': trough_price,
                            'type': 'trough',
                            'bullish_pct_avg': window_data['bullish_pct'].mean(),
                            'bearish_pct_avg': window_data['bearish_pct'].mean(),
                            'net_bullish_avg': window_data['net_bullish_pct'].mean(),
                            'extreme_bullish_avg': window_data['extreme_bullish_pct'].mean(),
                            'extreme_bearish_avg': window_data['extreme_bearish_pct'].mean(),
                            'bullish_pct_at_trough': merged_df.iloc[trough_idx]['bullish_pct'],
                            'bearish_pct_at_trough': merged_df.iloc[trough_idx]['bearish_pct'],
                            'days_before_after': window
                        })
                
                turning_points[symbol] = {
                    'name': idx_info['name'],
                    'peaks': peak_analysis,
                    'troughs': trough_analysis,
                    'merged_data': merged_df
                }
            
            logger.info(f"Identified turning points for {len(turning_points)} indices")
            return turning_points
            
        except Exception as e:
            logger.error(f"Error identifying turning points: {str(e)}")
            return {}

def create_sentiment_index_chart(sentiment_df, index_data, selected_index, sentiment_metric):
    """Create a dual-axis chart showing sentiment vs index performance."""
    try:
        if selected_index not in index_data:
            st.error(f"Index {selected_index} not found in data")
            return go.Figure()
        
        index_info = index_data[selected_index]
        index_df = index_info['data'].copy()
        
        # Merge data - both should be timezone-naive now
        merged_df = pd.merge(sentiment_df, index_df, left_on='date', right_on='Date', how='inner')
        
        if merged_df.empty:
            st.warning(f"No overlapping data found between sentiment and {selected_index}")
            return go.Figure()
        
        st.info(f"📊 Chart data: {len(merged_df)} overlapping days between {merged_df['date'].min().strftime('%Y-%m-%d')} and {merged_df['date'].max().strftime('%Y-%m-%d')}")
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.15,
            subplot_titles=[f'{index_info["name"]} Price', f'{sentiment_metric.replace("_", " ").title()}'],
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
        )
        
        # Add index price
        fig.add_trace(
            go.Scatter(
                x=merged_df['date'],
                y=merged_df['Close'],
                name=f'{index_info["name"]} Price',
                line=dict(color='white', width=2),
                hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Add sentiment metric
        fig.add_trace(
            go.Scatter(
                x=merged_df['date'],
                y=merged_df[sentiment_metric],
                name=sentiment_metric.replace('_', ' ').title(),
                line=dict(color='cyan', width=2),
                hovertemplate=f"Date: %{{x}}<br>{sentiment_metric.replace('_', ' ').title()}: %{{y:.1f}}%<extra></extra>"
            ),
            row=2, col=1
        )
        
        # Add sentiment moving average if available
        ma_col = f'{sentiment_metric}_ma20'
        if ma_col in merged_df.columns and not merged_df[ma_col].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=merged_df['date'],
                    y=merged_df[ma_col],
                    name=f'{sentiment_metric.replace("_", " ").title()} MA20',
                    line=dict(color='orange', width=1, dash='dash'),
                    hovertemplate=f"Date: %{{x}}<br>MA20: %{{y:.1f}}%<extra></extra>"
                ),
                row=2, col=1
            )
        
        # Highlight peaks and troughs if available
        if 'is_peak' in merged_df.columns and merged_df['is_peak'].any():
            peaks = merged_df[merged_df['is_peak']]
            if not peaks.empty:
                fig.add_trace(
                    go.Scatter(
                        x=peaks['date'],
                        y=peaks['Close'],
                        mode='markers',
                        name='Market Peaks',
                        marker=dict(color='red', size=10, symbol='triangle-down'),
                        showlegend=True,
                        hovertemplate="Peak: %{x}<br>Price: $%{y:.2f}<extra></extra>"
                    ),
                    row=1, col=1
                )
        
        if 'is_trough' in merged_df.columns and merged_df['is_trough'].any():
            troughs = merged_df[merged_df['is_trough']]
            if not troughs.empty:
                fig.add_trace(
                    go.Scatter(
                        x=troughs['date'],
                        y=troughs['Close'],
                        mode='markers',
                        name='Market Troughs',
                        marker=dict(color='green', size=10, symbol='triangle-up'),
                        showlegend=True,
                        hovertemplate="Trough: %{x}<br>Price: $%{y:.2f}<extra></extra>"
                    ),
                    row=1, col=1
                )
        
        fig.update_layout(
            title=f'Market Sentiment Analysis: {index_info["name"]} vs {sentiment_metric.replace("_", " ").title()}',
            template='plotly_dark',
            height=700,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text=f"{sentiment_metric.replace('_', ' ').title()} (%)", row=2, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating sentiment chart: {str(e)}")
        st.error(f"Chart creation error: {str(e)}")
        return go.Figure().update_layout(title="Error creating chart")

def create_correlation_heatmap(correlation_results, lookback_period):
    """Create a heatmap showing correlations between sentiment metrics and index returns."""
    try:
        correlation_data = []
        
        for index_symbol, results in correlation_results.items():
            index_name = results['name']
            if lookback_period in results['correlations']:
                correlations = results['correlations'][lookback_period]
                
                for metric_pair, corr_data in correlations.items():
                    correlation_data.append({
                        'Index': index_name,
                        'Metric_Pair': metric_pair,
                        'Correlation': corr_data['pearson'],
                        'P_Value': corr_data['p_value'],
                        'Significant': corr_data['significant'],
                        'Sample_Size': corr_data['sample_size']
                    })
        
        if not correlation_data:
            return go.Figure().update_layout(title="No correlations calculated")
        
        corr_df = pd.DataFrame(correlation_data)
        
        # Show all correlations but highlight significant ones
        pivot_df = corr_df.pivot(index='Index', columns='Metric_Pair', values='Correlation')
        significance_df = corr_df.pivot(index='Index', columns='Metric_Pair', values='Significant')
        
        # Create text annotations
        text_annotations = []
        for i in range(len(pivot_df.index)):
            row_text = []
            for j in range(len(pivot_df.columns)):
                corr_val = pivot_df.iloc[i, j]
                if pd.notna(corr_val):
                    significant = significance_df.iloc[i, j] if not pd.isna(significance_df.iloc[i, j]) else False
                    if significant:
                        row_text.append(f"{corr_val:.3f}*")
                    else:
                        row_text.append(f"{corr_val:.3f}")
                else:
                    row_text.append("")
            text_annotations.append(row_text)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=[col.replace('_vs_', ' vs\n').replace('_', ' ') for col in pivot_df.columns],
            y=pivot_df.index,
            colorscale='RdBu',
            zmid=0,
            text=text_annotations,
            texttemplate="%{text}",
            textfont={"size": 9},
            hovertemplate='Index: %{y}<br>Metric: %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Sentiment-Index Correlations ({lookback_period}) - * indicates p<0.05',
            template='plotly_dark',
            height=max(400, len(pivot_df.index) * 60),
            xaxis_title="Sentiment Metric vs Index Return",
            yaxis_title="Index"
        )
        
        # Rotate x-axis labels for readability
        fig.update_xaxes(tickangle=45)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {str(e)}")
        return go.Figure().update_layout(title="Error creating heatmap")

def create_turning_points_analysis(turning_points, selected_index):
    """Create visualization showing sentiment at market turning points."""
    try:
        if selected_index not in turning_points:
            return go.Figure().update_layout(title="No turning points data available")
        
        tp_data = turning_points[selected_index]
        peaks = pd.DataFrame(tp_data['peaks'])
        troughs = pd.DataFrame(tp_data['troughs'])
        
        if peaks.empty and troughs.empty:
            return go.Figure().update_layout(title="No turning points found")
        
        fig = go.Figure()
        
        # Plot peaks
        if not peaks.empty:
            fig.add_trace(go.Scatter(
                x=peaks['bullish_pct_at_peak'],
                y=peaks['bearish_pct_at_peak'],
                mode='markers',
                name='Market Peaks',
                marker=dict(
                    color='red',
                    size=12,
                    symbol='triangle-down',
                    line=dict(width=1, color='white')
                ),
                text=[f"Peak: {pd.to_datetime(date).strftime('%Y-%m-%d')}" for date in peaks['date']],
                hovertemplate='Bullish %: %{x:.1f}<br>Bearish %: %{y:.1f}<br>%{text}<extra></extra>'
            ))
        
        # Plot troughs
        if not troughs.empty:
            fig.add_trace(go.Scatter(
                x=troughs['bullish_pct_at_trough'],
                y=troughs['bearish_pct_at_trough'],
                mode='markers',
                name='Market Troughs',
                marker=dict(
                    color='green',
                    size=12,
                    symbol='triangle-up',
                    line=dict(width=1, color='white')
                ),
                text=[f"Trough: {pd.to_datetime(date).strftime('%Y-%m-%d')}" for date in troughs['date']],
                hovertemplate='Bullish %: %{x:.1f}<br>Bearish %: %{y:.1f}<br>%{text}<extra></extra>'
            ))
        
        # Add diagonal line for reference
        max_val = 100
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color="gray", width=1, dash="dash"),
        )
        
        fig.update_layout(
            title=f'Sentiment at Market Turning Points - {tp_data["name"]}',
            xaxis_title='Bullish Sentiment %',
            yaxis_title='Bearish Sentiment %',
            template='plotly_dark',
            height=500,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating turning points analysis: {str(e)}")
        return go.Figure().update_layout(title="Error creating chart")

def main():
    st.set_page_config(page_title="Market Sentiment Correlation Analysis", layout="wide")
    st.title("📊 Market Sentiment Correlation Analysis")
    st.markdown("*Analyzing correlations between stock scoring sentiment and market indices*")
    
    analyzer = MarketSentimentAnalyzer()
    
    try:
        # Sidebar controls
        st.sidebar.header("Analysis Settings")
        
        # Load data first to determine appropriate thresholds
        with st.spinner("Loading data to determine optimal thresholds..."):
            stock_df = analyzer.load_stock_data()
            
        # Calculate suggested thresholds based on data
        if 'bullish_score' in stock_df.columns:
            bullish_median = stock_df['bullish_score'].median()
            bullish_75th = stock_df['bullish_score'].quantile(0.75)
            bearish_median = stock_df['bearish_score'].median()
            bearish_75th = stock_df['bearish_score'].quantile(0.75)
            
            suggested_bullish = min(70, int(bullish_75th))
            suggested_bearish = min(70, int(bearish_75th))
        else:
            suggested_bullish = 60
            suggested_bearish = 60
        
        # Market cap filter
        cap_categories = [
            "Mega Cap", "Large Cap", "Mid Cap",
            "Small Cap", "Micro Cap", "Nano Cap"
        ]
        selected_caps = st.sidebar.multiselect(
            "Market Cap Categories",
            options=cap_categories,
            default=cap_categories,  # Default to ALL categories selected
            help="Select market cap categories to include in sentiment analysis"
        )
        
        # Sentiment thresholds
        st.sidebar.subheader("Sentiment Thresholds")
        st.sidebar.info(f"💡 Suggested based on your data: Bullish ~{suggested_bullish}, Bearish ~{suggested_bearish}")
        
        bullish_threshold = st.sidebar.slider(
            "Bullish Score Threshold",
            min_value=40, max_value=90, value=suggested_bullish, step=5,
            help="Minimum bullish score to count as bullish sentiment"
        )
        bearish_threshold = st.sidebar.slider(
            "Bearish Score Threshold", 
            min_value=40, max_value=90, value=suggested_bearish, step=5,
            help="Minimum bearish score to count as bearish sentiment"
        )
        
        # Index selection
        selected_indices = st.sidebar.multiselect(
            "Select Indices to Analyze",
            options=list(analyzer.indices.keys()),
            default=['SPY', 'IWM', 'QQQ'],
            format_func=lambda x: f"{x} - {analyzer.indices[x]}"
        )
        
        if not selected_indices:
            st.warning("Please select at least one index to analyze.")
            return
        
        # Fetch index data
        with st.spinner("Fetching index data..."):
            start_date = stock_df['date'].min() - timedelta(days=30)
            end_date = stock_df['date'].max() + timedelta(days=1)
            
            selected_index_dict = {k: v for k, v in analyzer.indices.items() if k in selected_indices}
            index_data = analyzer.fetch_index_data(selected_index_dict, start_date, end_date)
        
        if not index_data:
            st.error("Could not fetch index data. Please check your internet connection.")
            return
        
        # Calculate sentiment metrics
        with st.spinner("Calculating sentiment metrics..."):
            sentiment_df = analyzer.calculate_market_sentiment_metrics(
                stock_df, selected_caps, bullish_threshold, bearish_threshold
            )
        
        # Calculate correlations
        with st.spinner("Calculating correlations..."):
            correlation_results = analyzer.calculate_correlations(sentiment_df, index_data)
        
        # Identify turning points
        with st.spinner("Identifying market turning points..."):
            turning_points = analyzer.identify_market_turning_points(index_data, sentiment_df)
        
        # Display results with COOL SENTIMENT METER at the top
        st.markdown("## 🎯 **REAL-TIME MARKET SENTIMENT DASHBOARD**")
        
        # Calculate current sentiment metrics for the meter
        latest_sentiment = sentiment_df.iloc[-1]
        current_bullish = latest_sentiment['bullish_pct']
        current_bearish = latest_sentiment['bearish_pct']
        current_net = latest_sentiment['net_bullish_pct']
        
        # Calculate historical percentiles for context
        bullish_percentile = (sentiment_df['bullish_pct'] <= current_bullish).mean() * 100
        bearish_percentile = (sentiment_df['bearish_pct'] <= current_bearish).mean() * 100
        
        # Create the SICK sentiment meter
        def create_sentiment_meter():
            """Create an awesome circular sentiment meter"""
            import plotly.graph_objects as go
            import numpy as np
            
            # Create the gauge chart
            fig = go.Figure()
            
            # Main sentiment gauge (bullish sentiment)
            fig.add_trace(go.Indicator(
                mode = "gauge+number+delta",
                value = current_bullish,
                domain = {'x': [0, 0.48], 'y': [0.15, 0.85]},
                title = {'text': "🐂 BULLISH", 'font': {'size': 24, 'color': 'white'}},
                delta = {'reference': sentiment_df['bullish_pct'].mean(), 'increasing': {'color': "lightgreen"}, 'decreasing': {'color': "red"}},
                gauge = {
                    'axis': {'range': [None, 80], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "lightgreen" if current_bullish > 35 else "#FFD700" if current_bullish > 25 else "lightblue"},
                    'bgcolor': "rgba(0,0,0,0.3)",
                    'borderwidth': 2,
                    'bordercolor': "white",
                    'steps': [
                        {'range': [0, 25], 'color': "#00FF00"},      # Bright Green - Safe zone
                        {'range': [25, 35], 'color': "#FFD700"},     # Gold/Yellow - Caution
                        {'range': [35, 50], 'color': "#FF8C00"},     # Dark Orange - Warning
                        {'range': [50, 80], 'color': "#FF0000"}      # Bright Red - Danger
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 45
                    }
                },
                number = {'font': {'size': 32, 'color': 'white'}, 'suffix': '%'}
            ))
            
            # Bearish sentiment gauge
            fig.add_trace(go.Indicator(
                mode = "gauge+number+delta",
                value = current_bearish,
                domain = {'x': [0.52, 1], 'y': [0.15, 0.85]},
                title = {'text': "🐻 BEARISH", 'font': {'size': 24, 'color': 'white'}},
                delta = {'reference': sentiment_df['bearish_pct'].mean(), 'increasing': {'color': "red"}, 'decreasing': {'color': "lightgreen"}},
                gauge = {
                    'axis': {'range': [None, 80], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#FF0000" if current_bearish > 40 else "#FF8C00" if current_bearish > 25 else "lightcoral"},
                    'bgcolor': "rgba(0,0,0,0.3)",
                    'borderwidth': 2,
                    'bordercolor': "white",
                    'steps': [
                        {'range': [0, 15], 'color': "#FF0000"},      # Bright Red - Complacency
                        {'range': [15, 25], 'color': "#FF8C00"},     # Dark Orange - Normal
                        {'range': [25, 40], 'color': "#FFD700"},     # Gold/Yellow - Elevated fear
                        {'range': [40, 80], 'color': "#00FF00"}      # Bright Green - Opportunity zone
                    ],
                    'threshold': {
                        'line': {'color': "green", 'width': 4},
                        'thickness': 0.75,
                        'value': 35
                    }
                },
                number = {'font': {'size': 32, 'color': 'white'}, 'suffix': '%'}
            ))
            
            # Net sentiment bar (bottom)
            net_color = "green" if current_net > 10 else "red" if current_net < -10 else "orange"
            fig.add_trace(go.Indicator(
                mode = "number+delta",
                value = current_net,
                domain = {'x': [0.2, 0.8], 'y': [0, 0.12]},
                title = {'text': "⚖️ NET SENTIMENT", 'font': {'size': 20, 'color': 'white'}},
                delta = {'reference': 0, 'position': "right"},
                number = {'font': {'size': 36, 'color': net_color}, 'suffix': '%', 'prefix': '+' if current_net > 0 else ''}
            ))
            
            fig.update_layout(
                paper_bgcolor = "rgba(0,0,0,0)",
                plot_bgcolor = "rgba(0,0,0,0)",
                font = {'color': "white", 'family': "Arial Black"},
                height = 450,
                margin = dict(l=20, r=20, t=60, b=20)
            )
            
            return fig
        
        # Display the meter
        meter_fig = create_sentiment_meter()
        st.plotly_chart(meter_fig, use_container_width=True)
        
        # Add gauge legend/explanation
        st.markdown("### 🎨 **Gauge Color Legend**")
        
        col_legend1, col_legend2 = st.columns(2)
        
        with col_legend1:
            st.markdown("""
            **🐂 Bullish Gauge Zones:**
            - 🟢 **Green (0-25%)**: Safe Zone - Normal bullish sentiment
            - 🟡 **Yellow (25-35%)**: Caution Zone - Elevated optimism  
            - 🟠 **Orange (35-50%)**: Warning Zone - High risk territory
            - 🔴 **Red (50%+)**: Danger Zone - Extreme euphoria
            - **Red Line at 45%**: Historical peak threshold
            """)
        
        with col_legend2:
            st.markdown("""
            **🐻 Bearish Gauge Zones:**
            - 🔴 **Red (0-15%)**: Complacency Zone - Too little fear
            - 🟠 **Orange (15-25%)**: Normal Zone - Healthy skepticism
            - 🟡 **Yellow (25-40%)**: Elevated Fear - Market stress
            - 🟢 **Green (40%+)**: Opportunity Zone - Potential bottoms
            - **Green Line at 35%**: Historical opportunity threshold
            """)
        
        # Add context indicators below the meter
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            # Bullish percentile indicator
            if bullish_percentile > 90:
                st.error("🚨 EXTREME HIGH")
            elif bullish_percentile > 75:
                st.warning("⚠️ HIGH RISK")
            elif bullish_percentile > 50:
                st.info("📈 ELEVATED")
            else:
                st.success("✅ NORMAL")
            st.caption(f"Bullish {bullish_percentile:.0f}th percentile")
        
        with col2:
            # Bearish percentile indicator  
            if bearish_percentile > 90:
                st.success("🎯 EXTREME OPPORTUNITY")
            elif bearish_percentile > 75:
                st.info("📈 OPPORTUNITY ZONE")
            elif bearish_percentile > 50:
                st.warning("⚠️ ELEVATED FEAR")
            else:
                st.error("😴 COMPLACENT")
            st.caption(f"Bearish {bearish_percentile:.0f}th percentile")
        
        with col3:
            # Momentum indicator
            recent_momentum = sentiment_df['bullish_momentum_5d'].iloc[-1] if 'bullish_momentum_5d' in sentiment_df.columns else 0
            if abs(recent_momentum) > 5:
                momentum_emoji = "🚀" if recent_momentum > 0 else "📉"
                st.warning(f"{momentum_emoji} STRONG")
            elif abs(recent_momentum) > 2:
                momentum_emoji = "📈" if recent_momentum > 0 else "📉"
                st.info(f"{momentum_emoji} MODERATE")
            else:
                st.success("😴 STABLE")
            st.caption(f"5d Momentum: {recent_momentum:+.1f}%")
        
        with col4:
            # Volume indicator
            if 'extreme_bullish_pct' in latest_sentiment:
                extreme_bullish = latest_sentiment['extreme_bullish_pct']
                if extreme_bullish > 15:
                    st.error("🔥 EXTREME")
                elif extreme_bullish > 8:
                    st.warning("⚡ HIGH")
                elif extreme_bullish > 4:
                    st.info("📊 MODERATE")
                else:
                    st.success("😌 LOW")
                st.caption(f"Extreme Bulls: {extreme_bullish:.1f}%")
        
        with col5:
            # Overall market phase
            if current_bullish > 45 and current_bearish < 20:
                st.error("🔴 EUPHORIA")
                phase_text = "Potential Top"
            elif current_bullish < 15 and current_bearish > 40:
                st.success("🟢 CAPITULATION") 
                phase_text = "Potential Bottom"
            elif current_bullish > 35:
                st.warning("🟡 FROTHY")
                phase_text = "Caution Advised"
            elif current_bearish > 30:
                st.info("🔵 FEARFUL")
                phase_text = "Watch for Opportunity"
            else:
                st.success("⚪ NEUTRAL")
                phase_text = "Balanced Market"
            st.caption(phase_text)
        
        # Add a sleek separator
        st.markdown("---")
        
        st.markdown("## 📈 Sentiment vs Index Performance")
        
        # Chart controls
        col1, col2, col3 = st.columns(3)
        with col1:
            chart_index = st.selectbox(
                "Select Index for Chart",
                options=list(index_data.keys()),
                format_func=lambda x: f"{x} - {analyzer.indices[x]}"
            )
        
        with col2:
            sentiment_metric = st.selectbox(
                "Select Sentiment Metric",
                options=[
                    'bullish_pct', 'bearish_pct', 'net_bullish_pct',
                    'extreme_bullish_pct', 'extreme_bearish_pct',
                    'bullish_weighted_pct', 'sector_bullish_pct'
                ],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col3:
            correlation_period = st.selectbox(
                "Correlation Period",
                options=['20d', '50d', '252d'],
                index=1,
                format_func=lambda x: f"{x.replace('d', ' days')}"
            )
        
        # Main sentiment vs index chart
        fig_main = create_sentiment_index_chart(sentiment_df, index_data, chart_index, sentiment_metric)
        st.plotly_chart(fig_main, use_container_width=True)
        
        # Summary statistics
        st.markdown("## 📊 Current Market Sentiment Summary")
        
        latest_sentiment = sentiment_df.iloc[-1]
        latest_date = latest_sentiment['date'].strftime('%Y-%m-%d')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Bullish Stocks %",
                f"{latest_sentiment['bullish_pct']:.1f}%",
                f"{latest_sentiment['bullish_pct'] - sentiment_df.iloc[-6]['bullish_pct']:.1f}%" if len(sentiment_df) > 5 else None
            )
        
        with col2:
            st.metric(
                "Bearish Stocks %", 
                f"{latest_sentiment['bearish_pct']:.1f}%",
                f"{latest_sentiment['bearish_pct'] - sentiment_df.iloc[-6]['bearish_pct']:.1f}%" if len(sentiment_df) > 5 else None
            )
        
        with col3:
            st.metric(
                "Net Bullish %",
                f"{latest_sentiment['net_bullish_pct']:.1f}%",
                f"{latest_sentiment['net_bullish_pct'] - sentiment_df.iloc[-6]['net_bullish_pct']:.1f}%" if len(sentiment_df) > 5 else None
            )
        
        with col4:
            st.metric(
                "Extreme Bullish %",
                f"{latest_sentiment['extreme_bullish_pct']:.1f}%",
                f"{latest_sentiment['extreme_bullish_pct'] - sentiment_df.iloc[-6]['extreme_bullish_pct']:.1f}%" if len(sentiment_df) > 5 else None
            )
        
        # Correlation analysis
        st.markdown("## 🔗 Correlation Analysis")
        
        if correlation_results:
            # Show debug information
            total_correlations = 0
            significant_correlations = 0
            
            for index_symbol, results in correlation_results.items():
                if correlation_period in results['correlations']:
                    correlations = results['correlations'][correlation_period]
                    total_correlations += len(correlations)
                    significant_correlations += sum(1 for corr_data in correlations.values() if corr_data['significant'])
            
            st.info(f"📊 Analysis Summary: {total_correlations} correlations computed, {significant_correlations} statistically significant (p<0.05)")
            
            fig_corr = create_correlation_heatmap(correlation_results, correlation_period)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Show strongest correlations
            st.markdown("### 🎯 Notable Correlations")
            
            all_correlations = []
            for index_symbol, results in correlation_results.items():
                if correlation_period in results['correlations']:
                    correlations = results['correlations'][correlation_period]
                    for metric_pair, corr_data in correlations.items():
                        if abs(corr_data['pearson']) > 0.2:  # Show correlations > 0.2
                            all_correlations.append({
                                'Index': results['name'],
                                'Metric Pair': metric_pair.replace('_vs_', ' vs ').replace('_', ' '),
                                'Correlation': corr_data['pearson'],
                                'P-Value': corr_data['p_value'],
                                'Significant': "Yes" if corr_data['significant'] else "No",
                                'Sample Size': corr_data['sample_size']
                            })
            
            if all_correlations:
                corr_display_df = pd.DataFrame(all_correlations)
                corr_display_df = corr_display_df.sort_values('Correlation', key=abs, ascending=False)
                st.dataframe(corr_display_df.round(4), use_container_width=True)
            else:
                st.info("No notable correlations (>0.2) found for the selected period.")
        else:
            st.warning("No correlation data available. Check that indices data was fetched successfully.")
        
        # Market turning points analysis
        st.markdown("## 🔄 Market Turning Points Analysis")
        
        if turning_points:
            turning_point_index = st.selectbox(
                "Select Index for Turning Points Analysis",
                options=list(turning_points.keys()),
                format_func=lambda x: f"{x} - {analyzer.indices[x]}",
                key="turning_points_index"
            )
            
            fig_turning = create_turning_points_analysis(turning_points, turning_point_index)
            st.plotly_chart(fig_turning, use_container_width=True)
            
            # Enhanced turning points summary with ALL data
            if turning_point_index in turning_points:
                tp_data = turning_points[turning_point_index]
                peaks_df = pd.DataFrame(tp_data['peaks'])
                troughs_df = pd.DataFrame(tp_data['troughs'])
                
                # Combine ALL turning points across ALL indices for comprehensive analysis
                all_peaks = []
                all_troughs = []
                
                for idx_symbol, idx_tp_data in turning_points.items():
                    if idx_tp_data['peaks']:
                        for peak in idx_tp_data['peaks']:
                            peak['index_name'] = analyzer.indices[idx_symbol]
                            all_peaks.append(peak)
                    
                    if idx_tp_data['troughs']:
                        for trough in idx_tp_data['troughs']:
                            trough['index_name'] = analyzer.indices[idx_symbol]
                            all_troughs.append(trough)
                
                all_peaks_df = pd.DataFrame(all_peaks)
                all_troughs_df = pd.DataFrame(all_troughs)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Market Peaks Analysis")
                    if not all_peaks_df.empty:
                        # Enhanced peak statistics
                        avg_bullish_all_peaks = all_peaks_df['bullish_pct_at_peak'].mean()
                        avg_bearish_all_peaks = all_peaks_df['bearish_pct_at_peak'].mean()
                        max_bullish_peak = all_peaks_df['bullish_pct_at_peak'].max()
                        min_bullish_peak = all_peaks_df['bullish_pct_at_peak'].min()
                        
                        st.metric("📊 Total Peaks Identified", len(all_peaks_df))
                        st.metric("📈 Average Bullish % at Peaks", f"{avg_bullish_all_peaks:.1f}%")
                        st.metric("📉 Average Bearish % at Peaks", f"{avg_bearish_all_peaks:.1f}%")
                        
                        # Peak severity levels
                        extreme_peaks = len(all_peaks_df[all_peaks_df['bullish_pct_at_peak'] > 60])
                        high_peaks = len(all_peaks_df[all_peaks_df['bullish_pct_at_peak'] > 45])
                        moderate_peaks = len(all_peaks_df[all_peaks_df['bullish_pct_at_peak'] > 35])
                        
                        st.markdown("**📊 Peak Severity Levels:**")
                        st.write(f"🔴 Extreme (>60% bullish): {extreme_peaks} peaks")
                        st.write(f"🟠 High (>45% bullish): {high_peaks} peaks") 
                        st.write(f"🟡 Moderate (>35% bullish): {moderate_peaks} peaks")
                        
                        # Show range
                        st.markdown(f"**📈 Bullish Range:** {min_bullish_peak:.1f}% - {max_bullish_peak:.1f}%")
                        
                        # Recent peaks table
                        st.markdown("**🕒 Recent Major Peaks:**")
                        recent_peaks = all_peaks_df.nlargest(5, 'date')[['date', 'index_name', 'bullish_pct_at_peak', 'bearish_pct_at_peak']]
                        recent_peaks['date'] = pd.to_datetime(recent_peaks['date']).dt.strftime('%Y-%m-%d')
                        recent_peaks.columns = ['Date', 'Index', 'Bullish %', 'Bearish %']
                        st.dataframe(recent_peaks, hide_index=True)
                    else:
                        st.info("No peaks identified across all indices")
                
                with col2:
                    st.markdown("### 📉 Market Troughs Analysis")
                    if not all_troughs_df.empty:
                        # Enhanced trough statistics
                        avg_bullish_all_troughs = all_troughs_df['bullish_pct_at_trough'].mean()
                        avg_bearish_all_troughs = all_troughs_df['bearish_pct_at_trough'].mean()
                        min_bullish_trough = all_troughs_df['bullish_pct_at_trough'].min()
                        max_bearish_trough = all_troughs_df['bearish_pct_at_trough'].max()
                        
                        st.metric("📊 Total Troughs Identified", len(all_troughs_df))
                        st.metric("📉 Average Bullish % at Troughs", f"{avg_bullish_all_troughs:.1f}%")
                        st.metric("📈 Average Bearish % at Troughs", f"{avg_bearish_all_troughs:.1f}%")
                        
                        # Trough severity levels
                        extreme_troughs = len(all_troughs_df[all_troughs_df['bearish_pct_at_trough'] > 50])
                        high_troughs = len(all_troughs_df[all_troughs_df['bearish_pct_at_trough'] > 35])
                        moderate_troughs = len(all_troughs_df[all_troughs_df['bearish_pct_at_trough'] > 25])
                        
                        st.markdown("**📊 Trough Severity Levels:**")
                        st.write(f"🔴 Extreme (>50% bearish): {extreme_troughs} troughs")
                        st.write(f"🟠 High (>35% bearish): {high_troughs} troughs")
                        st.write(f"🟡 Moderate (>25% bearish): {moderate_troughs} troughs")
                        
                        # Show range
                        st.markdown(f"**📉 Bullish Range at Troughs:** {min_bullish_trough:.1f}% - {all_troughs_df['bullish_pct_at_trough'].max():.1f}%")
                        st.markdown(f"**📈 Bearish Range at Troughs:** {all_troughs_df['bearish_pct_at_trough'].min():.1f}% - {max_bearish_trough:.1f}%")
                        
                        # Recent troughs table
                        st.markdown("**🕒 Recent Major Troughs:**")
                        recent_troughs = all_troughs_df.nlargest(5, 'date')[['date', 'index_name', 'bullish_pct_at_trough', 'bearish_pct_at_trough']]
                        recent_troughs['date'] = pd.to_datetime(recent_troughs['date']).dt.strftime('%Y-%m-%d')
                        recent_troughs.columns = ['Date', 'Index', 'Bullish %', 'Bearish %']
                        st.dataframe(recent_troughs, hide_index=True)
                    else:
                        st.info("No troughs identified across all indices")
                
                # Cross-index summary
                st.markdown("### 🌍 Cross-Index Turning Points Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🎯 Total Turning Points", len(all_peaks_df) + len(all_troughs_df))
                
                with col2:
                    st.metric("📈 Total Peaks", len(all_peaks_df))
                
                with col3:
                    st.metric("📉 Total Troughs", len(all_troughs_df))
                
                with col4:
                    if not all_peaks_df.empty and not all_troughs_df.empty:
                        sentiment_spread = avg_bullish_all_peaks - avg_bullish_all_troughs
                        st.metric("📊 Peak-Trough Spread", f"{sentiment_spread:.1f}%")
                
                # Enhanced insights based on ALL data
                st.markdown("### 💡 Enhanced Market Timing Insights")
                
                if not all_peaks_df.empty and not all_troughs_df.empty:
                    # Calculate comprehensive statistics
                    bullish_peak_75th = np.percentile(all_peaks_df['bullish_pct_at_peak'], 75)
                    bullish_peak_90th = np.percentile(all_peaks_df['bullish_pct_at_peak'], 90)
                    bearish_trough_75th = np.percentile(all_troughs_df['bearish_pct_at_trough'], 75)
                    bearish_trough_90th = np.percentile(all_troughs_df['bearish_pct_at_trough'], 90)
                    
                    current_bullish = latest_sentiment['bullish_pct']
                    current_bearish = latest_sentiment['bearish_pct']
                    
                    insights = []
                    
                    # Peak warnings
                    if current_bullish >= bullish_peak_90th:
                        insights.append(f"🚨 **EXTREME PEAK ALERT**: Current bullish sentiment ({current_bullish:.1f}%) exceeds 90% of historical peaks")
                    elif current_bullish >= bullish_peak_75th:
                        insights.append(f"⚠️ **PEAK WARNING**: Current bullish sentiment ({current_bullish:.1f}%) exceeds 75% of historical peaks")
                    
                    # Trough opportunities  
                    if current_bearish >= bearish_trough_90th:
                        insights.append(f"🎯 **EXTREME BOTTOM SIGNAL**: Current bearish sentiment ({current_bearish:.1f}%) exceeds 90% of historical troughs")
                    elif current_bearish >= bearish_trough_75th:
                        insights.append(f"📈 **BOTTOM OPPORTUNITY**: Current bearish sentiment ({current_bearish:.1f}%) exceeds 75% of historical troughs")
                    
                    # Historical context
                    insights.append(f"📊 **Historical Context**: Peak bullish readings typically range {all_peaks_df['bullish_pct_at_peak'].min():.1f}%-{all_peaks_df['bullish_pct_at_peak'].max():.1f}%")
                    insights.append(f"📊 **Historical Context**: Trough bearish readings typically range {all_troughs_df['bearish_pct_at_trough'].min():.1f}%-{all_troughs_df['bearish_pct_at_trough'].max():.1f}%")
                    
                    for insight in insights:
                        st.markdown(insight)
                        
                    # Trading thresholds based on historical data
                    st.markdown("### 🎯 Data-Driven Trading Thresholds")
                    
                    st.info(f"""
                    **📈 Bullish Sentiment Levels (Based on {len(all_peaks_df)} Historical Peaks):**
                    - 🟢 Normal: < {np.percentile(all_peaks_df['bullish_pct_at_peak'], 50):.0f}%
                    - 🟡 Elevated: {np.percentile(all_peaks_df['bullish_pct_at_peak'], 50):.0f}% - {np.percentile(all_peaks_df['bullish_pct_at_peak'], 75):.0f}%
                    - 🟠 High Risk: {np.percentile(all_peaks_df['bullish_pct_at_peak'], 75):.0f}% - {np.percentile(all_peaks_df['bullish_pct_at_peak'], 90):.0f}%
                    - 🔴 Extreme Risk: > {np.percentile(all_peaks_df['bullish_pct_at_peak'], 90):.0f}%
                    
                    **📉 Bearish Sentiment Levels (Based on {len(all_troughs_df)} Historical Troughs):**
                    - 🟢 Normal: < {np.percentile(all_troughs_df['bearish_pct_at_trough'], 50):.0f}%
                    - 🟡 Elevated: {np.percentile(all_troughs_df['bearish_pct_at_trough'], 50):.0f}% - {np.percentile(all_troughs_df['bearish_pct_at_trough'], 75):.0f}%
                    - 🟠 Opportunity Zone: {np.percentile(all_troughs_df['bearish_pct_at_trough'], 75):.0f}% - {np.percentile(all_troughs_df['bearish_pct_at_trough'], 90):.0f}%
                    - 🔴 Extreme Opportunity: > {np.percentile(all_troughs_df['bearish_pct_at_trough'], 90):.0f}%
                    """)
        else:
            st.warning("No turning points data available.")
        
        # Market timing signals with adaptive thresholds
        st.markdown("## 🚨 Market Timing Signals")
        
        # Calculate adaptive percentile thresholds based on actual data distribution
        bullish_80th = np.percentile(sentiment_df['bullish_pct'], 80)
        bullish_90th = np.percentile(sentiment_df['bullish_pct'], 90)
        bullish_95th = np.percentile(sentiment_df['bullish_pct'], 95)
        bullish_20th = np.percentile(sentiment_df['bullish_pct'], 20)
        bullish_10th = np.percentile(sentiment_df['bullish_pct'], 10)
        bullish_5th = np.percentile(sentiment_df['bullish_pct'], 5)
        
        bearish_80th = np.percentile(sentiment_df['bearish_pct'], 80)
        bearish_90th = np.percentile(sentiment_df['bearish_pct'], 90)
        bearish_95th = np.percentile(sentiment_df['bearish_pct'], 95)
        bearish_20th = np.percentile(sentiment_df['bearish_pct'], 20)
        bearish_10th = np.percentile(sentiment_df['bearish_pct'], 10)
        bearish_5th = np.percentile(sentiment_df['bearish_pct'], 5)
        
        # Calculate current sentiment vs historical percentiles
        current_bullish = latest_sentiment['bullish_pct']
        current_bearish = latest_sentiment['bearish_pct']
        current_extreme_bullish = latest_sentiment['extreme_bullish_pct']
        current_extreme_bearish = latest_sentiment['extreme_bearish_pct']
        
        # Historical percentiles
        bullish_percentile = (sentiment_df['bullish_pct'] <= current_bullish).mean() * 100
        bearish_percentile = (sentiment_df['bearish_pct'] <= current_bearish).mean() * 100
        extreme_bullish_percentile = (sentiment_df['extreme_bullish_pct'] <= current_extreme_bullish).mean() * 100
        extreme_bearish_percentile = (sentiment_df['extreme_bearish_pct'] <= current_extreme_bearish).mean() * 100
        
        # Show adaptive thresholds
        st.info(f"""
        **📊 Adaptive Sentiment Thresholds (Based on Your Historical Data):**
        
        **Bullish Sentiment:**
        - 95th percentile (extreme high): {bullish_95th:.1f}%
        - 80th percentile (high): {bullish_80th:.1f}%
        - 20th percentile (low): {bullish_20th:.1f}%
        - 5th percentile (extreme low): {bullish_5th:.1f}%
        
        **Bearish Sentiment:**
        - 95th percentile (extreme high): {bearish_95th:.1f}%
        - 80th percentile (high): {bearish_80th:.1f}%
        - 20th percentile (low): {bearish_20th:.1f}%
        - 5th percentile (extreme low): {bearish_5th:.1f}%
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🟢 Potential Bottom Signals")
            bottom_signals = []
            
            # Use adaptive thresholds
            if current_bearish >= bearish_80th:
                percentile_label = "high" if current_bearish < bearish_90th else "very high" if current_bearish < bearish_95th else "extreme"
                bottom_signals.append(f"🔴 {percentile_label.title()} bearish sentiment ({current_bearish:.1f}% - {bearish_percentile:.0f}th percentile)")
            
            if current_extreme_bearish >= np.percentile(sentiment_df['extreme_bearish_pct'], 85):
                bottom_signals.append(f"🔴 Elevated extreme bearish sentiment ({current_extreme_bearish:.1f}% - {extreme_bearish_percentile:.0f}th percentile)")
            
            if current_bullish <= bullish_20th:
                percentile_label = "low" if current_bullish > bullish_10th else "very low" if current_bullish > bullish_5th else "extreme low"
                bottom_signals.append(f"📉 {percentile_label.title()} bullish sentiment ({current_bullish:.1f}% - {bullish_percentile:.0f}th percentile)")
            
            # Check for sentiment divergence
            recent_10d = sentiment_df.tail(10)
            if len(recent_10d) >= 10:
                bearish_trend = recent_10d['bearish_pct'].diff().mean()
                if bearish_trend > 1:  # Adjusted for your data scale
                    bottom_signals.append(f"📈 Rising bearish sentiment (+{bearish_trend:.1f}% avg daily)")
            
            # Check net sentiment
            net_sentiment = latest_sentiment['net_bullish_pct']
            net_5th = np.percentile(sentiment_df['net_bullish_pct'], 5)
            net_10th = np.percentile(sentiment_df['net_bullish_pct'], 10)
            
            if net_sentiment <= net_10th:
                percentile_label = "low" if net_sentiment > net_5th else "extremely low"
                bottom_signals.append(f"⚠️ {percentile_label.title()} net bullish sentiment ({net_sentiment:.1f}%)")
            
            if bottom_signals:
                for signal in bottom_signals:
                    st.write(signal)
                    
                # Calculate bottom signal strength
                signal_strength = len(bottom_signals)
                if signal_strength >= 3:
                    st.success("🎯 **STRONG BOTTOM SIGNAL** - Multiple indicators suggest potential market bottom")
                elif signal_strength >= 2:
                    st.warning("⚠️ **MODERATE BOTTOM SIGNAL** - Some indicators suggest oversold conditions")
            else:
                st.info("No clear bottom signals detected")
        
        with col2:
            st.markdown("### 🔴 Potential Top Signals")
            top_signals = []
            
            # Use adaptive thresholds
            if current_bullish >= bullish_80th:
                percentile_label = "high" if current_bullish < bullish_90th else "very high" if current_bullish < bullish_95th else "extreme"
                top_signals.append(f"🟢 {percentile_label.title()} bullish sentiment ({current_bullish:.1f}% - {bullish_percentile:.0f}th percentile)")
            
            if current_extreme_bullish >= np.percentile(sentiment_df['extreme_bullish_pct'], 85):
                top_signals.append(f"🟢 Elevated extreme bullish sentiment ({current_extreme_bullish:.1f}% - {extreme_bullish_percentile:.0f}th percentile)")
            
            if current_bearish <= bearish_20th:
                percentile_label = "low" if current_bearish > bearish_10th else "very low" if current_bearish > bearish_5th else "extreme low"
                top_signals.append(f"📈 {percentile_label.title()} bearish sentiment ({current_bearish:.1f}% - {bearish_percentile:.0f}th percentile)")
            
            # Check for sentiment divergence
            recent_10d = sentiment_df.tail(10)
            if len(recent_10d) >= 10:
                bullish_trend = recent_10d['bullish_pct'].diff().mean()
                if bullish_trend > 1:  # Adjusted for your data scale
                    top_signals.append(f"📈 Rising bullish sentiment (+{bullish_trend:.1f}% avg daily)")
            
            # Check net sentiment for euphoria
            net_95th = np.percentile(sentiment_df['net_bullish_pct'], 95)
            net_90th = np.percentile(sentiment_df['net_bullish_pct'], 90)
            
            if net_sentiment >= net_90th:
                percentile_label = "high" if net_sentiment < net_95th else "extremely high"
                top_signals.append(f"🚀 {percentile_label.title()} net bullish sentiment ({net_sentiment:.1f}%)")
            
            if top_signals:
                for signal in top_signals:
                    st.write(signal)
                    
                # Calculate top signal strength
                signal_strength = len(top_signals)
                if signal_strength >= 3:
                    st.error("🚨 **STRONG TOP SIGNAL** - Multiple indicators suggest potential market top")
                elif signal_strength >= 2:
                    st.warning("⚠️ **MODERATE TOP SIGNAL** - Some indicators suggest overbought conditions")
            else:
                st.info("No clear top signals detected")
        
        # Advanced analysis section
        st.markdown("## 🔬 Advanced Analysis")
        
        with st.expander("📊 Sentiment Distribution Analysis"):
            # Create distribution chart
            fig_dist = go.Figure()
            
            fig_dist.add_trace(go.Histogram(
                x=sentiment_df['bullish_pct'],
                name='Bullish %',
                opacity=0.7,
                nbinsx=20
            ))
            
            fig_dist.add_trace(go.Histogram(
                x=sentiment_df['bearish_pct'],
                name='Bearish %',
                opacity=0.7,
                nbinsx=20
            ))
            
            # Add current reading line
            fig_dist.add_vline(
                x=current_bullish,
                line_dash="dash",
                line_color="cyan",
                annotation_text=f"Current Bullish: {current_bullish:.1f}%"
            )
            
            fig_dist.add_vline(
                x=current_bearish,
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Current Bearish: {current_bearish:.1f}%"
            )
            
            fig_dist.update_layout(
                title="Historical Sentiment Distribution",
                xaxis_title="Sentiment %",
                yaxis_title="Frequency",
                template='plotly_dark',
                barmode='overlay'
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with st.expander("📈 Momentum Analysis"):
            # Sentiment momentum chart
            fig_momentum = go.Figure()
            
            fig_momentum.add_trace(go.Scatter(
                x=sentiment_df['date'],
                y=sentiment_df['bullish_momentum_5d'],
                name='Bullish Momentum (5d)',
                line=dict(color='green')
            ))
            
            fig_momentum.add_trace(go.Scatter(
                x=sentiment_df['date'],
                y=sentiment_df['bearish_momentum_5d'],
                name='Bearish Momentum (5d)',
                line=dict(color='red')
            ))
            
            fig_momentum.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            
            fig_momentum.update_layout(
                title="Sentiment Momentum (5-day change)",
                xaxis_title="Date",
                yaxis_title="Momentum (% points)",
                template='plotly_dark'
            )
            
            st.plotly_chart(fig_momentum, use_container_width=True)
        
        # Export functionality
        st.markdown("## 💾 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export sentiment data
            sentiment_csv = sentiment_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sentiment Data",
                data=sentiment_csv,
                file_name=f"market_sentiment_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export correlation results
            if correlation_results:
                all_correlations = []
                for index_symbol, results in correlation_results.items():
                    for period, correlations in results['correlations'].items():
                        for metric_pair, corr_data in correlations.items():
                            all_correlations.append({
                                'Index': results['name'],
                                'Period': period,
                                'Metric_Pair': metric_pair,
                                'Pearson_Correlation': corr_data['pearson'],
                                'Spearman_Correlation': corr_data['spearman'],
                                'P_Value': corr_data['p_value'],
                                'Significant': corr_data['significant'],
                                'Sample_Size': corr_data['sample_size']
                            })
                
                if all_correlations:
                    corr_csv = pd.DataFrame(all_correlations).to_csv(index=False)
                    st.download_button(
                        label="📥 Download Correlations",
                        data=corr_csv,
                        file_name=f"sentiment_correlations_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        with col3:
            # Export turning points
            if turning_points:
                all_turning_points = []
                for index_symbol, tp_data in turning_points.items():
                    for peak in tp_data['peaks']:
                        peak['index'] = tp_data['name']
                        peak['point_type'] = 'peak'
                        all_turning_points.append(peak)
                    
                    for trough in tp_data['troughs']:
                        trough['index'] = tp_data['name']
                        trough['point_type'] = 'trough'
                        all_turning_points.append(trough)
                
                if all_turning_points:
                    tp_csv = pd.DataFrame(all_turning_points).to_csv(index=False)
                    st.download_button(
                        label="📥 Download Turning Points",
                        data=tp_csv,
                        file_name=f"turning_points_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        # Analysis insights
        st.markdown("## 💡 Key Insights")
        
        insights = []
        
        # Sentiment extremes
        if bullish_percentile > 90:
            insights.append("🔴 **Extreme bullish sentiment detected** - Market may be overbought")
        elif bullish_percentile < 10:
            insights.append("🟢 **Extreme low bullish sentiment** - Potential buying opportunity")
        
        if bearish_percentile > 90:
            insights.append("🟢 **Extreme bearish sentiment detected** - Market may be oversold")
        elif bearish_percentile < 10:
            insights.append("🔴 **Very low bearish sentiment** - Market may be complacent")
        
        # Correlation insights
        if correlation_results:
            strong_negative_corr = []
            strong_positive_corr = []
            
            for index_symbol, results in correlation_results.items():
                if '50d' in results['correlations']:
                    correlations = results['correlations']['50d']
                    for metric_pair, corr_data in correlations.items():
                        if corr_data['significant']:
                            if corr_data['pearson'] < -0.4:
                                strong_negative_corr.append(f"{results['name']} vs {metric_pair}")
                            elif corr_data['pearson'] > 0.4:
                                strong_positive_corr.append(f"{results['name']} vs {metric_pair}")
            
            if strong_negative_corr:
                insights.append(f"📊 **Strong negative correlations found:** {', '.join(strong_negative_corr[:2])}...")
            
            if strong_positive_corr:
                insights.append(f"📊 **Strong positive correlations found:** {', '.join(strong_positive_corr[:2])}...")
        
        # Market timing insights
        net_sentiment = latest_sentiment['net_bullish_pct']
        if net_sentiment > 20:
            insights.append("📈 **Strong net bullish sentiment** - Monitor for potential top formation")
        elif net_sentiment < -10:
            insights.append("📉 **Strong net bearish sentiment** - Monitor for potential bottom formation")
        
        if insights:
            for insight in insights:
                st.markdown(insight)
        else:
            st.info("No significant market timing signals detected at current sentiment levels.")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred during analysis. Please check your data and try again.")
        if st.checkbox("Show error details"):
            st.exception(e)

if __name__ == "__main__":
    main()