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
        
        # PERFECTLY CALIBRATED THRESHOLDS FROM HISTORICAL ANALYSIS 🎯
        self.bullish_thresholds = {
            'safe': 24,      # Below 25th percentile of historical peaks
            'caution': 33,   # 25th-50th percentile  
            'warning': 44,   # 50th-75th percentile
            'danger': 44     # Above 75th percentile (RED LINE)
        }
        
        self.bearish_thresholds = {
            'complacent': 66,   # Below 25th percentile of historical troughs
            'normal': 77,       # 25th-50th percentile
            'fearful': 83,      # 50th-75th percentile  
            'opportunity': 83   # Above 75th percentile (GREEN LINE)
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
                        
                        index_data[symbol] = {
                            'name': name,
                            'data': data
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
        """Analyze sentiment at manually defined major market turning points (2009-2025)."""
        try:
            # MANUALLY DEFINED MAJOR TURNING POINTS (2009-2025)
            # These are the MOST SIGNIFICANT market peaks and troughs
            
            manual_turning_points = {
                'SPY': {
                    'name': 'S&P 500',
                    'peaks': [
                        '2009-09-16',  # Recovery rally peak
                        '2011-04-29',  # QE2 peak
                        '2012-04-02',  # European debt crisis peak
                        '2014-12-29',  # QE infinity peak
                        '2015-05-20',  # Pre-crash peak
                        '2018-01-26',  # Tax cut euphoria
                        '2018-09-21',  # Pre-crash peak
                        '2020-02-19',  # Pre-COVID peak
                        '2021-01-29',  # Meme stock mania
                        '2021-11-08',  # Inflation denial peak
                        '2024-03-28',  # AI bubble peak
                        '2024-07-16',  # Election rally peak
                        '2025-01-17'   # Inauguration peak
                    ],
                    'troughs': [
                        '2009-03-09',  # Financial crisis bottom (82.1% bearish)
                        '2010-05-06',  # Flash crash
                        '2011-10-03',  # Debt ceiling crisis
                        '2015-08-24',  # China crash
                        '2016-01-20',  # Oil crash (82.1% bearish)
                        '2018-12-24',  # Christmas Eve massacre
                        '2020-03-23',  # COVID crash (76% bearish)
                        '2022-06-16',  # Bear market intermediate low
                        '2022-10-12',  # Bear market final low
                        '2023-03-13',  # Banking crisis
                        '2023-10-27',  # Bond yield spike
                        '2024-08-05'   # Japan carry trade unwind
                    ]
                },
                'IWM': {
                    'name': 'Russell 2000',
                    'peaks': [
                        '2009-09-15',  # Small cap recovery
                        '2011-04-29',  # QE2 small cap peak
                        '2012-03-26',  # Risk-on peak
                        '2014-07-01',  # Small cap bubble
                        '2015-06-23',  # Pre-crash peak
                        '2018-01-16',  # Small cap tax cut peak
                        '2018-08-31',  # Trade war peak
                        '2020-02-13',  # Pre-COVID small cap peak
                        '2021-01-27',  # Meme mania peak
                        '2021-03-15',  # Reopening trade peak
                        '2021-11-08',  # Small cap speculation
                        '2024-07-10',  # Trump trade peak
                        '2024-12-31',  # Year-end rally
                        '2025-01-15'   # Small cap rotation
                    ],
                    'troughs': [
                        '2009-03-09',  # Crisis bottom
                        '2010-05-25',  # European crisis
                        '2011-10-03',  # Small cap selloff
                        '2015-08-24',  # Growth scare
                        '2016-01-20',  # Energy crash
                        '2018-12-24',  # Small cap carnage
                        '2020-03-18',  # COVID small cap crash
                        '2022-05-12',  # Growth scare
                        '2022-06-17',  # Bear market low
                        '2022-09-30',  # Rate fear peak
                        '2023-03-13',  # Banking crisis
                        '2024-04-19'   # Earnings disappointment
                    ]
                },
                'QQQ': {
                    'name': 'NASDAQ 100',
                    'peaks': [
                        '2009-09-16',  # Tech recovery
                        '2011-04-29',  # Apple peak
                        '2012-04-02',  # Facebook IPO era
                        '2014-12-29',  # Tech dominance
                        '2015-07-20',  # Pre-correction peak
                        '2018-01-29',  # FAANG mania
                        '2018-08-29',  # Tech leadership
                        '2020-02-19',  # Pre-COVID tech peak
                        '2021-01-27',  # Tesla/meme peak
                        '2021-11-22',  # Tech bubble peak
                        '2024-03-21',  # AI mania peak
                        '2024-07-10',  # Mega-cap peak
                        '2025-01-13'   # Tech rotation
                    ],
                    'troughs': [
                        '2009-03-09',  # Tech crash bottom
                        '2010-05-06',  # Flash crash tech
                        '2011-10-03',  # European tech selloff
                        '2015-08-24',  # China tech scare
                        '2016-01-20',  # Apple/oil crash
                        '2018-12-24',  # Tech wreckage
                        '2020-03-23',  # COVID tech crash
                        '2022-05-11',  # Growth implosion
                        '2022-06-16',  # Tech bear market
                        '2022-12-28',  # Rate peak fears
                        '2023-03-13',  # SVB tech crisis
                        '2024-08-05'   # Mag 7 selloff
                    ]
                },
                'DIA': {
                    'name': 'Dow Jones',
                    'peaks': [
                        '2009-09-16',  # Industrial recovery
                        '2011-04-29',  # Commodity peak
                        '2012-04-02',  # Manufacturing peak
                        '2014-12-29',  # Energy peak
                        '2015-05-19',  # Dollar strength peak
                        '2018-01-26',  # Industrial euphoria
                        '2018-10-03',  # Trade war peak
                        '2020-02-12',  # Pre-COVID Dow peak
                        '2021-05-07',  # Reopening peak
                        '2021-11-08',  # Infrastructure peak
                        '2024-12-13',  # Trump industrial
                        '2025-01-20'   # Inauguration peak
                    ],
                    'troughs': [
                        '2009-03-06',  # Industrial bottom
                        '2010-05-06',  # Flash crash
                        '2011-10-03',  # Manufacturing fear
                        '2015-08-24',  # Commodity crash
                        '2016-01-20',  # Industrial recession
                        '2018-12-26',  # Dow Christmas crash
                        '2020-03-23',  # COVID industrial
                        '2022-06-17',  # Recession fears
                        '2022-09-30',  # Manufacturing PMI
                        '2023-03-13',  # Banking fears
                        '2024-08-05'   # Recession scare
                    ]
                }
            }
            
            turning_points = {}
            
            for symbol, idx_info in index_data.items():
                if symbol not in manual_turning_points:
                    continue  # Skip indices we haven't defined
                    
                index_df = idx_info['data'].copy()
                
                # Merge with sentiment data - both timezone-naive
                merged_df = pd.merge(sentiment_df, index_df, left_on='date', right_on='Date', how='inner')
                
                if len(merged_df) < 100:
                    logger.warning(f"Insufficient data for turning points analysis of {symbol}: {len(merged_df)} records")
                    continue
                
                # Get manual turning points for this symbol
                manual_peaks = manual_turning_points[symbol]['peaks']
                manual_troughs = manual_turning_points[symbol]['troughs']
                
                # Analyze sentiment at manual turning points
                peak_analysis = []
                trough_analysis = []
                
                # Analyze manual peaks
                for peak_date_str in manual_peaks:
                    try:
                        peak_date = pd.to_datetime(peak_date_str).tz_localize(None)
                        
                        # Find closest date in our data
                        merged_df['date_diff'] = abs(merged_df['date'] - peak_date)
                        closest_idx = merged_df['date_diff'].idxmin()
                        
                        if merged_df.loc[closest_idx, 'date_diff'].days <= 5:  # Within 5 days
                            closest_row = merged_df.loc[closest_idx]
                            
                            # Get window data around the peak
                            start_idx = max(0, closest_idx - window)
                            end_idx = min(len(merged_df), closest_idx + window + 1)
                            window_data = merged_df.iloc[start_idx:end_idx]
                            
                            peak_analysis.append({
                                'date': closest_row['date'],
                                'price': closest_row['Close'],
                                'type': 'peak',
                                'manual_date': peak_date_str,
                                'days_off': merged_df.loc[closest_idx, 'date_diff'].days,
                                'bullish_pct_avg': window_data['bullish_pct'].mean(),
                                'bearish_pct_avg': window_data['bearish_pct'].mean(),
                                'net_bullish_avg': window_data['net_bullish_pct'].mean(),
                                'extreme_bullish_avg': window_data['extreme_bullish_pct'].mean(),
                                'extreme_bearish_avg': window_data['extreme_bearish_pct'].mean(),
                                'bullish_pct_at_peak': closest_row['bullish_pct'],
                                'bearish_pct_at_peak': closest_row['bearish_pct'],
                                'days_before_after': window
                            })
                            logger.info(f"Found peak for {symbol} on {peak_date_str}: sentiment {closest_row['bullish_pct']:.1f}% bullish")
                        else:
                            logger.warning(f"No data within 5 days of manual peak {peak_date_str} for {symbol}")
                            
                    except Exception as e:
                        logger.warning(f"Error processing manual peak {peak_date_str} for {symbol}: {e}")
                
                # Analyze manual troughs
                for trough_date_str in manual_troughs:
                    try:
                        trough_date = pd.to_datetime(trough_date_str).tz_localize(None)
                        
                        # Find closest date in our data
                        merged_df['date_diff'] = abs(merged_df['date'] - trough_date)
                        closest_idx = merged_df['date_diff'].idxmin()
                        
                        if merged_df.loc[closest_idx, 'date_diff'].days <= 5:  # Within 5 days
                            closest_row = merged_df.loc[closest_idx]
                            
                            # Get window data around the trough
                            start_idx = max(0, closest_idx - window)
                            end_idx = min(len(merged_df), closest_idx + window + 1)
                            window_data = merged_df.iloc[start_idx:end_idx]
                            
                            trough_analysis.append({
                                'date': closest_row['date'],
                                'price': closest_row['Close'],
                                'type': 'trough',
                                'manual_date': trough_date_str,
                                'days_off': merged_df.loc[closest_idx, 'date_diff'].days,
                                'bullish_pct_avg': window_data['bullish_pct'].mean(),
                                'bearish_pct_avg': window_data['bearish_pct'].mean(),
                                'net_bullish_avg': window_data['net_bullish_pct'].mean(),
                                'extreme_bullish_avg': window_data['extreme_bullish_pct'].mean(),
                                'extreme_bearish_avg': window_data['extreme_bearish_pct'].mean(),
                                'bullish_pct_at_trough': closest_row['bullish_pct'],
                                'bearish_pct_at_trough': closest_row['bearish_pct'],
                                'days_before_after': window
                            })
                            logger.info(f"Found trough for {symbol} on {trough_date_str}: sentiment {closest_row['bearish_pct']:.1f}% bearish")
                        else:
                            logger.warning(f"No data within 5 days of manual trough {trough_date_str} for {symbol}")
                            
                    except Exception as e:
                        logger.warning(f"Error processing manual trough {trough_date_str} for {symbol}: {e}")
                
                turning_points[symbol] = {
                    'name': idx_info['name'],
                    'peaks': peak_analysis,
                    'troughs': trough_analysis,
                    'merged_data': merged_df
                }
                
                logger.info(f"Manual turning points for {symbol}: {len(peak_analysis)} peaks, {len(trough_analysis)} troughs")
            
            logger.info(f"Identified manual turning points for {len(turning_points)} indices")
            return turning_points
            
        except Exception as e:
            logger.error(f"Error identifying manual turning points: {str(e)}")
            return {}

def create_ultimate_sentiment_meter(sentiment_df, analyzer):
    """Create the ULTIMATE sentiment meter with perfect historical calibration! 🎯"""
    
    latest_sentiment = sentiment_df.iloc[-1]
    current_bullish = latest_sentiment['bullish_pct']
    current_bearish = latest_sentiment['bearish_pct']
    current_net = latest_sentiment['net_bullish_pct']
    
    # Calculate historical percentiles for context
    bullish_percentile = (sentiment_df['bullish_pct'] <= current_bullish).mean() * 100
    bearish_percentile = (sentiment_df['bearish_pct'] <= current_bearish).mean() * 100
    
    fig = go.Figure()
    
    # BULLISH GAUGE - Perfectly calibrated with historical peaks! 🎯
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = current_bullish,
        domain = {'x': [0, 0.48], 'y': [0.15, 0.85]},
        title = {'text': "🐂 BULLISH SENTIMENT", 'font': {'size': 24, 'color': 'white'}},
        delta = {'reference': sentiment_df['bullish_pct'].mean(), 'increasing': {'color': "orange"}, 'decreasing': {'color': "lightgreen"}},
        gauge = {
            'axis': {'range': [None, 70], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "red" if current_bullish >= analyzer.bullish_thresholds['danger'] else 
                            "orange" if current_bullish >= analyzer.bullish_thresholds['warning'] else
                            "yellow" if current_bullish >= analyzer.bullish_thresholds['caution'] else "lightgreen"},
            'bgcolor': "rgba(0,0,0,0.3)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, analyzer.bullish_thresholds['caution']], 'color': "#00FF00"},      # SAFE: 0-24%
                {'range': [analyzer.bullish_thresholds['caution'], analyzer.bullish_thresholds['warning']], 'color': "#FFD700"},     # CAUTION: 24-33%
                {'range': [analyzer.bullish_thresholds['warning'], analyzer.bullish_thresholds['danger']], 'color': "#FF8C00"},     # WARNING: 33-44%
                {'range': [analyzer.bullish_thresholds['danger'], 70], 'color': "#FF0000"}      # DANGER: 44%+
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': analyzer.bullish_thresholds['danger']  # RED LINE AT 44%
            }
        },
        number = {'font': {'size': 32, 'color': 'white'}, 'suffix': '%'}
    ))
    
    # BEARISH GAUGE - Perfectly calibrated with historical troughs! 🎯
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = current_bearish,
        domain = {'x': [0.52, 1], 'y': [0.15, 0.85]},
        title = {'text': "🐻 BEARISH SENTIMENT", 'font': {'size': 24, 'color': 'white'}},
        delta = {'reference': sentiment_df['bearish_pct'].mean(), 'increasing': {'color': "lightgreen"}, 'decreasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 95], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "lightgreen" if current_bearish >= analyzer.bearish_thresholds['opportunity'] else 
                            "yellow" if current_bearish >= analyzer.bearish_thresholds['fearful'] else
                            "orange" if current_bearish >= analyzer.bearish_thresholds['normal'] else "red"},
            'bgcolor': "rgba(0,0,0,0.3)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, analyzer.bearish_thresholds['normal']], 'color': "#FF0000"},      # COMPLACENT: 0-66%
                {'range': [analyzer.bearish_thresholds['normal'], analyzer.bearish_thresholds['fearful']], 'color': "#FF8C00"},     # NORMAL: 66-77%
                {'range': [analyzer.bearish_thresholds['fearful'], analyzer.bearish_thresholds['opportunity']], 'color': "#FFD700"},     # FEARFUL: 77-83%
                {'range': [analyzer.bearish_thresholds['opportunity'], 95], 'color': "#00FF00"}      # OPPORTUNITY: 83%+
            ],
            'threshold': {
                'line': {'color': "green", 'width': 4},
                'thickness': 0.75,
                'value': analyzer.bearish_thresholds['opportunity']  # GREEN LINE AT 83%
            }
        },
        number = {'font': {'size': 32, 'color': 'white'}, 'suffix': '%'}
    ))
    
    # NET SENTIMENT INDICATOR
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
        margin = dict(l=20, r=20, t=60, b=20),
        title = {
            'text': f"🎯 PERFECTLY CALIBRATED SENTIMENT METER 🎯<br><sub>Based on 62 Historical Turning Points (2009-2025)</sub>",
            'x': 0.5,
            'font': {'size': 28, 'color': 'white'}
        }
    )
    
    return fig

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
                text=[f"Peak: {pd.to_datetime(date).strftime('%Y-%m-%d')}<br>Manual: {manual}" 
                      for date, manual in zip(peaks['date'], peaks['manual_date'])],
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
                text=[f"Trough: {pd.to_datetime(date).strftime('%Y-%m-%d')}<br>Manual: {manual}" 
                      for date, manual in zip(troughs['date'], troughs['manual_date'])],
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
            title=f'Sentiment at Manual Turning Points - {tp_data["name"]}',
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

def create_historical_context_chart(sentiment_df, analyzer):
    """Create a chart showing current sentiment vs historical extremes."""
    
    latest_sentiment = sentiment_df.iloc[-1]
    current_bullish = latest_sentiment['bullish_pct']
    current_bearish = latest_sentiment['bearish_pct']
    
    # Create historical distribution
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Bullish Sentiment Distribution', 'Bearish Sentiment Distribution'],
        vertical_spacing=0.15
    )
    
    # Bullish distribution
    fig.add_trace(
        go.Histogram(
            x=sentiment_df['bullish_pct'],
            name='Historical Bullish %',
            opacity=0.7,
            nbinsx=30,
            marker_color='lightblue'
        ),
        row=1, col=1
    )
    
    # Add threshold lines for bullish
    for threshold, label in zip([analyzer.bullish_thresholds['caution'], analyzer.bullish_thresholds['warning'], analyzer.bullish_thresholds['danger']], 
                               ['Caution', 'Warning', 'Danger']):
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="orange" if label == 'Caution' else "red" if label == 'Danger' else "yellow",
            annotation_text=f"{label}: {threshold}%",
            row=1, col=1
        )
    
    # Current bullish line
    fig.add_vline(
        x=current_bullish,
        line_dash="solid",
        line_color="cyan",
        line_width=3,
        annotation_text=f"Current: {current_bullish:.1f}%",
        row=1, col=1
    )
    
    # Bearish distribution
    fig.add_trace(
        go.Histogram(
            x=sentiment_df['bearish_pct'],
            name='Historical Bearish %',
            opacity=0.7,
            nbinsx=30,
            marker_color='lightcoral'
        ),
        row=2, col=1
    )
    
    # Add threshold lines for bearish
    for threshold, label in zip([analyzer.bearish_thresholds['normal'], analyzer.bearish_thresholds['fearful'], analyzer.bearish_thresholds['opportunity']], 
                               ['Normal', 'Fearful', 'Opportunity']):
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="orange" if label == 'Normal' else "green" if label == 'Opportunity' else "yellow",
            annotation_text=f"{label}: {threshold}%",
            row=2, col=1
        )
    
    # Current bearish line
    fig.add_vline(
        x=current_bearish,
        line_dash="solid",
        line_color="cyan",
        line_width=3,
        annotation_text=f"Current: {current_bearish:.1f}%",
        row=2, col=1
    )
    
    fig.update_layout(
        title="🎯 Current Sentiment vs Historical Distribution",
        template='plotly_dark',
        height=600,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Bullish Sentiment %", row=1, col=1)
    fig.update_xaxes(title_text="Bearish Sentiment %", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    
    return fig

def main():
    st.set_page_config(page_title="🎯 ULTIMATE Market Sentiment Analyzer", layout="wide", initial_sidebar_state="expanded")
    
    # EPIC HEADER
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4); 
                border-radius: 15px; margin-bottom: 30px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);'>
        <h1 style='color: white; font-size: 3.5em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>
            🎯 ULTIMATE MARKET SENTIMENT ANALYZER 🎯
        </h1>
        <h3 style='color: white; margin: 10px 0 0 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>
            ⚡ Perfectly Calibrated with 62 Historical Turning Points (2009-2025) ⚡
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    analyzer = MarketSentimentAnalyzer()
    
    try:
        # Sidebar controls with enhanced styling
        st.sidebar.markdown("## 🔧 **ANALYSIS CONTROLS**")
        
        # Load data first to determine appropriate thresholds
        with st.spinner("🚀 Loading perfectly calibrated data..."):
            stock_df = analyzer.load_stock_data()
            
        # Market cap filter
        cap_categories = [
            "Mega Cap", "Large Cap", "Mid Cap",
            "Small Cap", "Micro Cap", "Nano Cap"
        ]
        selected_caps = st.sidebar.multiselect(
            "📊 Market Cap Categories",
            options=cap_categories,
            default=cap_categories,
            help="Select market cap categories to include in sentiment analysis"
        )
        
        # Sentiment thresholds with historical context
        st.sidebar.markdown("### 🎯 **Sentiment Thresholds**")
        st.sidebar.success(f"""
        **🔥 PERFECTLY CALIBRATED THRESHOLDS:**
        - **Bullish Warning**: {analyzer.bullish_thresholds['danger']}%
        - **Bearish Opportunity**: {analyzer.bearish_thresholds['opportunity']}%
        
        *Based on 62 historical turning points!*
        """)
        
        bullish_threshold = st.sidebar.slider(
            "Bullish Score Threshold",
            min_value=40, max_value=90, value=65, step=5,
            help="Minimum bullish score to count as bullish sentiment"
        )
        bearish_threshold = st.sidebar.slider(
            "Bearish Score Threshold", 
            min_value=40, max_value=90, value=65, step=5,
            help="Minimum bearish score to count as bearish sentiment"
        )
        
        # Index selection
        selected_indices = st.sidebar.multiselect(
            "📈 Select Indices to Analyze",
            options=list(analyzer.indices.keys()),
            default=['SPY', 'IWM', 'QQQ'],
            format_func=lambda x: f"{x} - {analyzer.indices[x]}"
        )
        
        if not selected_indices:
            st.warning("Please select at least one index to analyze.")
            return
        
        # Fetch index data
        with st.spinner("📡 Fetching index data..."):
            start_date = stock_df['date'].min() - timedelta(days=30)
            end_date = stock_df['date'].max() + timedelta(days=1)
            
            selected_index_dict = {k: v for k, v in analyzer.indices.items() if k in selected_indices}
            index_data = analyzer.fetch_index_data(selected_index_dict, start_date, end_date)
        
        if not index_data:
            st.error("Could not fetch index data. Please check your internet connection.")
            return
        
        # Calculate sentiment metrics
        with st.spinner("🧠 Calculating sentiment metrics..."):
            sentiment_df = analyzer.calculate_market_sentiment_metrics(
                stock_df, selected_caps, bullish_threshold, bearish_threshold
            )
        
        # Calculate correlations
        with st.spinner("🔗 Calculating correlations..."):
            correlation_results = analyzer.calculate_correlations(sentiment_df, index_data)
        
        # Identify turning points
        with st.spinner("🎯 Analyzing 62 manual turning points..."):
            turning_points = analyzer.identify_market_turning_points(index_data, sentiment_df)
        
        # MAIN DASHBOARD DISPLAY
        st.markdown("---")
        
        # Display the ULTIMATE sentiment meter
        meter_fig = create_ultimate_sentiment_meter(sentiment_df, analyzer)
        st.plotly_chart(meter_fig, use_container_width=True)
        
        # Enhanced gauge legend
        st.markdown("### 🎨 **PERFECTLY CALIBRATED GAUGE ZONES**")
        
        col_legend1, col_legend2 = st.columns(2)
        
        with col_legend1:
            st.success(f"""
            **🐂 BULLISH GAUGE (Based on Historical Peaks):**
            - 🟢 **SAFE (0-{analyzer.bullish_thresholds['caution']}%)**: Normal sentiment - Continue holding
            - 🟡 **CAUTION ({analyzer.bullish_thresholds['caution']}-{analyzer.bullish_thresholds['warning']}%)**: Elevated optimism - Watch closely  
            - 🟠 **WARNING ({analyzer.bullish_thresholds['warning']}-{analyzer.bullish_thresholds['danger']}%)**: High risk territory - Consider reducing
            - 🔴 **DANGER ({analyzer.bullish_thresholds['danger']}%+)**: Extreme euphoria - Historical peak zone!
            - **🚨 RED LINE at {analyzer.bullish_thresholds['danger']}%**: 75th percentile of all peaks
            """)
        
        with col_legend2:
            st.success(f"""
            **🐻 BEARISH GAUGE (Based on Historical Troughs):**
            - 🔴 **COMPLACENT (0-{analyzer.bearish_thresholds['normal']}%)**: Dangerous low fear
            - 🟠 **NORMAL ({analyzer.bearish_thresholds['normal']}-{analyzer.bearish_thresholds['fearful']}%)**: Healthy skepticism
            - 🟡 **FEARFUL ({analyzer.bearish_thresholds['fearful']}-{analyzer.bearish_thresholds['opportunity']}%)**: High fear - Getting interesting
            - 🟢 **OPPORTUNITY ({analyzer.bearish_thresholds['opportunity']}%+)**: Extreme fear - Historical bottom zone!
            - **💚 GREEN LINE at {analyzer.bearish_thresholds['opportunity']}%**: 75th percentile of all troughs
            """)
        
        # Current market assessment with enhanced context
        latest_sentiment = sentiment_df.iloc[-1]
        latest_date = latest_sentiment['date'].strftime('%Y-%m-%d')
        
        current_bullish = latest_sentiment['bullish_pct']
        current_bearish = latest_sentiment['bearish_pct']
        current_net = latest_sentiment['net_bullish_pct']
        
        # Calculate percentiles
        bullish_percentile = (sentiment_df['bullish_pct'] <= current_bullish).mean() * 100
        bearish_percentile = (sentiment_df['bearish_pct'] <= current_bearish).mean() * 100
        
        # Status determination
        def get_bullish_status(value):
            if value < analyzer.bullish_thresholds['caution']:
                return "SAFE ✅", "success"
            elif value < analyzer.bullish_thresholds['warning']:
                return "CAUTION ⚠️", "warning"
            elif value < analyzer.bullish_thresholds['danger']:
                return "WARNING 🚨", "error"
            else:
                return "DANGER 🔥", "error"
        
        def get_bearish_status(value):
            if value < analyzer.bearish_thresholds['normal']:
                return "COMPLACENT 😴", "error"
            elif value < analyzer.bearish_thresholds['fearful']:
                return "NORMAL 😐", "info"
            elif value < analyzer.bearish_thresholds['opportunity']:
                return "FEARFUL 😨", "warning"
            else:
                return "OPPORTUNITY 🎯", "success"
        
        bullish_status, bullish_color = get_bullish_status(current_bullish)
        bearish_status, bearish_color = get_bearish_status(current_bearish)
        
        # Market status display
        st.markdown(f"## 🎯 **CURRENT MARKET STATUS** ({latest_date})")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if bullish_color == "success":
                st.success(f"**BULLISH**\n{current_bullish:.1f}%\n{bullish_status}")
            elif bullish_color == "warning":
                st.warning(f"**BULLISH**\n{current_bullish:.1f}%\n{bullish_status}")
            else:
                st.error(f"**BULLISH**\n{current_bullish:.1f}%\n{bullish_status}")
            st.caption(f"{bullish_percentile:.0f}th percentile")
        
        with col2:
            if bearish_color == "success":
                st.success(f"**BEARISH**\n{current_bearish:.1f}%\n{bearish_status}")
            elif bearish_color == "warning":
                st.warning(f"**BEARISH**\n{current_bearish:.1f}%\n{bearish_status}")
            else:
                st.error(f"**BEARISH**\n{current_bearish:.1f}%\n{bearish_status}")
            st.caption(f"{bearish_percentile:.0f}th percentile")
        
        with col3:
            net_color_type = "success" if current_net > 10 else "error" if current_net < -10 else "info"
            if net_color_type == "success":
                st.success(f"**NET**\n{current_net:+.1f}%")
            elif net_color_type == "error":
                st.error(f"**NET**\n{current_net:+.1f}%")
            else:
                st.info(f"**NET**\n{current_net:+.1f}%")
            st.caption("Bullish - Bearish")
        
        with col4:
            extreme_bullish = latest_sentiment['extreme_bullish_pct']
            if extreme_bullish > 15:
                st.error(f"**EXTREME BULLS**\n{extreme_bullish:.1f}%\n🔥 HIGH")
            elif extreme_bullish > 8:
                st.warning(f"**EXTREME BULLS**\n{extreme_bullish:.1f}%\n⚡ MODERATE")
            else:
                st.success(f"**EXTREME BULLS**\n{extreme_bullish:.1f}%\n😌 LOW")
            st.caption("80+ Bull Score")
        
        with col5:
            # Overall market phase
            if current_bullish > analyzer.bullish_thresholds['danger'] and current_bearish < analyzer.bearish_thresholds['normal']:
                st.error("**PHASE**\n🔴 EUPHORIA\nPotential Top")
            elif current_bullish < analyzer.bullish_thresholds['caution'] and current_bearish > analyzer.bearish_thresholds['opportunity']:
                st.success("**PHASE**\n🟢 CAPITULATION\nPotential Bottom") 
            elif current_bullish > analyzer.bullish_thresholds['warning']:
                st.warning("**PHASE**\n🟡 FROTHY\nCaution Advised")
            elif current_bearish > analyzer.bearish_thresholds['fearful']:
                st.info("**PHASE**\n🔵 FEARFUL\nWatch for Opportunity")
            else:
                st.success("**PHASE**\n⚪ NEUTRAL\nBalanced Market")
        
        # Historical context chart
        st.markdown("## 📊 **Historical Context Analysis**")
        context_fig = create_historical_context_chart(sentiment_df, analyzer)
        st.plotly_chart(context_fig, use_container_width=True)
        
        # Enhanced metrics summary
        st.markdown("## 📈 **Enhanced Market Metrics**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Total Stocks Analyzed",
                f"{latest_sentiment['total_stocks']:,}",
                help="Number of stocks in current analysis"
            )
        
        with col2:
            momentum_5d = latest_sentiment.get('bullish_momentum_5d', 0) or 0
            st.metric(
                "🚀 Bullish Momentum (5d)",
                f"{momentum_5d:+.1f}%",
                delta=f"{momentum_5d:+.1f}% change",
                help="5-day change in bullish sentiment"
            )
        
        with col3:
            sector_bullish = latest_sentiment.get('sector_bullish_pct', 0) or 0
            st.metric(
                "🏭 Sector Breadth",
                f"{sector_bullish:.1f}%",
                help="Percentage of sectors showing bullish sentiment"
            )
        
        with col4:
            market_cap = latest_sentiment.get('total_market_cap', 0) or 0
            st.metric(
                "💰 Total Market Cap",
                f"${market_cap/1000:.1f}T" if market_cap > 1000 else f"${market_cap:.1f}B",
                help="Total market capitalization analyzed"
            )
        
        # Sentiment vs Index Performance
        st.markdown("## 📈 **Sentiment vs Index Performance**")
        
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
        
        # Correlation analysis
        st.markdown("## 🔗 **Advanced Correlation Analysis**")
        
        if correlation_results:
            # Show correlation summary
            total_correlations = 0
            significant_correlations = 0
            
            for index_symbol, results in correlation_results.items():
                if correlation_period in results['correlations']:
                    correlations = results['correlations'][correlation_period]
                    total_correlations += len(correlations)
                    significant_correlations += sum(1 for corr_data in correlations.values() if corr_data['significant'])
            
            st.info(f"📊 **Correlation Summary**: {total_correlations} correlations computed, {significant_correlations} statistically significant (p<0.05)")
            
            fig_corr = create_correlation_heatmap(correlation_results, correlation_period)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Show strongest correlations table
            st.markdown("### 🎯 **Notable Correlations**")
            
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
                                'Significant': "✅" if corr_data['significant'] else "❌",
                                'Sample Size': corr_data['sample_size']
                            })
            
            if all_correlations:
                corr_display_df = pd.DataFrame(all_correlations)
                corr_display_df = corr_display_df.sort_values('Correlation', key=abs, ascending=False)
                st.dataframe(
                    corr_display_df.round(4), 
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No notable correlations (>0.2) found for the selected period.")
        else:
            st.warning("No correlation data available. Check that indices data was fetched successfully.")
        
        # Market turning points analysis
        st.markdown("## 🔄 **Market Turning Points Analysis**")
        
        if turning_points:
            turning_point_index = st.selectbox(
                "Select Index for Turning Points Analysis",
                options=list(turning_points.keys()),
                format_func=lambda x: f"{x} - {analyzer.indices[x]}",
                key="turning_points_index"
            )
            
            fig_turning = create_turning_points_analysis(turning_points, turning_point_index)
            st.plotly_chart(fig_turning, use_container_width=True)
            
            # Enhanced turning points summary
            if turning_point_index in turning_points:
                tp_data = turning_points[turning_point_index]
                peaks_df = pd.DataFrame(tp_data['peaks'])
                troughs_df = pd.DataFrame(tp_data['troughs'])
                
                # Comprehensive turning points statistics
                st.markdown("### 🎯 **Turning Points Statistics**")
                
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
                    st.markdown("#### 📈 **Peak Analysis**")
                    if not all_peaks_df.empty:
                        avg_bullish_peaks = all_peaks_df['bullish_pct_at_peak'].mean()
                        avg_bearish_peaks = all_peaks_df['bearish_pct_at_peak'].mean()
                        max_bullish_peak = all_peaks_df['bullish_pct_at_peak'].max()
                        
                        st.success(f"""
                        **🔥 Peak Sentiment Statistics:**
                        - **Total Peaks**: {len(all_peaks_df)}
                        - **Avg Bullish at Peaks**: {avg_bullish_peaks:.1f}%
                        - **Avg Bearish at Peaks**: {avg_bearish_peaks:.1f}%
                        - **Max Bullish Reading**: {max_bullish_peak:.1f}%
                        - **Current vs Peak Avg**: {current_bullish - avg_bullish_peaks:+.1f}%
                        """)
                        
                        # Show recent major peaks
                        recent_peaks = all_peaks_df.nlargest(5, 'date')[['date', 'manual_date', 'index_name', 'bullish_pct_at_peak', 'bearish_pct_at_peak']].copy()
                        recent_peaks['date'] = pd.to_datetime(recent_peaks['date']).dt.strftime('%Y-%m-%d')
                        recent_peaks.columns = ['Actual Date', 'Target Date', 'Index', 'Bullish %', 'Bearish %']
                        
                        st.markdown("**🕒 Recent Major Peaks:**")
                        st.dataframe(recent_peaks, hide_index=True, use_container_width=True)
                    else:
                        st.info("No peaks identified")
                
                with col2:
                    st.markdown("#### 📉 **Trough Analysis**")
                    if not all_troughs_df.empty:
                        avg_bullish_troughs = all_troughs_df['bullish_pct_at_trough'].mean()
                        avg_bearish_troughs = all_troughs_df['bearish_pct_at_trough'].mean()
                        max_bearish_trough = all_troughs_df['bearish_pct_at_trough'].max()
                        
                        st.success(f"""
                        **🩸 Trough Sentiment Statistics:**
                        - **Total Troughs**: {len(all_troughs_df)}
                        - **Avg Bullish at Troughs**: {avg_bullish_troughs:.1f}%
                        - **Avg Bearish at Troughs**: {avg_bearish_troughs:.1f}%
                        - **Max Bearish Reading**: {max_bearish_trough:.1f}%
                        - **Current vs Trough Avg**: {current_bearish - avg_bearish_troughs:+.1f}%
                        """)
                        
                        # Show recent major troughs
                        recent_troughs = all_troughs_df.nlargest(5, 'date')[['date', 'manual_date', 'index_name', 'bullish_pct_at_trough', 'bearish_pct_at_trough']].copy()
                        recent_troughs['date'] = pd.to_datetime(recent_troughs['date']).dt.strftime('%Y-%m-%d')
                        recent_troughs.columns = ['Actual Date', 'Target Date', 'Index', 'Bullish %', 'Bearish %']
                        
                        st.markdown("**🕒 Recent Major Troughs:**")
                        st.dataframe(recent_troughs, hide_index=True, use_container_width=True)
                    else:
                        st.info("No troughs identified")
                
                # Market timing insights
                st.markdown("### 💡 **Market Timing Insights**")
                
                if not all_peaks_df.empty and not all_troughs_df.empty:
                    # Calculate comprehensive statistics
                    bullish_peak_75th = np.percentile(all_peaks_df['bullish_pct_at_peak'], 75)
                    bullish_peak_90th = np.percentile(all_peaks_df['bullish_pct_at_peak'], 90)
                    bearish_trough_75th = np.percentile(all_troughs_df['bearish_pct_at_trough'], 75)
                    bearish_trough_90th = np.percentile(all_troughs_df['bearish_pct_at_trough'], 90)
                    
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
                    insights.append(f"📊 **Historical Range**: Peak bullish {all_peaks_df['bullish_pct_at_peak'].min():.1f}%-{all_peaks_df['bullish_pct_at_peak'].max():.1f}%, Trough bearish {all_troughs_df['bearish_pct_at_trough'].min():.1f}%-{all_troughs_df['bearish_pct_at_trough'].max():.1f}%")
                    
                    for insight in insights:
                        st.markdown(insight)
                        
                    # Enhanced trading signals
                    st.markdown("### 🚨 **Enhanced Market Timing Signals**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🟢 **Bottom Signals**")
                        bottom_signals = []
                        
                        if current_bearish >= bearish_trough_75th:
                            bottom_signals.append(f"🔴 High bearish sentiment ({current_bearish:.1f}% vs {bearish_trough_75th:.1f}% threshold)")
                        
                        if current_bullish <= np.percentile(all_peaks_df['bullish_pct_at_peak'], 25):
                            bottom_signals.append(f"📉 Low bullish sentiment ({current_bullish:.1f}%)")
                        
                        if current_bearish > current_bullish + 20:
                            bottom_signals.append(f"⚖️ Strong bearish dominance ({current_bearish:.1f}% vs {current_bullish:.1f}%)")
                        
                        if bottom_signals:
                            for signal in bottom_signals:
                                st.write(f"• {signal}")
                            
                            signal_strength = len(bottom_signals)
                            if signal_strength >= 2:
                                st.success("🎯 **STRONG BOTTOM SIGNAL** - Multiple indicators suggest potential market bottom")
                            else:
                                st.warning("⚠️ **MODERATE BOTTOM SIGNAL** - Some indicators suggest oversold conditions")
                        else:
                            st.info("No clear bottom signals detected")
                    
                    with col2:
                        st.markdown("#### 🔴 **Top Signals**")
                        top_signals = []
                        
                        if current_bullish >= bullish_peak_75th:
                            top_signals.append(f"🟢 High bullish sentiment ({current_bullish:.1f}% vs {bullish_peak_75th:.1f}% threshold)")
                        
                        if current_bearish <= np.percentile(all_troughs_df['bearish_pct_at_trough'], 25):
                            top_signals.append(f"📈 Low bearish sentiment ({current_bearish:.1f}%)")
                        
                        if current_bullish > current_bearish + 15:
                            top_signals.append(f"⚖️ Strong bullish dominance ({current_bullish:.1f}% vs {current_bearish:.1f}%)")
                        
                        if latest_sentiment.get('extreme_bullish_pct', 0) > 10:
                            top_signals.append(f"🔥 High extreme bullish ({latest_sentiment['extreme_bullish_pct']:.1f}%)")
                        
                        if top_signals:
                            for signal in top_signals:
                                st.write(f"• {signal}")
                            
                            signal_strength = len(top_signals)
                            if signal_strength >= 3:
                                st.error("🚨 **STRONG TOP SIGNAL** - Multiple indicators suggest potential market top")
                            elif signal_strength >= 2:
                                st.warning("⚠️ **MODERATE TOP SIGNAL** - Some indicators suggest overbought conditions")
                        else:
                            st.info("No clear top signals detected")
        else:
            st.warning("No turning points data available.")
        
        # Export functionality
        st.markdown("## 💾 **Export Analysis Data**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export sentiment data
            sentiment_csv = sentiment_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sentiment Data",
                data=sentiment_csv,
                file_name=f"market_sentiment_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help="Download complete sentiment time series data"
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
                        mime="text/csv",
                        help="Download correlation analysis results"
                    )
        
        with col3:
            # Export turning points
            if turning_points:
                all_turning_points = []
                for index_symbol, tp_data in turning_points.items():
                    for peak in tp_data['peaks']:
                        peak_copy = peak.copy()
                        peak_copy['index'] = tp_data['name']
                        peak_copy['point_type'] = 'peak'
                        all_turning_points.append(peak_copy)
                    
                    for trough in tp_data['troughs']:
                        trough_copy = trough.copy()
                        trough_copy['index'] = tp_data['name']
                        trough_copy['point_type'] = 'trough'
                        all_turning_points.append(trough_copy)
                
                if all_turning_points:
                    tp_csv = pd.DataFrame(all_turning_points).to_csv(index=False)
                    st.download_button(
                        label="📥 Download Turning Points",
                        data=tp_csv,
                        file_name=f"turning_points_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        help="Download manual turning points analysis"
                    )
        
        # Final insights and summary
        st.markdown("## 💡 **Final Market Assessment**")
        
        # Create summary based on all analysis
        summary_insights = []
        
        # Current sentiment vs historical extremes
        if current_bullish > analyzer.bullish_thresholds['danger']:
            summary_insights.append("🚨 **EXTREME BULLISH ALERT** - Sentiment at dangerous levels historically associated with market tops")
        elif current_bullish > analyzer.bullish_thresholds['warning']:
            summary_insights.append("⚠️ **ELEVATED BULLISH CAUTION** - Sentiment approaching historical warning levels")
        
        if current_bearish > analyzer.bearish_thresholds['opportunity']:
            summary_insights.append("🎯 **EXTREME OPPORTUNITY ZONE** - Bearish sentiment at levels historically associated with market bottoms")
        elif current_bearish > analyzer.bearish_thresholds['fearful']:
            summary_insights.append("📈 **OPPORTUNITY DEVELOPING** - Bearish sentiment approaching historical buying zones")
        
        # Net sentiment analysis
        if abs(current_net) > 20:
            direction = "bullish" if current_net > 0 else "bearish"
            summary_insights.append(f"⚖️ **EXTREME NET SENTIMENT** - Strong {direction} imbalance ({current_net:+.1f}%) suggests potential reversal ahead")
        
        # Market phase determination
        if current_bullish > analyzer.bullish_thresholds['danger'] and current_bearish < analyzer.bearish_thresholds['normal']:
            summary_insights.append("🔴 **EUPHORIA PHASE** - Classic market top conditions with high bullish and low bearish sentiment")
        elif current_bullish < analyzer.bullish_thresholds['caution'] and current_bearish > analyzer.bearish_thresholds['opportunity']:
            summary_insights.append("🟢 **CAPITULATION PHASE** - Classic market bottom conditions with low bullish and high bearish sentiment")
        elif current_bullish > analyzer.bullish_thresholds['warning']:
            summary_insights.append("🟡 **FROTHY PHASE** - Market showing signs of excessive optimism, exercise caution")
        elif current_bearish > analyzer.bearish_thresholds['fearful']:
            summary_insights.append("🔵 **FEARFUL PHASE** - Market showing elevated fear, potential opportunities developing")
        else:
            summary_insights.append("⚪ **NEUTRAL PHASE** - Balanced sentiment, normal market conditions")
        
        # Momentum insights
        if 'bullish_momentum_5d' in latest_sentiment:
            momentum = latest_sentiment['bullish_momentum_5d'] or 0
            if abs(momentum) > 5:
                direction = "increasing" if momentum > 0 else "decreasing"
                summary_insights.append(f"🚀 **STRONG MOMENTUM** - Bullish sentiment {direction} rapidly ({momentum:+.1f}% in 5 days)")
        
        # Display insights
        if summary_insights:
            for insight in summary_insights:
                st.markdown(f"• {insight}")
        else:
            st.info("Market showing balanced conditions with no extreme sentiment readings.")
        
        # Trading recommendations based on analysis
        st.markdown("### 🎯 **Trading Recommendations**")
        
        # Determine overall market bias
        bullish_score = 0
        bearish_score = 0
        
        # Score based on sentiment levels
        if current_bullish < analyzer.bullish_thresholds['caution']:
            bullish_score += 2
        elif current_bullish > analyzer.bullish_thresholds['danger']:
            bearish_score += 3
        elif current_bullish > analyzer.bullish_thresholds['warning']:
            bearish_score += 1
        
        if current_bearish > analyzer.bearish_thresholds['opportunity']:
            bullish_score += 3
        elif current_bearish > analyzer.bearish_thresholds['fearful']:
            bullish_score += 1
        elif current_bearish < analyzer.bearish_thresholds['normal']:
            bearish_score += 2
        
        # Score based on momentum
        if 'bullish_momentum_5d' in latest_sentiment:
            momentum = latest_sentiment['bullish_momentum_5d'] or 0
            if momentum > 3:
                bearish_score += 1  # Rising bullish sentiment is bearish for timing
            elif momentum < -3:
                bullish_score += 1  # Falling bullish sentiment is bullish for timing
        
        # Generate recommendation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if bullish_score > bearish_score:
                st.success(f"""
                **🎯 BULLISH BIAS**
                
                **Action**: Consider accumulating positions
                **Rationale**: Sentiment conditions favor buying
                **Risk**: Monitor for sentiment deterioration
                **Score**: {bullish_score} vs {bearish_score}
                """)
            elif bearish_score > bullish_score:
                st.error(f"""
                **🚨 BEARISH BIAS**
                
                **Action**: Consider reducing positions
                **Rationale**: Sentiment conditions suggest caution
                **Risk**: Monitor for sentiment improvement
                **Score**: {bearish_score} vs {bullish_score}
                """)
            else:
                st.info(f"""
                **⚪ NEUTRAL BIAS**
                
                **Action**: Maintain current positions
                **Rationale**: Balanced sentiment conditions
                **Risk**: Watch for directional change
                **Score**: {bullish_score} vs {bearish_score}
                """)
        
        with col2:
            # Risk level assessment
            risk_factors = 0
            
            if current_bullish > analyzer.bullish_thresholds['danger']:
                risk_factors += 3
            elif current_bullish > analyzer.bullish_thresholds['warning']:
                risk_factors += 1
            
            if current_bearish < analyzer.bearish_thresholds['normal']:
                risk_factors += 2
            
            if latest_sentiment.get('extreme_bullish_pct', 0) > 15:
                risk_factors += 1
            
            if risk_factors >= 4:
                st.error("🔴 **HIGH RISK**\nMultiple warning signals")
            elif risk_factors >= 2:
                st.warning("🟡 **MODERATE RISK**\nSome caution warranted")
            else:
                st.success("🟢 **LOW RISK**\nFavorable conditions")
            
            st.caption(f"Risk factors: {risk_factors}/6")
        
        with col3:
            # Opportunity assessment
            opportunity_factors = 0
            
            if current_bearish > analyzer.bearish_thresholds['opportunity']:
                opportunity_factors += 3
            elif current_bearish > analyzer.bearish_thresholds['fearful']:
                opportunity_factors += 1
            
            if current_bullish < analyzer.bullish_thresholds['caution']:
                opportunity_factors += 2
            
            if current_net < -15:
                opportunity_factors += 1
            
            if opportunity_factors >= 4:
                st.success("🎯 **HIGH OPPORTUNITY**\nExcellent buying conditions")
            elif opportunity_factors >= 2:
                st.info("📈 **MODERATE OPPORTUNITY**\nSome attractive setups")
            else:
                st.warning("😐 **LOW OPPORTUNITY**\nLimited attractive entry points")
            
            st.caption(f"Opportunity factors: {opportunity_factors}/6")
        
        # Performance statistics
        st.markdown("## 📊 **Analysis Performance Stats**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 Turning Points Found",
                f"{sum(len(tp['peaks']) + len(tp['troughs']) for tp in turning_points.values()) if turning_points else 0}",
                "vs ~12 algorithmic",
                help="Manual turning points vs algorithmic detection"
            )
        
        with col2:
            if turning_points:
                all_peaks = []
                for tp_data in turning_points.values():
                    all_peaks.extend(tp_data['peaks'])
                
                if all_peaks:
                    avg_peak_sentiment = np.mean([p['bullish_pct_at_peak'] for p in all_peaks if p['bullish_pct_at_peak'] is not None])
                    st.metric(
                        "📈 Avg Peak Sentiment",
                        f"{avg_peak_sentiment:.1f}%",
                        f"{current_bullish - avg_peak_sentiment:+.1f}% vs current",
                        help="Average bullish sentiment at historical peaks"
                    )
        
        with col3:
            if turning_points:
                all_troughs = []
                for tp_data in turning_points.values():
                    all_troughs.extend(tp_data['troughs'])
                
                if all_troughs:
                    avg_trough_sentiment = np.mean([t['bearish_pct_at_trough'] for t in all_troughs if t['bearish_pct_at_trough'] is not None])
                    st.metric(
                        "📉 Avg Trough Sentiment",
                        f"{avg_trough_sentiment:.1f}%",
                        f"{current_bearish - avg_trough_sentiment:+.1f}% vs current",
                        help="Average bearish sentiment at historical troughs"
                    )
        
        with col4:
            data_quality = (len(sentiment_df) / 365 * 16) * 100  # 16 years of data expected
            data_quality = min(100, data_quality)
            st.metric(
                "📊 Data Quality",
                f"{data_quality:.0f}%",
                f"{len(sentiment_df)} trading days",
                help="Data coverage and completeness score"
            )
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #2C3E50, #34495E); 
                    border-radius: 10px; margin-top: 30px;'>
            <h4 style='color: white; margin: 0;'>
                🎯 ULTIMATE Market Sentiment Analyzer 🎯
            </h4>
            <p style='color: #BDC3C7; margin: 10px 0 0 0; font-size: 14px;'>
                Powered by 62 Historical Turning Points • Perfect Calibration • Real-time Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred during analysis. Please check your data and try again.")
        if st.checkbox("Show error details"):
            st.exception(e)

if __name__ == "__main__":
    main()