"""
Signal Gauge Dashboard
======================
Quick-glance dashboard showing how close top backtested signals are to firing.
Features gauges/meters for each signal and a composite "opportunity score".

Author: Dave's Trading Analysis Suite
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance - it may fail on some cloud deployments
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Page config
st.set_page_config(
    layout="wide",
    page_title="Signal Gauge Dashboard",
    page_icon="🎯"
)

# =============================================================================
# CONFIGURATION
# =============================================================================

def find_data_directory():
    """Find the data directory."""
    possible_paths = [
        Path("Data/stock_scores"),
        Path("../Data/stock_scores"),
        Path(r"C:\Users\davet\Documents\GitHub\Industry-analysis\Data\stock_scores"),
        Path(r"C:\Users\davet\Documents\new_dev\Industry-analysis\score_analysis\data"),
    ]
    
    for path in possible_paths:
        try:
            if path.exists() and (path / "historical_data.parquet.gzip").exists():
                return path
        except:
            continue
    
    return Path("Data/stock_scores")

DATA_DIR = find_data_directory()

# Signal thresholds based on backtesting results
# Organized by tier based on win rate and sample size
SIGNALS = {
    # ===========================================
    # 🏆 TIER 1 - TOP SIGNALS (80%+ Win Rate)
    # ===========================================
    'net_score_turns_positive': {
        'name': 'Net Score Turns Positive',
        'description': 'Net score crosses from negative to positive',
        'type': 'bullish',
        'tier': 1,
        'win_rate': 91.7,
        'avg_return': 1.61,
        'threshold': 0,
        'direction': 'crosses_above',
        'metric': 'market_net'
    },
    'avg_bearish_extreme': {
        'name': 'Avg Bearish > 55',
        'description': 'Average bearish score exceeds 55 (extreme fear)',
        'type': 'bullish',
        'tier': 1,
        'win_rate': 90.0,
        'avg_return': 4.37,
        'threshold': 55,
        'direction': 'above',
        'metric': 'market_bear'
    },
    'bull_roc_washout': {
        'name': 'Bull ROC 5d < -10',
        'description': 'Bullish score drops 10+ points in 5 days (washout)',
        'type': 'bullish',
        'tier': 1,
        'win_rate': 85.2,
        'avg_return': 2.92,
        'threshold': -10,
        'direction': 'below',
        'metric': 'bull_roc_5d'
    },
    'real_estate_extreme_bearish': {
        'name': 'Real Estate Extremely Bearish',
        'description': 'Real Estate sector extremely bearish (leading indicator)',
        'type': 'bullish',
        'tier': 1,
        'win_rate': 83.3,
        'avg_return': 2.54,
        'threshold': 30,
        'direction': 'below',
        'metric': 'real_estate_breadth'
    },
    'utilities_extreme_bearish': {
        'name': 'Utilities Extremely Bearish',
        'description': 'Utilities sector extremely bearish (leading indicator)',
        'type': 'bullish',
        'tier': 1,
        'win_rate': 83.3,
        'avg_return': 1.45,
        'threshold': 30,
        'direction': 'below',
        'metric': 'utilities_breadth'
    },
    'sharp_rotation_to_value': {
        'name': 'Sharp Rotation to Value',
        'description': 'Quick shift from growth to value sectors',
        'type': 'bullish',
        'tier': 1,
        'win_rate': 80.4,
        'avg_return': 1.39,
        'threshold': -5,
        'direction': 'below',
        'metric': 'growth_value_diff'
    },
    'breadth_crosses_above_50': {
        'name': 'Breadth Crosses Above 50%',
        'description': 'Market breadth crosses above 50% (momentum shift)',
        'type': 'bullish',
        'tier': 1,
        'win_rate': 80.0,
        'avg_return': 1.47,
        'threshold': 50,
        'direction': 'crosses_above',
        'metric': 'market_breadth'
    },
    'bull_bear_convergence': {
        'name': 'Bull/Bear Convergence',
        'description': 'Bull and Bear scores converge (spread < 5)',
        'type': 'bullish',
        'tier': 1,
        'win_rate': 79.7,
        'avg_return': 1.50,
        'threshold': 5,
        'direction': 'below',
        'metric': 'bull_bear_spread'
    },
    
    # ===========================================
    # 🥈 TIER 2 - STRONG SIGNALS (65-80% Win Rate)
    # ===========================================
    'breadth_very_low': {
        'name': 'Very Low Breadth',
        'description': 'Market breadth < 20%',
        'type': 'bullish',
        'tier': 2,
        'win_rate': 67.8,
        'avg_return': 1.49,
        'threshold': 20,
        'direction': 'below',
        'metric': 'market_breadth'
    },
    'zscore_extreme_oversold': {
        'name': 'Extremely Oversold (Z-Score)',
        'description': 'Z-Score < -2 (extreme oversold)',
        'type': 'bullish',
        'tier': 2,
        'win_rate': 64.5,
        'avg_return': 1.61,
        'threshold': -2,
        'direction': 'below',
        'metric': 'market_zscore'
    },
    'loss_streak_long': {
        'name': 'Long Loss Streak',
        'description': '5+ consecutive down days',
        'type': 'bullish',
        'tier': 2,
        'win_rate': 62.5,
        'avg_return': 0.86,
        'threshold': -5,
        'direction': 'below',
        'metric': 'market_streak'
    },
    'zscore_oversold': {
        'name': 'Oversold (Z-Score)',
        'description': 'Z-Score < -1 (oversold)',
        'type': 'bullish',
        'tier': 2,
        'win_rate': 62.6,
        'avg_return': 0.99,
        'threshold': -1,
        'direction': 'below',
        'metric': 'market_zscore'
    },
    'breadth_low': {
        'name': 'Low Breadth',
        'description': 'Market breadth 20-40%',
        'type': 'bullish',
        'tier': 2,
        'win_rate': 61.8,
        'avg_return': 0.74,
        'threshold': 40,
        'direction': 'below',
        'metric': 'market_breadth'
    },
    'momentum_strong_negative': {
        'name': 'Strong Negative Momentum',
        'description': '5d Net ROC < -10 (washout)',
        'type': 'bullish',
        'tier': 2,
        'win_rate': 61.2,
        'avg_return': 0.77,
        'threshold': -10,
        'direction': 'below',
        'metric': 'net_roc_5d'
    },
    
    # ===========================================
    # 🔴 BEARISH/CAUTION SIGNALS
    # ===========================================
    'breadth_very_high': {
        'name': 'Very High Breadth',
        'description': 'Market breadth > 80% (overbought)',
        'type': 'bearish',
        'tier': 1,
        'win_rate': 52.9,
        'avg_return': 0.13,
        'threshold': 80,
        'direction': 'above',
        'metric': 'market_breadth'
    },
    'breadth_crosses_above_70': {
        'name': 'Breadth > 70%',
        'description': 'Breadth exceeds 70% (extreme optimism)',
        'type': 'bearish',
        'tier': 1,
        'win_rate': 28.6,
        'avg_return': -1.23,
        'threshold': 70,
        'direction': 'above',
        'metric': 'market_breadth'
    },
    'net_score_above_20': {
        'name': 'Net Score > 20',
        'description': 'Net score exceeds 20 (overextended)',
        'type': 'bearish',
        'tier': 1,
        'win_rate': 40.8,
        'avg_return': -0.48,
        'threshold': 20,
        'direction': 'above',
        'metric': 'market_net'
    },
    'zscore_overbought': {
        'name': 'Overbought (Z-Score)',
        'description': 'Z-Score > 1 (overbought)',
        'type': 'bearish',
        'tier': 2,
        'win_rate': 55.4,
        'avg_return': 0.28,
        'threshold': 1,
        'direction': 'above',
        'metric': 'market_zscore'
    },
    'win_streak_long': {
        'name': 'Long Win Streak',
        'description': '5+ consecutive up days',
        'type': 'bearish',
        'tier': 2,
        'win_rate': 56.3,
        'avg_return': 0.01,
        'threshold': 5,
        'direction': 'above',
        'metric': 'market_streak'
    },
    'financial_services_extreme_bullish': {
        'name': 'Financials Extremely Bullish',
        'description': 'Financial Services extremely bullish (fade signal)',
        'type': 'bearish',
        'tier': 2,
        'win_rate': 39.6,
        'avg_return': -0.35,
        'threshold': 70,
        'direction': 'above',
        'metric': 'financials_breadth'
    },
}


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=300)
def load_data():
    """Load and process all required data."""
    
    parquet_path = DATA_DIR / 'historical_data.parquet.gzip'
    
    if not parquet_path.exists():
        st.error(f"Data file not found: {parquet_path}")
        st.stop()
    
    df = pd.read_parquet(parquet_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df['net_score'] = df['bullish_score'] - df['bearish_score']
    
    return df


@st.cache_data(ttl=300)
def calculate_market_metrics(df):
    """Calculate all market-level metrics needed for signals."""
    
    # Daily market aggregates
    market_agg = df.groupby('date').agg({
        'bullish_score': 'mean',
        'bearish_score': 'mean',
        'net_score': 'mean',
        'symbol': 'count'
    }).reset_index()
    
    market_agg.columns = ['date', 'market_bull', 'market_bear', 'market_net', 'stock_count']
    
    # Breadth
    breadth = df.groupby('date').apply(
        lambda x: (x['bullish_score'] > x['bearish_score']).mean() * 100
    ).reset_index()
    breadth.columns = ['date', 'market_breadth']
    market_agg = market_agg.merge(breadth, on='date')
    
    # Sort by date
    market_agg = market_agg.sort_values('date').reset_index(drop=True)
    
    # Calculate momentum (5-day ROC)
    market_agg['net_roc_5d'] = market_agg['market_net'].diff(5)
    market_agg['breadth_roc_5d'] = market_agg['market_breadth'].diff(5)
    
    # Bull ROC 5d (for washout signal)
    market_agg['bull_roc_5d'] = market_agg['market_bull'].diff(5)
    
    # Bull/Bear Spread (for convergence signal)
    market_agg['bull_bear_spread'] = abs(market_agg['market_bull'] - market_agg['market_bear'])
    
    # Calculate Z-Score (20-day)
    market_agg['net_rolling_mean'] = market_agg['market_net'].rolling(20).mean()
    market_agg['net_rolling_std'] = market_agg['market_net'].rolling(20).std()
    market_agg['market_zscore'] = (market_agg['market_net'] - market_agg['net_rolling_mean']) / market_agg['net_rolling_std']
    
    # Calculate streak
    market_agg['daily_change'] = market_agg['market_net'].diff()
    market_agg['is_up'] = market_agg['daily_change'] > 0
    
    streak = 0
    streaks = []
    prev_up = None
    
    for is_up in market_agg['is_up']:
        if pd.isna(is_up):
            streaks.append(0)
            continue
        
        if prev_up is None:
            streak = 1 if is_up else -1
        elif is_up == prev_up:
            streak = streak + 1 if is_up else streak - 1
        else:
            streak = 1 if is_up else -1
        
        prev_up = is_up
        streaks.append(streak)
    
    market_agg['market_streak'] = streaks
    
    # Previous day values for crossover detection
    market_agg['prev_net'] = market_agg['market_net'].shift(1)
    market_agg['prev_breadth'] = market_agg['market_breadth'].shift(1)
    
    # ===========================================
    # SECTOR-LEVEL METRICS (for sector signals)
    # ===========================================
    
    # Define sector groupings
    growth_sectors = ['Technology', 'Communication Services', 'Consumer Cyclical']
    value_sectors = ['Utilities', 'Consumer Defensive', 'Energy', 'Financial Services']
    
    # Calculate sector breadths
    sector_breadth = df.groupby(['date', 'sector']).apply(
        lambda x: (x['bullish_score'] > x['bearish_score']).mean() * 100
    ).reset_index()
    sector_breadth.columns = ['date', 'sector', 'sector_breadth']
    
    # Pivot to get each sector as a column
    sector_pivot = sector_breadth.pivot(index='date', columns='sector', values='sector_breadth').reset_index()
    
    # Rename columns for specific sectors we track
    sector_mapping = {
        'Real Estate': 'real_estate_breadth',
        'Utilities': 'utilities_breadth',
        'Financial Services': 'financials_breadth',
        'Technology': 'technology_breadth',
        'Energy': 'energy_breadth',
        'Basic Materials': 'basic_materials_breadth'
    }
    
    for sector, col_name in sector_mapping.items():
        if sector in sector_pivot.columns:
            market_agg = market_agg.merge(
                sector_pivot[['date', sector]].rename(columns={sector: col_name}),
                on='date',
                how='left'
            )
        else:
            market_agg[col_name] = 50  # Default if sector not found
    
    # Calculate Growth vs Value differential
    growth_cols = [s for s in growth_sectors if s in sector_pivot.columns]
    value_cols = [s for s in value_sectors if s in sector_pivot.columns]
    
    if growth_cols and value_cols:
        sector_pivot['growth_avg'] = sector_pivot[growth_cols].mean(axis=1)
        sector_pivot['value_avg'] = sector_pivot[value_cols].mean(axis=1)
        sector_pivot['growth_value_diff'] = sector_pivot['growth_avg'] - sector_pivot['value_avg']
        
        market_agg = market_agg.merge(
            sector_pivot[['date', 'growth_value_diff']],
            on='date',
            how='left'
        )
        
        # Calculate 5-day change in growth/value diff for "sharp rotation" signal
        market_agg['growth_value_diff_roc'] = market_agg['growth_value_diff'].diff(5)
    else:
        market_agg['growth_value_diff'] = 0
        market_agg['growth_value_diff_roc'] = 0
    
    return market_agg


# =============================================================================
# GAUGE CREATION
# =============================================================================

def create_gauge(value, min_val, max_val, threshold, title, signal_type, direction, is_firing):
    """Create a gauge chart for a signal."""
    
    # Determine colors based on signal type and status
    if signal_type == 'bullish':
        bar_color = '#00ff00' if is_firing else '#00aa00'
        threshold_color = '#00ff00'
    else:
        bar_color = '#ff4444' if is_firing else '#aa0000'
        threshold_color = '#ffffff'  # White threshold for better visibility on bearish gauges
    
    # Create gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'font': {'size': 24, 'color': 'white'}},
        gauge={
            'axis': {
                'range': [min_val, max_val],
                'tickcolor': 'white',
                'tickfont': {'color': 'white', 'size': 10}
            },
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': '#1a1a2e',
            'borderwidth': 2,
            'bordercolor': '#333',
            'threshold': {
                'line': {'color': threshold_color, 'width': 4},
                'thickness': 0.8,
                'value': threshold
            },
            'steps': [
                {'range': [min_val, max_val], 'color': '#16213e'}
            ]
        },
        title={'text': title, 'font': {'size': 14, 'color': 'white'}}
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    
    return fig


def create_signal_gauge(result, signal):
    """Create a gauge for any signal based on its metric type."""
    
    metric = signal['metric']
    current_value = result['current_value']
    threshold = signal['threshold']
    signal_type = signal['type']
    is_firing = result['is_firing']
    direction = signal['direction']
    
    # Handle NaN values
    if pd.isna(current_value):
        current_value = 0
    
    # Define gauge parameters based on metric
    gauge_configs = {
        'market_breadth': (0, 100, f"Breadth: {current_value:.1f}%"),
        'market_zscore': (-3, 3, f"Z-Score: {current_value:.2f}"),
        'net_roc_5d': (-25, 25, f"Net ROC 5d: {current_value:.1f}"),
        'bull_roc_5d': (-25, 25, f"Bull ROC 5d: {current_value:.1f}"),
        'market_net': (-30, 30, f"Net Score: {current_value:.1f}"),
        'market_streak': (-10, 10, f"Streak: {current_value:.0f}"),
        'market_bear': (20, 70, f"Avg Bearish: {current_value:.1f}"),
        'bull_bear_spread': (0, 30, f"Spread: {current_value:.1f}"),
        'growth_value_diff': (-30, 30, f"G/V Diff: {current_value:.1f}"),
        'real_estate_breadth': (0, 100, f"RE Breadth: {current_value:.1f}%"),
        'utilities_breadth': (0, 100, f"Util Breadth: {current_value:.1f}%"),
        'financials_breadth': (0, 100, f"Fin Breadth: {current_value:.1f}%"),
        'technology_breadth': (0, 100, f"Tech Breadth: {current_value:.1f}%"),
        'energy_breadth': (0, 100, f"Energy Breadth: {current_value:.1f}%"),
        'basic_materials_breadth': (0, 100, f"Materials Breadth: {current_value:.1f}%"),
    }
    
    if metric in gauge_configs:
        min_val, max_val, title = gauge_configs[metric]
        return create_gauge(current_value, min_val, max_val, threshold, title, signal_type, direction, is_firing)
    
    return None


def create_proximity_bar(proximity, signal_name, signal_type, is_firing):
    """Create a horizontal proximity bar showing how close to firing."""
    
    if signal_type == 'bullish':
        color = '#00ff00' if is_firing else f'rgba(0, {int(155 + proximity)}, 0, 0.8)'
    else:
        color = '#ff4444' if is_firing else f'rgba({int(155 + proximity)}, 0, 0, 0.8)'
    
    fig = go.Figure(go.Bar(
        x=[proximity],
        y=[signal_name],
        orientation='h',
        marker_color=color,
        text=f'{proximity:.0f}%',
        textposition='inside',
        textfont={'color': 'white', 'size': 14}
    ))
    
    fig.update_layout(
        height=50,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis={'range': [0, 100], 'showticklabels': False, 'showgrid': False},
        yaxis={'showticklabels': False, 'showgrid': False},
        showlegend=False
    )
    
    return fig


def create_composite_gauge(score, num_bullish, num_bearish):
    """
    Create the main composite opportunity gauge.
    
    Score interpretation:
    - HIGH (70-100): Bullish signals firing = market oversold = BUY OPPORTUNITY
    - NEUTRAL (30-70): No strong signals either way = WAIT
    - LOW (0-30): Bearish/caution signals firing = market overbought = RISK
    """
    
    # Color and label based on score
    if score >= 70:
        color = '#00ff00'
        label = 'OVERSOLD - BUY'
    elif score >= 55:
        color = '#88ff00'
        label = 'LEANING BULLISH'
    elif score >= 45:
        color = '#ffaa00'
        label = 'NEUTRAL - WAIT'
    elif score >= 30:
        color = '#ff6600'
        label = 'LEANING BEARISH'
    else:
        color = '#ff0000'
        label = 'OVERBOUGHT - RISK'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': '%', 'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickvals': [0, 25, 50, 75, 100],
                'ticktext': ['SELL', 'CAUTION', 'NEUTRAL', 'BUY', 'STRONG'],
                'tickcolor': 'white',
                'tickfont': {'color': 'white', 'size': 12}
            },
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': '#1a1a2e',
            'borderwidth': 3,
            'bordercolor': color,
            'steps': [
                {'range': [0, 15], 'color': 'rgba(255,0,0,0.2)'},
                {'range': [15, 30], 'color': 'rgba(255,102,0,0.2)'},
                {'range': [30, 50], 'color': 'rgba(255,170,0,0.2)'},
                {'range': [50, 70], 'color': 'rgba(136,255,0,0.2)'},
                {'range': [70, 100], 'color': 'rgba(0,255,0,0.2)'},
            ]
        },
        title={'text': f'<b>OPPORTUNITY SCORE</b><br><span style="font-size:20px;color:{color}">{label}</span>', 
               'font': {'size': 20, 'color': 'white'}}
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=80, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    
    return fig


# =============================================================================
# SIGNAL ANALYSIS
# =============================================================================

def calculate_signal_proximity(current_value, threshold, direction, metric_range):
    """Calculate how close a signal is to firing (0-100%)."""
    
    if direction == 'below':
        # Signal fires when value drops BELOW threshold
        if current_value <= threshold:
            return 100  # Already firing
        else:
            # How far from threshold as % of typical range
            distance = current_value - threshold
            max_distance = metric_range[1] - threshold
            proximity = max(0, 100 - (distance / max_distance * 100))
            return proximity
            
    elif direction == 'above':
        # Signal fires when value rises ABOVE threshold
        if current_value >= threshold:
            return 100  # Already firing
        else:
            distance = threshold - current_value
            max_distance = threshold - metric_range[0]
            proximity = max(0, 100 - (distance / max_distance * 100))
            return proximity
            
    elif direction == 'crosses_above':
        # For crossover signals, check if close to crossing
        if current_value > threshold:
            return 100
        else:
            distance = threshold - current_value
            proximity = max(0, 100 - (abs(distance) * 10))  # Scale factor
            return proximity
    
    return 0


def analyze_signals(current_data, prev_data):
    """Analyze all signals and calculate their status."""
    
    results = {}
    
    # Define typical ranges for each metric
    metric_ranges = {
        'market_breadth': (0, 100),
        'market_zscore': (-3, 3),
        'net_roc_5d': (-20, 20),
        'bull_roc_5d': (-20, 20),
        'market_net': (-30, 30),
        'market_streak': (-10, 10),
        'market_bear': (20, 70),
        'bull_bear_spread': (0, 30),
        'growth_value_diff': (-30, 30),
        'real_estate_breadth': (0, 100),
        'utilities_breadth': (0, 100),
        'financials_breadth': (0, 100),
        'technology_breadth': (0, 100),
        'energy_breadth': (0, 100),
        'basic_materials_breadth': (0, 100),
    }
    
    for signal_id, signal in SIGNALS.items():
        metric = signal['metric']
        threshold = signal['threshold']
        direction = signal['direction']
        
        current_value = current_data.get(metric, 0)
        if pd.isna(current_value):
            current_value = 0
        prev_value = prev_data.get(metric, 0) if prev_data else current_value
        if pd.isna(prev_value):
            prev_value = current_value
        
        # For breadth crossover, use prev_breadth
        if metric == 'market_breadth' and direction == 'crosses_above':
            prev_value = current_data.get('prev_breadth', prev_value)
            if pd.isna(prev_value):
                prev_value = current_value
        
        # Check if signal is firing
        if direction == 'below':
            is_firing = current_value < threshold
        elif direction == 'above':
            is_firing = current_value > threshold
        elif direction == 'crosses_above':
            is_firing = (prev_value < threshold) and (current_value >= threshold)
        else:
            is_firing = False
        
        # Calculate proximity
        proximity = calculate_signal_proximity(
            current_value, 
            threshold, 
            direction,
            metric_ranges.get(metric, (0, 100))
        )
        
        results[signal_id] = {
            'signal': signal,
            'current_value': current_value,
            'threshold': threshold,
            'is_firing': is_firing,
            'proximity': proximity
        }
    
    return results


def calculate_composite_score(signal_results):
    """
    Calculate composite opportunity score based on all signals.
    
    IMPORTANT: Bullish signals are CONTRARIAN - they fire when market is OVERSOLD.
    So a HIGH opportunity score means:
    - Bullish signals ARE firing or close to firing (market oversold = BUY opportunity)
    - Bearish signals are NOT firing (market not overbought)
    
    A LOW opportunity score means:
    - Bullish signals are far from firing (market already recovered, opportunity passed)
    - Bearish signals ARE firing (market overbought = CAUTION)
    """
    
    bullish_opportunity = 0
    bearish_risk = 0
    bullish_firing_count = 0
    bearish_firing_count = 0
    
    # Track tier 1 signals separately (they're more important)
    tier1_bullish_firing = 0
    tier1_bearish_firing = 0
    
    for signal_id, result in signal_results.items():
        signal = result['signal']
        proximity = result['proximity']
        is_firing = result['is_firing']
        win_rate = signal['win_rate']
        tier = signal.get('tier', 2)
        
        # Weight by win rate (higher win rate = more important signal)
        weight = (win_rate - 50) / 50  # Normalize: 50% WR = 0, 100% WR = 1
        
        # Tier 1 signals get 2x weight
        tier_multiplier = 2.0 if tier == 1 else 1.0
        
        if signal['type'] == 'bullish':
            # Bullish signals FIRING = OPPORTUNITY (market is oversold)
            # High proximity to bullish threshold = getting close to buy signal
            bullish_opportunity += proximity * weight * tier_multiplier
            if is_firing:
                bullish_firing_count += 1
                if tier == 1:
                    tier1_bullish_firing += 1
        else:
            # Bearish signals FIRING = RISK (market is overbought)
            # High proximity to bearish threshold = getting close to caution
            bearish_risk += proximity * weight * tier_multiplier
            if is_firing:
                bearish_firing_count += 1
                if tier == 1:
                    tier1_bearish_firing += 1
    
    # Calculate max possible scores for normalization
    bullish_signals = [s for s in SIGNALS.values() if s['type'] == 'bullish']
    bearish_signals = [s for s in SIGNALS.values() if s['type'] == 'bearish']
    
    max_bullish = sum(
        100 * ((s['win_rate'] - 50) / 50) * (2.0 if s.get('tier', 2) == 1 else 1.0)
        for s in bullish_signals
    )
    max_bearish = sum(
        100 * ((s['win_rate'] - 50) / 50) * (2.0 if s.get('tier', 2) == 1 else 1.0)
        for s in bearish_signals
        if s['win_rate'] > 50  # Only count bearish signals with >50% as warnings
    )
    
    # Normalize to 0-100 scale
    # High bullish opportunity (signals firing) = HIGH score
    # High bearish risk (caution signals firing) = LOW score
    
    if max_bullish > 0:
        bullish_component = (bullish_opportunity / max_bullish) * 50
    else:
        bullish_component = 0
    
    # For bearish, we need to consider that LOW win rate bearish signals
    # (like Breadth >70% with 28.6% WR) are actually STRONG sell signals
    # So we calculate bearish risk differently
    
    bearish_risk_score = 0
    for signal_id, result in signal_results.items():
        signal = result['signal']
        if signal['type'] == 'bearish':
            proximity = result['proximity']
            win_rate = signal['win_rate']
            tier = signal.get('tier', 2)
            tier_mult = 2.0 if tier == 1 else 1.0
            
            # Lower win rate on bearish = STRONGER sell signal
            # e.g., 28.6% WR means 71.4% chance of DOWN move
            if win_rate < 50:
                # This is a real sell signal - weight by how bearish it is
                sell_strength = (50 - win_rate) / 50  # 28.6% WR -> 0.43 strength
                bearish_risk_score += proximity * sell_strength * tier_mult
            else:
                # Weak caution signal
                bearish_risk_score += proximity * 0.2 * tier_mult
    
    # Max bearish risk
    max_bearish_risk = sum(
        100 * ((50 - s['win_rate']) / 50 if s['win_rate'] < 50 else 0.2) * 
        (2.0 if s.get('tier', 2) == 1 else 1.0)
        for s in bearish_signals
    )
    
    if max_bearish_risk > 0:
        bearish_component = (bearish_risk_score / max_bearish_risk) * 50
    else:
        bearish_component = 0
    
    # Final score: Start at 50 (neutral)
    # + bullish component (signals close to firing = opportunity)
    # - bearish component (caution signals firing = risk)
    composite = 50 + bullish_component - bearish_component
    
    # Bonus/penalty for multiple tier 1 signals firing
    if tier1_bullish_firing >= 2:
        composite += 10  # Multiple strong buy signals
    if tier1_bearish_firing >= 2:
        composite -= 15  # Multiple strong caution signals
    
    composite = max(0, min(100, composite))
    
    return composite, bullish_firing_count, bearish_firing_count


# =============================================================================
# CONFLUENCE ALERTS - Based on backtested signal combinations
# =============================================================================

# Define high-probability signal combinations from backtesting
CONFLUENCE_RULES = {
    'bullish': {
        # 100% Win Rate Combos
        'bull_roc_convergence': {
            'name': '🔥 Bull Washout + Convergence',
            'signals': ['bull_roc_washout', 'bull_bear_convergence'],
            'win_rate': 100.0,
            'avg_return': 3.04,
            'occurrences': 7,
            'tier': 'S'
        },
        'net_positive_re_bearish': {
            'name': '🔥 Net Positive + RE Bearish',
            'signals': ['net_score_turns_positive', 'real_estate_extreme_bearish'],
            'win_rate': 100.0,
            'avg_return': 2.68,
            'occurrences': 6,
            'tier': 'S'
        },
        'net_positive_util_bearish': {
            'name': '🔥 Net Positive + Utilities Bearish',
            'signals': ['net_score_turns_positive', 'utilities_extreme_bearish'],
            'win_rate': 100.0,
            'avg_return': 1.36,
            'occurrences': 5,
            'tier': 'S'
        },
        # 90%+ Win Rate Combos
        'net_positive_rotation': {
            'name': '💪 Net Positive + Sharp Rotation',
            'signals': ['net_score_turns_positive', 'sharp_rotation_to_value'],
            'win_rate': 93.3,
            'avg_return': 1.97,
            'occurrences': 15,
            'tier': 'A'
        },
        'net_positive_breadth_cross': {
            'name': '💪 Net Positive + Breadth Cross 50%',
            'signals': ['net_score_turns_positive', 'breadth_crosses_above_50'],
            'win_rate': 91.7,
            'avg_return': 1.50,
            'occurrences': 12,
            'tier': 'A'
        },
        'extreme_fear_low_breadth': {
            'name': '💪 Extreme Fear + Very Low Breadth',
            'signals': ['avg_bearish_extreme', 'breadth_very_low'],
            'win_rate': 90.0,
            'avg_return': 4.51,
            'occurrences': 10,
            'tier': 'A'
        },
        're_bearish_low_breadth': {
            'name': '💪 RE Bearish + Very Low Breadth',
            'signals': ['real_estate_extreme_bearish', 'breadth_very_low'],
            'win_rate': 90.0,
            'avg_return': 4.51,
            'occurrences': 10,
            'tier': 'A'
        },
        # Triple Combos
        'triple_washout': {
            'name': '🎯 Triple: Washout + Convergence + Oversold',
            'signals': ['bull_roc_washout', 'bull_bear_convergence', 'zscore_oversold'],
            'win_rate': 100.0,
            'avg_return': 3.11,
            'occurrences': 4,
            'tier': 'S'
        },
        'triple_washout_momentum': {
            'name': '🎯 Triple: Washout + Convergence + Neg Momentum',
            'signals': ['bull_roc_washout', 'bull_bear_convergence', 'momentum_strong_negative'],
            'win_rate': 100.0,
            'avg_return': 3.04,
            'occurrences': 7,
            'tier': 'S'
        },
    },
    'bearish': {
        # Best Short Combos
        'breadth_high_overbought': {
            'name': '⚠️ Breadth >70% + Overbought Z',
            'signals': ['breadth_crosses_above_70', 'zscore_overbought'],
            'win_rate': 66.7,
            'avg_return': 1.49,
            'occurrences': 6,
            'tier': 'A'
        },
        'net_high_overbought': {
            'name': '⚠️ Net >20 + Overbought Z',
            'signals': ['net_score_above_20', 'zscore_overbought'],
            'win_rate': 61.1,
            'avg_return': 0.52,
            'occurrences': 36,
            'tier': 'B'
        },
        'net_high_financials': {
            'name': '⚠️ Net >20 + Financials Bullish',
            'signals': ['net_score_above_20', 'financial_services_extreme_bullish'],
            'win_rate': 60.4,
            'avg_return': 0.56,
            'occurrences': 48,
            'tier': 'B'
        },
        'triple_overbought': {
            'name': '🚨 Triple: Breadth >70 + Net >20 + Overbought Z',
            'signals': ['breadth_crosses_above_70', 'net_score_above_20', 'zscore_overbought'],
            'win_rate': 66.7,
            'avg_return': 1.49,
            'occurrences': 6,
            'tier': 'A'
        },
    },
    'confluence_levels': {
        'tier1_bullish_3plus': {
            'name': '🏆 3+ Tier 1 Bullish Signals',
            'description': 'Three or more Tier 1 bullish signals firing',
            'win_rate': 85.1,
            'avg_return': 2.25,
            'occurrences': 67,
            'tier': 'S'
        },
        'tier1_bullish_2plus': {
            'name': '🥇 2+ Tier 1 Bullish Signals',
            'description': 'Two or more Tier 1 bullish signals firing',
            'win_rate': 77.5,
            'avg_return': 1.84,
            'occurrences': 151,
            'tier': 'A'
        },
        'bearish_4plus': {
            'name': '🚨 4+ Bearish Signals',
            'description': 'Four or more bearish signals firing - SHORT SIGNAL',
            'win_rate': 75.0,
            'avg_return': 1.20,
            'occurrences': 8,
            'tier': 'S'
        },
        'bearish_3plus': {
            'name': '⚠️ 3+ Bearish Signals',
            'description': 'Three or more bearish signals firing',
            'win_rate': 59.5,
            'avg_return': 0.50,
            'occurrences': 37,
            'tier': 'B'
        },
    }
}


def check_confluence(signal_results):
    """Check which confluence patterns are currently firing."""
    
    firing_signals = set()
    for signal_id, result in signal_results.items():
        if result['is_firing']:
            firing_signals.add(signal_id)
    
    alerts = {
        'S_tier': [],  # 100% or highest conviction
        'A_tier': [],  # 90%+ win rate
        'B_tier': [],  # 60-90% win rate
    }
    
    # Check specific bullish combos
    for combo_id, combo in CONFLUENCE_RULES['bullish'].items():
        required_signals = set(combo['signals'])
        if required_signals.issubset(firing_signals):
            alert = {
                'name': combo['name'],
                'type': 'bullish',
                'win_rate': combo['win_rate'],
                'avg_return': combo['avg_return'],
                'occurrences': combo['occurrences'],
                'signals': [SIGNALS[s]['name'] for s in combo['signals'] if s in SIGNALS]
            }
            tier = combo['tier']
            alerts[f'{tier}_tier'].append(alert)
    
    # Check specific bearish combos
    for combo_id, combo in CONFLUENCE_RULES['bearish'].items():
        required_signals = set(combo['signals'])
        if required_signals.issubset(firing_signals):
            alert = {
                'name': combo['name'],
                'type': 'bearish',
                'win_rate': combo['win_rate'],
                'avg_return': combo['avg_return'],
                'occurrences': combo['occurrences'],
                'signals': [SIGNALS[s]['name'] for s in combo['signals'] if s in SIGNALS]
            }
            tier = combo['tier']
            alerts[f'{tier}_tier'].append(alert)
    
    # Check confluence levels (count-based)
    bullish_t1_firing = sum(1 for s, r in signal_results.items() 
                           if r['is_firing'] and r['signal']['type'] == 'bullish' 
                           and r['signal'].get('tier', 2) == 1)
    
    bearish_firing = sum(1 for s, r in signal_results.items() 
                        if r['is_firing'] and r['signal']['type'] == 'bearish')
    
    # Tier 1 bullish confluence
    if bullish_t1_firing >= 3:
        rule = CONFLUENCE_RULES['confluence_levels']['tier1_bullish_3plus']
        alerts['S_tier'].append({
            'name': rule['name'],
            'type': 'bullish',
            'win_rate': rule['win_rate'],
            'avg_return': rule['avg_return'],
            'occurrences': rule['occurrences'],
            'signals': [f"{bullish_t1_firing} Tier 1 signals firing"]
        })
    elif bullish_t1_firing >= 2:
        rule = CONFLUENCE_RULES['confluence_levels']['tier1_bullish_2plus']
        alerts['A_tier'].append({
            'name': rule['name'],
            'type': 'bullish',
            'win_rate': rule['win_rate'],
            'avg_return': rule['avg_return'],
            'occurrences': rule['occurrences'],
            'signals': [f"{bullish_t1_firing} Tier 1 signals firing"]
        })
    
    # Bearish confluence
    if bearish_firing >= 4:
        rule = CONFLUENCE_RULES['confluence_levels']['bearish_4plus']
        alerts['S_tier'].append({
            'name': rule['name'],
            'type': 'bearish',
            'win_rate': rule['win_rate'],
            'avg_return': rule['avg_return'],
            'occurrences': rule['occurrences'],
            'signals': [f"{bearish_firing} bearish signals firing"]
        })
    elif bearish_firing >= 3:
        rule = CONFLUENCE_RULES['confluence_levels']['bearish_3plus']
        alerts['B_tier'].append({
            'name': rule['name'],
            'type': 'bearish',
            'win_rate': rule['win_rate'],
            'avg_return': rule['avg_return'],
            'occurrences': rule['occurrences'],
            'signals': [f"{bearish_firing} bearish signals firing"]
        })
    
    return alerts, bullish_t1_firing, bearish_firing


def display_confluence_alerts(signal_results):
    """Display confluence alerts in the dashboard."""
    
    alerts, t1_bullish_count, bearish_count = check_confluence(signal_results)
    
    # Count total alerts
    total_alerts = len(alerts['S_tier']) + len(alerts['A_tier']) + len(alerts['B_tier'])
    
    if total_alerts == 0:
        st.info("📊 **No High-Conviction Confluence Patterns Detected**  \n"
                "Waiting for multiple signals to align...")
        return
    
    st.subheader("🚨 CONFLUENCE ALERTS")
    
    # S-Tier Alerts (Highest Conviction)
    if alerts['S_tier']:
        for alert in alerts['S_tier']:
            if alert['type'] == 'bullish':
                st.success(
                    f"### {alert['name']}  \n"
                    f"**Win Rate: {alert['win_rate']}%** | "
                    f"**Avg Return: +{alert['avg_return']:.2f}%** | "
                    f"Historical: {alert['occurrences']}x  \n"
                    f"Signals: {', '.join(alert['signals'])}"
                )
            else:
                st.error(
                    f"### {alert['name']}  \n"
                    f"**Short Win Rate: {alert['win_rate']}%** | "
                    f"**Avg Short Return: +{alert['avg_return']:.2f}%** | "
                    f"Historical: {alert['occurrences']}x  \n"
                    f"Signals: {', '.join(alert['signals'])}"
                )
    
    # A-Tier Alerts
    if alerts['A_tier']:
        for alert in alerts['A_tier']:
            if alert['type'] == 'bullish':
                st.success(
                    f"### {alert['name']}  \n"
                    f"**Win Rate: {alert['win_rate']}%** | "
                    f"**Avg Return: +{alert['avg_return']:.2f}%** | "
                    f"Historical: {alert['occurrences']}x  \n"
                    f"Signals: {', '.join(alert['signals'])}"
                )
            else:
                st.warning(
                    f"### {alert['name']}  \n"
                    f"**Short Win Rate: {alert['win_rate']}%** | "
                    f"**Avg Short Return: +{alert['avg_return']:.2f}%** | "
                    f"Historical: {alert['occurrences']}x  \n"
                    f"Signals: {', '.join(alert['signals'])}"
                )
    
    # B-Tier Alerts
    if alerts['B_tier']:
        with st.expander(f"📋 Lower Conviction Alerts ({len(alerts['B_tier'])})"):
            for alert in alerts['B_tier']:
                signal_type = "Long" if alert['type'] == 'bullish' else "Short"
                st.info(
                    f"**{alert['name']}**  \n"
                    f"{signal_type} WR: {alert['win_rate']}% | "
                    f"Avg: +{alert['avg_return']:.2f}% | "
                    f"{alert['occurrences']}x  \n"
                    f"Signals: {', '.join(alert['signals'])}"
                )
    
    # Summary stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tier 1 Bullish Signals Firing", t1_bullish_count, 
                  help="Need 3+ for 85% win rate signal")
    with col2:
        st.metric("Bearish Signals Firing", bearish_count,
                  help="4+ signals = 75% short win rate")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.title("🎯 Signal Gauge Dashboard")
    st.markdown("*Quick-glance view of top backtested signals*")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        market_agg = calculate_market_metrics(df)
    
    # Get current and previous day data
    latest_date = market_agg['date'].max()
    current_data = market_agg[market_agg['date'] == latest_date].iloc[0].to_dict()
    
    prev_date = market_agg[market_agg['date'] < latest_date]['date'].max()
    prev_data = market_agg[market_agg['date'] == prev_date].iloc[0].to_dict() if pd.notna(prev_date) else None
    
    # Analyze signals
    signal_results = analyze_signals(current_data, prev_data)
    composite_score, bullish_firing, bearish_firing = calculate_composite_score(signal_results)
    
    # Header info
    st.markdown(f"**Last Updated:** {latest_date.strftime('%Y-%m-%d')} | "
                f"**Bullish Signals Firing:** {bullish_firing} | "
                f"**Bearish Signals Firing:** {bearish_firing}")
    
    st.divider()
    
    # ===================
    # COMPOSITE GAUGE
    # ===================
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        fig = create_composite_gauge(composite_score, bullish_firing, bearish_firing)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # ===================
    # 🚨 CONFLUENCE ALERTS
    # ===================
    display_confluence_alerts(signal_results)
    
    st.divider()
    
    # ===================
    # CURRENT METRICS
    # ===================
    st.subheader("📊 Current Market Metrics")
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    with m1:
        breadth = current_data['market_breadth']
        st.metric("Breadth", f"{breadth:.1f}%", 
                  delta=f"{current_data.get('breadth_roc_5d', 0):.1f}% (5d)")
    
    with m2:
        net = current_data['market_net']
        st.metric("Net Score", f"{net:.1f}",
                  delta=f"{current_data.get('net_roc_5d', 0):.1f} (5d)")
    
    with m3:
        zscore = current_data.get('market_zscore', 0)
        if pd.isna(zscore):
            zscore = 0
        st.metric("Z-Score", f"{zscore:.2f}")
    
    with m4:
        streak = current_data['market_streak']
        st.metric("Streak", f"{streak:+.0f} days")
    
    with m5:
        bear = current_data.get('market_bear', 0)
        st.metric("Avg Bearish", f"{bear:.1f}")
    
    with m6:
        spread = current_data.get('bull_bear_spread', 0)
        if pd.isna(spread):
            spread = 0
        st.metric("Bull/Bear Spread", f"{spread:.1f}")
    
    st.divider()
    
    # ===================
    # 🏆 TIER 1 SIGNALS
    # ===================
    st.subheader("🏆 Tier 1 Bullish Signals (80%+ Win Rate)")
    
    tier1_bullish = {k: v for k, v in signal_results.items() 
                    if v['signal']['type'] == 'bullish' and v['signal'].get('tier', 2) == 1}
    
    # Sort by proximity (highest first)
    tier1_sorted = sorted(tier1_bullish.items(), key=lambda x: x[1]['proximity'], reverse=True)
    
    cols = st.columns(4)
    
    for idx, (signal_id, result) in enumerate(tier1_sorted):
        signal = result['signal']
        
        with cols[idx % 4]:
            # Status indicator with tier badge
            if result['is_firing']:
                st.markdown(f"### 🔥 **{signal['name']}**")
                st.success(f"**FIRING!** WR: {signal['win_rate']}% | Avg: +{signal['avg_return']}%")
            else:
                st.markdown(f"### {signal['name']}")
                st.info(f"Proximity: {result['proximity']:.0f}%")
            
            # Create gauge based on metric type
            fig = create_signal_gauge(result, signal)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.caption(signal['description'])
    
    st.divider()
    
    # ===================
    # 🥈 TIER 2 SIGNALS
    # ===================
    st.subheader("🥈 Tier 2 Bullish Signals (60-80% Win Rate)")
    
    tier2_bullish = {k: v for k, v in signal_results.items() 
                    if v['signal']['type'] == 'bullish' and v['signal'].get('tier', 2) == 2}
    
    tier2_sorted = sorted(tier2_bullish.items(), key=lambda x: x[1]['proximity'], reverse=True)
    
    cols = st.columns(4)
    
    for idx, (signal_id, result) in enumerate(tier2_sorted):
        signal = result['signal']
        
        with cols[idx % 4]:
            if result['is_firing']:
                st.markdown(f"### 🔥 **{signal['name']}**")
                st.success(f"**FIRING!** WR: {signal['win_rate']}% | Avg: +{signal['avg_return']}%")
            else:
                st.markdown(f"### {signal['name']}")
                st.info(f"Proximity: {result['proximity']:.0f}%")
            
            fig = create_signal_gauge(result, signal)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.caption(signal['description'])
    
    st.divider()
    
    # ===================
    # BEARISH SIGNALS
    # ===================
    st.subheader("🔴 Bearish / Caution Signals")
    
    bearish_signals = {k: v for k, v in signal_results.items() if v['signal']['type'] == 'bearish'}
    
    # Sort by proximity (highest first)
    bearish_sorted = sorted(bearish_signals.items(), key=lambda x: x[1]['proximity'], reverse=True)
    
    cols = st.columns(4)
    
    for idx, (signal_id, result) in enumerate(bearish_sorted):
        signal = result['signal']
        
        with cols[idx % 4]:
            # Status indicator
            tier_badge = "🏆" if signal.get('tier', 2) == 1 else "🥈"
            if result['is_firing']:
                st.markdown(f"### ⚠️ **{signal['name']}**")
                st.error(f"**FIRING!** WR: {signal['win_rate']}% | Avg: {signal['avg_return']:+.2f}%")
            else:
                st.markdown(f"### {signal['name']}")
                st.info(f"Proximity: {result['proximity']:.0f}%")
            
            # Create gauge using helper
            fig = create_signal_gauge(result, signal)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.caption(signal['description'])
    
    st.divider()
    
    # ===================
    # SIGNAL SUMMARY TABLE
    # ===================
    with st.expander("📋 Signal Reference Table"):
        st.markdown("### Backtested Signal Performance")
        
        table_data = []
        for signal_id, signal in SIGNALS.items():
            result = signal_results[signal_id]
            current_val = result['current_value']
            if pd.isna(current_val):
                current_val = 0
            table_data.append({
                'Tier': '🏆 T1' if signal.get('tier', 2) == 1 else '🥈 T2',
                'Signal': signal['name'],
                'Type': '🟢 Bullish' if signal['type'] == 'bullish' else '🔴 Bearish',
                'Win Rate': f"{signal['win_rate']}%",
                'Avg Return': f"{signal['avg_return']:+.2f}%",
                'Current': f"{current_val:.2f}",
                'Threshold': signal['threshold'],
                'Proximity': f"{result['proximity']:.0f}%",
                'Status': '🔥 FIRING' if result['is_firing'] else '⏳ Watching'
            })
        
        df_table = pd.DataFrame(table_data)
        # Sort by tier then win rate
        df_table = df_table.sort_values(['Tier', 'Win Rate'], ascending=[True, False])
        st.dataframe(df_table, use_container_width=True, hide_index=True)
    
    # ===================
    # HISTORICAL CONTEXT
    # ===================
    with st.expander("📈 Historical Context"):
        # Show recent market metrics
        st.markdown("### Last 10 Trading Days")
        
        recent = market_agg.tail(10)[['date', 'market_breadth', 'market_net', 'market_zscore', 'market_streak', 'net_roc_5d']]
        recent.columns = ['Date', 'Breadth', 'Net Score', 'Z-Score', 'Streak', '5d Momentum']
        recent = recent.sort_values('Date', ascending=False)
        
        st.dataframe(
            recent.style.format({
                'Breadth': '{:.1f}%',
                'Net Score': '{:.2f}',
                'Z-Score': '{:.2f}',
                'Streak': '{:.0f}',
                '5d Momentum': '{:+.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    st.divider()
    
    # ===================
    # HISTORICAL SIGNAL CHART
    # ===================
    st.subheader("📈 Historical Signal Chart")
    st.markdown("*See when signals fired in the past and what happened after*")
    
    # Build signal options including individual signals AND confluence patterns
    signal_options = {}
    
    # Add individual signals
    st.markdown("##### Individual Signals & Confluence Patterns")
    for k, s in SIGNALS.items():
        signal_options[f"📊 {s['name']} ({'Bullish' if s['type'] == 'bullish' else 'Bearish'})"] = ('single', k)
    
    # Add confluence patterns
    signal_options["─────── CONFLUENCE PATTERNS ───────"] = ('separator', None)
    
    # Bullish confluence
    for combo_id, combo in CONFLUENCE_RULES.get('bullish', {}).items():
        tier_emoji = "🔥" if combo['tier'] == 'S' else "💪" if combo['tier'] == 'A' else "📈"
        signal_options[f"{tier_emoji} {combo['name']} ({combo['win_rate']}% WR)"] = ('confluence_bullish', combo_id)
    
    # Bearish confluence  
    for combo_id, combo in CONFLUENCE_RULES.get('bearish', {}).items():
        tier_emoji = "🚨" if combo['tier'] == 'S' else "⚠️" if combo['tier'] == 'A' else "📉"
        signal_options[f"{tier_emoji} {combo['name']} ({combo['win_rate']}% WR)"] = ('confluence_bearish', combo_id)
    
    # Count-based confluence
    signal_options["─────── COUNT-BASED CONFLUENCE ───────"] = ('separator', None)
    signal_options["🏆 3+ Tier 1 Bullish Signals (85.1% WR)"] = ('count', 'tier1_bullish_3plus')
    signal_options["🥇 2+ Tier 1 Bullish Signals (77.5% WR)"] = ('count', 'tier1_bullish_2plus')
    signal_options["🚨 4+ Bearish Signals - SHORT (75.0% WR)"] = ('count', 'bearish_4plus')
    signal_options["⚠️ 3+ Bearish Signals (59.5% WR)"] = ('count', 'bearish_3plus')
    
    # Signal selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_signal_name = st.selectbox("Select Signal to View", list(signal_options.keys()))
        signal_type, signal_id = signal_options[selected_signal_name]
    
    with col2:
        lookback_days = st.selectbox("Lookback Period", [90, 180, 365, 'All'], index=1)
    
    # Handle separator selection
    if signal_type == 'separator':
        st.info("👆 Please select a signal or confluence pattern from the dropdown above")
        return
    
    # Get SPY data with robust error handling for cloud deployment
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_spy_data():
        """Fetch SPY data with multiple fallback methods."""
        if not YFINANCE_AVAILABLE:
            return None
            
        import time
        
        # Method 1: Standard yfinance download
        try:
            spy = yf.download('SPY', start='2024-01-01', progress=False, timeout=10)
            if len(spy) > 0:
                spy = spy.reset_index()
                spy.columns = [col[0] if isinstance(col, tuple) else col for col in spy.columns]
                spy['Date'] = pd.to_datetime(spy['Date']).dt.normalize()
                return spy
        except Exception as e:
            pass
        
        # Method 2: Try with Ticker object (sometimes more reliable)
        try:
            time.sleep(1)  # Brief pause before retry
            ticker = yf.Ticker('SPY')
            spy = ticker.history(start='2024-01-01', timeout=10)
            if len(spy) > 0:
                spy = spy.reset_index()
                spy['Date'] = pd.to_datetime(spy['Date']).dt.normalize()
                # Rename columns to match expected format
                spy = spy.rename(columns={'index': 'Date'})
                return spy
        except Exception as e:
            pass
        
        # Method 3: Try alternative ticker symbol
        try:
            time.sleep(1)
            spy = yf.download('^GSPC', start='2024-01-01', progress=False, timeout=10)  # S&P 500 index
            if len(spy) > 0:
                spy = spy.reset_index()
                spy.columns = [col[0] if isinstance(col, tuple) else col for col in spy.columns]
                spy['Date'] = pd.to_datetime(spy['Date']).dt.normalize()
                return spy
        except Exception as e:
            pass
        
        return None
    
    # Try to get SPY data
    spy_data = None
    has_spy = False
    
    try:
        with st.spinner("Loading SPY data..."):
            spy_data = get_spy_data()
            if spy_data is not None and len(spy_data) > 0:
                has_spy = True
    except Exception as e:
        pass
    
    if not has_spy:
        st.warning("⚠️ Could not fetch SPY data for overlay. This sometimes happens on cloud deployments due to rate limiting. The signal indicators below still work!")
    
    # Filter data by lookback
    if lookback_days != 'All':
        cutoff_date = market_agg['date'].max() - timedelta(days=int(lookback_days))
        chart_data = market_agg[market_agg['date'] >= cutoff_date].copy()
    else:
        chart_data = market_agg.copy()
    
    # Ensure dates are normalized datetime
    chart_data['date'] = pd.to_datetime(chart_data['date']).dt.normalize()
    
    # ===========================================
    # DETECT SIGNALS BASED ON TYPE
    # ===========================================
    
    # First, detect ALL individual signals for confluence detection
    for sig_id, sig in SIGNALS.items():
        metric = sig['metric']
        threshold = sig['threshold']
        direction = sig['direction']
        
        if metric not in chart_data.columns:
            chart_data[f'sig_{sig_id}'] = False
            continue
        
        current = chart_data[metric].fillna(0)
        
        if direction == 'below':
            chart_data[f'sig_{sig_id}'] = current < threshold
        elif direction == 'above':
            chart_data[f'sig_{sig_id}'] = current > threshold
        elif direction == 'crosses_above':
            if metric == 'market_net':
                prev = chart_data['prev_net'].fillna(0) if 'prev_net' in chart_data.columns else current.shift(1).fillna(0)
            elif metric == 'market_breadth':
                prev = chart_data['prev_breadth'].fillna(0) if 'prev_breadth' in chart_data.columns else current.shift(1).fillna(0)
            else:
                prev = current.shift(1).fillna(0)
            chart_data[f'sig_{sig_id}'] = (prev < threshold) & (current >= threshold)
        else:
            chart_data[f'sig_{sig_id}'] = False
    
    # Calculate signal counts
    bullish_signals = [s for s, sig in SIGNALS.items() if sig['type'] == 'bullish']
    bearish_signals = [s for s, sig in SIGNALS.items() if sig['type'] == 'bearish']
    tier1_bullish = [s for s, sig in SIGNALS.items() if sig['type'] == 'bullish' and sig.get('tier', 2) == 1]
    
    chart_data['bullish_count'] = chart_data[[f'sig_{s}' for s in bullish_signals]].sum(axis=1)
    chart_data['bearish_count'] = chart_data[[f'sig_{s}' for s in bearish_signals]].sum(axis=1)
    chart_data['tier1_bullish_count'] = chart_data[[f'sig_{s}' for s in tier1_bullish]].sum(axis=1)
    
    # Now detect the selected signal/confluence
    if signal_type == 'single':
        # Single signal
        selected_signal = SIGNALS[signal_id]
        metric = selected_signal['metric']
        threshold = selected_signal['threshold']
        direction = selected_signal['direction']
        chart_data['signal_fired'] = chart_data[f'sig_{signal_id}']
        signal_name = selected_signal['name']
        is_bullish = selected_signal['type'] == 'bullish'
        win_rate = selected_signal.get('win_rate', 0)
        avg_return = selected_signal.get('avg_return', 0)
        
    elif signal_type == 'confluence_bullish':
        # Bullish confluence pattern
        combo = CONFLUENCE_RULES['bullish'][signal_id]
        required_signals = combo['signals']
        # Signal fires when ALL required signals fire
        chart_data['signal_fired'] = chart_data[[f'sig_{s}' for s in required_signals if f'sig_{s}' in chart_data.columns]].all(axis=1)
        signal_name = combo['name']
        metric = 'market_breadth'  # Default display metric
        is_bullish = True
        win_rate = combo['win_rate']
        avg_return = combo['avg_return']
        
    elif signal_type == 'confluence_bearish':
        # Bearish confluence pattern
        combo = CONFLUENCE_RULES['bearish'][signal_id]
        required_signals = combo['signals']
        chart_data['signal_fired'] = chart_data[[f'sig_{s}' for s in required_signals if f'sig_{s}' in chart_data.columns]].all(axis=1)
        signal_name = combo['name']
        metric = 'market_breadth'
        is_bullish = False
        win_rate = combo['win_rate']
        avg_return = combo['avg_return']
        
    elif signal_type == 'count':
        # Count-based confluence
        if signal_id == 'tier1_bullish_3plus':
            chart_data['signal_fired'] = chart_data['tier1_bullish_count'] >= 3
            signal_name = "3+ Tier 1 Bullish Signals"
            metric = 'tier1_bullish_count'
            is_bullish = True
            win_rate = 85.1
            avg_return = 2.25
        elif signal_id == 'tier1_bullish_2plus':
            chart_data['signal_fired'] = chart_data['tier1_bullish_count'] >= 2
            signal_name = "2+ Tier 1 Bullish Signals"
            metric = 'tier1_bullish_count'
            is_bullish = True
            win_rate = 77.5
            avg_return = 1.84
        elif signal_id == 'bearish_4plus':
            chart_data['signal_fired'] = chart_data['bearish_count'] >= 4
            signal_name = "4+ Bearish Signals (SHORT)"
            metric = 'bearish_count'
            is_bullish = False
            win_rate = 75.0
            avg_return = 1.20
        elif signal_id == 'bearish_3plus':
            chart_data['signal_fired'] = chart_data['bearish_count'] >= 3
            signal_name = "3+ Bearish Signals"
            metric = 'bearish_count'
            is_bullish = False
            win_rate = 59.5
            avg_return = 0.50
        else:
            chart_data['signal_fired'] = False
            signal_name = "Unknown"
            metric = 'market_breadth'
            is_bullish = True
            win_rate = 0
            avg_return = 0
    else:
        chart_data['signal_fired'] = False
        signal_name = "Unknown"
        metric = 'market_breadth'
        is_bullish = True
        win_rate = 0
        avg_return = 0
    
    # Get signal dates - normalize to ensure consistent datetime format
    signal_dates = pd.to_datetime(chart_data[chart_data['signal_fired']]['date']).dt.normalize().tolist()
    
    # Determine which metric to show in middle panel
    if metric in chart_data.columns:
        display_metric = metric
    else:
        display_metric = 'market_breadth'
    
    # Create the chart
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('SPY Price with Signal Markers', f'{signal_name} Indicator', 'Market Breadth')
    )
    
    # Row 1: SPY with signals
    if has_spy:
        # Ensure date comparison works by normalizing both to datetime
        chart_min_date = pd.to_datetime(chart_data['date'].min()).normalize()
        spy_data['Date'] = pd.to_datetime(spy_data['Date']).dt.normalize()
        spy_filtered = spy_data[spy_data['Date'] >= chart_min_date]
        
        fig.add_trace(
            go.Candlestick(
                x=spy_filtered['Date'],
                open=spy_filtered['Open'],
                high=spy_filtered['High'],
                low=spy_filtered['Low'],
                close=spy_filtered['Close'],
                name='SPY',
                increasing_line_color='#00cc66',
                decreasing_line_color='#ff4444'
            ),
            row=1, col=1
        )
        
        # Add signal markers
        if signal_dates:
            # Normalize SPY dates for comparison
            spy_dates_normalized = pd.to_datetime(spy_filtered['Date']).dt.normalize()
            signal_dates_normalized = pd.to_datetime(signal_dates).normalize()
            signal_mask = spy_dates_normalized.isin(signal_dates_normalized)
            signal_spy = spy_filtered[signal_mask]
            
            if not signal_spy.empty:
                marker_color = '#00ff00' if is_bullish else '#ff4444'
                marker_symbol = 'triangle-up' if is_bullish else 'triangle-down'
                
                fig.add_trace(
                    go.Scatter(
                        x=signal_spy['Date'],
                        y=signal_spy['Low'] * 0.995 if is_bullish else signal_spy['High'] * 1.005,
                        mode='markers',
                        marker=dict(
                            symbol=marker_symbol,
                            size=12,
                            color=marker_color,
                            line=dict(color='white', width=1)
                        ),
                        name=f'{signal_name} Signal',
                        hovertemplate='%{x}<br>Signal Fired<extra></extra>'
                    ),
                    row=1, col=1
                )
    
    # Row 2: Signal indicator
    # Use cyan/blue for signal indicator
    if display_metric in chart_data.columns:
        fig.add_trace(
            go.Scatter(
                x=chart_data['date'],
                y=chart_data[display_metric],
                mode='lines',
                name=display_metric.replace('_', ' ').title(),
                line=dict(color='#00ffff', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 255, 0.15)'
            ),
            row=2, col=1
        )
        
        # Add threshold line for single signals
        if signal_type == 'single':
            selected_signal = SIGNALS[signal_id]
            threshold = selected_signal['threshold']
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color='#ffaa00',
                annotation_text=f"T",
                annotation_position="right",
                row=2, col=1
            )
        elif signal_type == 'count':
            # Add threshold for count-based
            if 'tier1_bullish' in signal_id:
                thresh = 3 if '3plus' in signal_id else 2
            else:
                thresh = 4 if '4plus' in signal_id else 3
            fig.add_hline(
                y=thresh,
                line_dash="dash",
                line_color='#ffaa00',
                annotation_text=f"Min: {thresh}",
                annotation_position="right",
                row=2, col=1
            )
    
    # Mark signal points on indicator
    signal_points = chart_data[chart_data['signal_fired']]
    if not signal_points.empty and display_metric in chart_data.columns:
        fig.add_trace(
            go.Scatter(
                x=signal_points['date'],
                y=signal_points[display_metric],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=8,
                    color='#00ff00' if is_bullish else '#ff4444'
                ),
                name='Signal Points',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Row 3: Breadth - use purple/magenta, different from signal indicator
    fig.add_trace(
        go.Scatter(
            x=chart_data['date'],
            y=chart_data['market_breadth'],
            mode='lines',
            name='Breadth %',
            line=dict(color='#aa55ff', width=2),
            fill='tonexty' if display_metric != 'market_breadth' else 'tozeroy',
            fillcolor='rgba(170, 85, 255, 0.2)'
        ),
        row=3, col=1
    )
    
    # Add horizontal bands for breadth zones
    fig.add_hrect(y0=0, y1=40, fillcolor="rgba(255,0,0,0.1)", line_width=0, row=3, col=1)
    fig.add_hrect(y0=40, y1=60, fillcolor="rgba(255,255,0,0.05)", line_width=0, row=3, col=1)
    fig.add_hrect(y0=60, y1=100, fillcolor="rgba(0,255,0,0.1)", line_width=0, row=3, col=1)
    
    # Add 50% line for breadth
    fig.add_hline(y=50, line_dash="dash", line_color='white', opacity=0.5, row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=700,
        template='plotly_dark',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    # Better y-axis labels based on metric type
    metric_labels = {
        'market_net': 'Market Net',
        'market_breadth': 'Breadth %',
        'market_zscore': 'Z-Score',
        'market_streak': 'Streak Days',
        'net_roc_5d': '5d ROC',
        'bull_roc_5d': 'Bull 5d ROC',
        'market_bear': 'Avg Bearish',
        'bull_bear_spread': 'Bull/Bear Spread',
        'growth_value_diff': 'Growth-Value',
        'real_estate_breadth': 'RE Breadth %',
        'utilities_breadth': 'Util Breadth %',
        'financials_breadth': 'Fin Breadth %',
        'tier1_bullish_count': 'T1 Bullish Count',
        'bullish_count': 'Bullish Count',
        'bearish_count': 'Bearish Count',
    }
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text=metric_labels.get(display_metric, display_metric), row=2, col=1)
    fig.update_yaxes(title_text="Breadth %", row=3, col=1, range=[0, 100])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Signal performance summary
    st.markdown(f"### 📊 {signal_name} - Historical Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Times Fired", len(signal_dates))
    
    with col2:
        st.metric("Win Rate", f"{win_rate}%")
    
    with col3:
        st.metric("Avg 10d Return", f"+{avg_return}%")
    
    with col4:
        signal_type_emoji = "🟢 Bullish" if is_bullish else "🔴 Bearish"
        st.metric("Signal Type", signal_type_emoji)
    
    # List recent signal dates
    if signal_dates:
        with st.expander(f"📅 All Signal Dates ({len(signal_dates)} occurrences)"):
            # Show most recent first
            recent_signals = sorted(signal_dates, reverse=True)[:20]
            
            signal_df = pd.DataFrame({
                'Date': [d.strftime('%Y-%m-%d') for d in recent_signals],
                'Signal': [signal_name] * len(recent_signals)
            })
            
            st.dataframe(signal_df, use_container_width=True, hide_index=True)
            
            if len(signal_dates) > 20:
                st.caption(f"Showing most recent 20 of {len(signal_dates)} signals")
    
    # Footer
    st.divider()
    st.caption(f"📂 Data source: {DATA_DIR} | Last updated: {latest_date.strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()