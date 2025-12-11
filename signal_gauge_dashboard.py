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
SIGNALS = {
    # BULLISH SIGNALS (from backtesting)
    'breadth_very_low': {
        'name': 'Very Low Breadth',
        'description': 'Market breadth < 20%',
        'type': 'bullish',
        'win_rate': 67.8,
        'avg_return': 1.49,
        'threshold': 20,
        'direction': 'below',  # Signal fires when BELOW threshold
        'metric': 'market_breadth'
    },
    'breadth_low': {
        'name': 'Low Breadth',
        'description': 'Market breadth 20-40%',
        'type': 'bullish',
        'win_rate': 61.8,
        'avg_return': 0.74,
        'threshold': 40,
        'direction': 'below',
        'metric': 'market_breadth'
    },
    'zscore_oversold': {
        'name': 'Oversold (Z-Score)',
        'description': 'Z-Score < -1 (oversold)',
        'type': 'bullish',
        'win_rate': 62.6,
        'avg_return': 0.99,
        'threshold': -1,
        'direction': 'below',
        'metric': 'market_zscore'
    },
    'zscore_extreme_oversold': {
        'name': 'Extremely Oversold',
        'description': 'Z-Score < -2 (extreme)',
        'type': 'bullish',
        'win_rate': 64.5,
        'avg_return': 1.61,
        'threshold': -2,
        'direction': 'below',
        'metric': 'market_zscore'
    },
    'momentum_strong_negative': {
        'name': 'Strong Negative Momentum',
        'description': '5d ROC < -10 (washout)',
        'type': 'bullish',
        'win_rate': 61.2,
        'avg_return': 0.77,
        'threshold': -10,
        'direction': 'below',
        'metric': 'net_roc_5d'
    },
    'net_score_turns_positive': {
        'name': 'Net Score Turns Positive',
        'description': 'Net score crosses above zero',
        'type': 'bullish',
        'win_rate': 91.7,
        'avg_return': 1.61,
        'threshold': 0,
        'direction': 'crosses_above',
        'metric': 'market_net'
    },
    'loss_streak_long': {
        'name': 'Long Loss Streak',
        'description': '5+ consecutive down days',
        'type': 'bullish',
        'win_rate': 62.5,
        'avg_return': 0.86,
        'threshold': -5,
        'direction': 'below',
        'metric': 'market_streak'
    },
    
    # BEARISH/CAUTION SIGNALS
    'breadth_very_high': {
        'name': 'Very High Breadth',
        'description': 'Market breadth > 80%',
        'type': 'bearish',
        'win_rate': 52.9,
        'avg_return': 0.13,
        'threshold': 80,
        'direction': 'above',
        'metric': 'market_breadth'
    },
    'breadth_high': {
        'name': 'High Breadth',
        'description': 'Market breadth 60-80%',
        'type': 'bearish',
        'win_rate': 56.0,
        'avg_return': 0.32,
        'threshold': 60,
        'direction': 'above',
        'metric': 'market_breadth'
    },
    'zscore_overbought': {
        'name': 'Overbought (Z-Score)',
        'description': 'Z-Score > 1 (overbought)',
        'type': 'bearish',
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
        'win_rate': 56.3,
        'avg_return': 0.01,
        'threshold': 5,
        'direction': 'above',
        'metric': 'market_streak'
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
        threshold_color = '#ff4444'
    
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
    """Create the main composite opportunity gauge."""
    
    # Color based on score
    if score >= 70:
        color = '#00ff00'
        label = 'STRONG BUY'
    elif score >= 50:
        color = '#88ff00'
        label = 'BUY'
    elif score >= 30:
        color = '#ffaa00'
        label = 'NEUTRAL'
    elif score >= 15:
        color = '#ff6600'
        label = 'CAUTION'
    else:
        color = '#ff0000'
        label = 'SELL'
    
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
        'market_net': (-30, 30),
        'market_streak': (-10, 10),
    }
    
    for signal_id, signal in SIGNALS.items():
        metric = signal['metric']
        threshold = signal['threshold']
        direction = signal['direction']
        
        current_value = current_data.get(metric, 0)
        prev_value = prev_data.get(metric, 0) if prev_data else current_value
        
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
    """Calculate composite opportunity score based on all signals."""
    
    bullish_score = 0
    bearish_score = 0
    bullish_count = 0
    bearish_count = 0
    
    for signal_id, result in signal_results.items():
        signal = result['signal']
        proximity = result['proximity']
        is_firing = result['is_firing']
        win_rate = signal['win_rate']
        
        # Weight by win rate and proximity
        weight = (win_rate - 50) / 50  # Normalize win rate to 0-1 scale
        contribution = proximity * weight
        
        if signal['type'] == 'bullish':
            bullish_score += contribution
            if is_firing:
                bullish_count += 1
        else:
            bearish_score += contribution
            if is_firing:
                bearish_count += 1
    
    # Calculate final score (bullish signals increase score, bearish decrease)
    # Scale to 0-100 where 50 is neutral
    num_bullish_signals = len([s for s in SIGNALS.values() if s['type'] == 'bullish'])
    num_bearish_signals = len([s for s in SIGNALS.values() if s['type'] == 'bearish'])
    
    max_bullish = num_bullish_signals * 100 * 0.5  # Max possible bullish contribution
    max_bearish = num_bearish_signals * 100 * 0.5
    
    normalized_bullish = (bullish_score / max_bullish * 50) if max_bullish > 0 else 0
    normalized_bearish = (bearish_score / max_bearish * 50) if max_bearish > 0 else 0
    
    # Final score: 50 + bullish contribution - bearish contribution
    composite = 50 + normalized_bullish - normalized_bearish
    composite = max(0, min(100, composite))
    
    return composite, bullish_count, bearish_count


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
    # CURRENT METRICS
    # ===================
    st.subheader("📊 Current Market Metrics")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        breadth = current_data['market_breadth']
        breadth_color = "🟢" if breadth < 40 else "🔴" if breadth > 70 else "🟡"
        st.metric("Breadth", f"{breadth:.1f}%", 
                  delta=f"{current_data.get('breadth_roc_5d', 0):.1f}% (5d)")
    
    with m2:
        net = current_data['market_net']
        net_color = "🟢" if net > 0 else "🔴"
        st.metric("Net Score", f"{net:.1f}",
                  delta=f"{current_data.get('net_roc_5d', 0):.1f} (5d)")
    
    with m3:
        zscore = current_data['market_zscore']
        z_color = "🟢" if zscore < -1 else "🔴" if zscore > 1 else "🟡"
        st.metric("Z-Score", f"{zscore:.2f}")
    
    with m4:
        streak = current_data['market_streak']
        streak_color = "🟢" if streak < -3 else "🔴" if streak > 3 else "🟡"
        st.metric("Streak", f"{streak:+.0f} days")
    
    with m5:
        roc = current_data.get('net_roc_5d', 0)
        roc_color = "🟢" if roc < -10 else "🔴" if roc > 10 else "🟡"
        st.metric("5d Momentum", f"{roc:+.1f}")
    
    st.divider()
    
    # ===================
    # BULLISH SIGNALS
    # ===================
    st.subheader("🟢 Bullish Signals")
    
    bullish_signals = {k: v for k, v in signal_results.items() if v['signal']['type'] == 'bullish'}
    
    # Sort by proximity (highest first)
    bullish_sorted = sorted(bullish_signals.items(), key=lambda x: x[1]['proximity'], reverse=True)
    
    cols = st.columns(4)
    
    for idx, (signal_id, result) in enumerate(bullish_sorted):
        signal = result['signal']
        
        with cols[idx % 4]:
            # Status indicator
            if result['is_firing']:
                st.markdown(f"### 🔥 **{signal['name']}**")
                st.success(f"**FIRING!** WR: {signal['win_rate']}% | Avg: +{signal['avg_return']}%")
            else:
                st.markdown(f"### {signal['name']}")
                st.info(f"Proximity: {result['proximity']:.0f}%")
            
            # Create mini gauge
            if signal['metric'] == 'market_breadth':
                fig = create_gauge(result['current_value'], 0, 100, signal['threshold'],
                                  f"Breadth: {result['current_value']:.1f}%", 'bullish',
                                  signal['direction'], result['is_firing'])
            elif signal['metric'] == 'market_zscore':
                fig = create_gauge(result['current_value'], -3, 3, signal['threshold'],
                                  f"Z-Score: {result['current_value']:.2f}", 'bullish',
                                  signal['direction'], result['is_firing'])
            elif signal['metric'] == 'net_roc_5d':
                fig = create_gauge(result['current_value'], -25, 25, signal['threshold'],
                                  f"5d ROC: {result['current_value']:.1f}", 'bullish',
                                  signal['direction'], result['is_firing'])
            elif signal['metric'] == 'market_net':
                fig = create_gauge(result['current_value'], -30, 30, signal['threshold'],
                                  f"Net: {result['current_value']:.1f}", 'bullish',
                                  signal['direction'], result['is_firing'])
            elif signal['metric'] == 'market_streak':
                fig = create_gauge(result['current_value'], -10, 10, signal['threshold'],
                                  f"Streak: {result['current_value']:.0f}", 'bullish',
                                  signal['direction'], result['is_firing'])
            else:
                continue
            
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
            if result['is_firing']:
                st.markdown(f"### ⚠️ **{signal['name']}**")
                st.error(f"**FIRING!** WR: {signal['win_rate']}% | Avg: +{signal['avg_return']}%")
            else:
                st.markdown(f"### {signal['name']}")
                st.info(f"Proximity: {result['proximity']:.0f}%")
            
            # Create mini gauge
            if signal['metric'] == 'market_breadth':
                fig = create_gauge(result['current_value'], 0, 100, signal['threshold'],
                                  f"Breadth: {result['current_value']:.1f}%", 'bearish',
                                  signal['direction'], result['is_firing'])
            elif signal['metric'] == 'market_zscore':
                fig = create_gauge(result['current_value'], -3, 3, signal['threshold'],
                                  f"Z-Score: {result['current_value']:.2f}", 'bearish',
                                  signal['direction'], result['is_firing'])
            elif signal['metric'] == 'market_streak':
                fig = create_gauge(result['current_value'], -10, 10, signal['threshold'],
                                  f"Streak: {result['current_value']:.0f}", 'bearish',
                                  signal['direction'], result['is_firing'])
            else:
                continue
            
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
            table_data.append({
                'Signal': signal['name'],
                'Type': '🟢 Bullish' if signal['type'] == 'bullish' else '🔴 Bearish',
                'Win Rate': f"{signal['win_rate']}%",
                'Avg Return': f"+{signal['avg_return']}%",
                'Current': f"{result['current_value']:.2f}",
                'Threshold': signal['threshold'],
                'Proximity': f"{result['proximity']:.0f}%",
                'Status': '🔥 FIRING' if result['is_firing'] else '⏳ Watching'
            })
        
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    
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
    
    # Footer
    st.divider()
    st.caption(f"📂 Data source: {DATA_DIR} | Last updated: {latest_date.strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()
