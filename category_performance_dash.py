#!/usr/bin/env python3
"""
Category Performance Dashboard - Complete Rewrite
Focused visualization for crypto category strength scores and trends

FIXES:
- AttributeError on timestamp sort_values()
- Enhanced date range selection (load multiple days)
- Improved error handling
- Better performance
- Cleaner code structure
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import glob
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ================================================================================
# PAGE CONFIGURATION
# ================================================================================

st.set_page_config(
    page_title="Category Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================================
# CUSTOM CSS
# ================================================================================

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        color: #ffffff;
        padding: 1.2rem;
        border-radius: 0.75rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    .strength-high {
        border-left-color: #00ff88;
        background: linear-gradient(135deg, #0d2818 0%, #1a3d2a 100%);
    }
    .strength-medium {
        border-left-color: #ffa726;
        background: linear-gradient(135deg, #2d2318 0%, #3d3118 100%);
    }
    .strength-low {
        border-left-color: #ff4444;
        background: linear-gradient(135deg, #2d1b1b 0%, #3d1f1f 100%);
    }
    
    .big-metric {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-label {
        font-size: 0.9rem;
        color: #b0b0b0;
        margin-top: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

def format_volume(volume: float) -> str:
    """Format volume with appropriate units"""
    if volume >= 1e9:
        return f"${volume/1e9:.2f}B"
    elif volume >= 1e6:
        return f"${volume/1e6:.2f}M"
    elif volume >= 1e3:
        return f"${volume/1e3:.2f}K"
    else:
        return f"${volume:.2f}"

def format_percentage(value: float, include_sign: bool = True, decimals: int = 1) -> str:
    """Format percentage with appropriate sign and precision"""
    if value is None or pd.isna(value):
        return "N/A"
    
    if include_sign:
        return f"{value:+.{decimals}f}%"
    else:
        return f"{value:.{decimals}f}%"

def get_trend_color(score: float) -> str:
    """Get color based on strength score"""
    if pd.isna(score):
        return "#808080"
    if score >= 70:
        return "#00ff88"  # Strong bullish
    elif score >= 60:
        return "#00cc66"  # Bullish
    elif score >= 50:
        return "#ffa726"  # Neutral-positive
    elif score >= 40:
        return "#ff9800"  # Neutral
    elif score >= 30:
        return "#ff6666"  # Bearish
    else:
        return "#ff4444"  # Strong bearish

# ================================================================================
# ADVANCED TRADING ANALYSIS FUNCTIONS
# ================================================================================

def calculate_short_candidates(momentum_rankings: pd.DataFrame, divergences: pd.DataFrame, 
                               turn_signals: pd.DataFrame, relative_strength: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all signals to identify best short candidates.
    Returns ranked list with short scores and confidence levels.
    """
    short_scores = []
    
    # Get all unique categories
    all_categories = set()
    if not momentum_rankings.empty:
        all_categories.update(momentum_rankings['category'].tolist())
    if not divergences.empty:
        all_categories.update(divergences['category'].tolist())
    if not turn_signals.empty:
        all_categories.update(turn_signals['category'].tolist())
    if not relative_strength.empty:
        all_categories.update(relative_strength['category'].tolist())
    
    for category in all_categories:
        score = 0
        signals = []
        confidence_factors = []
        
        # 1. Turn Signals (0-30 points)
        if not turn_signals.empty:
            turn_data = turn_signals[turn_signals['category'] == category]
            if not turn_data.empty:
                signal_type = turn_data.iloc[0]['signal_type']
                signal_strength = turn_data.iloc[0]['signal_strength']
                
                if 'Bearish Turn' in signal_type or 'Breakdown' in signal_type:
                    score += 30
                    signals.append(f"🔴 {signal_type}")
                    confidence_factors.append("Strong Turn Signal")
                elif 'Negative Divergence' in signal_type:
                    score += 20
                    signals.append(f"⚠️ {signal_type}")
                    confidence_factors.append("Turn Signal")
                elif 'Bullish' in signal_type or 'Breakout' in signal_type:
                    score -= 20  # Penalize shorts on bullish signals
                    signals.append(f"🟢 {signal_type} (AVOID SHORT)")
        
        # 2. Momentum Rankings (0-25 points)
        if not momentum_rankings.empty:
            momentum_data = momentum_rankings[momentum_rankings['category'] == category]
            if not momentum_data.empty:
                rank = momentum_data.iloc[0]['rank']
                total_ranks = len(momentum_rankings)
                accel_score = momentum_data.iloc[0]['acceleration_score']
                
                if rank > total_ranks * 0.7:  # Bottom 30%
                    score += 25
                    signals.append(f"🔴 Top Decelerator (#{rank})")
                    confidence_factors.append("Momentum Weakness")
                elif rank > total_ranks * 0.5:  # Bottom 50%
                    score += 15
                    signals.append(f"⚠️ Decelerating (#{rank})")
                elif rank <= 5:  # Top 5
                    score -= 15  # Penalize shorts on leaders
                    signals.append(f"🟢 Top Accelerator (AVOID SHORT)")
        
        # 3. Divergence (0-25 points)
        if not divergences.empty:
            div_data = divergences[divergences['category'] == category]
            if not div_data.empty:
                div_score = div_data.iloc[0]['divergence_score']
                div_type = div_data.iloc[0]['divergence_type']
                
                if 'Strong Under' in div_type:
                    score += 25
                    signals.append(f"🔴 Strong Underperformance ({div_score:.1f})")
                    confidence_factors.append("Strong Divergence")
                elif 'Moderate Under' in div_type:
                    score += 15
                    signals.append(f"⚠️ Underperforming ({div_score:.1f})")
                    confidence_factors.append("Divergence")
                elif 'Strong Out' in div_type:
                    score -= 15  # Penalize shorts on outperformers
                    signals.append(f"🟢 Outperforming (AVOID SHORT)")
        
        # 4. Relative Strength (0-20 points)
        if not relative_strength.empty:
            rs_data = relative_strength[relative_strength['category'] == category]
            if not rs_data.empty:
                rs_score = rs_data.iloc[0]['rs_score']
                rs_trend = rs_data.iloc[0]['rs_trend']
                
                if rs_score < 35:
                    score += 20
                    signals.append(f"🔴 Weak RS ({rs_score:.1f})")
                    confidence_factors.append("Chronic Weakness")
                elif rs_score < 45:
                    score += 12
                    signals.append(f"⚠️ Below Average RS ({rs_score:.1f})")
                elif rs_score > 55 and rs_trend > 0:
                    score -= 15  # Penalize shorts on strong RS
                    signals.append(f"🟢 Strong RS (AVOID SHORT)")
        
        # Confidence level based on number of bearish signals
        num_bearish = len([s for s in signals if '🔴' in s])
        
        if num_bearish >= 3 and score >= 70:
            confidence = "🔥🔥🔥 MAXIMUM"
        elif num_bearish >= 2 and score >= 50:
            confidence = "🔥🔥 HIGH"
        elif num_bearish >= 1 and score >= 30:
            confidence = "🔥 MODERATE"
        else:
            confidence = "⚪ LOW"
        
        # Action recommendation
        if score >= 70:
            action = "🎯 MAX SHORT"
            position_size = "Full Size"
        elif score >= 50:
            action = "🎯 SHORT"
            position_size = "Standard Size"
        elif score >= 30:
            action = "⚠️ LIGHT SHORT"
            position_size = "Reduced Size"
        elif score <= -20:
            action = "🟢 AVOID - BULLISH"
            position_size = "NO SHORT"
        else:
            action = "⚪ NEUTRAL"
            position_size = "Pass"
        
        short_scores.append({
            'category': category,
            'short_score': max(0, min(100, score)),  # Cap between 0-100
            'confidence': confidence,
            'action': action,
            'position_size': position_size,
            'signals': ' | '.join(signals) if signals else 'No clear signals',
            'num_bearish_signals': num_bearish,
            'confidence_factors': ', '.join(confidence_factors) if confidence_factors else 'None'
        })
    
    df = pd.DataFrame(short_scores)
    
    if not df.empty:
        df = df.sort_values('short_score', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
    
    return df

def calculate_long_candidates(momentum_rankings: pd.DataFrame, divergences: pd.DataFrame, 
                              turn_signals: pd.DataFrame, relative_strength: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all signals to identify best LONG candidates.
    Mirror of short candidates but looking for bullish signals.
    """
    long_scores = []
    
    # Get all unique categories
    all_categories = set()
    if not momentum_rankings.empty:
        all_categories.update(momentum_rankings['category'].tolist())
    if not divergences.empty:
        all_categories.update(divergences['category'].tolist())
    if not turn_signals.empty:
        all_categories.update(turn_signals['category'].tolist())
    if not relative_strength.empty:
        all_categories.update(relative_strength['category'].tolist())
    
    for category in all_categories:
        score = 0
        signals = []
        confidence_factors = []
        
        # 1. Turn Signals (0-30 points)
        if not turn_signals.empty:
            turn_data = turn_signals[turn_signals['category'] == category]
            if not turn_data.empty:
                signal_type = turn_data.iloc[0]['signal_type']
                signal_strength = turn_data.iloc[0]['signal_strength']
                
                if 'Bullish Turn' in signal_type or 'Breakout' in signal_type:
                    score += 30
                    signals.append(f"🟢 {signal_type}")
                    confidence_factors.append("Strong Turn Signal")
                elif 'Positive Divergence' in signal_type:
                    score += 20
                    signals.append(f"💡 {signal_type}")
                    confidence_factors.append("Turn Signal")
                elif 'Bearish' in signal_type or 'Breakdown' in signal_type:
                    score -= 20  # Penalize longs on bearish signals
                    signals.append(f"🔴 {signal_type} (AVOID LONG)")
        
        # 2. Momentum Rankings (0-25 points)
        if not momentum_rankings.empty:
            momentum_data = momentum_rankings[momentum_rankings['category'] == category]
            if not momentum_data.empty:
                rank = momentum_data.iloc[0]['rank']
                total_ranks = len(momentum_rankings)
                accel_score = momentum_data.iloc[0]['acceleration_score']
                
                if rank <= 5:  # Top 5
                    score += 25
                    signals.append(f"🟢 Top Accelerator (#{rank})")
                    confidence_factors.append("Strong Momentum")
                elif rank <= total_ranks * 0.3:  # Top 30%
                    score += 15
                    signals.append(f"💡 Accelerating (#{rank})")
                elif rank > total_ranks * 0.7:  # Bottom 30%
                    score -= 15  # Penalize longs on laggards
                    signals.append(f"🔴 Top Decelerator (AVOID LONG)")
        
        # 3. Divergence (0-25 points)
        if not divergences.empty:
            div_data = divergences[divergences['category'] == category]
            if not div_data.empty:
                div_score = div_data.iloc[0]['divergence_score']
                div_type = div_data.iloc[0]['divergence_type']
                
                if 'Strong Out' in div_type:
                    score += 25
                    signals.append(f"🟢 Strong Outperformance ({div_score:.1f})")
                    confidence_factors.append("Strong Divergence")
                elif 'Moderate Out' in div_type:
                    score += 15
                    signals.append(f"💡 Outperforming ({div_score:.1f})")
                    confidence_factors.append("Divergence")
                elif 'Strong Under' in div_type:
                    score -= 15  # Penalize longs on underperformers
                    signals.append(f"🔴 Underperforming (AVOID LONG)")
        
        # 4. Relative Strength (0-20 points)
        if not relative_strength.empty:
            rs_data = relative_strength[relative_strength['category'] == category]
            if not rs_data.empty:
                rs_score = rs_data.iloc[0]['rs_score']
                rs_trend = rs_data.iloc[0]['rs_trend']
                
                if rs_score > 65:
                    score += 20
                    signals.append(f"🟢 Strong RS ({rs_score:.1f})")
                    confidence_factors.append("Strong RS")
                elif rs_score > 55:
                    score += 12
                    signals.append(f"💡 Above Average RS ({rs_score:.1f})")
                elif rs_score < 45 and rs_trend < 0:
                    score -= 15  # Penalize longs on weak RS
                    signals.append(f"🔴 Weak RS (AVOID LONG)")
        
        # Confidence level based on number of bullish signals
        num_bullish = len([s for s in signals if '🟢' in s or '💡' in s])
        
        if num_bullish >= 3 and score >= 70:
            confidence = "🔥🔥🔥 MAXIMUM"
        elif num_bullish >= 2 and score >= 50:
            confidence = "🔥🔥 HIGH"
        elif num_bullish >= 1 and score >= 30:
            confidence = "🔥 MODERATE"
        else:
            confidence = "⚪ LOW"
        
        # Action recommendation
        if score >= 70:
            action = "🚀 MAX LONG"
            position_size = "Full Size"
        elif score >= 50:
            action = "🚀 LONG"
            position_size = "Standard Size"
        elif score >= 30:
            action = "💡 LIGHT LONG"
            position_size = "Reduced Size"
        elif score <= -20:
            action = "🔴 AVOID - BEARISH"
            position_size = "NO LONG"
        else:
            action = "⚪ NEUTRAL"
            position_size = "Pass"
        
        long_scores.append({
            'category': category,
            'long_score': max(0, min(100, score)),  # Cap between 0-100
            'confidence': confidence,
            'action': action,
            'position_size': position_size,
            'signals': ' | '.join(signals) if signals else 'No clear signals',
            'num_bullish_signals': num_bullish,
            'confidence_factors': ', '.join(confidence_factors) if confidence_factors else 'None'
        })
    
    df = pd.DataFrame(long_scores)
    
    if not df.empty:
        df = df.sort_values('long_score', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
    
    return df

def detect_cover_alerts(df: pd.DataFrame, turn_signals: pd.DataFrame, 
                        momentum_rankings: pd.DataFrame) -> pd.DataFrame:
    """
    Identify when short positions should be covered.
    Highlights reversals and momentum shifts.
    """
    alerts = []
    
    for category in df['category'].unique():
        alert_level = "HOLD"  # Default
        alert_signals = []
        urgency = 0
        
        # Check turn signals
        if not turn_signals.empty:
            turn_data = turn_signals[turn_signals['category'] == category]
            if not turn_data.empty:
                signal_type = turn_data.iloc[0]['signal_type']
                signal_strength = turn_data.iloc[0]['signal_strength']
                
                if 'Bullish Turn' in signal_type:
                    urgency += 50
                    alert_signals.append("🚨 BULLISH TURN DETECTED")
                elif 'Breakout' in signal_type:
                    urgency += 40
                    alert_signals.append("⚠️ BREAKOUT SIGNAL")
                elif 'Positive Divergence' in signal_type:
                    urgency += 25
                    alert_signals.append("⚠️ Positive Divergence")
        
        # Check momentum shift
        if not momentum_rankings.empty:
            momentum_data = momentum_rankings[momentum_rankings['category'] == category]
            if not momentum_data.empty:
                rank = momentum_data.iloc[0]['rank']
                accel = momentum_data.iloc[0]['acceleration_score']
                
                if rank <= 3 and accel > 10:
                    urgency += 35
                    alert_signals.append(f"🚨 TOP ACCELERATOR (#{rank})")
                elif rank <= 5:
                    urgency += 20
                    alert_signals.append(f"⚠️ Accelerating (#{rank})")
        
        # Check current strength
        cat_data = df[df['category'] == category]
        if not cat_data.empty:
            latest = cat_data.iloc[-1]
            
            if latest['strength_score'] > 70 and latest['momentum'] > 2:
                urgency += 30
                alert_signals.append("⚠️ BREAKING TO STRENGTH")
            elif latest['strength_score'] > 60:
                urgency += 15
                alert_signals.append("⚠️ Approaching Strong Zone")
        
        # Determine alert level
        if urgency >= 70:
            alert_level = "🚨 COVER NOW"
        elif urgency >= 40:
            alert_level = "⚠️ WATCH CLOSELY"
        elif urgency >= 20:
            alert_level = "💡 MONITOR"
        else:
            alert_level = "✅ HOLD SHORT"
        
        alerts.append({
            'category': category,
            'alert_level': alert_level,
            'urgency_score': urgency,
            'signals': ' | '.join(alert_signals) if alert_signals else 'No reversal signals',
            'num_alerts': len(alert_signals)
        })
    
    alerts_df = pd.DataFrame(alerts)
    
    if not alerts_df.empty:
        alerts_df = alerts_df.sort_values('urgency_score', ascending=False).reset_index(drop=True)
    
    return alerts_df

def detect_exit_alerts_longs(df: pd.DataFrame, turn_signals: pd.DataFrame, 
                             momentum_rankings: pd.DataFrame) -> pd.DataFrame:
    """
    Identify when LONG positions should be exited.
    Mirror of cover alerts but for longs.
    """
    alerts = []
    
    for category in df['category'].unique():
        alert_level = "HOLD"  # Default
        alert_signals = []
        urgency = 0
        
        # Check turn signals
        if not turn_signals.empty:
            turn_data = turn_signals[turn_signals['category'] == category]
            if not turn_data.empty:
                signal_type = turn_data.iloc[0]['signal_type']
                signal_strength = turn_data.iloc[0]['signal_strength']
                
                if 'Bearish Turn' in signal_type:
                    urgency += 50
                    alert_signals.append("🚨 BEARISH TURN DETECTED")
                elif 'Breakdown' in signal_type:
                    urgency += 40
                    alert_signals.append("⚠️ BREAKDOWN SIGNAL")
                elif 'Negative Divergence' in signal_type:
                    urgency += 25
                    alert_signals.append("⚠️ Negative Divergence")
        
        # Check momentum shift
        if not momentum_rankings.empty:
            momentum_data = momentum_rankings[momentum_rankings['category'] == category]
            if not momentum_data.empty:
                rank = momentum_data.iloc[0]['rank']
                total_ranks = len(momentum_rankings)
                accel = momentum_data.iloc[0]['acceleration_score']
                
                if rank > total_ranks * 0.8 and accel < -10:
                    urgency += 35
                    alert_signals.append(f"🚨 TOP DECELERATOR (#{rank})")
                elif rank > total_ranks * 0.7:
                    urgency += 20
                    alert_signals.append(f"⚠️ Decelerating (#{rank})")
        
        # Check current strength
        cat_data = df[df['category'] == category]
        if not cat_data.empty:
            latest = cat_data.iloc[-1]
            
            if latest['strength_score'] < 30 and latest['momentum'] < -2:
                urgency += 30
                alert_signals.append("⚠️ BREAKING TO WEAKNESS")
            elif latest['strength_score'] < 40:
                urgency += 15
                alert_signals.append("⚠️ Approaching Weak Zone")
        
        # Determine alert level
        if urgency >= 70:
            alert_level = "🚨 EXIT NOW"
        elif urgency >= 40:
            alert_level = "⚠️ WATCH CLOSELY"
        elif urgency >= 20:
            alert_level = "💡 MONITOR"
        else:
            alert_level = "✅ HOLD LONG"
        
        alerts.append({
            'category': category,
            'alert_level': alert_level,
            'urgency_score': urgency,
            'signals': ' | '.join(alert_signals) if alert_signals else 'No reversal signals',
            'num_alerts': len(alert_signals)
        })
    
    alerts_df = pd.DataFrame(alerts)
    
    if not alerts_df.empty:
        alerts_df = alerts_df.sort_values('urgency_score', ascending=False).reset_index(drop=True)
    
    return alerts_df

def calculate_market_bias(df: pd.DataFrame, turn_signals: pd.DataFrame, 
                          momentum_rankings: pd.DataFrame) -> Dict:
    """
    Calculate overall market bias - bullish, bearish, or neutral.
    Helps determine if it's a shorting market or not.
    """
    bias_data = {
        'bullish_categories': 0,
        'bearish_categories': 0,
        'neutral_categories': 0,
        'bullish_signals': 0,
        'bearish_signals': 0,
        'avg_strength': 0,
        'avg_momentum': 0,
        'market_bias': 'NEUTRAL',
        'bias_strength': 50,  # 0-100
        'recommendation': ''
    }
    
    if df.empty:
        return bias_data
    
    # Get latest data for each category
    latest_data = df.groupby('category').last()
    
    # Count bullish/bearish categories by strength
    bias_data['bullish_categories'] = len(latest_data[latest_data['strength_score'] >= 60])
    bias_data['bearish_categories'] = len(latest_data[latest_data['strength_score'] <= 40])
    bias_data['neutral_categories'] = len(latest_data) - bias_data['bullish_categories'] - bias_data['bearish_categories']
    
    # Average metrics
    bias_data['avg_strength'] = latest_data['strength_score'].mean()
    bias_data['avg_momentum'] = latest_data['momentum'].mean()
    
    # Count turn signals
    if not turn_signals.empty:
        bias_data['bullish_signals'] = len(turn_signals[
            turn_signals['signal_type'].str.contains('Bullish|Breakout|Positive', na=False)
        ])
        bias_data['bearish_signals'] = len(turn_signals[
            turn_signals['signal_type'].str.contains('Bearish|Breakdown|Negative', na=False)
        ])
    
    # Calculate overall bias
    total_cats = len(latest_data)
    bullish_pct = (bias_data['bullish_categories'] / total_cats * 100) if total_cats > 0 else 0
    bearish_pct = (bias_data['bearish_categories'] / total_cats * 100) if total_cats > 0 else 0
    
    # Bias score (0 = max bearish, 50 = neutral, 100 = max bullish)
    strength_bias = bias_data['avg_strength']  # Already 0-100
    momentum_bias = (bias_data['avg_momentum'] + 5) * 10  # Scale momentum to 0-100
    momentum_bias = max(0, min(100, momentum_bias))
    signal_bias = 50
    
    if bias_data['bullish_signals'] + bias_data['bearish_signals'] > 0:
        signal_bias = (bias_data['bullish_signals'] / (bias_data['bullish_signals'] + bias_data['bearish_signals'])) * 100
    
    # Weighted average
    bias_data['bias_strength'] = (strength_bias * 0.5) + (momentum_bias * 0.3) + (signal_bias * 0.2)
    
    # Determine market bias
    if bias_data['bias_strength'] >= 65:
        bias_data['market_bias'] = '🟢 BULLISH'
        bias_data['recommendation'] = '🚀 RISK-ON: Focus on longs, avoid shorts'
    elif bias_data['bias_strength'] >= 55:
        bias_data['market_bias'] = '🟢 SLIGHTLY BULLISH'
        bias_data['recommendation'] = '✅ Favor longs, selective shorts'
    elif bias_data['bias_strength'] >= 45:
        bias_data['market_bias'] = '⚪ NEUTRAL'
        bias_data['recommendation'] = '⚖️ Mixed market, trade both ways'
    elif bias_data['bias_strength'] >= 35:
        bias_data['market_bias'] = '🔴 SLIGHTLY BEARISH'
        bias_data['recommendation'] = '✅ Favor shorts, selective longs'
    else:
        bias_data['market_bias'] = '🔴 BEARISH'
        bias_data['recommendation'] = '🎯 RISK-OFF: Focus on shorts'
    
    # Risk environment
    if bias_data['bias_strength'] > 50:
        bias_data['risk_environment'] = 'RISK-ON'
    else:
        bias_data['risk_environment'] = 'RISK-OFF'
    
    return bias_data

def detect_momentum_shifts(momentum_rankings: pd.DataFrame, lookback_hours: int = 24) -> pd.DataFrame:
    """
    Identify categories that have significantly changed rankings.
    Shows bottom reversals and top reversals.
    """
    # This would ideally compare current rankings to previous lookback period
    # For now, we'll flag categories based on extreme positions with strong acceleration
    
    shifts = []
    
    if momentum_rankings.empty:
        return pd.DataFrame()
    
    total_ranks = len(momentum_rankings)
    
    for idx, row in momentum_rankings.iterrows():
        shift_type = "No Shift"
        significance = "Low"
        
        rank = row['rank']
        accel = row['acceleration_score']
        strength_rate = row['strength_rate']
        
        # Bottom reversal: Low rank but strong positive acceleration
        if rank > total_ranks * 0.7 and accel > 15:
            shift_type = "🟢 BOTTOM REVERSAL"
            significance = "🔥🔥 HIGH"
        elif rank > total_ranks * 0.5 and accel > 10:
            shift_type = "🟢 Reversal Signal"
            significance = "🔥 MODERATE"
        
        # Top reversal: High rank but weak/negative acceleration
        elif rank <= 5 and accel < -10:
            shift_type = "🔴 TOP REVERSAL"
            significance = "🔥🔥 HIGH"
        elif rank <= 10 and accel < -5:
            shift_type = "🔴 Topping Signal"
            significance = "🔥 MODERATE"
        
        # Momentum explosion: Top ranks with extreme acceleration
        elif rank <= 3 and accel > 20:
            shift_type = "🚀 MOMENTUM EXPLOSION"
            significance = "🔥🔥🔥 MAXIMUM"
        
        # Momentum collapse: Bottom ranks with extreme deceleration
        elif rank > total_ranks * 0.8 and accel < -20:
            shift_type = "💥 MOMENTUM COLLAPSE"
            significance = "🔥🔥🔥 MAXIMUM"
        
        if shift_type != "No Shift":
            shifts.append({
                'category': row['category'],
                'shift_type': shift_type,
                'significance': significance,
                'rank': rank,
                'acceleration': accel,
                'strength_rate': strength_rate,
                'action': 'BUY' if '🟢' in shift_type or '🚀' in shift_type else 'SHORT' if '🔴' in shift_type or '💥' in shift_type else 'WATCH'
            })
    
    shifts_df = pd.DataFrame(shifts)
    
    if not shifts_df.empty:
        # Sort by significance
        sig_order = {'🔥🔥🔥 MAXIMUM': 3, '🔥🔥 HIGH': 2, '🔥 MODERATE': 1, 'Low': 0}
        shifts_df['sig_order'] = shifts_df['significance'].map(sig_order)
        shifts_df = shifts_df.sort_values('sig_order', ascending=False).drop('sig_order', axis=1).reset_index(drop=True)
    
    return shifts_df

def calculate_momentum_rankings(df: pd.DataFrame, lookback_hours: int = 24) -> pd.DataFrame:
    """
    Calculate momentum rankings for each category.
    Shows which categories are accelerating/decelerating fastest.
    """
    rankings = []
    
    # Get latest timestamp
    latest_time = df['timestamp'].max()
    cutoff_time = latest_time - pd.Timedelta(hours=lookback_hours)
    
    for category in df['category'].unique():
        cat_data = df[df['category'] == category].copy()
        cat_data = cat_data[cat_data['timestamp'] >= cutoff_time].sort_values('timestamp')
        
        if len(cat_data) < 2:
            continue
        
        # Calculate various momentum metrics
        latest = cat_data.iloc[-1]
        first = cat_data.iloc[0]
        
        # Strength score change
        strength_change = latest['strength_score'] - first['strength_score']
        
        # Performance acceleration
        perf_change = latest['performance_24h'] - first['performance_24h']
        
        # Momentum trend
        momentum_change = latest['momentum'] - first['momentum']
        
        # Rate of change (strength per hour)
        hours_elapsed = (latest['timestamp'] - first['timestamp']).total_seconds() / 3600
        strength_rate = strength_change / hours_elapsed if hours_elapsed > 0 else 0
        
        # Volume momentum
        volume_change = latest['total_volume'] - first['total_volume']
        volume_pct_change = (volume_change / first['total_volume'] * 100) if first['total_volume'] > 0 else 0
        
        rankings.append({
            'category': category,
            'current_strength': latest['strength_score'],
            'strength_change': strength_change,
            'strength_rate': strength_rate,  # Points per hour
            'performance_change': perf_change,
            'momentum_change': momentum_change,
            'volume_change_pct': volume_pct_change,
            'current_momentum': latest['momentum'],
            'acceleration_score': strength_change + (momentum_change * 2) + (perf_change * 0.5)
        })
    
    rankings_df = pd.DataFrame(rankings)
    
    if not rankings_df.empty:
        # Rank by acceleration score
        rankings_df = rankings_df.sort_values('acceleration_score', ascending=False).reset_index(drop=True)
        rankings_df['rank'] = range(1, len(rankings_df) + 1)
    
    return rankings_df

def detect_divergences(df: pd.DataFrame, lookback_hours: int = 24) -> pd.DataFrame:
    """
    Detect when categories diverge from the market (major_coins).
    Positive divergence = Category stronger than market
    Negative divergence = Category weaker than market
    """
    divergences = []
    
    # Get market benchmark (major_coins as proxy for BTC)
    market_data = df[df['category'] == 'major_coins'].copy()
    
    if market_data.empty:
        return pd.DataFrame()
    
    latest_time = df['timestamp'].max()
    cutoff_time = latest_time - pd.Timedelta(hours=lookback_hours)
    
    market_recent = market_data[market_data['timestamp'] >= cutoff_time]
    
    if len(market_recent) < 2:
        return pd.DataFrame()
    
    market_strength_change = market_recent['strength_score'].iloc[-1] - market_recent['strength_score'].iloc[0]
    market_perf_change = market_recent['performance_24h'].iloc[-1] - market_recent['performance_24h'].iloc[0]
    
    for category in df['category'].unique():
        if category == 'major_coins':
            continue
        
        cat_data = df[df['category'] == category].copy()
        cat_recent = cat_data[cat_data['timestamp'] >= cutoff_time]
        
        if len(cat_recent) < 2:
            continue
        
        cat_strength_change = cat_recent['strength_score'].iloc[-1] - cat_recent['strength_score'].iloc[0]
        cat_perf_change = cat_recent['performance_24h'].iloc[-1] - cat_recent['performance_24h'].iloc[0]
        
        # Calculate divergence
        strength_divergence = cat_strength_change - market_strength_change
        perf_divergence = cat_perf_change - market_perf_change
        
        # Overall divergence score
        divergence_score = (strength_divergence * 0.6) + (perf_divergence * 0.4)
        
        # Classify divergence
        if abs(divergence_score) < 5:
            divergence_type = "In Line"
        elif divergence_score > 15:
            divergence_type = "Strong Outperformance"
        elif divergence_score > 5:
            divergence_type = "Moderate Outperformance"
        elif divergence_score < -15:
            divergence_type = "Strong Underperformance"
        else:
            divergence_type = "Moderate Underperformance"
        
        divergences.append({
            'category': category,
            'divergence_score': divergence_score,
            'strength_divergence': strength_divergence,
            'perf_divergence': perf_divergence,
            'divergence_type': divergence_type,
            'category_strength_change': cat_strength_change,
            'market_strength_change': market_strength_change
        })
    
    divergences_df = pd.DataFrame(divergences)
    
    if not divergences_df.empty:
        divergences_df = divergences_df.sort_values('divergence_score', ascending=False).reset_index(drop=True)
    
    return divergences_df

def detect_turn_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect early reversal patterns and turn signals.
    Looks for categories showing momentum shifts before price follows.
    """
    signals = []
    
    for category in df['category'].unique():
        cat_data = df[df['category'] == category].copy().sort_values('timestamp')
        
        if len(cat_data) < 10:
            continue
        
        # Get recent data points
        recent_20 = cat_data.tail(20)
        latest = recent_20.iloc[-1]
        prev_5 = recent_20.tail(5)
        
        # Calculate trends
        strength_trend = (latest['strength_score'] - recent_20.iloc[0]['strength_score']) / 20
        momentum_trend = (latest['momentum'] - recent_20.iloc[0]['momentum']) / 20
        
        # Detect patterns
        signal_type = "Neutral"
        signal_strength = 0
        
        # Bullish Turn Signal: Momentum turning positive while still down
        if (latest['performance_24h'] < -2 and  # Still negative
            latest['momentum'] > 0 and  # But momentum positive
            prev_5['momentum'].mean() > 0 and  # Sustained
            latest['strength_score'] > prev_5.iloc[0]['strength_score']):  # Strength improving
            signal_type = "🟢 Bullish Turn Signal"
            signal_strength = min(100, abs(latest['momentum']) * 10 + (latest['strength_score'] - 30))
        
        # Bearish Turn Signal: Momentum turning negative while still up
        elif (latest['performance_24h'] > 2 and  # Still positive
              latest['momentum'] < 0 and  # But momentum negative
              prev_5['momentum'].mean() < 0 and  # Sustained
              latest['strength_score'] < prev_5.iloc[0]['strength_score']):  # Strength deteriorating
            signal_type = "🔴 Bearish Turn Signal"
            signal_strength = min(100, abs(latest['momentum']) * 10 + (70 - latest['strength_score']))
        
        # Early Breakout: Accelerating through key level
        elif (latest['strength_score'] > 70 and
              strength_trend > 0.5 and
              momentum_trend > 0.1):
            signal_type = "🚀 Breakout Signal"
            signal_strength = min(100, (latest['strength_score'] - 70) * 3 + abs(latest['momentum']))
        
        # Breakdown Warning: Falling through key support
        elif (latest['strength_score'] < 30 and
              strength_trend < -0.5 and
              momentum_trend < -0.1):
            signal_type = "⚠️ Breakdown Warning"
            signal_strength = min(100, (30 - latest['strength_score']) * 3 + abs(latest['momentum']))
        
        # Momentum Divergence: Price down but momentum improving
        elif (latest['performance_24h'] < 0 and
              momentum_trend > 0.2 and
              latest['momentum'] > prev_5.iloc[0]['momentum']):
            signal_type = "💡 Positive Divergence"
            signal_strength = min(100, abs(momentum_trend) * 50)
        
        # Negative Divergence: Price up but momentum weakening
        elif (latest['performance_24h'] > 0 and
              momentum_trend < -0.2 and
              latest['momentum'] < prev_5.iloc[0]['momentum']):
            signal_type = "⚠️ Negative Divergence"
            signal_strength = min(100, abs(momentum_trend) * 50)
        
        signals.append({
            'category': category,
            'signal_type': signal_type,
            'signal_strength': signal_strength,
            'current_strength': latest['strength_score'],
            'current_momentum': latest['momentum'],
            'strength_trend': strength_trend,
            'momentum_trend': momentum_trend,
            'performance_24h': latest['performance_24h']
        })
    
    signals_df = pd.DataFrame(signals)
    
    if not signals_df.empty:
        # Sort by signal strength (non-neutral signals first)
        signals_df['is_signal'] = signals_df['signal_type'] != "Neutral"
        signals_df = signals_df.sort_values(['is_signal', 'signal_strength'], ascending=[False, False]).reset_index(drop=True)
        signals_df = signals_df.drop('is_signal', axis=1)
    
    return signals_df

def calculate_relative_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate relative strength vs major_coins (BTC proxy).
    Shows which categories are outperforming/underperforming the market.
    """
    rs_data = []
    
    # Get major_coins as benchmark
    benchmark = df[df['category'] == 'major_coins'].copy().sort_values('timestamp')
    
    if benchmark.empty:
        return pd.DataFrame()
    
    for category in df['category'].unique():
        if category == 'major_coins':
            continue
        
        cat_data = df[df['category'] == category].copy().sort_values('timestamp')
        
        if cat_data.empty:
            continue
        
        # Get latest values
        latest_cat = cat_data.iloc[-1]
        latest_bench = benchmark.iloc[-1]
        
        # Calculate relative metrics
        rs_strength = latest_cat['strength_score'] - latest_bench['strength_score']
        rs_performance = latest_cat['performance_24h'] - latest_bench['performance_24h']
        rs_momentum = latest_cat['momentum'] - latest_bench['momentum']
        
        # Overall RS Score (0-100, with 50 being neutral)
        rs_score = 50 + (rs_strength * 0.5) + (rs_performance * 0.3) + (rs_momentum * 0.2)
        rs_score = max(0, min(100, rs_score))
        
        # Calculate RS trend (how RS is changing)
        if len(cat_data) >= 10 and len(benchmark) >= 10:
            # Compare RS now vs 10 periods ago
            cat_10_ago = cat_data.iloc[-10]
            bench_10_ago = benchmark.iloc[-10]
            
            rs_strength_ago = cat_10_ago['strength_score'] - bench_10_ago['strength_score']
            rs_trend = rs_strength - rs_strength_ago
        else:
            rs_trend = 0
        
        # Classify RS
        if rs_score >= 65:
            rs_rating = "🟢 Strong Outperformer"
        elif rs_score >= 55:
            rs_rating = "🟢 Outperformer"
        elif rs_score >= 45:
            rs_rating = "⚪ In Line"
        elif rs_score >= 35:
            rs_rating = "🔴 Underperformer"
        else:
            rs_rating = "🔴 Weak Underperformer"
        
        rs_data.append({
            'category': category,
            'rs_score': rs_score,
            'rs_strength': rs_strength,
            'rs_performance': rs_performance,
            'rs_momentum': rs_momentum,
            'rs_trend': rs_trend,
            'rs_rating': rs_rating,
            'category_strength': latest_cat['strength_score'],
            'market_strength': latest_bench['strength_score']
        })
    
    rs_df = pd.DataFrame(rs_data)
    
    if not rs_df.empty:
        rs_df = rs_df.sort_values('rs_score', ascending=False).reset_index(drop=True)
    
    return rs_df

# ================================================================================
# DATA LOADING FUNCTIONS
# ================================================================================

def find_data_directory() -> Optional[str]:
    """Find the data directory - works on both Windows and Unix"""
    # Get user's home directory
    home = os.path.expanduser("~")
    
    possible_dirs = [
        # Windows-style paths
        os.path.join(home, "Crypto-Dev", "CB_hourly_volume_scanner", "hourly_volume_data"),
        "C:\\Users\\davet\\Crypto-Dev\\CB_hourly_volume_scanner\\hourly_volume_data",
        "hourly_volume_data",
        
        # Unix-style paths
        "/personal-accnt/Crypto-Dev/CB_hourly_volume_scanner/hourly_volume_data",
        os.path.join(home, "Crypto-Dev", "CB_hourly_volume_scanner", "hourly_volume_data"),
        
        # Relative paths
        "../hourly_volume_data",
        ".",
        ".."
    ]
    
    for directory in possible_dirs:
        if os.path.exists(directory):
            # Check if it has data files
            patterns = [
                os.path.join(directory, "hourly_volume_data_*.json"),
                os.path.join(directory, "volume_data_*.json")
            ]
            for pattern in patterns:
                found_files = glob.glob(pattern)
                if found_files:
                    return directory
    
    return None

def load_data_files(data_dir: str) -> Dict[str, str]:
    """
    Load list of available data files and map them to dates.
    Handles both YYYY-MM-DD and YYYYMMDD_HHMM formats.
    Returns: dict mapping date strings (YYYY-MM-DD) to file paths
    """
    if not data_dir or not os.path.exists(data_dir):
        return {}
    
    # Find all data files
    patterns = [
        os.path.join(data_dir, "hourly_volume_data_*.json"),
        os.path.join(data_dir, "volume_data_*.json")
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    
    if not files:
        return {}
    
    # Map dates to files (only keep ONE file per date - the latest one)
    date_file_map = {}
    
    for filepath in files:
        try:
            filename = os.path.basename(filepath)
            
            # Extract date from filename
            # Format 1: hourly_volume_data_20251030_1204.json (YYYYMMDD_HHMM)
            # Format 2: hourly_volume_data_2025-10-30.json (YYYY-MM-DD)
            
            if '_' in filename:
                parts = filename.replace('.json', '').split('_')
                
                # Try to find date part
                for i, part in enumerate(parts):
                    # Format 1: YYYYMMDD (8 digits)
                    if len(part) == 8 and part.isdigit():
                        try:
                            date_obj = datetime.strptime(part, '%Y%m%d')
                            date_str = date_obj.strftime('%Y-%m-%d')
                            
                            # Keep only the latest file for each date
                            if date_str not in date_file_map:
                                date_file_map[date_str] = filepath
                            else:
                                # Compare times if available (HHMM part)
                                if i + 1 < len(parts) and len(parts[i + 1]) == 4 and parts[i + 1].isdigit():
                                    current_time = parts[i + 1]
                                    existing_file = date_file_map[date_str]
                                    existing_parts = os.path.basename(existing_file).replace('.json', '').split('_')
                                    
                                    if len(existing_parts) > i + 1:
                                        existing_time = existing_parts[i + 1]
                                        # Keep the later file
                                        if current_time > existing_time:
                                            date_file_map[date_str] = filepath
                            break
                        except:
                            pass
                    
                    # Format 2: YYYY-MM-DD (with dashes)
                    elif '-' in part and len(part) == 10:
                        try:
                            datetime.strptime(part, '%Y-%m-%d')
                            date_file_map[part] = filepath
                            break
                        except:
                            pass
        except Exception as e:
            continue
    
    return date_file_map

def load_single_file(filepath: str) -> Optional[Dict]:
    """Load data from a single JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading file {filepath}: {e}")
        return None

def load_multiple_files_combined(file_paths: List[str]) -> Optional[Dict]:
    """
    Load and combine multiple data files.
    Combines unique timestamps from all files for complete historical coverage.
    """
    if not file_paths:
        return None
    
    # Sort files by date (oldest to newest)
    sorted_files = sorted(file_paths)
    
    try:
        # Load all files and collect all unique volume_history entries
        all_history = []
        seen_timestamps = set()
        
        for filepath in sorted_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                if data and 'volume_history' in data:
                    for entry in data['volume_history']:
                        timestamp = entry['timestamp']
                        # Only add if we haven't seen this timestamp before
                        if timestamp not in seen_timestamps:
                            all_history.append(entry)
                            seen_timestamps.add(timestamp)
            except Exception as e:
                # Skip files that fail to load
                continue
        
        if not all_history:
            return None
        
        # Sort by timestamp
        all_history.sort(key=lambda x: x['timestamp'])
        
        # Create combined data structure
        combined_data = {
            'volume_history': all_history,
            '_metadata': {
                'oldest_file': os.path.basename(sorted_files[0]),
                'newest_file': os.path.basename(sorted_files[-1]),
                'num_files_loaded': len(sorted_files),
                'num_unique_timestamps': len(all_history),
                'files': [os.path.basename(f) for f in sorted_files]
            }
        }
        
        return combined_data
        
    except Exception as e:
        st.error(f"Error combining files: {e}")
        return None

# ================================================================================
# CATEGORY METRICS CALCULATION
# ================================================================================

def calculate_category_metrics(category_assets: List[Dict]) -> Optional[Dict[str, Any]]:
    """Calculate comprehensive category performance metrics"""
    
    if not category_assets:
        return None
    
    metrics = {}
    
    # 1. PRICE PERFORMANCE (Multiple Timeframes)
    perf_1h = [a.get('price_change_1h', 0) for a in category_assets if a.get('price_change_1h') is not None]
    perf_4h = [a.get('price_change_4h', 0) for a in category_assets if a.get('price_change_4h') is not None]
    perf_24h = [a.get('price_change_24h', 0) for a in category_assets if a.get('price_change_24h') is not None]
    
    metrics['perf_1h'] = np.mean(perf_1h) if perf_1h else 0
    metrics['perf_4h'] = np.mean(perf_4h) if perf_4h else 0
    metrics['perf_24h'] = np.mean(perf_24h) if perf_24h else 0
    
    # 2. MOMENTUM
    metrics['momentum_short'] = metrics['perf_4h']
    
    if metrics['perf_24h'] != 0:
        metrics['momentum_trend'] = metrics['perf_4h'] - (metrics['perf_24h'] / 6)
    else:
        metrics['momentum_trend'] = metrics['perf_4h']
    
    # 3. ACCELERATION
    if metrics['perf_4h'] != 0:
        expected_1h = metrics['perf_4h'] / 4
        metrics['acceleration'] = metrics['perf_1h'] - expected_1h
    else:
        metrics['acceleration'] = metrics['perf_1h']
    
    # 4. VOLUME SENTIMENT
    green = sum([a.get('green_candles', 0) for a in category_assets])
    red = sum([a.get('red_candles', 0) for a in category_assets])
    total_candles = green + red
    metrics['volume_sentiment'] = (green / total_candles) if total_candles > 0 else 0.5
    
    # 5. VOLUME-WEIGHTED PERFORMANCE
    total_vol = sum([a.get('total_dollar_volume', 0) for a in category_assets])
    if total_vol > 0:
        weighted_perf = sum([
            a.get('price_change_24h', 0) * a.get('total_dollar_volume', 0)
            for a in category_assets
        ])
        metrics['weighted_performance'] = weighted_perf / total_vol
    else:
        metrics['weighted_performance'] = 0
    
    # 6. VOLATILITY
    volatilities = [a.get('volatility_24h', 0) for a in category_assets if a.get('volatility_24h') is not None]
    metrics['avg_volatility'] = np.mean(volatilities) if volatilities else 0
    
    # 7. BREADTH
    positive_count = sum([1 for a in category_assets if a.get('price_change_24h', 0) > 0])
    metrics['breadth_pct'] = (positive_count / len(category_assets)) * 100
    metrics['breadth_ratio'] = positive_count / len(category_assets)
    
    # 8. STRENGTH SCORE (0-100)
    perf_score = np.clip(metrics['perf_24h'] / 20, -1, 1)
    sentiment_score = (metrics['volume_sentiment'] - 0.5) * 2
    breadth_score = (metrics['breadth_ratio'] - 0.5) * 2
    momentum_score = np.clip(metrics['momentum_short'] / 10, -1, 1)
    
    raw_strength = (
        perf_score * 0.4 +
        sentiment_score * 0.25 +
        breadth_score * 0.2 +
        momentum_score * 0.15
    )
    
    metrics['strength_score'] = (raw_strength + 1) * 50
    
    # 9. TOTAL VOLUME
    metrics['total_volume'] = total_vol
    
    # 10. ASSET COUNT
    metrics['asset_count'] = len(category_assets)
    
    return metrics

def process_historical_data(volume_history: List[Dict]) -> pd.DataFrame:
    """Process historical data to create category strength time series"""
    
    historical_data = []
    
    for entry in volume_history:
        timestamp = pd.to_datetime(entry['timestamp'])
        
        # Get volume by category for this timestamp
        volume_by_category = entry.get('volume_by_category', {})
        
        # Get assets for metric calculation
        assets = entry.get('assets', {})
        
        for category, volume in volume_by_category.items():
            # Find assets in this category
            category_assets = [
                asset for asset in assets.values()
                if asset.get('category') == category
            ]
            
            # Calculate metrics
            metrics = calculate_category_metrics(category_assets)
            
            if metrics:
                historical_data.append({
                    'timestamp': timestamp,
                    'category': category,
                    'strength_score': metrics['strength_score'],
                    'performance_24h': metrics['perf_24h'],
                    'performance_4h': metrics['perf_4h'],
                    'performance_1h': metrics['perf_1h'],
                    'momentum': metrics['momentum_short'],
                    'volume_sentiment': metrics['volume_sentiment'] * 100,
                    'breadth': metrics['breadth_pct'],
                    'total_volume': metrics['total_volume'],
                    'asset_count': metrics['asset_count'],
                    'acceleration': metrics['acceleration'],
                    'volatility': metrics['avg_volatility']
                })
    
    df = pd.DataFrame(historical_data)
    
    # Sort by timestamp
    if not df.empty:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

# ================================================================================
# VISUALIZATION FUNCTIONS
# ================================================================================

def create_strength_score_timeline(df: pd.DataFrame, categories: Optional[List[str]] = None) -> go.Figure:
    """Create line chart of strength scores over time with distinct colors"""
    
    if categories:
        df_plot = df[df['category'].isin(categories)]
    else:
        df_plot = df
    
    fig = go.Figure()
    
    # Use a distinct color palette for each category
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    
    unique_categories = df_plot['category'].unique()
    
    for idx, category in enumerate(unique_categories):
        cat_data = df_plot[df_plot['category'] == category]
        
        # Get average score for reference (but use distinct color from palette)
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=cat_data['timestamp'],
            y=cat_data['strength_score'],
            mode='lines',
            name=category,
            line=dict(color=color, width=2),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Time: %{x|%Y-%m-%d %H:%M}<br>' +
                         'Score: %{y:.1f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title="Category Strength Scores Timeline",
        xaxis_title="Time",
        yaxis_title="Strength Score",
        hovermode='closest',  # Show only the line being hovered over
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        ),
        yaxis=dict(range=[0, 100])
    )
    
    # Add threshold lines
    fig.add_hline(y=70, line_dash="dash", line_color="green", opacity=0.3, annotation_text="Strong")
    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.3, annotation_text="Neutral")
    fig.add_hline(y=30, line_dash="dash", line_color="red", opacity=0.3, annotation_text="Weak")
    
    return fig

def create_performance_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create heatmap of category performance"""
    
    # Pivot data for heatmap
    pivot = df.pivot_table(
        values='performance_24h',
        index='category',
        columns=df['timestamp'].dt.strftime('%m-%d %H:%M'),
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        zmid=0,
        text=pivot.values,
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        hovertemplate='Category: %{y}<br>Time: %{x}<br>Performance: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="24h Performance Heatmap",
        xaxis_title="Time",
        yaxis_title="Category",
        height=max(400, len(pivot.index) * 30)
    )
    
    return fig

def create_momentum_scatter(df: pd.DataFrame) -> go.Figure:
    """Create scatter plot of momentum vs performance"""
    
    latest_data = df.groupby('category').last().reset_index()
    
    fig = go.Figure()
    
    for _, row in latest_data.iterrows():
        color = get_trend_color(row['strength_score'])
        
        fig.add_trace(go.Scatter(
            x=[row['momentum']],
            y=[row['performance_24h']],
            mode='markers+text',
            name=row['category'],
            marker=dict(size=15, color=color, line=dict(width=2, color='white')),
            text=row['category'],
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>' +
                         'Momentum: %{x:.2f}%<br>' +
                         'Performance: %{y:.2f}%<br>' +
                         f"Score: {row['strength_score']:.1f}<br>" +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title="Momentum vs Performance",
        xaxis_title="Momentum (4h Change)",
        yaxis_title="24h Performance",
        hovermode='closest',
        height=600,
        showlegend=False
    )
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig

def create_volume_sentiment_bars(df: pd.DataFrame) -> go.Figure:
    """Create bar chart of volume sentiment by category"""
    
    latest_data = df.groupby('category').last().reset_index()
    latest_data = latest_data.sort_values('volume_sentiment', ascending=True)
    
    colors = [get_trend_color(s) for s in latest_data['volume_sentiment']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=latest_data['category'],
        x=latest_data['volume_sentiment'],
        orientation='h',
        marker=dict(color=colors),
        text=latest_data['volume_sentiment'].apply(lambda x: f'{x:.1f}%'),
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Sentiment: %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Volume Sentiment by Category",
        xaxis_title="Volume Sentiment (%)",
        yaxis_title="Category",
        height=max(400, len(latest_data) * 30),
        showlegend=False
    )
    
    fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Neutral")
    
    return fig

def create_breadth_gauge(df: pd.DataFrame, category: str) -> go.Figure:
    """Create gauge chart for market breadth"""
    
    cat_data = df[df['category'] == category]
    if cat_data.empty:
        return go.Figure()
    
    latest_breadth = cat_data['breadth'].iloc[-1]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_breadth,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{category} Market Breadth"},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': get_trend_color(latest_breadth)},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=250)
    
    return fig

# ================================================================================
# SIDEBAR CONFIGURATION
# ================================================================================

def configure_sidebar() -> Tuple[Optional[List[str]], List[str], int]:
    """
    Configure sidebar with date range selection and filters.
    Returns: (list of file paths, category_filter, lookback_hours)
    """
    
    st.sidebar.header("⚙️ Settings")
    
    # Find data directory
    data_dir = find_data_directory()
    
    if not data_dir:
        st.sidebar.error("❌ Data directory not found!")
        st.sidebar.info("""
        **Looking for data in:**
        - Current directory: `hourly_volume_data/`
        - Parent directory: `../hourly_volume_data/`
        - Home directory: `~/Crypto-Dev/CB_hourly_volume_scanner/hourly_volume_data/`
        
        **Files should match pattern:**
        - `hourly_volume_data_YYYY-MM-DD.json`
        
        **To fix:**
        ```bash
        # Make sure you're in the correct directory:
        cd C:\\Users\\davet\\Crypto-Dev\\CB_hourly_volume_scanner
        
        # Then run:
        streamlit run category_performance_dash.py
        ```
        
        **Or check that your data files exist:**
        ```bash
        dir hourly_volume_data
        ```
        """)
        return None, [], 168
    
    # Load available files
    date_file_map = load_data_files(data_dir)
    
    if not date_file_map:
        st.sidebar.error("❌ No data files found!")
        st.sidebar.caption(f"Searched in: {data_dir}")
        return None, [], 168
    
    # Sort dates
    available_dates = sorted(date_file_map.keys())
    
    if not available_dates:
        st.sidebar.error("❌ No valid dates found in files!")
        return None, [], 168
    
    # Show data info
    st.sidebar.success(f"📊 {len(available_dates)} days of data available")
    st.sidebar.caption(f"From {available_dates[0]} to {available_dates[-1]}")
    
    # ========================================
    # DATA SELECTION MODE
    # ========================================
    
    st.sidebar.subheader("📅 Data Selection")
    
    selection_mode = st.sidebar.radio(
        "Selection Mode",
        ["📅 Date Range", "📆 Single Date"],
        index=0,
        key="selection_mode"
    )
    
    selected_files = []
    
    if selection_mode == "📅 Date Range":
        st.sidebar.caption("Load multiple days for trend analysis")
        
        # Quick presets
        col1, col2, col3 = st.sidebar.columns(3)
        
        with col1:
            if st.button("7d", use_container_width=True):
                start_idx = max(0, len(available_dates) - 7)
                st.session_state['date_range'] = (available_dates[start_idx], available_dates[-1])
        
        with col2:
            if st.button("14d", use_container_width=True):
                start_idx = max(0, len(available_dates) - 14)
                st.session_state['date_range'] = (available_dates[start_idx], available_dates[-1])
        
        with col3:
            if st.button("All", use_container_width=True):
                st.session_state['date_range'] = (available_dates[0], available_dates[-1])
        
        # Initialize session state
        if 'date_range' not in st.session_state:
            # Default: last 7 days
            start_idx = max(0, len(available_dates) - 7)
            st.session_state['date_range'] = (available_dates[start_idx], available_dates[-1])
        
        # Date range selectors
        start_date = st.sidebar.selectbox(
            "Start Date",
            available_dates,
            index=available_dates.index(st.session_state['date_range'][0]),
            key="start_date"
        )
        
        end_date = st.sidebar.selectbox(
            "End Date",
            available_dates,
            index=available_dates.index(st.session_state['date_range'][1]),
            key="end_date"
        )
        
        # Update session state
        st.session_state['date_range'] = (start_date, end_date)
        
        # Get files in range
        for date_str in available_dates:
            if start_date <= date_str <= end_date:
                selected_files.append(date_file_map[date_str])
        
        # Show selection info
        num_days = len(selected_files)
        st.sidebar.info(f"📊 Selected: **{num_days} days**")
        st.sidebar.caption(f"{start_date} to {end_date}")
        
    else:  # Single Date
        st.sidebar.caption("Load one day's accumulated history")
        
        # Quick access
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("Latest", use_container_width=True):
                st.session_state['selected_date'] = available_dates[-1]
        
        with col2:
            if st.button("Oldest", use_container_width=True):
                st.session_state['selected_date'] = available_dates[0]
        
        # Initialize session state
        if 'selected_date' not in st.session_state:
            st.session_state['selected_date'] = available_dates[-1]
        
        # Date selector
        selected_date = st.sidebar.selectbox(
            "Select Date",
            available_dates,
            index=available_dates.index(st.session_state['selected_date']),
            key="single_date_select"
        )
        
        st.session_state['selected_date'] = selected_date
        selected_files = [date_file_map[selected_date]]
        
        # Parse date for day of week
        try:
            date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
            day_name = date_obj.strftime('%A')
            st.sidebar.info(f"📅 {selected_date} ({day_name})")
        except:
            st.sidebar.info(f"📅 {selected_date}")
    
    # ========================================
    # TIME RANGE FILTER
    # ========================================
    
    st.sidebar.subheader("⏱️ Analysis Window")
    
    time_preset = st.sidebar.selectbox(
        "Time Range",
        ["Last 6 Hours", "Last 12 Hours", "Last 24 Hours", "Last 3 Days", 
         "Last Week", "Last 2 Weeks", "Last Month", "All Available"],
        index=2,  # Default to 24 hours
        key="time_preset"
    )
    
    preset_hours = {
        "Last 6 Hours": 6,
        "Last 12 Hours": 12,
        "Last 24 Hours": 24,
        "Last 3 Days": 72,
        "Last Week": 168,
        "Last 2 Weeks": 336,
        "Last Month": 720,
        "All Available": 9999
    }
    
    lookback_hours = preset_hours[time_preset]
    
    # ========================================
    # CATEGORY FILTER
    # ========================================
    
    st.sidebar.subheader("🏷️ Categories")
    
    # Load data to get available categories
    if selected_files:
        try:
            # Load the latest file to get categories
            if len(selected_files) == 1:
                data = load_single_file(selected_files[0])
            else:
                data = load_multiple_files_combined(selected_files)
            
            if data and 'volume_history' in data and len(data['volume_history']) > 0:
                latest_entry = data['volume_history'][-1]
                available_categories = sorted(list(latest_entry.get('volume_by_category', {}).keys()))
                
                # Initialize session state for categories
                if 'category_filter' not in st.session_state:
                    st.session_state['category_filter'] = available_categories
                
                # Category selection
                select_all = st.sidebar.checkbox("Select All", value=True, key="select_all_cats")
                
                if select_all:
                    category_filter = st.sidebar.multiselect(
                        "Select Categories",
                        available_categories,
                        default=available_categories,
                        key="category_multiselect"
                    )
                else:
                    category_filter = st.sidebar.multiselect(
                        "Select Categories",
                        available_categories,
                        default=[],
                        key="category_multiselect_partial"
                    )
                
                st.session_state['category_filter'] = category_filter
                st.sidebar.caption(f"✅ {len(category_filter)} of {len(available_categories)} selected")
            else:
                category_filter = []
                st.sidebar.warning("⚠️ No category data found")
        except Exception as e:
            category_filter = []
            st.sidebar.error(f"❌ Error loading categories: {e}")
    else:
        category_filter = []
    
    # ========================================
    # INFO SECTION
    # ========================================
    
    st.sidebar.subheader("ℹ️ Data Info")
    
    if selected_files:
        total_size = sum(os.path.getsize(f) for f in selected_files if os.path.exists(f)) / 1024
        st.sidebar.caption(f"💾 Total Size: {total_size:.1f} KB")
        st.sidebar.caption(f"📁 Files: {len(selected_files)}")
    
    return selected_files, category_filter, lookback_hours

# ================================================================================
# MAIN APPLICATION
# ================================================================================

def main():
    """Main application function"""
    
    # App header
    st.title("📊 Category Performance Dashboard")
    st.markdown("*Real-time crypto category strength analysis and trends*")
    st.markdown("---")
    
    # Configure sidebar
    selected_files, category_filter, lookback_hours = configure_sidebar()
    
    if not selected_files:
        st.error("❌ No data files selected!")
        
        st.info("""
        ### 📊 Getting Started
        
        **This dashboard requires data files from your scanner.**
        
        The app looks for files in:
        - `/personal-accnt/Crypto-Dev/CB_hourly_volume_scanner/hourly_volume_data/`
        - `~/Crypto-Dev/CB_hourly_volume_scanner/hourly_volume_data/`
        
        **File format expected:**
        ```
        hourly_volume_data_YYYY-MM-DD.json
        ```
        
        **To start collecting data:**
        ```bash
        cd ~/Crypto-Dev/CB_hourly_volume_scanner
        python your_scanner_script.py
        ```
        """)
        
        st.caption(f"Current directory: {os.getcwd()}")
        return
    
    # Load data
    with st.spinner("📥 Loading data..."):
        if len(selected_files) == 1:
            data = load_single_file(selected_files[0])
        else:
            data = load_multiple_files_combined(selected_files)
    
    if not data or 'volume_history' not in data:
        st.error("❌ Invalid data format!")
        st.info("Data must contain 'volume_history' field")
        return
    
    # Process historical data
    volume_history = data['volume_history']
    
    if not volume_history:
        st.warning("⚠️ No historical data found in file")
        return
    
    # Show data range info
    all_timestamps = [pd.to_datetime(entry['timestamp']) for entry in volume_history]
    earliest_time = min(all_timestamps)
    latest_time = max(all_timestamps)
    total_points = len(volume_history)
    span_days = (latest_time - earliest_time).days
    
    # Filter by time range
    if lookback_hours == 9999:  # All Data
        filtered_history = volume_history
        time_info = f"📊 Showing **all available data**: {span_days} days ({total_points} data points)"
    else:
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        filtered_history = [
            entry for entry in volume_history 
            if pd.to_datetime(entry['timestamp']) >= cutoff_time
        ]
        
        if filtered_history:
            shown_start = pd.to_datetime(filtered_history[0]['timestamp'])
            shown_end = pd.to_datetime(filtered_history[-1]['timestamp'])
            time_info = f"📊 Showing **{lookback_hours/24:.1f} days**: {shown_start.strftime('%b %d')} to {shown_end.strftime('%b %d, %Y')} ({len(filtered_history)} points)"
        else:
            time_info = "⚠️ No data in selected time range"
    
    st.info(time_info)
    
    if not filtered_history:
        st.warning("⚠️ No data available for selected time range")
        st.info(f"Your data spans from {earliest_time.strftime('%b %d')} to {latest_time.strftime('%b %d, %Y')}")
        return
    
    # Process data into DataFrame
    with st.spinner("🔄 Processing category metrics..."):
        df = process_historical_data(filtered_history)
    
    if df.empty:
        st.warning("⚠️ No category data available")
        return
    
    # Apply category filter
    if category_filter:
        df_filtered = df[df['category'].isin(category_filter)]
    else:
        df_filtered = df.copy()
    
    if df_filtered.empty:
        st.warning("⚠️ No data for selected categories")
        return
    
    # Get latest data for freshness check
    latest_entry = filtered_history[-1]
    timestamp = pd.to_datetime(latest_entry['timestamp'])
    
    # Display freshness and stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        age = datetime.now() - timestamp.replace(tzinfo=None)
        age_minutes = age.total_seconds() / 60
        
        if age_minutes < 10:
            st.success(f"✅ Fresh ({age_minutes:.0f} min)")
        elif age_minutes < 60:
            st.warning(f"⚠️ {age_minutes:.0f} min old")
        else:
            st.error(f"⚠️ {age.total_seconds()/3600:.1f}h old")
    
    with col2:
        num_categories = len(df_filtered['category'].unique())
        st.metric("Categories", num_categories)
    
    with col3:
        st.metric("Data Points", len(df_filtered))
    
    with col4:
        st.caption(f"📅 {timestamp.strftime('%Y-%m-%d %H:%M')}")
    
    st.markdown("---")
    
    # ========================================
    # TABS FOR DIFFERENT VIEWS
    # ========================================
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Strength Timeline", 
        "🔥 Heatmap", 
        "📊 Performance Matrix",
        "💹 Market Breadth",
        "📉 Advanced Analytics",
        "📅 Data Statistics",
        "🎯 Trading Signals"  # NEW!
    ])
    
    # TAB 1: Strength Timeline
    with tab1:
        st.subheader("Category Strength Scores Over Time")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            show_all = st.checkbox("Show All Categories", value=False, key="show_all_timeline")
        
        if show_all:
            fig = create_strength_score_timeline(df_filtered)
        else:
            # Show only top 8 by average strength
            top_categories = df_filtered.groupby('category')['strength_score'].mean().nlargest(8).index.tolist()
            fig = create_strength_score_timeline(df_filtered, top_categories)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Current rankings
        st.subheader("Current Category Rankings")
        latest_scores = df_filtered.groupby('category').last().sort_values('strength_score', ascending=False)
        
        cols = st.columns(3)
        for idx, (cat, row) in enumerate(latest_scores.head(9).iterrows()):
            col = cols[idx % 3]
            with col:
                score = row['strength_score']
                perf = row['performance_24h']
                vol = row['total_volume']
                
                # Determine card style
                if score >= 70:
                    card_class = "strength-high"
                elif score >= 40:
                    card_class = "strength-medium"
                else:
                    card_class = "strength-low"
                
                st.markdown(f"""
                <div class="metric-card {card_class}">
                    <h3 style="margin:0; color:#ffffff;">{cat}</h3>
                    <p class="big-metric">{score:.1f}</p>
                    <p class="metric-label">STRENGTH SCORE</p>
                    <p style="margin:5px 0; color:#ffffff;">24h: {format_percentage(perf)} | Vol: {format_volume(vol)}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # TAB 2: Heatmap
    with tab2:
        st.subheader("24h Performance Heatmap")
        
        if len(df_filtered) > 0:
            fig = create_performance_heatmap(df_filtered)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data for heatmap")
    
    # TAB 3: Performance Matrix
    with tab3:
        st.subheader("Momentum vs Performance Analysis")
        
        fig = create_momentum_scatter(df_filtered)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Quadrant Analysis:**
        - **Top Right**: Strong momentum + positive performance (🚀 Bullish)
        - **Top Left**: Negative momentum + positive performance (⚠️ Losing steam)
        - **Bottom Right**: Strong momentum + negative performance (🔄 Potential reversal)
        - **Bottom Left**: Negative momentum + negative performance (🔻 Bearish)
        """)
    
    # TAB 4: Market Breadth
    with tab4:
        st.subheader("Volume Sentiment Analysis")
        
        fig = create_volume_sentiment_bars(df_filtered)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Market Breadth Gauges")
        
        categories = df_filtered['category'].unique()[:6]  # Show top 6
        cols = st.columns(3)
        
        for idx, category in enumerate(categories):
            with cols[idx % 3]:
                fig = create_breadth_gauge(df_filtered, category)
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 5: Advanced Analytics
    with tab5:
        st.subheader("Advanced Analytics")
        
        # Correlation analysis
        st.markdown("#### Category Correlation Matrix")
        
        # Create correlation matrix
        pivot = df_filtered.pivot_table(
            values='strength_score',
            index='timestamp',
            columns='category',
            aggfunc='mean'
        )
        
        if not pivot.empty and len(pivot.columns) > 1:
            corr = pivot.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale='RdBu',
                zmid=0,
                text=corr.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Strength Score Correlations",
                height=max(400, len(corr) * 40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("Correlation shows how categories move together. +1 = perfect correlation, -1 = perfect inverse correlation")
        else:
            st.info("Not enough data for correlation analysis")
        
        # Volatility analysis
        st.markdown("#### Category Volatility")
        
        latest_vol = df_filtered.groupby('category')['volatility'].last().sort_values(ascending=False)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=latest_vol.index,
            x=latest_vol.values,
            orientation='h',
            marker=dict(color='rgba(255, 107, 107, 0.8)'),
            text=latest_vol.values.round(2),
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Current Volatility by Category",
            xaxis_title="Volatility",
            yaxis_title="Category",
            height=max(400, len(latest_vol) * 30)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 6: Data Statistics
    with tab6:
        st.subheader("Data Coverage & Quality")
        
        # Overall statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Data Points", len(df_filtered))
            st.metric("Categories Tracked", len(df_filtered['category'].unique()))
        
        with col2:
            st.metric("Time Span", f"{(df_filtered['timestamp'].max() - df_filtered['timestamp'].min()).days} days")
            st.metric("Earliest Data", df_filtered['timestamp'].min().strftime('%Y-%m-%d'))
        
        with col3:
            st.metric("Latest Data", df_filtered['timestamp'].max().strftime('%Y-%m-%d'))
            avg_points = len(df_filtered) / len(df_filtered['category'].unique())
            st.metric("Avg Points/Category", f"{avg_points:.0f}")
        
        st.markdown("---")
        
        # Category-by-category stats
        st.markdown("#### Statistics by Category")
        
        category_stats = []
        for category in df_filtered['category'].unique():
            cat_data = df_filtered[df_filtered['category'] == category]
            
            stats = {
                'Category': category,
                'Data Points': len(cat_data),
                'Avg Strength': f"{cat_data['strength_score'].mean():.1f}",
                'Avg Performance': format_percentage(cat_data['performance_24h'].mean()),
                'Avg Volume': format_volume(cat_data['total_volume'].mean()),
                'Avg Assets': f"{cat_data['asset_count'].mean():.0f}"
            }
            category_stats.append(stats)
        
        stats_df = pd.DataFrame(category_stats)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Data quality metrics
        st.markdown("#### Data Quality Metrics")
        
        quality_cols = st.columns(3)
        
        with quality_cols[0]:
            # Check for gaps in data - FIXED: Remove sort_values() on DatetimeArray
            timestamps = sorted(pd.to_datetime(df_filtered['timestamp'].unique()))  # ← FIX HERE
            
            if len(timestamps) > 1:
                gaps = np.diff(timestamps)
                max_gap = pd.Timedelta(gaps.max()).total_seconds() / 3600
                avg_gap = pd.Timedelta(gaps.mean()).total_seconds() / 3600
                st.metric("Max Gap Between Data", f"{max_gap:.1f} hours")
                st.metric("Avg Gap Between Data", f"{avg_gap:.1f} hours")
        
        with quality_cols[1]:
            # Categories with consistent data
            consistent_cats = 0
            for cat in df_filtered['category'].unique():
                cat_timestamps = df_filtered[df_filtered['category'] == cat]['timestamp'].nunique()
                if cat_timestamps >= len(timestamps) * 0.8:  # 80% coverage
                    consistent_cats += 1
            st.metric("Categories with >80% Coverage", consistent_cats)
            st.metric("Total Unique Timestamps", len(timestamps))
        
        with quality_cols[2]:
            # Data completeness
            total_possible = len(df_filtered['category'].unique()) * len(timestamps)
            completeness = (len(df_filtered) / total_possible) * 100 if total_possible > 0 else 0
            st.metric("Data Completeness", f"{completeness:.1f}%")
            st.caption("Percentage of expected data points present")
    
    # TAB 7: Trading Signals (NEW!)
    with tab7:
        st.subheader("🎯 Trading Edge Analysis")
        st.markdown("*Find leaders, detect divergences, spot turn signals, and track relative strength*")
        
        # Select lookback period
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("#### Analysis Period")
        with col2:
            lookback_select = st.selectbox(
                "Lookback",
                [6, 12, 24, 48, 72],
                index=2,
                format_func=lambda x: f"{x}h",
                key="trading_signals_lookback"
            )
        
        # Calculate all signals
        with st.spinner("🔍 Analyzing market for trading signals..."):
            momentum_rankings = calculate_momentum_rankings(df_filtered, lookback_hours=lookback_select)
            divergences = detect_divergences(df_filtered, lookback_hours=lookback_select)
            turn_signals = detect_turn_signals(df_filtered)
            relative_strength = calculate_relative_strength(df_filtered)
            
            # Calculate BOTH directions
            short_candidates = calculate_short_candidates(momentum_rankings, divergences, turn_signals, relative_strength)
            long_candidates = calculate_long_candidates(momentum_rankings, divergences, turn_signals, relative_strength)
            cover_alerts = detect_cover_alerts(df_filtered, turn_signals, momentum_rankings)
            exit_alerts_longs = detect_exit_alerts_longs(df_filtered, turn_signals, momentum_rankings)
            market_bias = calculate_market_bias(df_filtered, turn_signals, momentum_rankings)
            momentum_shifts = detect_momentum_shifts(momentum_rankings, lookback_hours=lookback_select)
        
        # Determine which side to emphasize based on market bias
        is_bearish_market = market_bias['bias_strength'] < 50
        is_bullish_market = market_bias['bias_strength'] > 50
        
        st.markdown("---")
        
        # ========================================
        # MARKET OVERVIEW - NEW!
        # ========================================
        
        st.markdown("### 📊 MARKET OVERVIEW")
        
        # Market Bias Gauge
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### Market Bias")
            bias_color = "#00ff88" if market_bias['bias_strength'] > 55 else "#ff4444" if market_bias['bias_strength'] < 45 else "#808080"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: {bias_color}22; border: 2px solid {bias_color}; border-radius: 10px;">
                <h2 style="margin: 0; color: {bias_color};">{market_bias['market_bias']}</h2>
                <p style="margin: 5px 0; font-size: 2rem; color: {bias_color};">{market_bias['bias_strength']:.0f}</p>
                <p style="margin: 0; font-size: 0.8rem;">Bias Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Risk Environment")
            risk_color = "#00ff88" if market_bias['risk_environment'] == 'RISK-ON' else "#ff4444"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: {risk_color}22; border: 2px solid {risk_color}; border-radius: 10px;">
                <h2 style="margin: 0; color: {risk_color};">{market_bias['risk_environment']}</h2>
                <p style="margin: 5px 0;">Avg Strength: {market_bias['avg_strength']:.1f}</p>
                <p style="margin: 0;">Avg Momentum: {market_bias['avg_momentum']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("#### Category Distribution")
            st.metric("🟢 Bullish", market_bias['bullish_categories'])
            st.metric("🔴 Bearish", market_bias['bearish_categories'])
            st.metric("⚪ Neutral", market_bias['neutral_categories'])
        
        with col4:
            st.markdown("#### Active Signals")
            st.metric("🟢 Bullish Signals", market_bias['bullish_signals'])
            st.metric("🔴 Bearish Signals", market_bias['bearish_signals'])
            total_signals = market_bias['bullish_signals'] + market_bias['bearish_signals']
            st.caption(f"Total: {total_signals}")
        
        # Recommendation
        st.info(f"**💡 Trading Recommendation:** {market_bias['recommendation']}")
        
        st.markdown("---")
        
        # ========================================
        # QUICK TRADE SUMMARY - NEW! (BIDIRECTIONAL)
        # ========================================
        
        st.markdown("### ⚡ QUICK TRADE SUMMARY")
        
        if is_bearish_market:
            st.info(f"🔴 **BEARISH MARKET** (Bias: {market_bias['bias_strength']:.0f}) - Focus on SHORTS below")
        elif is_bullish_market:
            st.success(f"🟢 **BULLISH MARKET** (Bias: {market_bias['bias_strength']:.0f}) - Focus on LONGS below")
        else:
            st.warning(f"⚪ **NEUTRAL MARKET** (Bias: {market_bias['bias_strength']:.0f}) - Trade both directions")
        
        col1, col2 = st.columns(2)
        
        # LEFT COLUMN - Context dependent
        with col1:
            if is_bearish_market:
                # Show TOP SHORTS prominently when bearish
                st.markdown("#### 🎯 TOP SHORTS ⭐")
                if not short_candidates.empty:
                    max_shorts = short_candidates[short_candidates['short_score'] >= 50].head(5)
                    
                    if not max_shorts.empty:
                        for idx, row in max_shorts.iterrows():
                            confidence_color = "#ff4444" if "MAXIMUM" in row['confidence'] else "#ff6666" if "HIGH" in row['confidence'] else "#ff8888"
                            st.markdown(f"""
                            <div style="padding: 10px; margin: 5px 0; background: {confidence_color}22; border-left: 4px solid {confidence_color}; border-radius: 5px;">
                                <strong>{row['category']}</strong> {row['confidence']}<br/>
                                Score: {row['short_score']:.0f} | {row['action']}<br/>
                                <small>{row['position_size']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("✅ No strong short setups found")
                else:
                    st.warning("⚠️ Not enough data")
            
            else:
                # Show TOP LONGS prominently when bullish/neutral
                st.markdown("#### 🚀 TOP LONGS ⭐")
                if not long_candidates.empty:
                    max_longs = long_candidates[long_candidates['long_score'] >= 50].head(5)
                    
                    if not max_longs.empty:
                        for idx, row in max_longs.iterrows():
                            confidence_color = "#00ff88" if "MAXIMUM" in row['confidence'] else "#00cc66" if "HIGH" in row['confidence'] else "#00aa44"
                            st.markdown(f"""
                            <div style="padding: 10px; margin: 5px 0; background: {confidence_color}22; border-left: 4px solid {confidence_color}; border-radius: 5px;">
                                <strong>{row['category']}</strong> {row['confidence']}<br/>
                                Score: {row['long_score']:.0f} | {row['action']}<br/>
                                <small>{row['position_size']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("✅ No strong long setups found")
                else:
                    st.warning("⚠️ Not enough data")
        
        # RIGHT COLUMN - Opposite side (less emphasis)
        with col2:
            if is_bearish_market:
                # Show TOP LONGS as secondary when bearish
                st.markdown("#### 🚀 Top Longs (Less Favorable)")
                if not long_candidates.empty:
                    max_longs = long_candidates[long_candidates['long_score'] >= 50].head(3)
                    
                    if not max_longs.empty:
                        for idx, row in max_longs.iterrows():
                            st.markdown(f"""
                            <div style="padding: 8px; margin: 5px 0; background: #80808822; border-left: 2px solid #808080; border-radius: 5px; opacity: 0.7;">
                                <strong>{row['category']}</strong><br/>
                                Score: {row['long_score']:.0f} | {row['action']}<br/>
                                <small>⚠️ Risky in bearish market</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("✅ No longs recommended")
            
            else:
                # Show TOP SHORTS as secondary when bullish/neutral
                st.markdown("#### 🎯 Top Shorts (Less Favorable)")
                if not short_candidates.empty:
                    max_shorts = short_candidates[short_candidates['short_score'] >= 50].head(3)
                    
                    if not max_shorts.empty:
                        for idx, row in max_shorts.iterrows():
                            st.markdown(f"""
                            <div style="padding: 8px; margin: 5px 0; background: #80808822; border-left: 2px solid #808080; border-radius: 5px; opacity: 0.7;">
                                <strong>{row['category']}</strong><br/>
                                Score: {row['short_score']:.0f} | {row['action']}<br/>
                                <small>⚠️ Risky in bullish market</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("✅ No shorts recommended")
        
        st.markdown("---")
        
        # ========================================
        # MOMENTUM SHIFTS - NEW!
        # ========================================
        
        st.markdown("### 🔄 MOMENTUM SHIFTS - Catch the Reversals!")
        st.caption("Categories jumping rankings - your edge to catch turns FAST")
        
        if not momentum_shifts.empty:
            for idx, row in momentum_shifts.iterrows():
                action_color = "#00ff88" if row['action'] == 'BUY' else "#ff4444" if row['action'] == 'SHORT' else "#808080"
                st.markdown(f"""
                <div style="padding: 15px; margin: 10px 0; background: {action_color}22; border: 2px solid {action_color}; border-radius: 10px;">
                    <h3 style="margin: 0; color: {action_color};">{row['category']} - {row['shift_type']}</h3>
                    <p style="margin: 5px 0;">
                        <strong>Significance:</strong> {row['significance']}<br/>
                        <strong>Rank:</strong> #{row['rank']} | <strong>Acceleration:</strong> {row['acceleration']:+.1f} | <strong>Rate:</strong> {row['strength_rate']:+.2f} pts/hr
                    </p>
                    <p style="margin: 5px 0; padding: 10px; background: {action_color}44; border-radius: 5px;">
                        <strong>ACTION: {row['action']}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("✅ No significant momentum shifts detected - market in steady state")
        
        st.markdown("---")
        
        # ========================================
        # COVER/EXIT ALERTS - NEW! (BIDIRECTIONAL)
        # ========================================
        
        st.markdown("### ⚠️ POSITION MANAGEMENT ALERTS")
        st.caption("Monitor your positions - know when to exit BEFORE it's too late!")
        
        # Show relevant alerts based on market type
        col1, col2 = st.columns(2)
        
        with col1:
            if is_bearish_market:
                st.markdown("#### 🎯 SHORT POSITION ALERTS ⭐")
            else:
                st.markdown("#### 🎯 Short Position Alerts")
            
            if not cover_alerts.empty:
                urgent_alerts = cover_alerts[cover_alerts['urgency_score'] >= 20]
                
                if not urgent_alerts.empty:
                    for idx, row in urgent_alerts.iterrows():
                        if "COVER NOW" in row['alert_level']:
                            alert_color = "#ff4444"
                            urgency_emoji = "🚨🚨🚨"
                        elif "WATCH CLOSELY" in row['alert_level']:
                            alert_color = "#ffa726"
                            urgency_emoji = "⚠️⚠️"
                        elif "MONITOR" in row['alert_level']:
                            alert_color = "#ffcc00"
                            urgency_emoji = "💡"
                        else:
                            continue
                        
                        opacity = "1.0" if is_bearish_market else "0.6"
                        st.markdown(f"""
                        <div style="padding: 12px; margin: 8px 0; background: {alert_color}22; border: 2px solid {alert_color}; border-radius: 8px; opacity: {opacity};">
                            <h4 style="margin: 0; color: {alert_color};">{urgency_emoji} {row['category']}</h4>
                            <p style="margin: 5px 0; font-size: 1.1rem; color: {alert_color};"><strong>{row['alert_level']}</strong></p>
                            <p style="margin: 5px 0;">Urgency: {row['urgency_score']:.0f}/100</p>
                            <p style="margin: 5px 0; font-size: 0.85rem;">{row['signals']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No urgent short position alerts")
            else:
                st.info("⚪ No alert data available")
        
        with col2:
            if is_bullish_market:
                st.markdown("#### 🚀 LONG POSITION ALERTS ⭐")
            else:
                st.markdown("#### 🚀 Long Position Alerts")
            
            if not exit_alerts_longs.empty:
                urgent_alerts = exit_alerts_longs[exit_alerts_longs['urgency_score'] >= 20]
                
                if not urgent_alerts.empty:
                    for idx, row in urgent_alerts.iterrows():
                        if "EXIT NOW" in row['alert_level']:
                            alert_color = "#ff4444"
                            urgency_emoji = "🚨🚨🚨"
                        elif "WATCH CLOSELY" in row['alert_level']:
                            alert_color = "#ffa726"
                            urgency_emoji = "⚠️⚠️"
                        elif "MONITOR" in row['alert_level']:
                            alert_color = "#ffcc00"
                            urgency_emoji = "💡"
                        else:
                            continue
                        
                        opacity = "1.0" if is_bullish_market else "0.6"
                        st.markdown(f"""
                        <div style="padding: 12px; margin: 8px 0; background: {alert_color}22; border: 2px solid {alert_color}; border-radius: 8px; opacity: {opacity};">
                            <h4 style="margin: 0; color: {alert_color};">{urgency_emoji} {row['category']}</h4>
                            <p style="margin: 5px 0; font-size: 1.1rem; color: {alert_color};"><strong>{row['alert_level']}</strong></p>
                            <p style="margin: 5px 0;">Urgency: {row['urgency_score']:.0f}/100</p>
                            <p style="margin: 5px 0; font-size: 0.85rem;">{row['signals']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No urgent long position alerts")
            else:
                st.info("⚪ No alert data available")
        
        # Show all alerts in expandable tables
        with st.expander("📊 View All Alert Levels"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Short Position Alerts**")
                if not cover_alerts.empty:
                    display_cols = ['category', 'alert_level', 'urgency_score', 'signals']
                    st.dataframe(
                        cover_alerts[display_cols].style.background_gradient(
                            subset=['urgency_score'], cmap='YlOrRd'
                        ),
                        use_container_width=True,
                        hide_index=True
                    )
            
            with col2:
                st.markdown("**Long Position Alerts**")
                if not exit_alerts_longs.empty:
                    display_cols = ['category', 'alert_level', 'urgency_score', 'signals']
                    st.dataframe(
                        exit_alerts_longs[display_cols].style.background_gradient(
                            subset=['urgency_score'], cmap='YlOrRd'
                        ),
                        use_container_width=True,
                        hide_index=True
                    )
        
        st.markdown("---")
        
        # ========================================
        # BEST CANDIDATES - NEW! (BOTH DIRECTIONS)
        # ========================================
        
        st.markdown("### 🎯 BEST TRADING OPPORTUNITIES")
        st.caption("All 4 signals combined into one actionable score - for BOTH directions!")
        
        # Create tabs for Shorts vs Longs
        if is_bearish_market:
            cand_tab1, cand_tab2 = st.tabs(["🎯 SHORTS ⭐ (Focus Here)", "🚀 Longs (Secondary)"])
        elif is_bullish_market:
            cand_tab1, cand_tab2 = st.tabs(["🚀 LONGS ⭐ (Focus Here)", "🎯 Shorts (Secondary)"])
        else:
            cand_tab1, cand_tab2 = st.tabs(["🎯 SHORT Candidates", "🚀 LONG Candidates"])
        
        # TAB 1 - Primary direction based on market
        with cand_tab1:
            if is_bearish_market or (not is_bullish_market and not short_candidates.empty):
                # Show SHORT candidates
                st.markdown("#### 🔴 Top Short Opportunities")
                
                if not short_candidates.empty:
                    top_shorts = short_candidates.head(10)
                    
                    for idx, row in top_shorts.iterrows():
                        if row['short_score'] >= 70:
                            box_color = "#ff4444"
                        elif row['short_score'] >= 50:
                            box_color = "#ff6666"
                        elif row['short_score'] >= 30:
                            box_color = "#ff8888"
                        else:
                            box_color = "#808080"
                        
                        st.markdown(f"""
                        <div style="padding: 15px; margin: 10px 0; background: {box_color}22; border-left: 4px solid {box_color}; border-radius: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h3 style="margin: 0;">#{row['rank']}. {row['category']}</h3>
                                <span style="font-size: 1.5rem; color: {box_color};">{row['confidence']}</span>
                            </div>
                            <p style="margin: 10px 0;">
                                <strong>Short Score:</strong> {row['short_score']:.0f}/100 | 
                                <strong>Action:</strong> <span style="color: {box_color};">{row['action']}</span>
                            </p>
                            <p style="margin: 5px 0; padding: 10px; background: {box_color}11; border-radius: 5px;">
                                <strong>Position Size:</strong> {row['position_size']}<br/>
                                <strong>Bearish Signals:</strong> {row['num_bearish_signals']}<br/>
                                <strong>Confidence Factors:</strong> {row['confidence_factors']}
                            </p>
                            <p style="margin: 10px 0; font-size: 0.85rem; color: #ccc;">
                                {row['signals']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with st.expander("📊 View Complete Short Candidates Table"):
                        display_cols = ['rank', 'category', 'short_score', 'confidence', 'action', 
                                       'position_size', 'num_bearish_signals', 'confidence_factors']
                        st.dataframe(
                            short_candidates[display_cols].style.background_gradient(
                                subset=['short_score'], cmap='Reds'
                            ),
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    st.warning("⚠️ Not enough data for short candidates")
            
            else:
                # Show LONG candidates
                st.markdown("#### 🟢 Top Long Opportunities")
                
                if not long_candidates.empty:
                    top_longs = long_candidates.head(10)
                    
                    for idx, row in top_longs.iterrows():
                        if row['long_score'] >= 70:
                            box_color = "#00ff88"
                        elif row['long_score'] >= 50:
                            box_color = "#00cc66"
                        elif row['long_score'] >= 30:
                            box_color = "#00aa44"
                        else:
                            box_color = "#808080"
                        
                        st.markdown(f"""
                        <div style="padding: 15px; margin: 10px 0; background: {box_color}22; border-left: 4px solid {box_color}; border-radius: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h3 style="margin: 0;">#{row['rank']}. {row['category']}</h3>
                                <span style="font-size: 1.5rem; color: {box_color};">{row['confidence']}</span>
                            </div>
                            <p style="margin: 10px 0;">
                                <strong>Long Score:</strong> {row['long_score']:.0f}/100 | 
                                <strong>Action:</strong> <span style="color: {box_color};">{row['action']}</span>
                            </p>
                            <p style="margin: 5px 0; padding: 10px; background: {box_color}11; border-radius: 5px;">
                                <strong>Position Size:</strong> {row['position_size']}<br/>
                                <strong>Bullish Signals:</strong> {row['num_bullish_signals']}<br/>
                                <strong>Confidence Factors:</strong> {row['confidence_factors']}
                            </p>
                            <p style="margin: 10px 0; font-size: 0.85rem; color: #ccc;">
                                {row['signals']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with st.expander("📊 View Complete Long Candidates Table"):
                        display_cols = ['rank', 'category', 'long_score', 'confidence', 'action', 
                                       'position_size', 'num_bullish_signals', 'confidence_factors']
                        st.dataframe(
                            long_candidates[display_cols].style.background_gradient(
                                subset=['long_score'], cmap='Greens'
                            ),
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    st.warning("⚠️ Not enough data for long candidates")
        
        # TAB 2 - Secondary direction
        with cand_tab2:
            if is_bearish_market or (not is_bullish_market and not short_candidates.empty):
                # Show LONG candidates as secondary
                st.markdown("#### 🟢 Long Opportunities (Less Favorable in Bearish Market)")
                st.caption("⚠️ These longs may work but shorts are favored based on market bias")
                
                if not long_candidates.empty:
                    top_longs = long_candidates.head(5)
                    
                    for idx, row in top_longs.iterrows():
                        box_color = "#808080"  # Grayed out
                        
                        st.markdown(f"""
                        <div style="padding: 12px; margin: 8px 0; background: {box_color}11; border-left: 2px solid {box_color}; border-radius: 6px; opacity: 0.7;">
                            <h4 style="margin: 0;">#{row['rank']}. {row['category']}</h4>
                            <p style="margin: 5px 0;">
                                Long Score: {row['long_score']:.0f}/100 | {row['action']}
                            </p>
                            <p style="margin: 5px 0; font-size: 0.8rem;">
                                {row['position_size']} | {row['num_bullish_signals']} bullish signals
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("⚪ No long candidates")
            
            else:
                # Show SHORT candidates as secondary
                st.markdown("#### 🔴 Short Opportunities (Less Favorable in Bullish Market)")
                st.caption("⚠️ These shorts may work but longs are favored based on market bias")
                
                if not short_candidates.empty:
                    top_shorts = short_candidates.head(5)
                    
                    for idx, row in top_shorts.iterrows():
                        box_color = "#808080"  # Grayed out
                        
                        st.markdown(f"""
                        <div style="padding: 12px; margin: 8px 0; background: {box_color}11; border-left: 2px solid {box_color}; border-radius: 6px; opacity: 0.7;">
                            <h4 style="margin: 0;">#{row['rank']}. {row['category']}</h4>
                            <p style="margin: 5px 0;">
                                Short Score: {row['short_score']:.0f}/100 | {row['action']}
                            </p>
                            <p style="margin: 5px 0; font-size: 0.8rem;">
                                {row['position_size']} | {row['num_bearish_signals']} bearish signals
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("⚪ No short candidates")
        
        st.markdown("---")
        
        # ========================================
        # ORIGINAL FEATURES (Turn Signals, Momentum, etc.)
        # ========================================
        
        st.markdown("### 📈 DETAILED ANALYSIS")
        st.caption("Dive deeper into individual signal types")
        
        # ========================================
        # 1. TURN SIGNALS - Most Important First!
        # ========================================
        
        st.markdown("#### 🚨 Turn Signals - Early Reversal Patterns")
        st.caption("Categories showing momentum shifts BEFORE price follows - your edge at market turns!")
        
        if not turn_signals.empty:
            # Filter to only active signals
            active_signals = turn_signals[turn_signals['signal_type'] != "Neutral"]
            
            if not active_signals.empty:
                # Display top signals
                for idx, row in active_signals.head(10).iterrows():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    
                    with col1:
                        st.markdown(f"**{row['category']}**")
                        st.caption(row['signal_type'])
                    
                    with col2:
                        st.metric("Strength", f"{row['current_strength']:.1f}")
                    
                    with col3:
                        st.metric("Momentum", f"{row['current_momentum']:.2f}%")
                    
                    with col4:
                        # Signal strength gauge
                        if row['signal_strength'] >= 70:
                            st.markdown("🔥🔥🔥")
                        elif row['signal_strength'] >= 40:
                            st.markdown("🔥🔥")
                        else:
                            st.markdown("🔥")
                    
                    st.markdown("---")
            else:
                st.info("✅ No active turn signals - market in steady state")
        else:
            st.warning("⚠️ Not enough data to detect turn signals")
        
        st.markdown("---")
        
        # ========================================
        # 2. MOMENTUM RANKINGS
        # ========================================
        
        st.markdown("### 🚀 Momentum Rankings - Fastest Movers")
        st.caption("Which categories are accelerating/decelerating fastest")
        
        if not momentum_rankings.empty:
            # Create two columns for leaders and laggards
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🟢 Top Accelerators")
                top_5 = momentum_rankings.head(5)
                
                for idx, row in top_5.iterrows():
                    strength_color = get_trend_color(row['current_strength'])
                    st.markdown(f"""
                    <div style="padding: 10px; margin: 5px 0; background: linear-gradient(90deg, {strength_color}22, transparent); border-left: 3px solid {strength_color}; border-radius: 5px;">
                        <strong>{row['rank']}. {row['category']}</strong><br/>
                        Strength: {row['current_strength']:.1f} ({row['strength_change']:+.1f})<br/>
                        Rate: {row['strength_rate']:+.2f} pts/hr | Acceleration: {row['acceleration_score']:+.1f}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 🔴 Top Decelerators")
                bottom_5 = momentum_rankings.tail(5).sort_values('acceleration_score')
                
                for idx, row in bottom_5.iterrows():
                    strength_color = get_trend_color(row['current_strength'])
                    st.markdown(f"""
                    <div style="padding: 10px; margin: 5px 0; background: linear-gradient(90deg, {strength_color}22, transparent); border-left: 3px solid {strength_color}; border-radius: 5px;">
                        <strong>{row['category']}</strong><br/>
                        Strength: {row['current_strength']:.1f} ({row['strength_change']:+.1f})<br/>
                        Rate: {row['strength_rate']:+.2f} pts/hr | Acceleration: {row['acceleration_score']:+.1f}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Full rankings table
            with st.expander("📊 View Full Rankings Table"):
                display_cols = ['rank', 'category', 'current_strength', 'strength_change', 
                               'strength_rate', 'acceleration_score', 'momentum_change']
                st.dataframe(
                    momentum_rankings[display_cols].style.background_gradient(
                        subset=['acceleration_score'], cmap='RdYlGn'
                    ),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning("⚠️ Not enough data for momentum rankings")
        
        st.markdown("---")
        
        # ========================================
        # 3. DIVERGENCE DETECTOR
        # ========================================
        
        st.markdown("### 🔍 Divergence Detector - Breaking from the Pack")
        st.caption("Categories moving differently than major_coins (BTC) - early signs of rotation")
        
        if not divergences.empty:
            # Show strong divergences
            strong_divergences = divergences[
                (divergences['divergence_type'].str.contains('Strong')) | 
                (divergences['divergence_type'].str.contains('Moderate'))
            ]
            
            if not strong_divergences.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🟢 Strong Outperformers")
                    outperformers = strong_divergences[
                        strong_divergences['divergence_type'].str.contains('Outperformance')
                    ].head(5)
                    
                    for idx, row in outperformers.iterrows():
                        st.markdown(f"""
                        **{row['category']}**  
                        Divergence: {row['divergence_score']:+.1f} | {row['divergence_type']}  
                        Category: {row['category_strength_change']:+.1f} vs Market: {row['market_strength_change']:+.1f}
                        """)
                        st.progress(min(1.0, max(0.0, (row['divergence_score'] + 30) / 60)))
                
                with col2:
                    st.markdown("#### 🔴 Strong Underperformers")
                    underperformers = strong_divergences[
                        strong_divergences['divergence_type'].str.contains('Underperformance')
                    ].tail(5)
                    
                    for idx, row in underperformers.iterrows():
                        st.markdown(f"""
                        **{row['category']}**  
                        Divergence: {row['divergence_score']:+.1f} | {row['divergence_type']}  
                        Category: {row['category_strength_change']:+.1f} vs Market: {row['market_strength_change']:+.1f}
                        """)
                        st.progress(min(1.0, max(0.0, (30 - abs(row['divergence_score'])) / 60)))
            else:
                st.info("✅ All categories moving in line with the market")
            
            # Full divergence table
            with st.expander("📊 View All Divergences"):
                display_cols = ['category', 'divergence_score', 'divergence_type', 
                               'strength_divergence', 'perf_divergence']
                st.dataframe(
                    divergences[display_cols].style.background_gradient(
                        subset=['divergence_score'], cmap='RdYlGn'
                    ),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning("⚠️ Need major_coins data for divergence detection")
        
        st.markdown("---")
        
        # ========================================
        # 4. RELATIVE STRENGTH vs BTC
        # ========================================
        
        st.markdown("### 📊 Relative Strength vs Market")
        st.caption("Which categories are outperforming/underperforming major_coins (BTC)")
        
        if not relative_strength.empty:
            # Create RS chart
            fig = go.Figure()
            
            # Sort by RS score
            rs_sorted = relative_strength.sort_values('rs_score', ascending=True)
            
            # Color code by RS rating
            colors = []
            for rating in rs_sorted['rs_rating']:
                if 'Strong Outperformer' in rating:
                    colors.append('#00ff88')
                elif 'Outperformer' in rating:
                    colors.append('#00cc66')
                elif 'In Line' in rating:
                    colors.append('#808080')
                elif 'Weak' in rating:
                    colors.append('#ff4444')
                else:
                    colors.append('#ff6666')
            
            fig.add_trace(go.Bar(
                y=rs_sorted['category'],
                x=rs_sorted['rs_score'],
                orientation='h',
                marker=dict(color=colors),
                text=rs_sorted['rs_score'].apply(lambda x: f'{x:.1f}'),
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>' +
                             'RS Score: %{x:.1f}<br>' +
                             '<extra></extra>'
            ))
            
            fig.update_layout(
                title="Relative Strength Scores (50 = Neutral)",
                xaxis_title="RS Score",
                yaxis_title="Category",
                height=max(400, len(rs_sorted) * 30),
                showlegend=False
            )
            
            # Add reference line at 50 (neutral)
            fig.add_vline(x=50, line_dash="dash", line_color="white", opacity=0.5)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Top and bottom RS
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏆 Strongest RS")
                top_rs = relative_strength.head(5)
                
                for idx, row in top_rs.iterrows():
                    st.markdown(f"""
                    **{row['category']}** - {row['rs_rating']}  
                    RS Score: {row['rs_score']:.1f} | Trend: {row['rs_trend']:+.1f}  
                    Category: {row['category_strength']:.1f} vs Market: {row['market_strength']:.1f}
                    """)
            
            with col2:
                st.markdown("#### ⚠️ Weakest RS")
                bottom_rs = relative_strength.tail(5)
                
                for idx, row in bottom_rs.iterrows():
                    st.markdown(f"""
                    **{row['category']}** - {row['rs_rating']}  
                    RS Score: {row['rs_score']:.1f} | Trend: {row['rs_trend']:+.1f}  
                    Category: {row['category_strength']:.1f} vs Market: {row['market_strength']:.1f}
                    """)
            
            # Full RS table
            with st.expander("📊 View Full Relative Strength Table"):
                display_cols = ['category', 'rs_score', 'rs_rating', 'rs_trend', 
                               'rs_strength', 'rs_performance', 'rs_momentum']
                st.dataframe(
                    relative_strength[display_cols].style.background_gradient(
                        subset=['rs_score'], cmap='RdYlGn'
                    ),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning("⚠️ Need major_coins data for relative strength analysis")
        
        # Add interpretation guide
        st.markdown("---")
        st.markdown("""
        ### 📖 How to Use These Signals
        
        **NEW FEATURES:**
        
        **📊 Market Overview**
        - Shows overall market bias (bullish/bearish/neutral)
        - Risk environment (RISK-ON = favor longs, RISK-OFF = favor shorts)
        - Category distribution and active signals count
        - Use this FIRST to understand the big picture!
        
        **⚡ Quick Trade Summary**
        - Top Shorts: Highest conviction short setups RIGHT NOW
        - Avoid Shorts: Categories showing bullish signals - DO NOT SHORT
        - Perfect for quick decision making
        
        **🔄 Momentum Shifts**
        - 🟢 Bottom Reversals: Categories bouncing from weakness → BUY opportunities
        - 🔴 Top Reversals: Categories losing momentum from strength → SHORT opportunities
        - 🚀 Momentum Explosions: Extreme acceleration → Ride the trend
        - 💥 Momentum Collapses: Extreme deceleration → Short or exit
        - **THIS IS YOUR EDGE** - catch turns before the crowd!
        
        **⚠️ Cover Alerts**
        - 🚨 COVER NOW: Your short is reversing, exit IMMEDIATELY
        - ⚠️ WATCH CLOSELY: Early reversal signs, tighten stops
        - 💡 MONITOR: Keep an eye on it
        - ✅ HOLD SHORT: Safe to hold
        - Prevents you from holding shorts into reversals!
        
        **🎯 Best Short Candidates**
        - Aggregates ALL 4 signals into one score (0-100)
        - 🔥🔥🔥 MAXIMUM Confidence = Max size shorts
        - 🔥🔥 HIGH Confidence = Standard size
        - 🔥 MODERATE Confidence = Reduced size
        - Shows position sizing recommendations
        - One-stop shop for short selection!
        
        ---
        
        **ORIGINAL FEATURES:**
        
        **🚨 Turn Signals** - Your #1 Edge!
        - **Bullish Turn**: Category showing momentum reversal while still down → Early entry opportunity
        - **Bearish Turn**: Category losing momentum while still up → Time to exit
        - **Breakout**: Accelerating through resistance → Ride the trend
        - **Breakdown**: Falling through support → Stay away or short
        
        **🚀 Momentum Rankings**
        - **Top Accelerators**: Leaders at market turns - get in early
        - **Top Decelerators**: Laggards or early warnings - avoid or exit
        - Watch for categories jumping from bottom to top = reversal happening
        
        **🔍 Divergences**
        - **Outperformance**: Category stronger than BTC → Sector rotation into this
        - **Underperformance**: Category weaker than BTC → Money flowing out
        - Use for sector rotation plays when BTC consolidates
        
        **📊 Relative Strength**
        - **Strong RS + Rising**: Leaders - stay in these
        - **Weak RS + Falling**: Laggards - avoid these
        - **RS Turning Up**: Early accumulation - get in before crowd
        - **RS Turning Down**: Distribution starting - get out
        
        ---
        
        **💡 TRADING PLAYBOOK:**
        
        **FOR SHORTING (Your Current Strategy):**
        
        1. **Check Market Bias FIRST**
           - If BEARISH (< 45) → Great shorting environment
           - If BULLISH (> 55) → Be careful, shorts risky
           - If NEUTRAL (45-55) → Selective shorts only
        
        2. **Use Quick Trade Summary**
           - Go straight to "TOP SHORTS" section
           - Pick from categories with 🔥🔥🔥 or 🔥🔥 confidence
           - Avoid anything in "Avoid Shorts" list
        
        3. **Check Momentum Shifts**
           - Look for 🔴 TOP REVERSALS → Perfect short entries
           - Avoid 🟢 BOTTOM REVERSALS → Will bounce
        
        4. **Monitor Cover Alerts**
           - If ANY of your shorts show "COVER NOW" → EXIT IMMEDIATELY
           - "WATCH CLOSELY" → Tighten stops, reduce size
        
        5. **Use Best Short Candidates for Position Sizing**
           - 70+ score → Full size
           - 50-70 → Standard size
           - 30-50 → Reduced size
           - <30 → Skip it
        
        **FOR BUYING/LONGS:**
        
        1. **Check Market Bias**
           - If BULLISH (> 55) → Great long environment
           - Look for RISK-ON signal
        
        2. **Find Bottom Reversals**
           - 🟢 in Momentum Shifts = Early entry
           - 🟢 Bullish Turn Signal = Reversal confirmed
        
        3. **Check RS and Divergence**
           - Rising RS + Outperforming = Strongest longs
        
        **BEST SETUPS (Highest Probability):**
        
        **Perfect Short:**
        - Market Bias: BEARISH
        - Quick Trade Summary: In "TOP SHORTS"
        - Momentum Shift: 🔴 TOP REVERSAL
        - Best Short Candidates: 70+ score with 🔥🔥🔥
        - No Cover Alerts
        = **🎯 MAX SHORT POSITION**
        
        **Perfect Long:**
        - Market Bias: BULLISH
        - Momentum Shift: 🟢 BOTTOM REVERSAL
        - Turn Signal: 🟢 BULLISH TURN
        - RS: Rising and strong
        - In "Avoid Shorts" section
        = **🚀 MAX LONG POSITION**
        
        **Cover Short Immediately If:**
        - Cover Alert: 🚨 COVER NOW
        - Turn Signal flips: 🟢 BULLISH TURN appears
        - Jumps to Top 3 Accelerators
        - Momentum Shift: 🟢 BOTTOM REVERSAL
        = **⚠️ EXIT SHORT NOW!**
        
        ---
        
        **🎯 YOUR WORKFLOW:**
        
        **Morning Routine (6h lookback):**
        1. Check Market Bias → Is it a shorting day?
        2. Review Quick Trade Summary → What to short today
        3. Check Momentum Shifts → Any reversals overnight?
        4. Review Cover Alerts → Any shorts at risk?
        5. Plan your entries from Best Short Candidates
        
        **During Trading (6h lookback):**
        1. Monitor Cover Alerts CONSTANTLY
        2. Watch for Momentum Shifts (new reversals)
        3. Re-check Market Bias if market feels different
        4. Add to positions when signals align
        
        **End of Day (24h or 72h lookback):**
        1. Review what worked / what didn't
        2. Check longer-term trends
        3. Plan tomorrow's watchlist
        4. Set alerts for key categories
        
        ---
        
        **Remember:** 
        - Use 6h for day trading (most responsive)
        - Use 24h for swing trading
        - Use 72h for context/bigger picture
        - When ALL signals align = MAXIMUM CONVICTION
        - When signals conflict = WAIT or reduce size
        
        **You've got professional-grade trading signals - USE THEM!** 🚀
        """)

# ================================================================================
# RUN APP
# ================================================================================

if __name__ == "__main__":
    main()