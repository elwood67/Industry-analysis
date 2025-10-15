#!/usr/bin/env python3
"""
Enhanced Crypto Volume Analyzer Dashboard - Section 1: Core Fixes & Initialization
Fixed price change lookback issues and enhanced data loading
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
import random
from collections import defaultdict
warnings.filterwarnings('ignore')

# ================================================================================
# PAGE CONFIGURATION
# ================================================================================

st.set_page_config(
    page_title="Elwoods CB Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================================
# ENHANCED CUSTOM CSS
# ================================================================================

st.markdown("""
<style>
    /* Main Theme */
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
    
    /* Sentiment Cards */
    .bullish-card {
        border-left-color: #00ff88;
        background: linear-gradient(135deg, #0d2818 0%, #1a3d2a 100%);
    }
    .bearish-card {
        border-left-color: #ff4444;
        background: linear-gradient(135deg, #2d1b1b 0%, #3d1f1f 100%);
    }
    .neutral-card {
        border-left-color: #ffa726;
        background: linear-gradient(135deg, #2d2318 0%, #3d3118 100%);
    }
    
    /* Metrics */
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
    .metric-delta {
        font-size: 0.85rem;
        margin-top: 0.3rem;
        font-weight: 600;
    }
    
    /* Trends */
    .trend-up {
        color: #00ff88;
    }
    .trend-down {
        color: #ff4444;
    }
    .trend-neutral {
        color: #ffa726;
    }
    
    /* Data Quality Indicators */
    .quality-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .quality-high {
        background-color: #00ff88;
        color: #000000;
    }
    .quality-medium {
        background-color: #ffa726;
        color: #000000;
    }
    .quality-low {
        background-color: #ff4444;
        color: #ffffff;
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Market Structure Badge */
    .structure-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    .structure-uptrend {
        background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
        color: #000000;
    }
    .structure-downtrend {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: #ffffff;
    }
    .structure-sideways {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# UTILITY FUNCTIONS - ENHANCED
# ================================================================================

def format_volume(volume: float) -> str:
    """Format volume numbers for display"""
    if volume is None or volume == 0:
        return "$0"
    
    if volume >= 1e9:
        return f"${volume/1e9:.2f}B"
    elif volume >= 1e6:
        return f"${volume/1e6:.2f}M"
    elif volume >= 1e3:
        return f"${volume/1e3:.2f}K"
    else:
        return f"${volume:.2f}"

def format_price(price: float) -> str:
    """Format price based on magnitude"""
    if price is None:
        return "N/A"
    
    # Handle very small prices (meme coins)
    if price < 0.00001:
        return f"{price:.2E}"
    elif price < 0.01:
        return f"${price:.8f}"
    elif price < 1:
        return f"${price:.6f}"
    elif price < 100:
        return f"${price:.4f}"
    else:
        return f"${price:.2f}"

def format_percentage(value: float, include_sign: bool = True, decimals: int = 1) -> str:
    """Format percentage with appropriate sign and precision"""
    if value is None:
        return "N/A"
    
    if include_sign:
        return f"{value:+.{decimals}f}%"
    else:
        return f"{value:.{decimals}f}%"

def get_sentiment_color(ratio: float) -> str:
    """Get color based on sentiment ratio"""
    if ratio >= 0.65:
        return "#00ff88"  # Strong bullish
    elif ratio >= 0.55:
        return "#00cc66"  # Bullish
    elif ratio <= 0.35:
        return "#ff4444"  # Strong bearish
    elif ratio <= 0.45:
        return "#ff6666"  # Bearish
    else:
        return "#ffa726"  # Neutral

def get_quality_badge(rate: float) -> str:
    """Get quality badge HTML based on rate"""
    if rate >= 0.8:
        return '<span class="quality-badge quality-high">HIGH</span>'
    elif rate >= 0.5:
        return '<span class="quality-badge quality-medium">MEDIUM</span>'
    else:
        return '<span class="quality-badge quality-low">LOW</span>'

# ================================================================================
# FIXED PRICE CHANGE FIELD MAPPING
# ================================================================================

def get_price_change_field_for_period(period_hours: int) -> str:
    """Get the exact price change field name for a given period"""
    # Map to exact field names from scanner
    field_map = {
        1: 'price_change_1h',
        2: 'price_change_2h',
        4: 'price_change_4h',
        6: 'price_change_6h',
        8: 'price_change_8h',
        12: 'price_change_12h',
        24: 'price_change_24h'
    }
    return field_map.get(period_hours, 'price_change_24h')

def get_available_price_periods(asset_data: Dict) -> List[int]:
    """Get available price change periods from asset data"""
    periods = []
    for hours in [1, 2, 4, 6, 8, 12, 24]:
        field = f'price_change_{hours}h'
        if field in asset_data and asset_data[field] is not None:
            periods.append(hours)
    return periods

def validate_asset_data(asset_data: Dict) -> bool:
    """Validate that asset has required data fields"""
    required_fields = ['symbol', 'current_price', 'total_dollar_volume']
    return all(field in asset_data for field in required_fields)

# ================================================================================
# ENHANCED DATA LOADING WITH VALIDATION
# ================================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_enhanced_volume_data(file_path: str) -> Optional[Dict]:
    """Load enhanced volume data with validation"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert timestamp strings to datetime objects
        if 'volume_history' in data:
            for entry in data['volume_history']:
                if isinstance(entry.get('timestamp'), str):
                    entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
                if isinstance(entry.get('end_time_used'), str):
                    entry['end_time_used'] = datetime.fromisoformat(entry['end_time_used'])
                
                # Convert asset timestamps
                for asset_data in entry.get('assets', {}).values():
                    if 'last_updated' in asset_data and isinstance(asset_data['last_updated'], str):
                        asset_data['last_updated'] = datetime.fromisoformat(asset_data['last_updated'])
        
        # Validate data structure
        if 'volume_history' in data and data['volume_history']:
            latest = data['volume_history'][-1]
            
            # Check for enhanced fields
            has_quality = 'data_quality' in latest
            has_market_stats = 'market_stats' in latest
            has_enhanced_prices = any('price_change_1h' in asset for asset in latest.get('assets', {}).values())
            
            data['_metadata'] = {
                'has_quality_metrics': has_quality,
                'has_market_stats': has_market_stats,
                'has_enhanced_prices': has_enhanced_prices,
                'scanner_version': 'enhanced' if has_quality else 'legacy'
            }
        
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data(ttl=300)
def get_available_data_files() -> Dict[str, List[str]]:
    """Get all available data files organized by type"""
    files = {
        'hourly': [],
        'snapshots': [],
        'scanner_data': []
    }
    
    # Check hourly volume data
    hourly_pattern = "hourly_volume_data/hourly_volume_data_*.json"
    files['hourly'] = sorted(glob.glob(hourly_pattern), reverse=True)
    
    # Check for snapshots
    snapshot_pattern = "scanner_data/snapshot_*.json"
    files['snapshots'] = sorted(glob.glob(snapshot_pattern), reverse=True)
    
    # Check for persistent scanner data
    if os.path.exists("scanner_data/persistent/hourly/hourly_database.json"):
        files['scanner_data'].append("scanner_data/persistent/hourly/hourly_database.json")
    
    return files

def get_data_freshness(data_entry: Dict) -> Tuple[str, str, float]:
    """Get data freshness status, emoji, and age in hours"""
    try:
        timestamp = data_entry.get('timestamp')
        if isinstance(timestamp, datetime):
            age_hours = (datetime.now() - timestamp).total_seconds() / 3600
            
            if age_hours < 1:
                return "🟢 Live", "success", age_hours
            elif age_hours < 2:
                return "🟡 Recent", "warning", age_hours
            elif age_hours < 6:
                return "🟠 Aging", "warning", age_hours
            else:
                return "🔴 Stale", "error", age_hours
        
    except:
        pass
    
    return "⚫ Unknown", "info", 0

def get_data_quality_score(data_entry: Dict) -> float:
    """Calculate overall data quality score"""
    if 'data_quality' not in data_entry:
        return 0.5  # Legacy data
    
    quality = data_entry['data_quality']
    ticker_rate = quality.get('ticker_success_rate', 0)
    stats_rate = quality.get('stats_success_rate', 0)
    completeness = quality.get('candle_completeness', 0)
    
    # Weighted average
    score = (ticker_rate * 0.4 + stats_rate * 0.3 + completeness * 0.3)
    return score

# ================================================================================
# ENHANCED ASSET FILTERING AND VALIDATION
# ================================================================================

def filter_assets_for_analysis(assets: Dict, min_volume: float = 0, 
                              min_data_points: int = 1,
                              required_fields: List[str] = None) -> List[Dict]:
    """Filter and validate assets for analysis"""
    filtered_assets = []
    
    for symbol, asset_data in assets.items():
        # Basic validation
        if not validate_asset_data(asset_data):
            continue
        
        # Volume filter
        if asset_data.get('total_dollar_volume', 0) < min_volume:
            continue
        
        # Data points filter
        if asset_data.get('data_points', 0) < min_data_points:
            continue
        
        # Required fields filter
        if required_fields:
            if not all(field in asset_data and asset_data[field] is not None 
                      for field in required_fields):
                continue
        
        # Add symbol to data
        asset_data['symbol'] = symbol
        filtered_assets.append(asset_data)
    
    return filtered_assets

def get_asset_categories(assets: Dict) -> Dict[str, List[str]]:
    """Categorize assets including new categories"""
    categories = {
        'major_coins': [],
        'altcoins': [],
        'stablecoins': [],
        'meme_coins': [],
        'defi': [],
        'layer2': [],
        'gaming': []
    }
    
    # Define categories
    major_coins = {'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'LINK', 'LTC', 'BCH', 'AVAX', 'MATIC', 'UNI', 'ATOM'}
    stablecoins = {'USDC', 'USDT', 'DAI', 'BUSD', 'TUSD', 'GUSD', 'USDD', 'FRAX', 'PYUSD'}
    meme_coins = {'DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'MEM', 'BABYDOGE', 'GIGA', 'TOSHI', 'DEGEN', 'TURBO', 'KEYCAT', 'DOGINME' 'PENGU', 'TRUMP', 'MIGGLES', 'BOME', 'MOG'}
    defi_coins = {'AAVE', 'COMP', 'CRV', 'MKR', 'SNX', 'SUSHI', 'YFI', 'UNI', 'BAL', 'LDO'}
    layer2_coins = {'MATIC', 'ARB', 'OP', 'IMX', 'LRC', 'SKL', 'CTSI'}
    gaming_coins = {'AXS', 'SAND', 'MANA', 'ENJ', 'GALA', 'ILV', 'ALICE', 'SLP'}
    
    for symbol in assets.keys():
        if symbol in major_coins:
            categories['major_coins'].append(symbol)
        elif symbol in stablecoins:
            categories['stablecoins'].append(symbol)
        elif symbol in meme_coins:
            categories['meme_coins'].append(symbol)
        elif symbol in defi_coins:
            categories['defi'].append(symbol)
        elif symbol in layer2_coins:
            categories['layer2'].append(symbol)
        elif symbol in gaming_coins:
            categories['gaming'].append(symbol)
        else:
            categories['altcoins'].append(symbol)
    
    return categories

# ================================================================================
# MARKET STRUCTURE ANALYSIS
# ================================================================================

def analyze_market_structure(data_entry: Dict, lookback_hours: int = 24) -> Dict[str, Any]:
    """
    Enhanced market structure analysis with configurable lookback periods
    Updated with better thresholds for crypto markets - NO VOLATILE CATEGORY
    """
    
    assets = data_entry.get('assets', {})
    
    # Initialize structure counts for different patterns
    structure_analysis = {
        'strong_uptrend': 0,    # Score > 0.5
        'uptrend': 0,            # Score 0.3 to 0.5
        'weak_uptrend': 0,       # Score 0.1 to 0.3
        'strong_downtrend': 0,   # Score < -0.5
        'downtrend': 0,          # Score -0.5 to -0.3
        'weak_downtrend': 0,     # Score -0.3 to -0.1
        'sideways': 0,           # Score -0.1 to 0.1
        'unknown': 0
    }
    
    # Additional metrics
    momentum_scores = []
    volatility_scores = []
    trend_strengths = []
    asset_details = []  # Store details for debugging
    
    for symbol, asset_data in assets.items():
        # Determine structure based on available timeframe data
        structure_score = calculate_structure_score(asset_data, lookback_hours)
        
        # Get volatility for metrics (but not for classification)
        volatility = asset_data.get('volatility_24h', 0)
        
        # Classify based on score with adjusted thresholds
        if structure_score >= 0.5:
            structure_analysis['strong_uptrend'] += 1
            classification = 'strong_uptrend'
        elif structure_score >= 0.3:
            structure_analysis['uptrend'] += 1
            classification = 'uptrend'
        elif structure_score >= 0.1:
            structure_analysis['weak_uptrend'] += 1
            classification = 'weak_uptrend'
        elif structure_score <= -0.5:
            structure_analysis['strong_downtrend'] += 1
            classification = 'strong_downtrend'
        elif structure_score <= -0.3:
            structure_analysis['downtrend'] += 1
            classification = 'downtrend'
        elif structure_score <= -0.1:
            structure_analysis['weak_downtrend'] += 1
            classification = 'weak_downtrend'
        else:
            structure_analysis['sideways'] += 1
            classification = 'sideways'
        
        # Store details for potential debugging
        asset_details.append({
            'symbol': symbol,
            'score': structure_score,
            'classification': classification,
            'price_24h': asset_data.get('price_change_24h', 0),
            'momentum': asset_data.get('bullish_momentum', 0.5)
        })
        
        # Collect additional metrics
        momentum_scores.append(asset_data.get('bullish_momentum', 0.5))
        volatility_scores.append(volatility)
        trend_strengths.append(abs(structure_score))
    
    total = sum(structure_analysis.values())
    
    if total == 0:
        return {
            'dominant_structure': 'unknown',
            'strength': 0,
            'detailed_breakdown': structure_analysis,
            'market_character': 'insufficient_data'
        }
    
    # Calculate percentages
    structure_percentages = {k: (v/total)*100 for k, v in structure_analysis.items()}
    
    # Determine dominant structure with more nuance
    bullish_total = (structure_analysis['strong_uptrend'] + 
                     structure_analysis['uptrend'] + 
                     structure_analysis['weak_uptrend'])
    bearish_total = (structure_analysis['strong_downtrend'] + 
                     structure_analysis['downtrend'] + 
                     structure_analysis['weak_downtrend'])
    sideways_total = structure_analysis['sideways']
    
    # Calculate market character with better thresholds
    if bullish_total > total * 0.5:
        market_character = 'bullish_trending'
        dominant = 'uptrend'
    elif bearish_total > total * 0.5:
        market_character = 'bearish_trending'
        dominant = 'downtrend'
    elif sideways_total > total * 0.4:
        market_character = 'consolidating'
        dominant = 'sideways'
    else:
        market_character = 'mixed'
        dominant = 'mixed'
    
    # Calculate trend strength (0-1 scale)
    if dominant in ['uptrend', 'downtrend']:
        strong_count = structure_analysis[f'strong_{dominant}']
        normal_count = structure_analysis[dominant]
        weak_count = structure_analysis[f'weak_{dominant}']
        total_trending = strong_count + normal_count + weak_count
        
        if total_trending > 0:
            # Weighted average: strong=1.0, normal=0.6, weak=0.3
            strength = (strong_count * 1.0 + normal_count * 0.6 + weak_count * 0.3) / total_trending
        else:
            strength = 0
    else:
        strength = sideways_total / total if total > 0 else 0
    
    # Calculate additional market metrics
    avg_momentum = np.mean(momentum_scores) if momentum_scores else 0.5
    avg_volatility = np.mean(volatility_scores) if volatility_scores else 0
    avg_trend_strength = np.mean(trend_strengths) if trend_strengths else 0
    
    # Add debug info for top movers to help validate
    top_bullish = sorted(asset_details, key=lambda x: x['score'], reverse=True)[:5]
    top_bearish = sorted(asset_details, key=lambda x: x['score'])[:5]
    
    return {
        'dominant_structure': dominant,
        'market_character': market_character,
        'strength': strength,
        'lookback_hours': lookback_hours,
        'detailed_breakdown': structure_analysis,
        'percentages': structure_percentages,
        'bullish_assets': bullish_total,
        'bearish_assets': bearish_total,
        'ranging_assets': sideways_total,
        'bullish_pct': (bullish_total/total)*100 if total > 0 else 0,
        'bearish_pct': (bearish_total/total)*100 if total > 0 else 0,
        'sideways_pct': (sideways_total/total)*100 if total > 0 else 0,
        'avg_momentum': avg_momentum,
        'avg_volatility': avg_volatility,
        'avg_trend_strength': avg_trend_strength,
        'total_assets': total,
        'top_bullish': top_bullish,
        'top_bearish': top_bearish
    }

def calculate_structure_score(asset_data: Dict, lookback_hours: int) -> float:
    """
    Calculate a structure score for an individual asset
    Returns: -1.0 (strong downtrend) to +1.0 (strong uptrend)
    """
    
    score = 0.0
    weights_total = 0.0
    
    # Map lookback hours to appropriate price change fields
    # Adjust weights based on lookback period
    if lookback_hours <= 24:
        timeframe_weights = {
            1: 0.05,
            2: 0.05,
            4: 0.10,
            6: 0.15,
            8: 0.20,
            12: 0.20,
            24: 0.25
        }
    elif lookback_hours <= 48:
        timeframe_weights = {
            1: 0.02,
            2: 0.03,
            4: 0.05,
            6: 0.10,
            8: 0.15,
            12: 0.25,
            24: 0.40
        }
    else:  # 72 hours
        timeframe_weights = {
            1: 0.01,
            2: 0.02,
            4: 0.04,
            6: 0.08,
            8: 0.10,
            12: 0.25,
            24: 0.50
        }
    
    # Calculate weighted score based on price changes
    price_scores = []
    for hours, weight in timeframe_weights.items():
        if hours <= lookback_hours:
            price_field = f'price_change_{hours}h'
            if price_field in asset_data and asset_data[price_field] is not None:
                price_change = asset_data[price_field]
                
                # More aggressive normalization for better distribution
                # Most crypto moves are within ±10% daily, so use that as baseline
                if hours <= 6:
                    normalized_change = price_change / 5  # ±5% for short term
                elif hours <= 12:
                    normalized_change = price_change / 10  # ±10% for medium term
                else:
                    normalized_change = price_change / 20  # ±20% for 24h
                
                # Cap at ±1 but allow the full range
                normalized_change = max(-1, min(1, normalized_change))
                
                score += normalized_change * weight
                weights_total += weight
                price_scores.append(price_change)
    
    # Include momentum in structure calculation with more weight
    if 'bullish_momentum' in asset_data:
        momentum = asset_data['bullish_momentum']
        # Convert 0-1 momentum to -1 to 1 scale
        momentum_score = (momentum - 0.5) * 2
        
        # Give momentum more weight if we have limited price data
        momentum_weight = 0.3 if len(price_scores) < 3 else 0.15
        
        score += momentum_score * momentum_weight
        weights_total += momentum_weight
    
    # Add volume trend indicator if available
    if 'recent_green_candles' in asset_data and 'recent_red_candles' in asset_data:
        green = asset_data.get('recent_green_candles', 0)
        red = asset_data.get('recent_red_candles', 0)
        if green + red > 0:
            volume_trend_score = (green - red) / (green + red)
            score += volume_trend_score * 0.1
            weights_total += 0.1
    
    # Include volatility as a modifier (but less aggressively)
    if 'volatility_24h' in asset_data and asset_data['volatility_24h'] > 0:
        volatility = asset_data['volatility_24h']
        if volatility > 30:  # Very high volatility
            score *= 0.8  # Reduce trend confidence slightly
        elif volatility < 5:  # Very low volatility
            score *= 1.2  # Increase confidence in trend
    
    # Normalize final score
    if weights_total > 0:
        score = score / weights_total
    
    # Additional classification based on multiple timeframes
    # If we have enough data, check for consistency
    if len(price_scores) >= 3:
        # Check if all timeframes agree on direction
        all_positive = all(p > 0 for p in price_scores)
        all_negative = all(p < 0 for p in price_scores)
        
        if all_positive:
            score = max(score, 0.3)  # Ensure at least weak uptrend
        elif all_negative:
            score = min(score, -0.3)  # Ensure at least weak downtrend
    
    return score
   

# ================================================================================
# MOMENTUM DIVERGENCE DETECTION
# ================================================================================

def detect_momentum_divergence(volume_history: List[Dict], lookback: int = 4) -> Dict[str, Any]:
    """Detect momentum divergences in recent data"""
    if len(volume_history) < lookback:
        return {'has_divergence': False}
    
    recent = volume_history[-lookback:]
    
    # Extract metrics
    volumes = [entry['total_dollar_volume'] for entry in recent]
    momentums = [entry['momentum_metrics']['bullish_volume_ratio'] for entry in recent]
    
    # Calculate trends
    volume_trend = (volumes[-1] - volumes[0]) / volumes[0] if volumes[0] > 0 else 0
    momentum_trend = momentums[-1] - momentums[0]
    
    # Detect divergence
    bullish_divergence = volume_trend > 0.1 and momentum_trend < -0.05  # Volume up, momentum down
    bearish_divergence = volume_trend < -0.1 and momentum_trend > 0.05  # Volume down, momentum up
    
    return {
        'has_divergence': bullish_divergence or bearish_divergence,
        'type': 'bullish' if bullish_divergence else 'bearish' if bearish_divergence else None,
        'volume_trend': volume_trend,
        'momentum_trend': momentum_trend,
        'strength': abs(volume_trend - momentum_trend) if bullish_divergence or bearish_divergence else 0
    }

# ================================================================================
# SESSION STATE INITIALIZATION
# ================================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.selected_lookback = 24
        st.session_state.selected_category = 'all'
        st.session_state.min_volume_filter = 0
        st.session_state.show_quality_metrics = True
        st.session_state.auto_refresh = False
        st.session_state.last_refresh = datetime.now()
        st.session_state.theme = 'dark'
        st.session_state.analysis_view = 'overview'

# ================================================================================
# END OF SECTION 1
# ================================================================================

# ================================================================================
# SECTION 2: ENHANCED MARKET OVERVIEW & REAL-TIME METRICS
# New summary cards, market health indicators, and advanced overview
# ================================================================================

# ================================================================================
# MARKET OVERVIEW COMPONENTS
# ================================================================================

def create_market_health_card(data_entry: Dict, volume_history: List[Dict] = None) -> None:
    """Create comprehensive market health indicator card with trends"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate market health score
    momentum = data_entry['momentum_metrics']['bullish_volume_ratio']
    breadth = data_entry['momentum_metrics'].get('market_breadth', 0.5)
    quality_score = get_data_quality_score(data_entry)
    
    # Composite health score
    health_score = (momentum * 0.4 + breadth * 0.4 + quality_score * 0.2) * 100
    
    # Calculate trends if we have history
    health_trend = None
    ad_trend = None
    vol_trend = None
    
    if volume_history and len(volume_history) >= 2:
        # Get previous entry
        prev_entry = volume_history[-2]
        
        # Calculate previous health score
        prev_momentum = prev_entry['momentum_metrics']['bullish_volume_ratio']
        prev_breadth = prev_entry['momentum_metrics'].get('market_breadth', 0.5)
        prev_quality = get_data_quality_score(prev_entry)
        prev_health = (prev_momentum * 0.4 + prev_breadth * 0.4 + prev_quality * 0.2) * 100
        
        # Calculate trends
        health_trend = health_score - prev_health
        
        # A/D ratio trend
        advance_decline = data_entry['momentum_metrics'].get('advance_decline_ratio', 1)
        prev_ad = prev_entry['momentum_metrics'].get('advance_decline_ratio', 1)
        ad_trend = advance_decline - prev_ad
        
        # Volatility trend
        volatility = data_entry.get('market_stats', {}).get('avg_volatility', 0)
        prev_vol = prev_entry.get('market_stats', {}).get('avg_volatility', 0)
        vol_trend = volatility - prev_vol
    
    # Determine health status
    if health_score >= 70:
        health_status = "🟢 Excellent"
        health_class = "bullish-card"
    elif health_score >= 50:
        health_status = "🟡 Good"
        health_class = "neutral-card"
    else:
        health_status = "🔴 Poor"
        health_class = "bearish-card"
    
    with col1:
        # Add trend indicator
        if health_trend is not None:
            if health_trend > 0:
                trend_icon = "↑"
                trend_color = "#00ff88"
            elif health_trend < 0:
                trend_icon = "↓"
                trend_color = "#ff4444"
            else:
                trend_icon = "→"
                trend_color = "#ffa726"
            
            trend_text = f"""
            <div style="font-size: 0.8rem; color: {trend_color};">
                {trend_icon} {health_trend:+.1f}
            </div>
            """
        else:
            trend_text = ""
        
        st.markdown(f"""
        <div class="metric-card {health_class}">
            <div class="big-metric">{health_score:.0f}</div>
            <div class="metric-label">Market Health Score</div>
            <div class="metric-delta">{health_status}</div>
            {trend_text}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        advance_decline = data_entry['momentum_metrics'].get('advance_decline_ratio', 1)
        if advance_decline > 1.5:
            ad_status = "Strong Advance"
            ad_class = "trend-up"
        elif advance_decline < 0.66:
            ad_status = "Strong Decline"
            ad_class = "trend-down"
        else:
            ad_status = "Balanced"
            ad_class = "trend-neutral"
        
        # Add trend indicator
        if ad_trend is not None:
            if ad_trend > 0.1:
                trend_icon = "↑"
                trend_color = "#00ff88"
            elif ad_trend < -0.1:
                trend_icon = "↓"
                trend_color = "#ff4444"
            else:
                trend_icon = "→"
                trend_color = "#ffa726"
            
            trend_text = f"""
            <div style="font-size: 0.8rem; color: {trend_color};">
                {trend_icon} {ad_trend:+.2f}
            </div>
            """
        else:
            trend_text = ""
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-metric">{advance_decline:.2f}</div>
            <div class="metric-label">Advance/Decline</div>
            <div class="metric-delta {ad_class}">{ad_status}</div>
            {trend_text}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        volatility = data_entry.get('market_stats', {}).get('avg_volatility', 0)
        if volatility > 10:
            vol_status = "High Volatility"
            vol_emoji = "🔥"
        elif volatility > 5:
            vol_status = "Moderate"
            vol_emoji = "⚡"
        else:
            vol_status = "Low"
            vol_emoji = "😴"
        
        # Add trend indicator
        if vol_trend is not None:
            if vol_trend > 1:
                trend_icon = "↑"
                trend_color = "#ff4444"  # Rising volatility is typically concerning
            elif vol_trend < -1:
                trend_icon = "↓"
                trend_color = "#00ff88"  # Decreasing volatility is typically good
            else:
                trend_icon = "→"
                trend_color = "#ffa726"
            
            trend_text = f"""
            <div style="font-size: 0.8rem; color: {trend_color};">
                {trend_icon} {vol_trend:+.1f}%
            </div>
            """
        else:
            trend_text = ""
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-metric">{volatility:.1f}%</div>
            <div class="metric-label">Avg Volatility {vol_emoji}</div>
            <div class="metric-delta">{vol_status}</div>
            {trend_text}
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        freshness_text, freshness_type, age_hours = get_data_freshness(data_entry)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-metric">{age_hours:.1f}h</div>
            <div class="metric-label">Data Age</div>
            <div class="metric-delta">{freshness_text}</div>
        </div>
        """, unsafe_allow_html=True)


def create_enhanced_momentum_cards(data_entry: Dict) -> None:
    """Create enhanced momentum indicator cards"""
    st.subheader("🎯 Bullish Volume Senmtiment")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        overall = data_entry['momentum_metrics']['bullish_volume_ratio']
        card_class = "bullish-card" if overall > 0.6 else "bearish-card" if overall < 0.4 else "neutral-card"
        
        st.markdown(f"""
        <div class="metric-card {card_class}">
            <div class="big-metric">{overall:.1%}</div>
            <div class="metric-label">Overall Market</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        large_cap = data_entry['momentum_metrics']['large_cap_sentiment']
        card_class = "bullish-card" if large_cap > 0.6 else "bearish-card" if large_cap < 0.4 else "neutral-card"
        
        st.markdown(f"""
        <div class="metric-card {card_class}">
            <div class="big-metric">{large_cap:.1%}</div>
            <div class="metric-label">Large Cap</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        alt = data_entry['momentum_metrics']['alt_momentum']
        card_class = "bullish-card" if alt > 0.6 else "bearish-card" if alt < 0.4 else "neutral-card"
        
        st.markdown(f"""
        <div class="metric-card {card_class}">
            <div class="big-metric">{alt:.1%}</div>
            <div class="metric-label">Altcoins</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Calculate meme coin momentum if available
        meme_data = data_entry['volume_by_category'].get('meme_coins', {})
        if meme_data:
            meme_green = meme_data.get('green_candles', 0)
            meme_red = meme_data.get('red_candles', 0)
            meme_total = meme_green + meme_red
            meme_momentum = meme_green / meme_total if meme_total > 0 else 0.5
            card_class = "bullish-card" if meme_momentum > 0.6 else "bearish-card" if meme_momentum < 0.4 else "neutral-card"
            
            st.markdown(f"""
            <div class="metric-card {card_class}">
                <div class="big-metric">{meme_momentum:.1%}</div>
                <div class="metric-label">Meme Coins 🐕</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="big-metric">N/A</div>
                <div class="metric-label">Meme Coins</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col5:
        weighted = data_entry['momentum_metrics'].get('volume_weighted_sentiment', 0.5)
        card_class = "bullish-card" if weighted > 0.6 else "bearish-card" if weighted < 0.4 else "neutral-card"
        
        st.markdown(f"""
        <div class="metric-card {card_class}">
            <div class="big-metric">{weighted:.1%}</div>
            <div class="metric-label">Vol-Weighted</div>
        </div>
        """, unsafe_allow_html=True)

def create_volume_summary_cards(data_entry: Dict, volume_history: List[Dict]) -> None:
    """Create comprehensive volume summary cards"""
    st.subheader("💰 Volume Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_volume = data_entry['total_dollar_volume']
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-metric">{format_volume(total_volume)}</div>
            <div class="metric-label">Total Volume (24h)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        green_volume = data_entry['green_volume']
        green_pct = (green_volume / total_volume * 100) if total_volume > 0 else 0
        
        st.markdown(f"""
        <div class="metric-card bullish-card">
            <div class="big-metric">{format_volume(green_volume)}</div>
            <div class="metric-label">Bullish Volume</div>
            <div class="metric-delta trend-up">{green_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        red_volume = data_entry['red_volume']
        red_pct = (red_volume / total_volume * 100) if total_volume > 0 else 0
        
        st.markdown(f"""
        <div class="metric-card bearish-card">
            <div class="big-metric">{format_volume(red_volume)}</div>
            <div class="metric-label">Bearish Volume</div>
            <div class="metric-delta trend-down">{red_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Calculate volume trend
        if len(volume_history) >= 2:
            prev_volume = volume_history[-2]['total_dollar_volume']
            volume_change = ((total_volume / prev_volume) - 1) * 100 if prev_volume > 0 else 0
            trend_class = "trend-up" if volume_change > 0 else "trend-down"
            trend_symbol = "↑" if volume_change > 0 else "↓"
        else:
            volume_change = 0
            trend_class = "trend-neutral"
            trend_symbol = "→"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-metric">{abs(volume_change):.1f}%</div>
            <div class="metric-label">Volume Change {trend_symbol}</div>
            <div class="metric-delta {trend_class}">vs Previous Hour</div>
        </div>
        """, unsafe_allow_html=True)

def create_market_structure_display(data_entry: Dict, volume_history: List[Dict]) -> None:
    """Enhanced market structure display with multiple timeframes - NO VOLATILE CATEGORY"""
    
    st.subheader("🗂️ Enhanced Market Structure Analysis")
    
    # Timeframe selector
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        lookback = st.select_slider(
            "Analysis Lookback Period",
            options=[4, 8, 12, 24, 48, 72],
            value=24,
            format_func=lambda x: f"{x} hours",
            key="structure_lookback"
        )
    
    # Get structure analysis
    structure = analyze_market_structure(data_entry, lookback)
    
    # Display main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Market character card
        char_emoji = {
            'bullish_trending': '🚀',
            'bearish_trending': '📉',
            'consolidating': '➡️',
            'mixed': '🔄'
        }.get(structure['market_character'], '❓')
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem;">{char_emoji}</div>
            <div class="metric-label">Market Character</div>
            <div class="big-metric" style="font-size: 1.2rem;">
                {structure['market_character'].replace('_', ' ').title()}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Trend strength gauge
        strength_pct = structure['strength'] * 100
        strength_color = '#00ff88' if structure['dominant_structure'] == 'uptrend' else '#ff4444' if structure['dominant_structure'] == 'downtrend' else '#ffa726'
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-metric" style="color: {strength_color};">
                {strength_pct:.0f}%
            </div>
            <div class="metric-label">Trend Strength</div>
            <div class="metric-delta">{structure['dominant_structure'].title()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Distribution
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-around; padding: 0.5rem;">
                <div style="text-align: center;">
                    <div style="color: #00ff88; font-size: 1.5rem; font-weight: bold;">
                        {structure['bullish_assets']}
                    </div>
                    <div style="font-size: 0.7rem; opacity: 0.7;">Bullish</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: #ffa726; font-size: 1.5rem; font-weight: bold;">
                        {structure['ranging_assets']}
                    </div>
                    <div style="font-size: 0.7rem; opacity: 0.7;">Ranging</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: #ff4444; font-size: 1.5rem; font-weight: bold;">
                        {structure['bearish_assets']}
                    </div>
                    <div style="font-size: 0.7rem; opacity: 0.7;">Bearish</div>
                </div>
            </div>
            <div class="metric-label">Asset Distribution</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Market metrics
        st.markdown(f"""
        <div class="metric-card">
            <div style="padding: 0.5rem;">
                <div style="display: flex; justify-content: space-between; margin: 0.2rem 0;">
                    <span style="font-size: 0.8rem;">Momentum:</span>
                    <span style="font-size: 0.8rem; color: {'#00ff88' if structure['avg_momentum'] > 0.6 else '#ff4444' if structure['avg_momentum'] < 0.4 else '#ffa726'};">
                        {structure['avg_momentum']:.0%}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 0.2rem 0;">
                    <span style="font-size: 0.8rem;">Volatility:</span>
                    <span style="font-size: 0.8rem; color: {'#ff4444' if structure['avg_volatility'] > 15 else '#ffa726' if structure['avg_volatility'] > 10 else '#00ff88'};">
                        {structure['avg_volatility']:.1f}%
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 0.2rem 0;">
                    <span style="font-size: 0.8rem;">Trend:</span>
                    <span style="font-size: 0.8rem;">
                        {structure['avg_trend_strength']:.0%}
                    </span>
                </div>
            </div>
            <div class="metric-label">Market Metrics</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed breakdown
    with st.expander("📊 Detailed Structure Breakdown", expanded=False):
        # Create bar chart of structure distribution
        breakdown = structure['detailed_breakdown']
        
        # Sort by count
        sorted_structures = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
        
        for struct_type, count in sorted_structures:
            if count > 0:
                pct = (count / structure['total_assets']) * 100
                
                # Color coding
                if 'uptrend' in struct_type:
                    color = '#00ff88'
                elif 'downtrend' in struct_type:
                    color = '#ff4444'
                else:
                    color = '#ffa726'
                
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.2rem;">
                        <span>{struct_type.replace('_', ' ').title()}</span>
                        <span>{count} ({pct:.1f}%)</span>
                    </div>
                    <div style="background: #333; height: 8px; border-radius: 4px; overflow: hidden;">
                        <div style="background: {color}; width: {pct}%; height: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Historical trend if we have enough data
    if volume_history and len(volume_history) >= 24:
        st.subheader("📈 Structure Evolution")
        
        # Add controls for the evolution chart
        evo_col1, evo_col2, evo_col3 = st.columns([1, 2, 1])
        
        with evo_col2:
            # Determine max available history
            max_history = len(volume_history)
            
            # Create options based on available data
            evolution_options = []
            if max_history >= 24:
                evolution_options.append(24)
            if max_history >= 48:
                evolution_options.append(48)
            if max_history >= 72:
                evolution_options.append(72)
            if max_history >= 168:  # 1 week
                evolution_options.append(168)
            if max_history >= 336:  # 2 weeks
                evolution_options.append(336)
            if max_history >= 720:  # 30 days
                evolution_options.append(720)
            
            # Add "All" option if we have significant history
            if max_history > 72 and max_history not in evolution_options:
                evolution_options.append(max_history)
            
            # Set default value to the closest available option
            if 72 in evolution_options:
                default_value = 72
            elif len(evolution_options) > 0:
                # Use the largest value that's less than or equal to 72
                default_value = max([x for x in evolution_options if x <= 72] + [evolution_options[0]])
            else:
                default_value = 24
            
            evolution_lookback = st.select_slider(
                "Evolution Chart Period",
                options=evolution_options,
                value=default_value,
                format_func=lambda x: f"All ({x}h)" if x == max_history else f"{x//24}d" if x >= 168 else f"{x}h",
                key="evolution_lookback"
            )
        
        # Show data info
        with evo_col1:
            st.caption(f"📊 {evolution_lookback} data points")
        
        with evo_col3:
            st.caption(f"Total history: {max_history}h")
        
        # Calculate structure for selected period
        structure_evolution = []
        
        # Get the entries for the selected lookback period
        start_idx = max(0, len(volume_history) - evolution_lookback)
        selected_entries = volume_history[start_idx:]
        
        for entry in selected_entries:
            # Use consistent 24h lookback for each point's structure calculation
            struct = analyze_market_structure(entry, 24)
            
            structure_evolution.append({
                'timestamp': entry['timestamp'],
                'bullish_pct': struct['bullish_pct'],
                'bearish_pct': struct['bearish_pct'],
                'sideways_pct': struct['sideways_pct'],
                'strength': struct['strength']
            })
        
        # Create line chart
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        timestamps = [s['timestamp'] for s in structure_evolution]
        
        # Add traces with fills
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[s['bullish_pct'] for s in structure_evolution],
            name='Bullish %',
            line=dict(color='#00ff88', width=2),
            fill='tonexty',
            fillcolor='rgba(0, 255, 136, 0.1)',
            hovertemplate='%{y:.1f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[s['bearish_pct'] for s in structure_evolution],
            name='Bearish %',
            line=dict(color='#ff4444', width=2),
            fill='tonexty',
            fillcolor='rgba(255, 68, 68, 0.1)',
            hovertemplate='%{y:.1f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[s['sideways_pct'] for s in structure_evolution],
            name='Sideways %',
            line=dict(color='#ffa726', width=2),
            fill='tonexty',
            fillcolor='rgba(255, 167, 38, 0.1)',
            hovertemplate='%{y:.1f}%<extra></extra>'
        ))
        
        # Add horizontal reference lines
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.3)
        fig.add_hline(y=25, line_dash="dot", line_color="gray", opacity=0.2)
        fig.add_hline(y=75, line_dash="dot", line_color="gray", opacity=0.2)
        
        # Calculate date range for title
        if structure_evolution:
            start_date = structure_evolution[0]['timestamp'].strftime('%b %d')
            end_date = structure_evolution[-1]['timestamp'].strftime('%b %d %H:%M')
            title_text = f"Market Structure Evolution ({start_date} to {end_date})"
        else:
            title_text = f"Market Structure Evolution ({evolution_lookback}h)"
        
        fig.update_layout(
            title=title_text,
            yaxis_title="Percentage of Assets",
            xaxis_title="Time",
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary stats for the evolution period
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate trends over the period
        if len(structure_evolution) > 1:
            start_bullish = structure_evolution[0]['bullish_pct']
            end_bullish = structure_evolution[-1]['bullish_pct']
            bullish_change = end_bullish - start_bullish
            
            start_bearish = structure_evolution[0]['bearish_pct']
            end_bearish = structure_evolution[-1]['bearish_pct']
            bearish_change = end_bearish - start_bearish
            
            with col1:
                st.metric(
                    "Bullish Trend",
                    f"{end_bullish:.1f}%",
                    f"{bullish_change:+.1f}%",
                    delta_color="normal"
                )
            
            with col2:
                st.metric(
                    "Bearish Trend",
                    f"{end_bearish:.1f}%",
                    f"{bearish_change:+.1f}%",
                    delta_color="inverse"
                )
            
            with col3:
                avg_bullish = sum(s['bullish_pct'] for s in structure_evolution) / len(structure_evolution)
                st.metric(
                    "Avg Bullish",
                    f"{avg_bullish:.1f}%"
                )
            
            with col4:
                # Find most dominant structure over period
                total_bullish = sum(s['bullish_pct'] for s in structure_evolution)
                total_bearish = sum(s['bearish_pct'] for s in structure_evolution)
                total_sideways = sum(s['sideways_pct'] for s in structure_evolution)
                
                if total_bullish > max(total_bearish, total_sideways):
                    dominant_period = "Bullish"
                    dominant_emoji = "🟢"
                elif total_bearish > max(total_bullish, total_sideways):
                    dominant_period = "Bearish"
                    dominant_emoji = "🔴"
                else:
                    dominant_period = "Sideways"
                    dominant_emoji = "🟠"
                
                st.metric(
                    "Period Character",
                    f"{dominant_emoji} {dominant_period}"
                )

def detect_structure_cross(volume_history: List[Dict]) -> bool:
    """Detect when bearish % crosses below sideways % - bullish signal"""
    if len(volume_history) < 2:
        return False
    
    current = volume_history[-1]
    previous = volume_history[-2]
    
    # Get structure percentages
    if 'market_stats' in current and 'market_stats' in previous:
        curr_bearish_pct = current['market_stats'].get('trending_down', 0) / max(current.get('total_assets', 1), 1)
        curr_sideways_pct = current['market_stats'].get('sideways', 0) / max(current.get('total_assets', 1), 1)
        
        prev_bearish_pct = previous['market_stats'].get('trending_down', 0) / max(previous.get('total_assets', 1), 1)
        prev_sideways_pct = previous['market_stats'].get('sideways', 0) / max(previous.get('total_assets', 1), 1)
        
        # Check for bearish crossing below sideways
        if (prev_bearish_pct > prev_sideways_pct and 
            curr_bearish_pct < curr_sideways_pct):
            return True
    
    return False

def create_divergence_monitor(data_entry: Dict, volume_history: List[Dict]) -> None:
    """
    Real-time divergence monitor for the dashboard overview section
    Shows price/sentiment divergences and structure crosses
    """
    
    st.subheader("🔄 Divergence Monitor & Trading Signals")
    
    # Need at least 4 hours of history for divergence detection
    if len(volume_history) < 4:
        st.warning("Insufficient data for divergence detection (need 4+ hours)")
        return
    
    # Calculate true price momentum for recent entries
    recent_history = volume_history[-12:]  # Last 12 hours
    price_momentums = []
    vol_sentiments = []
    
    for entry in recent_history:
        # Get volume sentiment
        vol_sentiments.append(entry['momentum_metrics']['bullish_volume_ratio'])
        
        # Calculate average price momentum from assets
        if 'assets' in entry and entry['assets']:
            momentums_4h = []
            for asset in entry['assets'].values():
                if 'price_change_4h' in asset and asset['price_change_4h'] is not None:
                    momentums_4h.append(asset['price_change_4h'])
            
            if momentums_4h:
                # Use median for robustness
                price_momentum = np.median(momentums_4h)
            else:
                price_momentum = 0
        else:
            price_momentum = 0
        
        price_momentums.append(price_momentum)
    
    # Current values
    current_sentiment = vol_sentiments[-1]
    current_momentum = price_momentums[-1]
    
    # Calculate recent trends (3-hour lookback)
    if len(price_momentums) >= 4:
        sentiment_trend = vol_sentiments[-1] - vol_sentiments[-4]
        momentum_trend = price_momentums[-1] - price_momentums[-4]
        
        # Detect divergence
        divergence_detected = False
        divergence_type = None
        divergence_strength = 0
        
        # Bullish divergence: price falling, sentiment rising
        if momentum_trend < -1 and sentiment_trend > 0.02:
            divergence_detected = True
            divergence_type = "BULLISH"
            divergence_strength = abs(momentum_trend) + (sentiment_trend * 100)
        
        # Bearish divergence: price rising, sentiment falling  
        elif momentum_trend > 1 and sentiment_trend < -0.02:
            divergence_detected = True
            divergence_type = "BEARISH"
            divergence_strength = momentum_trend + abs(sentiment_trend * 100)
    else:
        sentiment_trend = 0
        momentum_trend = 0
        divergence_detected = False
        divergence_type = None
        divergence_strength = 0
    
    # Check for structure cross signal (NEW)
    structure_cross = detect_structure_cross(volume_history)
    
    # Display current state and signals
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Divergence/Cross Alert
        if structure_cross:
            # Structure cross takes priority as it's rarer and stronger
            alert_color = "#00ff88"
            alert_emoji = "🔄"
            signal = "BOUNCE SIGNAL"
            alert_type = "STRUCTURE CROSS"
        elif divergence_detected:
            if divergence_type == "BULLISH":
                alert_color = "#00ff88"
                alert_emoji = "🟢"
                signal = "BUY SIGNAL"
                alert_type = divergence_type
            else:
                alert_color = "#ff4444"
                alert_emoji = "🔴"
                signal = "SELL SIGNAL"
                alert_type = divergence_type
        else:
            alert_color = None
            alert_emoji = "⚪"
            signal = "NO SIGNAL"
            alert_type = None
        
        if alert_color:
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid {alert_color}; animation: pulse 2s infinite;">
                <div style="font-size: 2rem;">{alert_emoji}</div>
                <div class="big-metric" style="color: {alert_color}; font-size: 1.2rem;">
                    {alert_type}
                </div>
                <div class="metric-label">{"DIVERGENCE" if divergence_detected else "STRUCTURE"}</div>
                <div class="metric-delta" style="color: {alert_color};">{signal}</div>
            </div>
            <style>
            @keyframes pulse {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.7; }}
                100% {{ opacity: 1; }}
            }}
            </style>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">{alert_emoji}</div>
                <div class="big-metric">{signal}</div>
                <div class="metric-label">Signal Status</div>
                <div class="metric-delta">Monitoring...</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Price Momentum
        momentum_color = "#00ff88" if current_momentum > 1 else "#ff4444" if current_momentum < -1 else "#ffa726"
        trend_arrow = "↗" if momentum_trend > 0 else "↘" if momentum_trend < 0 else "→"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-metric" style="color: {momentum_color};">
                {current_momentum:.1f}%
            </div>
            <div class="metric-label">Price Momentum (4h)</div>
            <div class="metric-delta">
                3h trend: {momentum_trend:+.1f}% {trend_arrow}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Volume Sentiment
        sentiment_color = "#00ff88" if current_sentiment > 0.52 else "#ff4444" if current_sentiment < 0.48 else "#ffa726"
        sentiment_arrow = "↗" if sentiment_trend > 0 else "↘" if sentiment_trend < 0 else "→"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-metric" style="color: {sentiment_color};">
                {current_sentiment:.1%}
            </div>
            <div class="metric-label">Volume Sentiment</div>
            <div class="metric-delta">
                3h trend: {sentiment_trend*100:+.1f}% {sentiment_arrow}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Trading Recommendation
        if structure_cross:
            action = "BUY"
            action_color = "#00ff88"
            hold_time = "3-6 hours"
            confidence = 85  # High confidence for structure cross
        elif divergence_detected:
            if divergence_type == "BULLISH":
                action = "BUY"
                action_color = "#00ff88"
                hold_time = "3 hours"
                confidence = min(divergence_strength * 10, 90)  # Cap at 90%
            else:
                action = "SELL"
                action_color = "#ff4444"
                hold_time = "3 hours"
                confidence = min(divergence_strength * 10, 90)
        else:
            # Check for overbought/oversold
            if current_sentiment > 0.53 and current_momentum > 2:
                action = "SELL"
                action_color = "#ff4444"
                hold_time = "1-3 hours"
                confidence = 60
            elif current_sentiment < 0.47 and current_momentum < -2:
                action = "BUY"
                action_color = "#00ff88"
                hold_time = "1-3 hours"
                confidence = 60
            else:
                action = "HOLD"
                action_color = "#ffa726"
                hold_time = "Wait"
                confidence = 0
        
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {action_color};">
            <div class="big-metric" style="color: {action_color}; font-size: 1.5rem;">
                {action}
            </div>
            <div class="metric-label">Recommendation</div>
            <div class="metric-delta">Hold: {hold_time}</div>
            <div class="metric-delta" style="font-size: 0.8rem;">
                Confidence: {confidence:.0f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Divergence History Chart
    with st.expander("📊 Divergence History", expanded=False):
        # Add slider for lookback period
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            # Determine max available history
            max_history = len(volume_history)
            
            # Create options based on available data
            lookback_options = []
            if max_history >= 12:
                lookback_options.append(12)
            if max_history >= 24:
                lookback_options.append(24)
            if max_history >= 48:
                lookback_options.append(48)
            if max_history >= 72:
                lookback_options.append(72)
            if max_history >= 168:  # 1 week
                lookback_options.append(168)
            
            # Add max option if we have more data
            if max_history > 168:
                lookback_options.append(max_history)
            
            # Create slider
            if len(lookback_options) > 1:
                divergence_lookback = st.select_slider(
                    "History Lookback",
                    options=lookback_options,
                    value=min(24, max_history),  # Default to 24h or max available
                    format_func=lambda x: f"All ({x}h)" if x == max_history else f"{x//24}d" if x >= 168 else f"{x}h",
                    key="divergence_lookback"
                )
            else:
                divergence_lookback = min(12, max_history)
                st.caption(f"Showing {divergence_lookback} hours (limited data available)")
        
        with col1:
            st.caption(f"📊 {divergence_lookback} data points")
        
        with col3:
            st.caption(f"Total: {max_history}h")
        
        # Get the selected history range
        divergence_history = volume_history[-divergence_lookback:] if divergence_lookback <= len(volume_history) else volume_history
        
        # Recalculate momentum data for selected range
        div_price_momentums = []
        div_vol_sentiments = []
        
        for entry in divergence_history:
            # Get volume sentiment
            div_vol_sentiments.append(entry['momentum_metrics']['bullish_volume_ratio'])
            
            # Calculate average price momentum from assets
            if 'assets' in entry and entry['assets']:
                momentums_4h = []
                for asset in entry['assets'].values():
                    if 'price_change_4h' in asset and asset['price_change_4h'] is not None:
                        momentums_4h.append(asset['price_change_4h'])
                
                if momentums_4h:
                    price_momentum = np.median(momentums_4h)
                else:
                    price_momentum = 0
            else:
                price_momentum = 0
            
            div_price_momentums.append(price_momentum)
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Add price momentum line
        timestamps = [entry['timestamp'] for entry in divergence_history]
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=div_price_momentums,
            name='Price Momentum',
            line=dict(color='#1f77b4', width=2),
            yaxis='y'
        ))
        
        # Add volume sentiment line (scaled to match momentum range)
        sentiment_scaled = [(s - 0.5) * 20 for s in div_vol_sentiments]  # Scale to roughly match momentum
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=sentiment_scaled,
            name='Volume Sentiment (scaled)',
            line=dict(color='#00ff88', width=2),
            yaxis='y'
        ))
        
        # Mark divergences
        for i in range(3, len(divergence_history)):
            sent_change = div_vol_sentiments[i] - div_vol_sentiments[i-3]
            mom_change = div_price_momentums[i] - div_price_momentums[i-3]
            
            # Bullish divergence
            if mom_change < -1 and sent_change > 0.02:
                fig.add_vline(x=timestamps[i], line_dash="dash", line_color="cyan", opacity=0.5)
                fig.add_annotation(
                    x=timestamps[i],
                    y=div_price_momentums[i],
                    text="Bull Div",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="cyan",
                    ax=0,
                    ay=-20
                )
            
            # Bearish divergence
            elif mom_change > 1 and sent_change < -0.02:
                fig.add_vline(x=timestamps[i], line_dash="dash", line_color="magenta", opacity=0.5)
                fig.add_annotation(
                    x=timestamps[i],
                    y=div_price_momentums[i],
                    text="Bear Div",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="magenta",
                    ax=0,
                    ay=-20
                )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3)
        
        # Calculate date range for title
        if divergence_history:
            start_date = divergence_history[0]['timestamp'].strftime('%b %d %H:%M')
            end_date = divergence_history[-1]['timestamp'].strftime('%b %d %H:%M')
            title_text = f"Price Momentum vs Volume Sentiment ({start_date} to {end_date})"
        else:
            title_text = "Price Momentum vs Volume Sentiment"
        
        fig.update_layout(
            title=title_text,
            yaxis_title="Momentum % / Sentiment (scaled)",
            xaxis_title="Time",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary statistics for the selected period
        if len(divergence_history) > 3:
            divergence_count = 0
            bullish_div_count = 0
            bearish_div_count = 0
            
            for i in range(3, len(divergence_history)):
                sent_change = div_vol_sentiments[i] - div_vol_sentiments[i-3]
                mom_change = div_price_momentums[i] - div_price_momentums[i-3]
                
                if mom_change < -1 and sent_change > 0.02:
                    bullish_div_count += 1
                    divergence_count += 1
                elif mom_change > 1 and sent_change < -0.02:
                    bearish_div_count += 1
                    divergence_count += 1
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Divergences", divergence_count)
            with col2:
                st.metric("Bullish Divergences", bullish_div_count)
            with col3:
                st.metric("Bearish Divergences", bearish_div_count)
    
    # Trading Tips
    hour = data_entry['timestamp'].hour
    
    # Check if current hour is a high reversal hour (from backtesting)
    high_reversal_hours = [19, 15, 20]  # From your backtest results
    
    tips = []
    
    if structure_cross:
        tips.append("🔄 STRUCTURE CROSS: Bearish % crossed below Sideways % - Strong bounce signal!")
        tips.append("📊 Based on historical data, this pattern often marks local bottoms")
    
    if divergence_detected:
        tips.append(f"⚠️ {divergence_type} DIVERGENCE detected - Strong signal!")
        tips.append(f"📊 Recommended hold time: 3 hours based on backtesting")
    
    if hour in high_reversal_hours:
        tips.append(f"🕐 Current hour ({hour}:00) is a high reversal probability time")
    
    if current_sentiment > 0.53:
        tips.append("📈 Volume sentiment elevated - watch for potential top")
    elif current_sentiment < 0.47:
        tips.append("📉 Volume sentiment low - watch for potential bottom")
    
    if abs(momentum_trend) > 2:
        direction = "bullish" if momentum_trend > 0 else "bearish"
        tips.append(f"💨 Strong {direction} momentum trend developing")
    
    if tips:
        st.info("💡 **Trading Insights:**\n" + "\n".join(f"• {tip}" for tip in tips))


def create_market_momentum_oscillator(data_entry: Dict, volume_history: List[Dict]) -> None:
    """Create a comprehensive market-wide momentum oscillator chart"""
    
    st.subheader("📈 Market Momentum Oscillator")
    
    # Controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Lookback period for historical data
        history_periods = min(72, len(volume_history))
        lookback = st.slider(
            "Historical Periods",
            min_value=12,
            max_value=history_periods,
            value=min(24, history_periods),
            step=1,
            key="momentum_lookback"
        )
    
    with col1:
        # Option to use top assets only for clearer signals
        use_top_only = st.checkbox("Top 50 Assets Only", value=True, key="momentum_top_only")
    
    with col3:
        # Signal amplification
        amplify = st.select_slider("Signal Strength", [1, 2, 3], value=2, key="momentum_amplify")
    
    # Calculate momentum for different timeframes over history
    momentum_data = []
    
    for i in range(lookback):
        idx = -(lookback - i)
        entry = volume_history[idx]
        assets = entry.get('assets', {})
        
        if not assets:
            continue
        
        # Filter to top assets by volume if selected
        if use_top_only:
            sorted_assets = sorted(assets.items(), key=lambda x: x[1].get('total_dollar_volume', 0), reverse=True)
            assets_to_use = dict(sorted_assets[:50])
        else:
            assets_to_use = assets
        
        # Calculate various momentum metrics
        momentum_1h = []
        momentum_4h = []
        momentum_12h = []
        momentum_24h = []
        volumes = []
        
        for asset_data in assets_to_use.values():
            # Get price changes (these ARE momentum values)
            if 'price_change_1h' in asset_data and asset_data['price_change_1h'] is not None:
                momentum_1h.append(asset_data['price_change_1h'])
            if 'price_change_4h' in asset_data and asset_data['price_change_4h'] is not None:
                momentum_4h.append(asset_data['price_change_4h'])
            if 'price_change_12h' in asset_data and asset_data['price_change_12h'] is not None:
                momentum_12h.append(asset_data['price_change_12h'])
            if 'price_change_24h' in asset_data and asset_data['price_change_24h'] is not None:
                momentum_24h.append(asset_data['price_change_24h'])
                volumes.append(asset_data.get('total_dollar_volume', 0))
        
        # Use median for robustness, but also track percentiles
        if momentum_1h:
            # Use 60th percentile to lean toward stronger signals
            avg_mom_1h = np.percentile(momentum_1h, 60)
        else:
            avg_mom_1h = 0
            
        if momentum_4h:
            avg_mom_4h = np.percentile(momentum_4h, 60)
        else:
            avg_mom_4h = 0
            
        if momentum_12h:
            avg_mom_12h = np.percentile(momentum_12h, 60)
        else:
            avg_mom_12h = 0
            
        if momentum_24h:
            avg_mom_24h = np.percentile(momentum_24h, 60)
        else:
            avg_mom_24h = 0
        
        # Volume-weighted momentum (this is usually more dramatic)
        if momentum_24h and volumes and sum(volumes) > 0:
            weighted_mom = sum(m * v for m, v in zip(momentum_24h, volumes)) / sum(volumes)
        else:
            weighted_mom = 0
        
        # Momentum divergence (amplified for visibility)
        divergence = (avg_mom_4h - avg_mom_24h) * amplify if avg_mom_4h and avg_mom_24h else 0
        
        momentum_data.append({
            'timestamp': entry['timestamp'],
            'mom_1h': avg_mom_1h * amplify,
            'mom_4h': avg_mom_4h * amplify,
            'mom_12h': avg_mom_12h * amplify,
            'mom_24h': avg_mom_24h * amplify,
            'weighted_mom': weighted_mom * amplify,
            'divergence': divergence,
            'advancing': sum(1 for m in momentum_24h if m > 0) if momentum_24h else 0,
            'declining': sum(1 for m in momentum_24h if m < 0) if momentum_24h else 0
        })
    
    if not momentum_data:
        st.warning("Insufficient data for momentum oscillator")
        return
    
    # Create the oscillator chart
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Multi-Timeframe Volume Sentiment", "Sentiment Divergence", "Market Breadth"),
        vertical_spacing=0.05
    )
    
    timestamps = [d['timestamp'] for d in momentum_data]
    
    # Row 1: Main momentum oscillator
    # Use weighted combination for smoother signals
    fast_momentum = []
    slow_momentum = []
    
    for d in momentum_data:
        # Fast: Heavily weight 4h (more stable than 1h)
        fast = d['mom_4h'] * 0.8 + d['mom_1h'] * 0.2
        fast_momentum.append(fast)
        
        # Slow: Heavily weight 24h
        slow = d['mom_24h'] * 0.8 + d['mom_12h'] * 0.2
        slow_momentum.append(slow)
    
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=fast_momentum,
            name='Fast Momentum (1-4h)',
            line=dict(color='#00ff88', width=2),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=slow_momentum,
            name='Slow Momentum (12-24h)',
            line=dict(color='#ff4444', width=2),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=[d['weighted_mom'] for d in momentum_data],
            name='Volume-Weighted',
            line=dict(color='#ffa726', width=3),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.3, row=1, col=1)
    
    # Add overbought/oversold zones (adjusted for amplification)
    ob_level = 5 * amplify
    os_level = -5 * amplify
    fig.add_hrect(y0=ob_level, y1=ob_level*2, fillcolor="green", opacity=0.1, row=1, col=1)
    fig.add_hrect(y0=os_level*2, y1=os_level, fillcolor="red", opacity=0.1, row=1, col=1)
    
    # Row 2: Momentum divergence histogram
    divergence_colors = ['#00ff88' if d > 0 else '#ff4444' for d in [d['divergence'] for d in momentum_data]]
    
    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=[d['divergence'] for d in momentum_data],
            name='Divergence',
            marker_color=divergence_colors,
            hovertemplate='%{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.3, row=2, col=1)
    
    # Row 3: Advance/Decline breadth
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=[d['advancing'] for d in momentum_data],
            name='Advancing',
            line=dict(color='#00ff88', width=2),
            fill='tonexty',
            fillcolor='rgba(0, 255, 136, 0.1)',
            hovertemplate='%{y} assets<extra></extra>'
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=[-d['declining'] for d in momentum_data],
            name='Declining',
            line=dict(color='#ff4444', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.1)',
            hovertemplate='%{y} assets<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_yaxes(title_text="Momentum %", row=1, col=1)
    fig.update_yaxes(title_text="Divergence", row=2, col=1)
    fig.update_yaxes(title_text="Assets", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Current momentum signals
    if momentum_data:
        latest = momentum_data[-1]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fast_val = latest['mom_4h'] * 0.8 + latest['mom_1h'] * 0.2
            slow_val = latest['mom_24h'] * 0.8 + latest['mom_12h'] * 0.2
            signal = "Bullish" if fast_val > slow_val and fast_val > 0 else "Bearish" if fast_val < slow_val and fast_val < 0 else "Neutral"
            signal_color = "#00ff88" if signal == "Bullish" else "#ff4444" if signal == "Bearish" else "#ffa726"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="big-metric" style="color: {signal_color};">{signal}</div>
                <div class="metric-label">Momentum Signal</div>
                <div class="metric-delta">Fast: {fast_val/amplify:.1f}% | Slow: {slow_val/amplify:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            divergence = latest['divergence']
            div_text = "Positive" if divergence > 1 else "Negative" if divergence < -1 else "Neutral"
            div_color = "#00ff88" if divergence > 1 else "#ff4444" if divergence < -1 else "#ffa726"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="big-metric" style="color: {div_color};">{divergence:.1f}%</div>
                <div class="metric-label">Divergence</div>
                <div class="metric-delta">{div_text}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            breadth_ratio = latest['advancing'] / (latest['advancing'] + latest['declining']) if (latest['advancing'] + latest['declining']) > 0 else 0.5
            breadth_text = "Bullish" if breadth_ratio > 0.6 else "Bearish" if breadth_ratio < 0.4 else "Neutral"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="big-metric">{breadth_ratio:.0%}</div>
                <div class="metric-label">Breadth</div>
                <div class="metric-delta">{latest['advancing']}↑ {latest['declining']}↓</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            weighted = latest['weighted_mom'] / amplify  # Show unamplified value
            weighted_color = "#00ff88" if weighted > 2 else "#ff4444" if weighted < -2 else "#ffa726"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="big-metric" style="color: {weighted_color};">{weighted:.1f}%</div>
                <div class="metric-label">Weighted Mom</div>
                <div class="metric-delta">Volume-Adjusted</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Trading signals explanation
    with st.expander("📊 Understanding Momentum Signals"):
        st.markdown("""
        **Momentum Oscillator Components:**
        
        1. **Fast vs Slow Momentum**: 
           - Fast (1-4h) crossing above Slow (12-24h) = Bullish signal
           - Fast crossing below Slow = Bearish signal
           
        2. **Divergence**: 
           - Positive = Short-term momentum improving faster than long-term
           - Negative = Short-term momentum weakening
           
        3. **Market Breadth**: 
           - Shows percentage of advancing vs declining assets
           - Confirms if momentum is broad-based or narrow
           
        4. **Volume-Weighted Momentum**: 
           - Accounts for where the big money is flowing
           - Often leads the market direction
        
        **Settings:**
        - **Top 50 Only**: Focus on highest volume assets for clearer signals
        - **Signal Strength**: Amplifies oscillator movement for better visibility
        """)
             

def create_data_quality_dashboard(data_entry: Dict) -> None:
    """Create data quality metrics dashboard"""
    if 'data_quality' not in data_entry:
        return
    
    quality = data_entry['data_quality']
    
    with st.expander("📊 Data Quality Metrics", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ticker_rate = quality.get('ticker_success_rate', 0)
            badge = get_quality_badge(ticker_rate)
            st.markdown(f"""
            <div class="info-box">
                <h4>Ticker Data {badge}</h4>
                <div style="font-size: 2rem; font-weight: bold;">{ticker_rate:.0%}</div>
                <small>Real-time price accuracy</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            stats_rate = quality.get('stats_success_rate', 0)
            badge = get_quality_badge(stats_rate)
            st.markdown(f"""
            <div class="info-box">
                <h4>Stats Data {badge}</h4>
                <div style="font-size: 2rem; font-weight: bold;">{stats_rate:.0%}</div>
                <small>24hr statistics coverage</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            completeness = quality.get('candle_completeness', 0)
            badge = get_quality_badge(completeness)
            st.markdown(f"""
            <div class="info-box">
                <h4>Completeness {badge}</h4>
                <div style="font-size: 2rem; font-weight: bold;">{completeness:.0%}</div>
                <small>Asset data coverage</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            overall_score = get_data_quality_score(data_entry)
            badge = get_quality_badge(overall_score)
            st.markdown(f"""
            <div class="info-box">
                <h4>Overall Score {badge}</h4>
                <div style="font-size: 2rem; font-weight: bold;">{overall_score:.0%}</div>
                <small>Composite quality rating</small>
            </div>
            """, unsafe_allow_html=True)

# ================================================================================
# ENHANCED CATEGORY METRICS - NEW!
# ================================================================================

def calculate_enhanced_category_metrics(category_assets: List[Dict]) -> Dict[str, Any]:
    """
    Calculate comprehensive category performance metrics
    """
    
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
    
    # 2. TRUE MOMENTUM (Rate of Change)
    metrics['momentum_short'] = metrics['perf_4h']
    
    if metrics['perf_24h'] != 0:
        metrics['momentum_trend'] = metrics['perf_4h'] - (metrics['perf_24h'] / 6)
    else:
        metrics['momentum_trend'] = metrics['perf_4h']
    
    # 3. ACCELERATION (Is momentum increasing?)
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
    
    # 9. TOP MOVERS
    sorted_assets = sorted(category_assets, 
                          key=lambda x: x.get('price_change_24h', 0), 
                          reverse=True)
    metrics['top_gainer'] = sorted_assets[0] if sorted_assets else None
    metrics['top_loser'] = sorted_assets[-1] if sorted_assets else None
    
    # 10. TREND CLASSIFICATION
    metrics['trend'] = classify_category_trend(metrics)
    
    return metrics


def classify_category_trend(metrics: Dict) -> Dict[str, str]:
    """Classify the overall trend of a category"""
    
    perf = metrics['perf_24h']
    momentum = metrics['momentum_short']
    sentiment = metrics['volume_sentiment']
    
    if perf > 5 and momentum > 2 and sentiment > 0.6:
        return {'name': 'Strong Uptrend', 'emoji': '🚀', 'color': '#00ff88'}
    elif perf > 2 and (momentum > 1 or sentiment > 0.55):
        return {'name': 'Uptrend', 'emoji': '📈', 'color': '#00cc66'}
    elif perf > 0 and sentiment > 0.5:
        return {'name': 'Weak Uptrend', 'emoji': '↗️', 'color': '#66ff66'}
    elif perf < -5 and momentum < -2 and sentiment < 0.4:
        return {'name': 'Strong Downtrend', 'emoji': '💥', 'color': '#ff0000'}
    elif perf < -2 and (momentum < -1 or sentiment < 0.45):
        return {'name': 'Downtrend', 'emoji': '📉', 'color': '#ff4444'}
    elif perf < 0 and sentiment < 0.5:
        return {'name': 'Weak Downtrend', 'emoji': '↘️', 'color': '#ff6666'}
    else:
        return {'name': 'Sideways', 'emoji': '➡️', 'color': '#ffa726'}

def create_category_performance_grid(data_entry: Dict, volume_history: List[Dict] = None) -> None:
    """Create enhanced category performance comparison grid with expandable details"""
    
    st.subheader("🏷️ Category Performance")
    
    categories = data_entry.get('volume_by_category', {})
    assets = data_entry.get('assets', {})
    
    # Filter out empty categories
    non_empty_categories = {k: v for k, v in categories.items() if v.get('volume', 0) > 0}
    
    if not non_empty_categories:
        st.info("No category data available")
        return
    
    # Calculate enhanced metrics for each category
    category_metrics = {}
    for cat_name, cat_data in non_empty_categories.items():
        category_asset_list = []
        for symbol in cat_data.get('assets', []):
            if symbol in assets:
                asset_data = assets[symbol].copy()
                asset_data['symbol'] = symbol
                category_asset_list.append(asset_data)
        
        if category_asset_list:
            category_metrics[cat_name] = calculate_enhanced_category_metrics(category_asset_list)
    
    # Sort categories by strength score
    sorted_categories = sorted(
        category_metrics.items(),
        key=lambda x: x[1]['strength_score'] if x[1] else 0,
        reverse=True
    )
    
    # ================================================================================
    # CSV EXPORT BUTTON
    # ================================================================================
    
    # Prepare data for CSV export
    export_data = []
    for cat_name, metrics in sorted_categories:
        if metrics:
            cat_data = non_empty_categories[cat_name]
            export_data.append({
                'Category': cat_name.replace('_', ' ').title(),
                'Strength_Score': round(metrics['strength_score'], 2),
                'Trend': metrics['trend']['name'],
                'Performance_1h': round(metrics['perf_1h'], 2),
                'Performance_4h': round(metrics['perf_4h'], 2),
                'Performance_24h': round(metrics['perf_24h'], 2),
                'Momentum_Short': round(metrics['momentum_short'], 2),
                'Momentum_Trend': round(metrics['momentum_trend'], 2),
                'Acceleration': round(metrics['acceleration'], 2),
                'Volume_Sentiment': round(metrics['volume_sentiment'] * 100, 1),
                'Weighted_Performance': round(metrics['weighted_performance'], 2),
                'Breadth_Pct': round(metrics['breadth_pct'], 1),
                'Avg_Volatility': round(metrics['avg_volatility'], 2),
                'Total_Volume': cat_data['volume'],
                'Asset_Count': len(cat_data.get('assets', [])),
                'Top_Gainer': metrics['top_gainer']['symbol'] if metrics['top_gainer'] else 'N/A',
                'Top_Gainer_Change': round(metrics['top_gainer'].get('price_change_24h', 0), 2) if metrics['top_gainer'] else 0,
                'Top_Loser': metrics['top_loser']['symbol'] if metrics['top_loser'] else 'N/A',
                'Top_Loser_Change': round(metrics['top_loser'].get('price_change_24h', 0), 2) if metrics['top_loser'] else 0,
            })
    
    # Create DataFrame for export
    if export_data:
        export_df = pd.DataFrame(export_data)
        
        # Create CSV
        csv = export_df.to_csv(index=False).encode('utf-8')
        
        # Download button at the top
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.download_button(
                label="📥 Download Category Data (CSV)",
                data=csv,
                file_name=f"category_performance_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download all category metrics for analysis in Excel/Sheets"
            )
        
        st.markdown("---")
    
    # ================================================================================
    # END CSV EXPORT
    # ================================================================================
    
    # Determine grid layout - USE FEWER COLUMNS FOR MORE SPACE
    num_categories = len(sorted_categories)
    if num_categories <= 3:
        cols_per_row = num_categories
    else:
        cols_per_row = 3  # Max 3 per row for readability
    
    # Create category cards
    for i in range(0, len(sorted_categories), cols_per_row):
        row_categories = sorted_categories[i:i + cols_per_row]
        cols = st.columns(len(row_categories))
        
        for idx, (cat_name, metrics) in enumerate(row_categories):
            if not metrics:
                continue
                
            with cols[idx]:
                cat_data = non_empty_categories[cat_name]
                
                # Get display info
                cat_emojis = {
                    'major_coins': '👑', 'large_caps': '💎', 'altcoins': '🪙',
                    'other_alts': '🔷', 'stablecoins': '💵', 'meme_coins': '🐕',
                    'defi': '🦄', 'layer2': '⚡', 'layer1_alts': '🌐',
                    'gaming_metaverse': '🎮', 'ai_compute': '🤖',
                    'solana_ecosystem': '☀️', 'cosmos_ecosystem': '🌌',
                    'web3_social': '💬', 'nft_ecosystem': '🖼️', 'privacy': '🔒',
                    'exchange_tokens': '💱', 'oracle_data': '🔮',
                    'storage_infra': '💾', 'cross_chain': '🌉',
                    'liquid_staking': '💧', 'other_defi': '📊'
                }
                
                cat_emoji = cat_emojis.get(cat_name, '🪙')
                cat_display = cat_name.replace('_', ' ').title()
                
                display_name_map = {
                    'Gaming Metaverse': 'Gaming', 'Solana Ecosystem': 'Solana',
                    'Cosmos Ecosystem': 'Cosmos', 'Storage Infra': 'Storage',
                    'Exchange Tokens': 'Exchange', 'Liquid Staking': 'Liquid Stake',
                    'Layer1 Alts': 'Alt L1s'
                }
                cat_display = display_name_map.get(cat_display, cat_display)
                
                trend = metrics['trend']
                strength = metrics['strength_score']
                
                # Use Streamlit container
                with st.container():
                    # Category header - BIGGER
                    st.markdown(f"## {cat_emoji} {cat_display}")
                    
                    # Strength score - USE SINGLE COLUMN
                    st.metric(
                        label="💪 Strength",
                        value=f"{strength:.0f}",
                        delta=f"{trend['emoji']} {trend['name']}"
                    )
                    
                    # Stack metrics vertically instead of columns
                    st.metric("📈 24h", f"{metrics['perf_24h']:+.1f}%")
                    st.metric("⚡ Momentum", f"{metrics['momentum_short']:+.1f}%")
                    st.metric("🎯 Sentiment", f"{metrics['volume_sentiment']:.0%}")
                    
                    # Volume and asset count
                    st.caption(f"💰 {format_volume(cat_data['volume'])}")
                    st.caption(f"📊 {len(cat_data.get('assets', []))} assets")
                    
                    st.markdown("---")
                
                # Expandable details
                with st.expander("📊 Details", expanded=False):
                    st.markdown("**📈 Performance**")
                    st.write(f"1h: {metrics['perf_1h']:+.2f}%")
                    st.write(f"4h: {metrics['perf_4h']:+.2f}%")
                    st.write(f"24h: {metrics['perf_24h']:+.2f}%")
                    st.write(f"Vol-Weighted: {metrics['weighted_performance']:+.2f}%")
                    
                    st.markdown("**⚡ Momentum**")
                    st.write(f"Short-term: {metrics['momentum_short']:+.2f}%")
                    st.write(f"Trend: {metrics['momentum_trend']:+.2f}%")
                    st.write(f"Acceleration: {metrics['acceleration']:+.2f}%")
                    
                    st.markdown("**📊 Health**")
                    st.write(f"Sentiment: {metrics['volume_sentiment']:.1%}")
                    st.write(f"Breadth: {metrics['breadth_pct']:.0f}% advancing")
                    st.write(f"Volatility: {metrics['avg_volatility']:.1f}%")
                    
                    st.markdown("**🏆 Top Movers**")
                    if metrics['top_gainer']:
                        gainer = metrics['top_gainer']
                        st.success(f"↗ {gainer['symbol']}: {gainer.get('price_change_24h', 0):+.2f}%")
                    
                    if metrics['top_loser']:
                        loser = metrics['top_loser']
                        st.error(f"↘ {loser['symbol']}: {loser.get('price_change_24h', 0):+.2f}%")

def create_divergence_alert(volume_history: List[Dict]) -> None:
    """Create momentum divergence alert if detected"""
    divergence = detect_momentum_divergence(volume_history)
    
    if divergence['has_divergence']:
        div_type = divergence['type']
        strength = divergence['strength']
        
        if div_type == 'bullish':
            alert_color = "#00ff88"
            alert_emoji = "🟢"
            alert_text = "Bullish Divergence Detected"
            description = "Volume increasing while momentum decreasing - potential reversal"
        else:
            alert_color = "#ff4444"
            alert_emoji = "🔴"
            alert_text = "Bearish Divergence Detected"
            description = "Volume decreasing while momentum increasing - potential reversal"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {alert_color}22 0%, {alert_color}11 100%);
                    border: 2px solid {alert_color};
                    border-radius: 0.5rem;
                    padding: 1rem;
                    margin: 1rem 0;">
            <h3 style="margin: 0; color: {alert_color};">
                {alert_emoji} {alert_text}
            </h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                {description}
            </p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                Strength: {strength:.1%} | Volume Trend: {divergence['volume_trend']:.1%} | 
                Momentum Trend: {divergence['momentum_trend']*100:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

#!/usr/bin/env python3
"""
"""

def create_market_health_card(data_entry: Dict, volume_history: List[Dict] = None) -> None:
    """Create comprehensive market health indicator card with trends"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate market health score
    momentum = data_entry['momentum_metrics']['bullish_volume_ratio']
    breadth = data_entry['momentum_metrics'].get('market_breadth', 0.5)
    quality_score = get_data_quality_score(data_entry)
    
    # Composite health score
    health_score = (momentum * 0.4 + breadth * 0.4 + quality_score * 0.2) * 100
    
    # Calculate trends if we have history
    health_trend = None
    ad_trend = None
    vol_trend = None
    
    if volume_history and len(volume_history) >= 2:
        # Get previous entry
        prev_entry = volume_history[-2]
        
        # Calculate previous health score
        prev_momentum = prev_entry['momentum_metrics']['bullish_volume_ratio']
        prev_breadth = prev_entry['momentum_metrics'].get('market_breadth', 0.5)
        prev_quality = get_data_quality_score(prev_entry)
        prev_health = (prev_momentum * 0.4 + prev_breadth * 0.4 + prev_quality * 0.2) * 100
        
        # Calculate trends
        health_trend = health_score - prev_health
        
        # A/D ratio trend
        advance_decline = data_entry['momentum_metrics'].get('advance_decline_ratio', 1)
        prev_ad = prev_entry['momentum_metrics'].get('advance_decline_ratio', 1)
        ad_trend = advance_decline - prev_ad
        
        # Volatility trend
        volatility = data_entry.get('market_stats', {}).get('avg_volatility', 0)
        prev_vol = prev_entry.get('market_stats', {}).get('avg_volatility', 0)
        vol_trend = volatility - prev_vol
    
    # Determine health status
    if health_score >= 70:
        health_status = "🟢 Excellent"
        health_class = "bullish-card"
    elif health_score >= 50:
        health_status = "🟡 Good"
        health_class = "neutral-card"
    else:
        health_status = "🔴 Poor"
        health_class = "bearish-card"
    
    with col1:
        # Format trend for health score
        trend_str = ""
        if health_trend is not None:
            if health_trend > 0:
                trend_str = f" ↑ +{health_trend:.1f}"
            elif health_trend < 0:
                trend_str = f" ↓ {health_trend:.1f}"
            else:
                trend_str = " → 0.0"
        
        st.markdown(f"""
        <div class="metric-card {health_class}">
            <div class="big-metric">{health_score:.0f}{trend_str}</div>
            <div class="metric-label">Market Health Score</div>
            <div class="metric-delta">{health_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        advance_decline = data_entry['momentum_metrics'].get('advance_decline_ratio', 1)
        if advance_decline > 1.5:
            ad_status = "Strong Advance"
            ad_class = "trend-up"
        elif advance_decline < 0.66:
            ad_status = "Strong Decline"
            ad_class = "trend-down"
        else:
            ad_status = "Balanced"
            ad_class = "trend-neutral"
        
        # Format trend for A/D ratio
        trend_str = ""
        if ad_trend is not None:
            if ad_trend > 0.1:
                trend_str = f" ↑"
            elif ad_trend < -0.1:
                trend_str = f" ↓"
            else:
                trend_str = " →"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-metric">{advance_decline:.2f}{trend_str}</div>
            <div class="metric-label">Advance/Decline</div>
            <div class="metric-delta {ad_class}">{ad_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        volatility = data_entry.get('market_stats', {}).get('avg_volatility', 0)
        if volatility > 10:
            vol_status = "High Volatility"
            vol_emoji = "🔥"
        elif volatility > 5:
            vol_status = "Moderate"
            vol_emoji = "⚡"
        else:
            vol_status = "Low"
            vol_emoji = "😴"
        
        # Format trend for volatility
        trend_str = ""
        if vol_trend is not None:
            if vol_trend > 1:
                trend_str = f" ↑"
            elif vol_trend < -1:
                trend_str = f" ↓"
            else:
                trend_str = " →"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-metric">{volatility:.1f}%{trend_str}</div>
            <div class="metric-label">Avg Volatility {vol_emoji}</div>
            <div class="metric-delta">{vol_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        freshness_text, freshness_type, age_hours = get_data_freshness(data_entry)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-metric">{age_hours:.1f}h</div>
            <div class="metric-label">Data Age</div>
            <div class="metric-delta">{freshness_text}</div>
        </div>
        """, unsafe_allow_html=True)

# ================================================================================
# Also update the show_enhanced_market_overview function call
# ================================================================================

def show_enhanced_market_overview(data_entry: Dict, volume_history: List[Dict]) -> None:
    """Main function to display enhanced market overview - UPDATED"""
    st.header("🌐 Enhanced Market Overview")
    
    # Market health card - passes volume_history for trends
    create_market_health_card(data_entry, volume_history)
    
    # Divergence alert if present
    if len(volume_history) >= 4:
        create_divergence_alert(volume_history)
    
    # Momentum cards
    create_enhanced_momentum_cards(data_entry)
    
    # Volume summary
    create_volume_summary_cards(data_entry, volume_history)
    
    # Market structure
    create_market_structure_display(data_entry, volume_history)

    # Divergence Monitor
    create_divergence_monitor(data_entry, volume_history)

    # Market Momentum
    create_market_momentum_oscillator(data_entry, volume_history)
    
    # Category performance - now with trends!
    create_category_performance_grid(data_entry, volume_history)
    
    # Data quality dashboard
    if st.session_state.show_quality_metrics:
        create_data_quality_dashboard(data_entry)

# ================================================================================
# END OF SECTION 2
# ================================================================================

# ================================================================================
# SECTION 3: ADVANCED VOLUME ANALYSIS
# Enhanced volume charts, quality metrics visualization, and volume profiling
# ================================================================================

# ================================================================================
# VOLUME PROFILE ANALYSIS
# ================================================================================

def create_volume_profile_heatmap(volume_history: List[Dict], hours_back: int = 24) -> go.Figure:
    """Create volume profile heatmap showing hourly patterns"""
    if len(volume_history) < 2:
        return go.Figure().add_annotation(text="Insufficient data for heatmap", x=0.5, y=0.5)
    
    # Prepare data for heatmap
    recent_history = volume_history[-min(hours_back, len(volume_history)):]
    
    # Extract hourly data
    hours = []
    volumes = []
    momentums = []
    timestamps = []
    
    for entry in recent_history:
        timestamp = entry['timestamp']
        hour = timestamp.hour if isinstance(timestamp, datetime) else datetime.fromisoformat(timestamp).hour
        
        hours.append(hour)
        volumes.append(entry['total_dollar_volume'])
        momentums.append(entry['momentum_metrics']['bullish_volume_ratio'])
        timestamps.append(timestamp)
    
    # Create 24-hour profile
    hour_profile = defaultdict(lambda: {'volume': [], 'momentum': []})
    
    for h, v, m in zip(hours, volumes, momentums):
        hour_profile[h]['volume'].append(v)
        hour_profile[h]['momentum'].append(m)
    
    # Calculate averages
    profile_data = []
    for hour in range(24):
        if hour in hour_profile:
            avg_volume = np.mean(hour_profile[hour]['volume'])
            avg_momentum = np.mean(hour_profile[hour]['momentum'])
        else:
            avg_volume = 0
            avg_momentum = 0.5
        
        profile_data.append({
            'hour': f"{hour:02d}:00",
            'avg_volume': avg_volume,
            'avg_momentum': avg_momentum,
            'count': len(hour_profile[hour]['volume'])
        })
    
    # Create heatmap
    fig = go.Figure()
    
    # Volume heatmap
    fig.add_trace(go.Bar(
        x=[d['hour'] for d in profile_data],
        y=[d['avg_volume'] for d in profile_data],
        name='Avg Volume',
        marker=dict(
            color=[d['avg_momentum'] for d in profile_data],
            colorscale='RdYlGn',
            colorbar=dict(title="Momentum", x=1.1),
            cmin=0.3,
            cmax=0.7
        ),
        text=[f"Vol: {format_volume(d['avg_volume'])}<br>Mom: {d['avg_momentum']:.1%}<br>Samples: {d['count']}" 
              for d in profile_data],
        hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>'
    ))
    
    # Add momentum line
    fig.add_trace(go.Scatter(
        x=[d['hour'] for d in profile_data],
        y=[d['avg_momentum'] * max(d['avg_volume'] for d in profile_data) for d in profile_data],
        name='Momentum Trend',
        yaxis='y2',
        line=dict(color='white', width=2, dash='dot'),
        opacity=0.7
    ))
    
    fig.update_layout(
        title="24-Hour Volume Profile Heat Map",
        xaxis_title="Hour of Day",
        yaxis_title="Average Volume ($)",
        yaxis2=dict(
            overlaying='y',
            side='right',
            showgrid=False,
            showticklabels=False
        ),
        height=400,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    
    return fig

def create_enhanced_volume_distribution_chart(data_entry: Dict) -> go.Figure:
    """Create enhanced volume distribution with quality indicators"""
    categories = data_entry.get('volume_by_category', {})
    
    # Prepare data
    cat_names = []
    volumes = []
    green_ratios = []
    asset_counts = []
    colors = []
    
    # Color mapping for categories
    color_map = {
        'major_coins': '#FFD700',  # Gold
        'altcoins': '#00CED1',     # Dark Turquoise
        'stablecoins': '#90EE90',  # Light Green
        'meme_coins': '#FF69B4',   # Hot Pink
        'defi': '#9370DB',         # Medium Purple
        'layer2': '#FF8C00',       # Dark Orange
        'gaming': '#FF1493'        # Deep Pink
    }
    
    for cat, cat_data in categories.items():
        total_candles = cat_data['green_candles'] + cat_data['red_candles']
        if total_candles > 0:
            cat_names.append(cat.replace('_', ' ').title())
            volumes.append(cat_data['volume'])
            green_ratios.append(cat_data['green_candles'] / total_candles)
            asset_counts.append(len(cat_data.get('assets', [])))
            colors.append(color_map.get(cat, '#808080'))
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": True}], [{}]],
        subplot_titles=("Volume by Category", "Asset Distribution"),
        vertical_spacing=0.15
    )
    
    # Main volume bars
    fig.add_trace(
        go.Bar(
            x=cat_names,
            y=volumes,
            name="Volume",
            marker_color=colors,
            text=[format_volume(v) for v in volumes],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Volume: %{text}<br>Assets: ' + 
                         '%{customdata}<extra></extra>',
            customdata=asset_counts
        ),
        row=1, col=1, secondary_y=False
    )
    
    # Momentum line
    fig.add_trace(
        go.Scatter(
            x=cat_names,
            y=[r * 100 for r in green_ratios],
            mode='lines+markers+text',
            name="Bullish %",
            line=dict(color='#00ff88', width=3),
            marker=dict(size=12, color='#00ff88', line=dict(width=2, color='white')),
            text=[f"{r:.0%}" for r in green_ratios],
            textposition='top center',
            hovertemplate='<b>%{x}</b><br>Bullish: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=1, secondary_y=True
    )
    
    # Asset count bars
    fig.add_trace(
        go.Bar(
            x=cat_names,
            y=asset_counts,
            name="Asset Count",
            marker_color=colors,
            opacity=0.6,
            text=asset_counts,
            textposition='outside',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Category", row=2, col=1)
    fig.update_yaxes(title_text="Volume ($)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Bullish %", row=1, col=1, secondary_y=True, range=[0, 100])
    fig.update_yaxes(title_text="Number of Assets", row=2, col=1)
    
    fig.update_layout(
        title="Enhanced Category Analysis",
        height=600,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    
    # Add 50% reference line
    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5, 
                  secondary_y=True, row=1, col=1)
    
    return fig

def create_volume_momentum_scatter(assets_df: pd.DataFrame) -> go.Figure:
    """Create volume vs momentum scatter with quality indicators"""
    # Add quality scores if available
    if 'spread_percentage' in assets_df.columns:
        # Use spread as quality indicator (lower is better)
        quality_scores = 1 - (assets_df['spread_percentage'] / assets_df['spread_percentage'].max())
    else:
        quality_scores = [0.5] * len(assets_df)
    
    fig = go.Figure()
    
    # Create scatter plot
    fig.add_trace(go.Scatter(
        x=assets_df['total_dollar_volume'],
        y=assets_df['bullish_momentum'] * 100,
        mode='markers+text',
        marker=dict(
            size=assets_df['volatility_24h'] if 'volatility_24h' in assets_df.columns else 10,
            color=assets_df['price_change_24h'] if 'price_change_24h' in assets_df.columns else 0,
            colorscale='RdYlGn',
            colorbar=dict(title="24h Change %"),
            cmin=-10,
            cmax=10,
            line=dict(width=1, color='white'),
            opacity=0.8
        ),
        text=assets_df['symbol'],
        textposition='top center',
        textfont=dict(size=8),
        hovertemplate='<b>%{text}</b><br>' +
                      'Volume: $%{x:,.0f}<br>' +
                      'Momentum: %{y:.1f}%<br>' +
                      'Price Change: %{marker.color:.1f}%<br>' +
                      '<extra></extra>'
    ))
    
    # Add quadrant lines
    median_volume = assets_df['total_dollar_volume'].median()
    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.3)
    fig.add_vline(x=median_volume, line_dash="dash", line_color="gray", opacity=0.3)
    
    # Add quadrant labels
    fig.add_annotation(x=median_volume*2, y=75, text="High Vol + Bullish", 
                      showarrow=False, font=dict(color='#00ff88', size=12), opacity=0.5)
    fig.add_annotation(x=median_volume*2, y=25, text="High Vol + Bearish", 
                      showarrow=False, font=dict(color='#ff4444', size=12), opacity=0.5)
    fig.add_annotation(x=median_volume/4, y=75, text="Low Vol + Bullish", 
                      showarrow=False, font=dict(color='#ffa726', size=12), opacity=0.5)
    fig.add_annotation(x=median_volume/4, y=25, text="Low Vol + Bearish", 
                      showarrow=False, font=dict(color='#666666', size=12), opacity=0.5)
    
    fig.update_layout(
        title="Volume-Momentum Analysis Matrix",
        xaxis_title="Total Dollar Volume (log scale)",
        yaxis_title="Bullish Momentum (%)",
        xaxis_type="log",
        yaxis=dict(range=[0, 100]),
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_cumulative_volume_flow(volume_history: List[Dict]) -> go.Figure:
    """Create cumulative volume flow chart"""
    if len(volume_history) < 2:
        return go.Figure().add_annotation(text="Insufficient data", x=0.5, y=0.5)
    
    # Calculate cumulative flows
    timestamps = []
    cumulative_total = []
    cumulative_green = []
    cumulative_red = []
    net_flow = []
    
    total = 0
    green = 0
    red = 0
    
    for entry in volume_history:
        timestamp = entry['timestamp']
        total += entry['total_dollar_volume']
        green += entry['green_volume']
        red += entry['red_volume']
        
        timestamps.append(timestamp)
        cumulative_total.append(total)
        cumulative_green.append(green)
        cumulative_red.append(red)
        net_flow.append(green - red)
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Cumulative Volume Flow", "Net Flow Direction"),
        vertical_spacing=0.1
    )
    
    # Cumulative volumes
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=cumulative_total,
            name='Total Volume',
            line=dict(color='#1f77b4', width=3),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.1)'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=cumulative_green,
            name='Bullish Flow',
            line=dict(color='#00ff88', width=2),
            fill='tonexty',
            fillcolor='rgba(0, 255, 136, 0.1)'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=cumulative_red,
            name='Bearish Flow',
            line=dict(color='#ff4444', width=2),
            fill='tonexty',
            fillcolor='rgba(255, 68, 68, 0.1)'
        ),
        row=1, col=1
    )
    
    # Net flow bars
    colors = ['#00ff88' if x > 0 else '#ff4444' for x in net_flow]
    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=net_flow,
            name='Net Flow',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.3, row=2, col=1)
    
    # Update layout
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Volume ($)", row=1, col=1)
    fig.update_yaxes(title_text="Net Flow ($)", row=2, col=1)
    
    fig.update_layout(
        title="Volume Flow Analysis",
        height=600,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    
    return fig

def create_volume_sentiment_heatmap(assets_df: pd.DataFrame) -> go.Figure:
    """Create volume sentiment heatmap for different timeframes"""
    
    # Get available timeframes
    timeframes = []
    for period in [1, 2, 4, 6, 8, 12, 24]:
        field = f'price_change_{period}h'
        if field in assets_df.columns:
            timeframes.append(period)
    
    if len(timeframes) < 2:
        return go.Figure().add_annotation(text="Insufficient timeframe data", x=0.5, y=0.5)
    
    # Get top 30 assets by volume
    top_assets = assets_df.nlargest(30, 'total_dollar_volume')
    
    # Create matrix
    matrix = []
    for _, row in top_assets.iterrows():
        asset_data = []
        for period in timeframes:
            field = f'price_change_{period}h'
            value = row.get(field, 0)
            asset_data.append(value)
        matrix.append(asset_data)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f'{p}h' for p in timeframes],
        y=top_assets['symbol'].tolist(),
        colorscale='RdYlGn',
        zmid=0,
        text=matrix,
        texttemplate='%{text:.1f}%',
        textfont={"size": 8},
        colorbar=dict(title="Price Change %"),
        hovertemplate='%{y} @ %{x}: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Multi-Timeframe Volume Sentiment Heatmap (Top 30 by Volume)",
        xaxis_title="Timeframe",
        yaxis_title="Asset",
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_volume_velocity_chart(volume_history: List[Dict]) -> go.Figure:
    """Create volume velocity (rate of change) chart"""
    if len(volume_history) < 3:
        return go.Figure().add_annotation(text="Insufficient data", x=0.5, y=0.5)
    
    # Calculate velocity metrics
    timestamps = []
    volume_velocity = []
    momentum_velocity = []
    acceleration = []
    
    for i in range(1, len(volume_history)):
        current = volume_history[i]
        previous = volume_history[i-1]
        
        timestamps.append(current['timestamp'])
        
        # Volume velocity (rate of change)
        vol_change = ((current['total_dollar_volume'] / previous['total_dollar_volume']) - 1) * 100
        volume_velocity.append(vol_change)
        
        # Momentum velocity
        mom_change = (current['momentum_metrics']['bullish_volume_ratio'] - 
                     previous['momentum_metrics']['bullish_volume_ratio']) * 100
        momentum_velocity.append(mom_change)
        
        # Acceleration (2nd derivative)
        if i >= 2:
            prev_velocity = volume_velocity[-2] if len(volume_velocity) >= 2 else 0
            accel = vol_change - prev_velocity
            acceleration.append(accel)
        else:
            acceleration.append(0)
    
    # Create figure
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=("Volume Velocity", "Momentum Velocity", "Volume Acceleration"),
        vertical_spacing=0.1
    )
    
    # Volume velocity
    colors_vol = ['#00ff88' if x > 0 else '#ff4444' for x in volume_velocity]
    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=volume_velocity,
            name='Volume Velocity',
            marker_color=colors_vol,
            hovertemplate='%{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Momentum velocity
    colors_mom = ['#00ff88' if x > 0 else '#ff4444' for x in momentum_velocity]
    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=momentum_velocity,
            name='Momentum Velocity',
            marker_color=colors_mom,
            hovertemplate='%{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Acceleration
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=acceleration,
            mode='lines+markers',
            name='Acceleration',
            line=dict(color='#ffa726', width=2),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(255, 167, 38, 0.1)'
        ),
        row=3, col=1
    )
    
    # Add zero lines
    for row in [1, 2, 3]:
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3, row=row, col=1)
    
    # Update axes
    fig.update_yaxes(title_text="Change (%)", row=1, col=1)
    fig.update_yaxes(title_text="Change (%)", row=2, col=1)
    fig.update_yaxes(title_text="Acceleration", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    
    fig.update_layout(
        title="Volume Velocity & Acceleration Analysis",
        height=700,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    
    return fig

def show_advanced_volume_analysis(data_entry: Dict, volume_history: List[Dict], assets_df: pd.DataFrame) -> None:
    """Main function to show advanced volume analysis"""
    st.header("📊 Advanced Volume Analysis")
    
    # Volume analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Volume Profile", 
        "🎯 Category Analysis", 
        "🔄 Flow Analysis",
        "⚡ Velocity Analysis",
        "🎨 Momentum Matrix"
    ])
    
    with tab1:
        st.subheader("24-Hour Volume Profile")
        hours_profile = st.slider("Profile History (hours)", 6, 168, 24, 6, key="profile_hours")
        fig = create_volume_profile_heatmap(volume_history, hours_profile)
        st.plotly_chart(fig, use_container_width=True)
        
        # Profile insights
        st.info("""
        **📊 Volume Profile Insights:**
        - Color intensity shows momentum strength at each hour
        - Identify peak trading hours and momentum patterns
        - Sample count indicates data reliability for each hour
        """)
    
    with tab2:
        st.subheader("Enhanced Category Distribution")
        fig = create_enhanced_volume_distribution_chart(data_entry)
        st.plotly_chart(fig, use_container_width=True)
        
        # Category insights
        total_vol = data_entry['total_dollar_volume']
        top_category = max(data_entry['volume_by_category'].items(), 
                          key=lambda x: x[1]['volume'])
        
        st.success(f"""
        **🏆 Category Leader:** {top_category[0].replace('_', ' ').title()} 
        ({format_volume(top_category[1]['volume'])} - {top_category[1]['volume']/total_vol:.1%} of total)
        """)
    
    with tab3:
        st.subheader("Cumulative Volume Flow")
        fig = create_cumulative_volume_flow(volume_history)
        st.plotly_chart(fig, use_container_width=True)
        
        # Flow analysis
        if len(volume_history) > 0:
            latest = volume_history[-1]
            net = latest['green_volume'] - latest['red_volume']
            flow_direction = "Bullish" if net > 0 else "Bearish"
            
            st.markdown(f"""
            **💰 Current Flow Status:**
            - Direction: **{flow_direction}**
            - Net Flow: {format_volume(abs(net))}
            - Ratio: {latest['green_volume']/latest['total_dollar_volume']:.1%} bullish
            """)
    
    with tab4:
        st.subheader("Volume Velocity & Acceleration")
        fig = create_volume_velocity_chart(volume_history)
        st.plotly_chart(fig, use_container_width=True)
        
        # Velocity insights
        st.info("""
        **⚡ Velocity Metrics Explained:**
        - **Velocity**: Rate of volume change (1st derivative)
        - **Acceleration**: Rate of velocity change (2nd derivative)
        - Positive acceleration indicates increasing momentum
        """)
    
    with tab5:
        st.subheader("Volume-Momentum Matrix")
        
        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            min_vol = st.number_input("Min Volume ($)", 0, 10000000, 1000, 1000, key="matrix_min_vol")
        with col2:
            show_labels = st.checkbox("Show Asset Labels", True, key="matrix_labels")
        
        # Filter assets
        filtered_df = assets_df[assets_df['total_dollar_volume'] >= min_vol]
        
        if len(filtered_df) > 0:
            fig = create_volume_momentum_scatter(filtered_df)
            if not show_labels:
                fig.update_traces(text="", textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No assets match the filter criteria")

# ================================================================================
# END OF SECTION 3
# ================================================================================

# ================================================================================
# SECTION 4: FIXED PRICE ANALYSIS WITH ALL TIMEFRAMES
# Complete price vs volume fix, spread analysis, and price distribution
# ================================================================================

# ================================================================================
# FIXED PRICE VS VOLUME ANALYSIS
# ================================================================================

def create_fixed_price_volume_scatter(assets_df: pd.DataFrame, lookback_period: int) -> go.Figure:
    """Fixed price vs volume scatter that works with all timeframes"""
    
    # Get the correct price field for the selected period
    price_field = get_price_change_field_for_period(lookback_period)
    
    # Validate that the field exists in the dataframe
    if price_field not in assets_df.columns:
        # Fallback to closest available field
        available_periods = []
        for period in [1, 2, 4, 6, 8, 12, 24]:
            field = f'price_change_{period}h'
            if field in assets_df.columns:
                available_periods.append(period)
        
        if available_periods:
            # Use closest available period
            closest_period = min(available_periods, key=lambda x: abs(x - lookback_period))
            price_field = f'price_change_{closest_period}h'
            actual_period = closest_period
        else:
            # No price data available
            return go.Figure().add_annotation(
                text=f"No price data available for {lookback_period}h timeframe",
                x=0.5, y=0.5, showarrow=False
            )
    else:
        actual_period = lookback_period
    
    # Filter out assets with missing price data
    valid_df = assets_df[assets_df[price_field].notna() & (assets_df[price_field] != 0)]
    
    if len(valid_df) == 0:
        return go.Figure().add_annotation(
            text=f"No valid price data for {lookback_period}h timeframe",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Create the figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=valid_df[price_field],
        y=valid_df['total_dollar_volume'],
        mode='markers',
        marker=dict(
            size=valid_df['bullish_momentum'] * 20 + 5,
            color=valid_df['bullish_momentum'],
            colorscale='RdYlGn',
            colorbar=dict(title="Bullish<br>Momentum"),
            cmin=0.2,
            cmax=0.8,
            line=dict(width=1, color='white'),
            opacity=0.8
        ),
        text=valid_df['symbol'],
        customdata=np.column_stack((
            valid_df['current_price'],
            valid_df['green_candles'],
            valid_df['red_candles'],
            valid_df.get('volatility_24h', 0),
            valid_df.get('high_vol_bullish_ratio', 0.5) * 100
        )),
        hovertemplate='<b>%{text}</b><br>' +
                      f'Price Change ({actual_period}h): %{{x:.2f}}%<br>' +
                      'Volume: $%{y:,.0f}<br>' +
                      'Current Price: %{customdata[0]}<br>' +
                      'Green/Red: %{customdata[1]:.0f}/%{customdata[2]:.0f}<br>' +
                      'Volatility: %{customdata[3]:.1f}%<br>' +
                      'High Vol Bullish: %{customdata[4]:.0f}%<br>' +
                      '<extra></extra>'
    ))
    
    # Add quadrant lines
    median_volume = valid_df['total_dollar_volume'].median()
    fig.add_hline(y=median_volume, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant annotations
    max_x = valid_df[price_field].max()
    min_x = valid_df[price_field].min()
    max_y = valid_df['total_dollar_volume'].max()
    
    # Position annotations dynamically
    fig.add_annotation(x=max_x * 0.7, y=max_y * 0.8, 
                      text="🚀 High Volume<br>Price Up", 
                      showarrow=False, font=dict(color='#00ff88', size=10), opacity=0.6)
    fig.add_annotation(x=min_x * 0.7, y=max_y * 0.8, 
                      text="📉 High Volume<br>Price Down", 
                      showarrow=False, font=dict(color='#ff4444', size=10), opacity=0.6)
    fig.add_annotation(x=max_x * 0.7, y=median_volume * 0.2, 
                      text="⚠️ Low Volume<br>Price Up", 
                      showarrow=False, font=dict(color='#ffa726', size=10), opacity=0.6)
    fig.add_annotation(x=min_x * 0.7, y=median_volume * 0.2, 
                      text="💤 Low Volume<br>Price Down", 
                      showarrow=False, font=dict(color='#666666', size=10), opacity=0.6)
    
    # Update layout
    fig.update_layout(
        title=f"Price Performance vs Volume ({actual_period}h Lookback)",
        xaxis_title=f"Price Change {actual_period}h (%)",
        yaxis_title="Total Dollar Volume (log scale)",
        yaxis_type="log",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='closest'
    )
    
    # Add note if using different period than requested
    if actual_period != lookback_period:
        fig.add_annotation(
            text=f"Note: Showing {actual_period}h data (requested {lookback_period}h not available)",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=10, color='#ffa726')
        )
    
    return fig

def create_spread_analysis_chart(assets_df: pd.DataFrame) -> go.Figure:
    """Create spread analysis chart showing bid-ask spreads"""
    
    # Filter assets with spread data
    if 'spread_percentage' in assets_df.columns:
        spread_df = assets_df[assets_df['spread_percentage'].notna()].copy()
    else:
        return go.Figure().add_annotation(text="No spread data available", x=0.5, y=0.5)
    
    if len(spread_df) == 0:
        return go.Figure().add_annotation(text="No valid spread data", x=0.5, y=0.5)
    
    # Sort by spread
    spread_df = spread_df.sort_values('spread_percentage', ascending=False)
    
    # Take top 20 highest spreads
    top_spreads = spread_df.head(20)
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    colors = ['#ff4444' if x > 0.5 else '#ffa726' if x > 0.2 else '#00ff88' 
              for x in top_spreads['spread_percentage']]
    
    fig.add_trace(go.Bar(
        x=top_spreads['symbol'],
        y=top_spreads['spread_percentage'],
        marker_color=colors,
        text=[f"{x:.3f}%" for x in top_spreads['spread_percentage']],
        textposition='outside',
        customdata=np.column_stack((
            top_spreads['current_price'],
            top_spreads['bid'],
            top_spreads['ask'],
            top_spreads['total_dollar_volume']
        )),
        hovertemplate='<b>%{x}</b><br>' +
                      'Spread: %{y:.3f}%<br>' +
                      'Price: $%{customdata[0]:.6f}<br>' +
                      'Bid: $%{customdata[1]:.6f}<br>' +
                      'Ask: $%{customdata[2]:.6f}<br>' +
                      'Volume: $%{customdata[3]:,.0f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add reference lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.3,
                  annotation_text="High Spread (>0.5%)")
    fig.add_hline(y=0.2, line_dash="dash", line_color="orange", opacity=0.3,
                  annotation_text="Medium Spread")
    
    fig.update_layout(
        title="Bid-Ask Spread Analysis (Top 20 Widest Spreads)",
        xaxis_title="Asset",
        yaxis_title="Spread (%)",
        height=400,
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_price_distribution_chart(assets_df: pd.DataFrame, period: int) -> go.Figure:
    """Create price change distribution histogram"""
    
    price_field = get_price_change_field_for_period(period)
    
    if price_field not in assets_df.columns:
        return go.Figure().add_annotation(
            text=f"No price data for {period}h period",
            x=0.5, y=0.5
        )
    
    # Filter valid data
    valid_prices = assets_df[price_field].dropna()
    
    if len(valid_prices) == 0:
        return go.Figure().add_annotation(text="No valid price data", x=0.5, y=0.5)
    
    # Create histogram
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=valid_prices,
        nbinsx=30,
        name='Distribution',
        marker_color='#1f77b4',
        opacity=0.7,
        hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    # Add vertical lines for statistics
    mean_val = valid_prices.mean()
    median_val = valid_prices.median()
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color="yellow", 
                  annotation_text=f"Mean: {mean_val:.1f}%")
    fig.add_vline(x=median_val, line_dash="dash", line_color="cyan",
                  annotation_text=f"Median: {median_val:.1f}%")
    fig.add_vline(x=0, line_dash="solid", line_color="white", opacity=0.5)
    
    # Add statistics box
    positive_count = (valid_prices > 0).sum()
    negative_count = (valid_prices < 0).sum()
    
    fig.add_annotation(
        text=f"Positive: {positive_count} ({positive_count/len(valid_prices):.1%})<br>" +
             f"Negative: {negative_count} ({negative_count/len(valid_prices):.1%})<br>" +
             f"Std Dev: {valid_prices.std():.1f}%",
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        showarrow=False,
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="white",
        borderwidth=1,
        font=dict(size=10, color='white'),
        align="right"
    )
    
    fig.update_layout(
        title=f"Price Change Distribution ({period}h Period)",
        xaxis_title=f"Price Change (%)",
        yaxis_title="Frequency",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        bargap=0.1
    )
    
    return fig

def create_volatility_analysis(assets_df: pd.DataFrame) -> go.Figure:
    """Create volatility analysis chart"""
    
    if 'volatility_24h' not in assets_df.columns:
        return go.Figure().add_annotation(text="No volatility data available", x=0.5, y=0.5)
    
    # Filter and sort by volatility
    vol_df = assets_df[assets_df['volatility_24h'].notna()].copy()
    vol_df = vol_df.sort_values('volatility_24h', ascending=False).head(30)
    
    # Create figure
    fig = go.Figure()
    
    # Add bars colored by momentum
    fig.add_trace(go.Bar(
        x=vol_df['symbol'],
        y=vol_df['volatility_24h'],
        marker=dict(
            color=vol_df['bullish_momentum'],
            colorscale='RdYlGn',
            cmin=0.3,
            cmax=0.7,
            colorbar=dict(title="Momentum")
        ),
        text=[f"{x:.1f}%" for x in vol_df['volatility_24h']],
        textposition='outside',
        customdata=np.column_stack((
            vol_df['current_price'],
            vol_df['high_24h'],
            vol_df['low_24h'],
            vol_df['total_dollar_volume']
        )),
        hovertemplate='<b>%{x}</b><br>' +
                      'Volatility: %{y:.1f}%<br>' +
                      'Current: $%{customdata[0]:.6f}<br>' +
                      'High 24h: $%{customdata[1]:.6f}<br>' +
                      'Low 24h: $%{customdata[2]:.6f}<br>' +
                      'Volume: $%{customdata[3]:,.0f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add reference lines
    avg_volatility = assets_df['volatility_24h'].mean()
    fig.add_hline(y=avg_volatility, line_dash="dash", line_color="yellow", opacity=0.5,
                  annotation_text=f"Market Avg: {avg_volatility:.1f}%")
    
    fig.update_layout(
        title="Top 30 Assets by Volatility (24h Range)",
        xaxis_title="Asset",
        yaxis_title="Volatility (%)",
        height=400,
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def show_fixed_price_analysis(assets_df: pd.DataFrame) -> None:
    """Main function to show fixed price analysis"""
    st.header("💹 Price Analysis")
    
    # Period selector
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col1:
        # Get available periods from data
        available_periods = []
        for period in [1, 2, 4, 6, 8, 12, 24]:
            if f'price_change_{period}h' in assets_df.columns:
                if assets_df[f'price_change_{period}h'].notna().any():
                    available_periods.append(period)
        
        if not available_periods:
            st.error("No price change data available")
            return
        
        selected_period = st.selectbox(
            "Select Analysis Period",
            available_periods,
            index=len(available_periods)-1,  # Default to 24h
            format_func=lambda x: f"{x} hour{'s' if x > 1 else ''}",
            key="price_period_selector"
        )
    
    with col2:
        st.info(f"📊 Analyzing {len(assets_df)} assets with {selected_period}h price data")
    
    with col3:
        min_volume = st.number_input(
            "Min Volume Filter ($)",
            min_value=0,
            max_value=1000000,
            value=1000,
            step=1000,
            key="price_min_volume"
        )
    
    # Filter by minimum volume
    filtered_df = assets_df[assets_df['total_dollar_volume'] >= min_volume].copy()
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Price vs Volume",
        "📊 Distribution",
        "💱 Spread Analysis",
        "🌊 Volatility"
    ])
    
    with tab1:
        st.subheader(f"Price vs Volume Analysis ({selected_period}h)")
        
        # Additional info
        price_field = get_price_change_field_for_period(selected_period)
        if price_field in filtered_df.columns:
            gainers = (filtered_df[price_field] > 0).sum()
            losers = (filtered_df[price_field] < 0).sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Gainers", gainers, f"{gainers/(gainers+losers):.1%}")
            with col2:
                st.metric("Losers", losers, f"{losers/(gainers+losers):.1%}")
            with col3:
                avg_change = filtered_df[price_field].mean()
                st.metric("Avg Change", f"{avg_change:.2f}%")
        
        # Create and show the fixed chart
        fig = create_fixed_price_volume_scatter(filtered_df, selected_period)
        st.plotly_chart(fig, use_container_width=True)
        
        # Quadrant analysis
        if price_field in filtered_df.columns:
            median_vol = filtered_df['total_dollar_volume'].median()
            
            quadrants = {
                'High Vol + Up': filtered_df[(filtered_df[price_field] > 0) & 
                                            (filtered_df['total_dollar_volume'] > median_vol)],
                'High Vol + Down': filtered_df[(filtered_df[price_field] < 0) & 
                                              (filtered_df['total_dollar_volume'] > median_vol)],
                'Low Vol + Up': filtered_df[(filtered_df[price_field] > 0) & 
                                           (filtered_df['total_dollar_volume'] <= median_vol)],
                'Low Vol + Down': filtered_df[(filtered_df[price_field] < 0) & 
                                             (filtered_df['total_dollar_volume'] <= median_vol)]
            }
            
            st.subheader("Quadrant Analysis")
            cols = st.columns(4)
            for idx, (name, data) in enumerate(quadrants.items()):
                with cols[idx]:
                    st.markdown(f"**{name}**")
                    st.write(f"Count: {len(data)}")
                    if len(data) > 0:
                        top_asset = data.nlargest(1, 'total_dollar_volume')
                        if len(top_asset) > 0:
                            symbol = top_asset.iloc[0]['symbol']
                            change = top_asset.iloc[0][price_field]
                            st.write(f"Top: {symbol} ({change:+.1f}%)")
    
    with tab2:
        st.subheader(f"Price Distribution ({selected_period}h)")
        fig = create_price_distribution_chart(filtered_df, selected_period)
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution stats
        if price_field in filtered_df.columns:
            st.subheader("Distribution Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            price_data = filtered_df[price_field].dropna()
            
            with col1:
                st.metric("Mean", f"{price_data.mean():.2f}%")
            with col2:
                st.metric("Median", f"{price_data.median():.2f}%")
            with col3:
                st.metric("Std Dev", f"{price_data.std():.2f}%")
            with col4:
                st.metric("Skewness", f"{price_data.skew():.2f}")
    
    with tab3:
        st.subheader("Bid-Ask Spread Analysis")
        
        if 'spread_percentage' in filtered_df.columns:
            # Spread statistics
            avg_spread = filtered_df['spread_percentage'].mean()
            median_spread = filtered_df['spread_percentage'].median()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Spread", f"{avg_spread:.3f}%")
            with col2:
                st.metric("Median Spread", f"{median_spread:.3f}%")
            with col3:
                tight_spreads = (filtered_df['spread_percentage'] < 0.1).sum()
                st.metric("Tight Spreads (<0.1%)", tight_spreads)
            
            fig = create_spread_analysis_chart(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Spread data not available. Run scanner with enhanced mode for spread analysis.")
    
    with tab4:
        st.subheader("Volatility Analysis (24h)")
        
        if 'volatility_24h' in filtered_df.columns:
            # Volatility stats
            avg_vol = filtered_df['volatility_24h'].mean()
            high_vol_count = (filtered_df['volatility_24h'] > 10).sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Volatility", f"{avg_vol:.1f}%")
            with col2:
                st.metric("High Vol Assets (>10%)", high_vol_count)
            with col3:
                max_vol = filtered_df['volatility_24h'].max()
                max_vol_asset = filtered_df[filtered_df['volatility_24h'] == max_vol]['symbol'].iloc[0]
                st.metric("Most Volatile", f"{max_vol_asset} ({max_vol:.1f}%)")
            
            fig = create_volatility_analysis(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Volatility data not available in current dataset.")

# ================================================================================
# END OF SECTION 4
# ================================================================================

# ================================================================================
# SECTION 5: NEW FEATURES - MARKET STRUCTURE, TOP MOVERS, ADVANCED INSIGHTS
# Market structure visualization, top movers tracking, and new analytical tools
# ================================================================================

# ================================================================================
# TOP MOVERS ANALYSIS
# ================================================================================

def create_top_movers_display(assets_df: pd.DataFrame, period: int = 24) -> None:
    """Create comprehensive top movers display"""
    st.header("🚀 Top Movers")
    
    price_field = get_price_change_field_for_period(period)
    
    if price_field not in assets_df.columns:
        st.warning(f"No price data available for {period}h period")
        return
    
    # Filter valid data
    valid_df = assets_df[assets_df[price_field].notna()].copy()
    
    if len(valid_df) == 0:
        st.warning("No valid price data available")
        return
    
    # Get top gainers and losers
    top_gainers = valid_df.nlargest(10, price_field)
    top_losers = valid_df.nsmallest(10, price_field)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🟢 Top Gainers ({period}h)")
        
        for idx, row in top_gainers.iterrows():
            # Determine momentum emoji
            if row['bullish_momentum'] > 0.7:
                mom_emoji = "🔥"
            elif row['bullish_momentum'] > 0.5:
                mom_emoji = "✅"
            else:
                mom_emoji = "⚠️"
            
            # Create card
            change = row[price_field]
            vol_str = format_volume(row['total_dollar_volume'])
            
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #00ff8822 0%, transparent 100%);
                        border-left: 3px solid #00ff88;
                        padding: 0.5rem;
                        margin: 0.5rem 0;
                        border-radius: 0.25rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="font-size: 1.1rem;">{row['symbol']}</strong> {mom_emoji}
                        <div style="font-size: 0.9rem; color: #b0b0b0;">
                            {format_price(row['current_price'])} | Vol: {vol_str}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.3rem; color: #00ff88; font-weight: bold;">
                            +{change:.2f}%
                        </div>
                        <div style="font-size: 0.8rem; color: #b0b0b0;">
                            Mom: {row['bullish_momentum']:.0%}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader(f"🔴 Top Losers ({period}h)")
        
        for idx, row in top_losers.iterrows():
            # Determine momentum emoji
            if row['bullish_momentum'] < 0.3:
                mom_emoji = "💥"
            elif row['bullish_momentum'] < 0.5:
                mom_emoji = "❌"
            else:
                mom_emoji = "🔄"
            
            # Create card
            change = row[price_field]
            vol_str = format_volume(row['total_dollar_volume'])
            
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #ff444422 0%, transparent 100%);
                        border-left: 3px solid #ff4444;
                        padding: 0.5rem;
                        margin: 0.5rem 0;
                        border-radius: 0.25rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="font-size: 1.1rem;">{row['symbol']}</strong> {mom_emoji}
                        <div style="font-size: 0.9rem; color: #b0b0b0;">
                            {format_price(row['current_price'])} | Vol: {vol_str}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.3rem; color: #ff4444; font-weight: bold;">
                            {change:.2f}%
                        </div>
                        <div style="font-size: 0.8rem; color: #b0b0b0;">
                            Mom: {row['bullish_momentum']:.0%}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def create_volume_leaders_chart(assets_df: pd.DataFrame) -> go.Figure:
    """Create volume leaders visualization"""
    # Get top 20 by volume
    top_volume = assets_df.nlargest(20, 'total_dollar_volume')
    
    # Create figure
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=top_volume['symbol'],
        y=top_volume['total_dollar_volume'],
        marker=dict(
            color=top_volume['bullish_momentum'],
            colorscale='RdYlGn',
            cmin=0.3,
            cmax=0.7,
            colorbar=dict(title="Momentum")
        ),
        text=[format_volume(x) for x in top_volume['total_dollar_volume']],
        textposition='outside',
        customdata=np.column_stack((
            top_volume['current_price'],
            top_volume.get('price_change_24h', 0),
            top_volume['green_candles'],
            top_volume['red_candles']
        )),
        hovertemplate='<b>%{x}</b><br>' +
                      'Volume: %{text}<br>' +
                      'Price: %{customdata[0]}<br>' +
                      '24h Change: %{customdata[1]:.1f}%<br>' +
                      'Candles: %{customdata[2]:.0f} green / %{customdata[3]:.0f} red<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title="Top 20 Assets by Volume",
        xaxis_title="Asset",
        yaxis_title="Volume ($)",
        height=400,
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ================================================================================
# MARKET STRUCTURE VISUALIZATION
# ================================================================================

def create_market_structure_asset_table(data_entry: Dict, assets_df: pd.DataFrame, timeframe: str = 'current') -> None:
    """Create a simple table showing which assets are in which trend category - WITH TRUE MOMENTUM"""
    
    st.subheader("Market Structure by Asset")
    
    # Add lookback period selector
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        lookback_hours = st.select_slider(
            "Analysis Lookback Period",
            options=[4, 8, 12, 24, 48, 72],
            value=24,
            format_func=lambda x: f"{x} hours",
            key="asset_table_lookback"
        )
    
    with col1:
        st.caption(f"📊 Analyzing {len(assets_df)} assets")
    
    with col3:
        st.caption(f"⏰ {lookback_hours}h lookback")
    
    # Categorize each asset
    categorized_assets = {
        'Strong Uptrend 🚀': [],
        'Uptrend 📈': [],
        'Weak Uptrend ↗️': [],
        'Sideways ➡️': [],
        'Weak Downtrend ↘️': [],
        'Downtrend 📉': [],
        'Strong Downtrend 💥': []
    }
    
    for symbol in assets_df['symbol'].values:
        if symbol in data_entry['assets']:
            asset_data = data_entry['assets'][symbol]
            score = calculate_structure_score(asset_data, lookback_hours)
            price_change = asset_data.get('price_change_24h', 0)
            volume = asset_data.get('total_dollar_volume', 0)
            
            # CALCULATE TRUE MOMENTUM - Like a momentum oscillator
            # Compare recent 4h movement to what we'd expect based on the full period average
            
            # Get 4h price change (recent movement)
            recent_change = asset_data.get('price_change_4h', 0)
            
            # Get the full period change
            momentum_field = get_price_change_field_for_period(lookback_hours)
            period_change = asset_data.get(momentum_field, 0)
            
            # Calculate what 4h SHOULD be if moving at the average pace
            expected_4h_change = (period_change / lookback_hours) * 4
            
            # MOMENTUM = actual recent movement - expected movement
            # Positive = accelerating, Negative = decelerating
            true_momentum = recent_change - expected_4h_change
            
            # Format for display
            momentum_display = f"{true_momentum:+.2f}%"
            
            # Create asset info dict
            asset_info = {
                'Symbol': symbol,
                '24h Change': f"{price_change:+.1f}%",
                'Volume': f"${volume/1e6:.1f}M" if volume >= 1e6 else f"${volume/1e3:.0f}K",
                'Momentum': momentum_display,  # TRUE MOMENTUM!
                'Score': score,
                'RawMomentum': true_momentum  # For sorting
            }
            
            # Classify based on score (no volatility check)
            if score >= 0.5:
                categorized_assets['Strong Uptrend 🚀'].append(asset_info)
            elif score >= 0.3:
                categorized_assets['Uptrend 📈'].append(asset_info)
            elif score >= 0.1:
                categorized_assets['Weak Uptrend ↗️'].append(asset_info)
            elif score <= -0.5:
                categorized_assets['Strong Downtrend 💥'].append(asset_info)
            elif score <= -0.3:
                categorized_assets['Downtrend 📉'].append(asset_info)
            elif score <= -0.1:
                categorized_assets['Weak Downtrend ↘️'].append(asset_info)
            else:
                categorized_assets['Sideways ➡️'].append(asset_info)
    
    # Create tabs for each category with assets
    non_empty_categories = {k: v for k, v in categorized_assets.items() if v}
    
    if not non_empty_categories:
        st.warning("No assets to display")
        return
    
    # Show summary first
    col1, col2, col3 = st.columns(3)
    with col1:
        total_bullish = (len(categorized_assets['Strong Uptrend 🚀']) + 
                        len(categorized_assets['Uptrend 📈']) + 
                        len(categorized_assets['Weak Uptrend ↗️']))
        st.metric("Bullish Assets", total_bullish)
    
    with col2:
        total_bearish = (len(categorized_assets['Strong Downtrend 💥']) + 
                        len(categorized_assets['Downtrend 📉']) + 
                        len(categorized_assets['Weak Downtrend ↘️']))
        st.metric("Bearish Assets", total_bearish)
    
    with col3:
        total_neutral = len(categorized_assets['Sideways ➡️'])
        st.metric("Sideways Assets", total_neutral)
    
    # Add momentum explanation
    with st.expander("ℹ️ Understanding Momentum Values"):
        st.markdown(f"""
        **Momentum shows recent price acceleration vs {lookback_hours}h average:**
        
        Calculation: `4h price change - expected 4h change based on {lookback_hours}h average`
        
        - **Positive values**: Price is moving **faster** than the {lookback_hours}h average pace (gaining momentum)
        - **Negative values**: Price is moving **slower** than the {lookback_hours}h average pace (losing momentum)
        - **Near zero**: Price is moving at the same pace as the {lookback_hours}h average
        
        **Examples:**
        - **+2.5%**: Last 4h was 2.5% stronger than expected (accelerating!)
        - **-1.8%**: Last 4h was 1.8% weaker than expected (decelerating)
        - **+0.1%**: Steady momentum, matching the average pace
        
        This mimics a traditional momentum oscillator!
        """)
    
    # Create tabs for categories
    tab_names = list(non_empty_categories.keys())
    tabs = st.tabs(tab_names)
    
    for tab, (category, assets) in zip(tabs, non_empty_categories.items()):
        with tab:
            if assets:
                # Sort by score (strength of trend)
                sorted_assets = sorted(assets, key=lambda x: x['Score'], reverse=True)
                
                # Create DataFrame for display (without Score and RawMomentum columns)
                display_assets = [{k: v for k, v in asset.items() if k not in ['Score', 'RawMomentum']} 
                                 for asset in sorted_assets]
                
                df = pd.DataFrame(display_assets)
                
                # Display as a nice table
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                        "24h Change": st.column_config.TextColumn("24h Change", width="small"),
                        "Volume": st.column_config.TextColumn("Volume", width="small"),
                        "Momentum": st.column_config.TextColumn("Momentum (4h)", width="small"),
                    }
                )
                
                # Show top movers in this category
                if len(sorted_assets) > 0:
                    st.caption(f"Strongest in category: {sorted_assets[0]['Symbol']} ({sorted_assets[0]['Momentum']} momentum)")
                
                # Add TradingView watchlist export
                col1, col2 = st.columns([3, 1])
                with col2:
                    # Create TradingView compatible watchlist
                    symbols_for_tv = [asset['Symbol'] for asset in sorted_assets]
                    
                    # TradingView format: "COINBASE:BTCUSD,COINBASE:ETHUSD,..."
                    tv_watchlist = ",".join([f"COINBASE:{symbol}USD" for symbol in symbols_for_tv])
                    
                    # Create downloadable text file
                    watchlist_content = tv_watchlist
                    
                    # Clean category name for filename
                    clean_category = category.replace(' ', '_').replace('🚀', '').replace('📈', '').replace('📉', '').replace('💥', '').replace('↗️', '').replace('↘️', '').replace('➡️', '').strip()
                    
                    st.download_button(
                        label="📥 TradingView List",
                        data=watchlist_content,
                        file_name=f"tv_watchlist_{clean_category}_{lookback_hours}h.txt",
                        mime="text/plain",
                        help="Download symbols in TradingView format. Copy the contents and paste into TradingView's watchlist import."
                    )
            else:
                st.info(f"No assets in {category}")
    
    # Add a simple view option - just lists
    with st.expander("Quick View - Asset Lists"):
        cols = st.columns(2)
        
        with cols[0]:
            st.markdown("### 🟢 Bullish Assets")
            for category in ['Strong Uptrend 🚀', 'Uptrend 📈', 'Weak Uptrend ↗️']:
                if categorized_assets[category]:
                    st.markdown(f"**{category}:**")
                    symbols = [a['Symbol'] for a in categorized_assets[category]]
                    # Create rows of 5 symbols
                    for i in range(0, len(symbols), 5):
                        st.text(" ".join(symbols[i:i+5]))
        
        with cols[1]:
            st.markdown("### 🔴 Bearish Assets")
            for category in ['Strong Downtrend 💥', 'Downtrend 📉', 'Weak Downtrend ↘️']:
                if categorized_assets[category]:
                    st.markdown(f"**{category}:**")
                    symbols = [a['Symbol'] for a in categorized_assets[category]]
                    # Create rows of 5 symbols
                    for i in range(0, len(symbols), 5):
                        st.text(" ".join(symbols[i:i+5]))
        
        # Sideways below
        if categorized_assets['Sideways ➡️']:
            st.markdown("### 🟠 Sideways Assets")
            st.markdown(f"**Sideways ➡️:**")
            symbols = [a['Symbol'] for a in categorized_assets['Sideways ➡️']]
            for i in range(0, len(symbols), 5):
                st.text(" ".join(symbols[i:i+5]))

# ================================================================================
# CORRELATION ANALYSIS
# ================================================================================

def create_correlation_matrix(assets_df: pd.DataFrame, metric: str = 'price_change_24h') -> go.Figure:
    """Create correlation matrix for top assets"""
    
    # Get top 20 by volume
    top_assets = assets_df.nlargest(20, 'total_dollar_volume')
    
    # Create correlation matrix
    metrics = ['bullish_momentum', 'total_dollar_volume', metric, 'volatility_24h']
    available_metrics = [m for m in metrics if m in top_assets.columns]
    
    if len(available_metrics) < 2:
        return go.Figure().add_annotation(text="Insufficient metrics for correlation", x=0.5, y=0.5)
    
    corr_matrix = top_assets[available_metrics].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        colorbar=dict(title="Correlation"),
        hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Metric Correlation Matrix",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ================================================================================
# MOMENTUM STRENGTH INDICATOR
# ================================================================================

def create_volume_sentiment_gauge(data_entry: Dict) -> go.Figure:
    """Create comprehensive volume sentiment gauge"""
    
    # Calculate composite sentiment score
    bullish_vol = data_entry['momentum_metrics']['bullish_volume_ratio']
    market_breadth = data_entry['momentum_metrics'].get('market_breadth', 0.5)
    large_cap = data_entry['momentum_metrics']['large_cap_sentiment']
    alt_sentiment = data_entry['momentum_metrics']['alt_momentum']
    
    # Weighted composite
    composite = (bullish_vol * 0.3 + market_breadth * 0.3 + large_cap * 0.2 + alt_sentiment * 0.2) * 100
    
    # Create gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = composite,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Composite Market Strength"},
        delta = {'reference': 50, 'increasing': {'color': "#00ff88"}, 'decreasing': {'color': "#ff4444"}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': get_sentiment_color(composite/100)},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#ff0000'},
                {'range': [20, 40], 'color': '#ff6666'},
                {'range': [40, 60], 'color': '#ffff00'},
                {'range': [60, 80], 'color': '#66ff66'},
                {'range': [80, 100], 'color': '#00ff00'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': composite
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'size': 14}
    )
    
    return fig

# ================================================================================
# UNUSUAL ACTIVITY DETECTION
# ================================================================================

def detect_unusual_activity(assets_df: pd.DataFrame) -> pd.DataFrame:
    """Detect assets with unusual activity patterns"""
    unusual = []
    
    for _, row in assets_df.iterrows():
        flags = []
        score = 0
        
        # Check for unusual volume
        if 'avg_volume' in row and row['total_volume'] > row['avg_volume'] * 2:
            flags.append("High Volume")
            score += 2
        
        # Check for unusual momentum vs price action
        if row['bullish_momentum'] > 0.7 and row.get('price_change_24h', 0) < -5:
            flags.append("Bullish Divergence")
            score += 3
        elif row['bullish_momentum'] < 0.3 and row.get('price_change_24h', 0) > 5:
            flags.append("Bearish Divergence")
            score += 3
        
        # Check for high volatility
        if row.get('volatility_24h', 0) > 15:
            flags.append("High Volatility")
            score += 1
        
        # Check for unusual spread
        if row.get('spread_percentage', 0) > 0.5:
            flags.append("Wide Spread")
            score += 1
        
        if flags:
            unusual.append({
                'symbol': row['symbol'],
                'flags': ', '.join(flags),
                'score': score,
                'price': row['current_price'],
                'volume': row['total_dollar_volume'],
                'momentum': row['bullish_momentum']
            })
    
    return pd.DataFrame(unusual).sort_values('score', ascending=False)

def show_new_features_analysis(data_entry: Dict, assets_df: pd.DataFrame, volume_history: List[Dict]) -> None:
    """Main function to show new feature analysis"""
    st.header("🎯 Advanced Market Analysis")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Top Movers",
        "🗂️ Market Structure",
        "🔥 Volume Sentiment Analysis",
        "🔍 Unusual Activity",
        "📊 Correlations"
    ])
    
    with tab1:
        # Period selector for top movers
        col1, col2 = st.columns([1, 3])
        with col1:
            mover_period = st.selectbox(
                "Timeframe",
                [1, 4, 24],
                index=2,
                format_func=lambda x: f"{x}h",
                key="mover_period"
            )
        
        create_top_movers_display(assets_df, mover_period)
        
        st.subheader("📊 Volume Leaders")
        fig = create_volume_leaders_chart(assets_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Market Structure Analysis")
        
        # Add timeframe selector for treemap
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Check available timeframes
            available_tf = ['Current (4h)']
            if 'market_structure_short' in assets_df.columns:
                available_tf.append('Short Term (24h)')
            if 'market_structure_medium' in assets_df.columns:
                available_tf.append('Medium Term (3d)')
            if 'market_structure_long' in assets_df.columns:
                available_tf.append('Long Term (1w)')
            
            tf_map = {
                'Current (4h)': 'current',
                'Short Term (24h)': 'short',
                'Medium Term (3d)': 'medium',
                'Long Term (1w)': 'long'
            }
            
            selected_tf_display = st.selectbox(
                "Treemap Timeframe",
                available_tf,
                key="treemap_timeframe"
            )
            
            selected_tf = tf_map[selected_tf_display]
        
        # Structure stats for selected timeframe
        structure = analyze_market_structure(data_entry, 24)

        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dominant Pattern", structure['dominant_structure'].title())
        with col2:
            st.metric("Pattern Strength", f"{structure['strength']:.1%}")
        with col3:
            st.metric("Total Assets", len(assets_df))
        
        # Market Structure Table
        create_market_structure_asset_table(data_entry, assets_df, selected_tf)
    
    with tab3:
        st.subheader("Multi-Timeframe Volume Sentiment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment gauge (renamed from momentum gauge)
            fig = create_volume_sentiment_gauge(data_entry)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment stats (renamed from momentum stats)
            st.markdown("""
            **📈 Sentiment Components:**
            """)
            
            components = [
                ("Volume Ratio", data_entry['momentum_metrics']['bullish_volume_ratio'], 0.3),
                ("Market Breadth", data_entry['momentum_metrics'].get('market_breadth', 0.5), 0.3),
                ("Large Cap", data_entry['momentum_metrics']['large_cap_sentiment'], 0.2),
                ("Altcoins", data_entry['momentum_metrics']['alt_momentum'], 0.2)
            ]
            
            for name, value, weight in components:
                color = get_sentiment_color(value)
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>{name} (×{weight:.0%})</span>
                        <span style="color: {color}; font-weight: bold;">{value:.1%}</span>
                    </div>
                    <div style="background: #333; height: 8px; border-radius: 4px; overflow: hidden;">
                        <div style="background: {color}; width: {value*100}%; height: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Heatmap (renamed)
        st.subheader("Volume Sentiment Heatmap")
        fig = create_volume_sentiment_heatmap(assets_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🔍 Unusual Activity Detection")
        
        unusual_df = detect_unusual_activity(assets_df)
        
        if len(unusual_df) > 0:
            st.warning(f"Found {len(unusual_df)} assets with unusual activity patterns")
            
            # Display unusual activity
            for _, row in unusual_df.head(10).iterrows():
                alert_color = "#ff4444" if row['score'] >= 3 else "#ffa726"
                
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, {alert_color}22 0%, transparent 100%);
                            border-left: 3px solid {alert_color};
                            padding: 0.75rem;
                            margin: 0.5rem 0;
                            border-radius: 0.25rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <strong style="font-size: 1.1rem;">{row['symbol']}</strong>
                            <div style="font-size: 0.9rem; color: #ffa726; margin: 0.25rem 0;">
                                ⚠️ {row['flags']}
                            </div>
                            <div style="font-size: 0.85rem; color: #b0b0b0;">
                                Price: {format_price(row['price'])} | Volume: {format_volume(row['volume'])}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.9rem; color: {alert_color};">
                                Score: {row['score']}/7
                            </div>
                            <div style="font-size: 0.85rem; color: #b0b0b0;">
                                Sentiment: {row['momentum']:.0%}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No unusual activity patterns detected")
    
    with tab5:
        st.subheader("Correlation Analysis")
        
        # Correlation matrix
        fig = create_correlation_matrix(assets_df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **📊 Correlation Insights:**
        - Values close to 1: Strong positive correlation
        - Values close to -1: Strong negative correlation
        - Values near 0: No correlation
        """)

# ================================================================================
# END OF SECTION 5
# ================================================================================

# ================================================================================
# SECTION 6: FINAL INTEGRATION - MAIN APP, SIDEBAR, EXPORT FUNCTIONS
# Bringing everything together with navigation and performance optimizations
# ================================================================================

# ================================================================================
# SIDEBAR CONFIGURATION
# ================================================================================

def configure_enhanced_sidebar():
    """Configure enhanced sidebar with all controls"""
    st.sidebar.header("🎛️ Control Panel")
    
    # Data status
    st.sidebar.subheader("📊 Data Status")
    
    available_files = get_available_data_files()
    
    # Show data availability
    if available_files['hourly']:
        st.sidebar.success(f"✅ {len(available_files['hourly'])} hourly files")
    else:
        st.sidebar.error("❌ No hourly data")
    
    if available_files['scanner_data']:
        st.sidebar.info("✅ Persistent scanner data found")
    
    # File selection
    st.sidebar.subheader("📁 Data Selection")
    
    if available_files['hourly']:
        selected_file = st.sidebar.selectbox(
            "Select Data File",
            available_files['hourly'],
            format_func=lambda x: os.path.basename(x)[:30] + "...",
            key="data_file_selector"
        )
    else:
        st.sidebar.error("No data files found!")
        st.sidebar.code("python enhanced_scanner.py", language="bash")
        return None
    
    # Analysis options
    st.sidebar.subheader("🎯 Analysis Options")
    
    analysis_view = st.sidebar.radio(
        "View Mode",
        ["📈 Overview", "📊 Volume Analysis", "💹 Price Analysis", 
         "🚀 Top Movers", "🎯 Advanced", "📋 Reports"],
        key="view_mode"
    )
    
    # Global filters
    st.sidebar.subheader("🔍 Global Filters")
    
    min_volume = st.sidebar.number_input(
        "Min Volume ($)",
        min_value=0,
        max_value=10000000,
        value=1000,
        step=1000,
        key="global_min_volume"
    )
    
    selected_categories = st.sidebar.multiselect(
        "Categories",
        ["BTC & ETH", "Altcoins", "Stablecoins", "Meme Coins", "DeFi", "Layer2", "Gaming"],
        default=["BTC & ETH", "Altcoins", "Meme Coins"],
        key="category_filter"
    )
    
    # Display settings
    st.sidebar.subheader("⚙️ Display Settings")
    
    show_quality = st.sidebar.checkbox(
        "Show Data Quality Metrics",
        value=True,
        key="show_quality"
    )
    
    auto_refresh = st.sidebar.checkbox(
        "Auto Refresh (5 min)",
        value=False,
        key="auto_refresh"
    )
    
    if auto_refresh:
        st.sidebar.info("🔄 Auto-refresh enabled")
    
    # Export options
    st.sidebar.subheader("💾 Export Options")
    
    if st.sidebar.button("📥 Export Report", key="export_report"):
        st.session_state.export_requested = True
    
    # Info section
    st.sidebar.subheader("ℹ️ Information")
    
    if selected_file:
        try:
            file_stats = os.stat(selected_file)
            file_size = file_stats.st_size / 1024
            mod_time = datetime.fromtimestamp(file_stats.st_mtime)
            
            st.sidebar.caption(f"Size: {file_size:.1f} KB")
            st.sidebar.caption(f"Modified: {mod_time.strftime('%Y-%m-%d %H:%M')}")
        except:
            pass
    
    return selected_file

# ================================================================================
# EXPORT FUNCTIONS
# ================================================================================

def generate_market_report(data_entry: Dict, assets_df: pd.DataFrame) -> str:
    """Generate comprehensive market report"""
    
    report = f"""
# 📊 CRYPTO MARKET ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🌍 MARKET OVERVIEW
- **Total Volume**: {format_volume(data_entry['total_dollar_volume'])}
- **Market Sentiment**: {data_entry['momentum_metrics']['bullish_volume_ratio']:.1%} Bullish
- **Active Assets**: {data_entry['successful_collections']}/{data_entry['total_assets']}
- **Data Quality**: {get_data_quality_score(data_entry):.1%}

## 📈 MOMENTUM METRICS
- **Overall Momentum**: {data_entry['momentum_metrics']['bullish_volume_ratio']:.1%}
- **Large Cap Sentiment**: {data_entry['momentum_metrics']['large_cap_sentiment']:.1%}
- **Alt Momentum**: {data_entry['momentum_metrics']['alt_momentum']:.1%}
- **Market Breadth**: {data_entry['momentum_metrics'].get('market_breadth', 0):.1%}
- **Advance/Decline**: {data_entry['momentum_metrics'].get('advance_decline_ratio', 1):.2f}

## 🏗️ MARKET STRUCTURE
"""
    
    structure = analyze_market_structure(data_entry)
    report += f"""
- **Dominant Pattern**: {structure['dominant_structure'].title()}
- **Pattern Strength**: {structure['strength']:.1%}
- **Uptrending**: {structure['trending_up']} assets ({structure['trending_up_pct']:.1f}%)
- **Downtrending**: {structure['trending_down']} assets ({structure['trending_down_pct']:.1f}%)
- **Sideways**: {structure['sideways']} assets ({structure['sideways_pct']:.1f}%)

## 🚀 TOP PERFORMERS (24H)
"""
    
    # Add top movers
    if 'price_change_24h' in assets_df.columns:
        top_gainers = assets_df.nlargest(5, 'price_change_24h')
        for _, row in top_gainers.iterrows():
            report += f"- **{row['symbol']}**: +{row['price_change_24h']:.2f}% | Volume: {format_volume(row['total_dollar_volume'])}\n"
    
    report += "\n## 📉 BIGGEST DECLINERS (24H)\n"
    
    if 'price_change_24h' in assets_df.columns:
        top_losers = assets_df.nsmallest(5, 'price_change_24h')
        for _, row in top_losers.iterrows():
            report += f"- **{row['symbol']}**: {row['price_change_24h']:.2f}% | Volume: {format_volume(row['total_dollar_volume'])}\n"
    
    report += f"""

## 📊 CATEGORY BREAKDOWN
"""
    
    for cat_name, cat_data in data_entry['volume_by_category'].items():
        if cat_data['volume'] > 0:
            total_candles = cat_data['green_candles'] + cat_data['red_candles']
            momentum = (cat_data['green_candles'] / total_candles * 100) if total_candles > 0 else 50
            report += f"- **{cat_name.replace('_', ' ').title()}**: {format_volume(cat_data['volume'])} ({momentum:.0f}% bullish)\n"
    
    return report

def export_to_csv(assets_df: pd.DataFrame) -> bytes:
    """Export assets data to CSV"""
    # Select relevant columns
    export_columns = [
        'symbol', 'current_price', 'total_dollar_volume', 
        'bullish_momentum', 'price_change_24h', 'volatility_24h',
        'green_candles', 'red_candles', 'market_structure'
    ]
    
    available_columns = [col for col in export_columns if col in assets_df.columns]
    export_df = assets_df[available_columns].copy()
    
    # Format numbers
    if 'current_price' in export_df.columns:
        export_df['current_price'] = export_df['current_price'].round(6)
    if 'total_dollar_volume' in export_df.columns:
        export_df['total_dollar_volume'] = export_df['total_dollar_volume'].round(2)
    
    return export_df.to_csv(index=False).encode('utf-8')

# ================================================================================
# MAIN APPLICATION
# ================================================================================

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.title("🚀 Elwood's CB Dash")
    st.markdown("*Hourly TF Analysis*")
    
    # Configure sidebar and get selected file
    selected_file = configure_enhanced_sidebar()
    
    if not selected_file:
        st.error("❌ No data files available!")
        st.info("""
        ### 📊 Getting Started
        
        Run the enhanced scanner to collect data:
        ```bash
        python enhanced_scanner.py --test
        ```
        
        Or start continuous collection:
        ```bash
        python enhanced_scanner.py
        ```
        """)
        return
    
    # Load data
    with st.spinner("Loading data..."):
        data = load_enhanced_volume_data(selected_file)
    
    if not data or 'volume_history' not in data or not data['volume_history']:
        st.error("Failed to load data or data is empty!")
        return
    
    # Get latest entry and prepare dataframes
    volume_history = data['volume_history']
    latest_entry = volume_history[-1]
    
    # Show data freshness
    freshness_text, freshness_type, age_hours = get_data_freshness(latest_entry)
    
    col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
        if freshness_type == "success":
            st.success(freshness_text)
        elif freshness_type == "warning":
            st.warning(freshness_text)
        else:
            st.error(freshness_text)
    
    with col2:
        quality_score = get_data_quality_score(latest_entry)
        quality_text = "High" if quality_score > 0.8 else "Medium" if quality_score > 0.5 else "Low"
        st.info(f"📊 Data Quality: {quality_text} ({quality_score:.0%})")
    
    with col3:
        st.info(f"📅 {latest_entry['timestamp'].strftime('%Y-%m-%d %H:%M')}")
    
    # Prepare assets dataframe
    assets_list = filter_assets_for_analysis(latest_entry['assets'])
    
    if not assets_list:
        st.error("No valid asset data available!")
        return
    
    assets_df = pd.DataFrame(assets_list)
    
    # Apply category filter
    category_map = {
        "BTC & ETH": "major_coins",
        "Altcoins": "altcoins",
        "Stablecoins": "stablecoins",
        "Meme Coins": "meme_coins",
        "DeFi": "defi",
        "Layer2": "layer2",
        "Gaming": "gaming"
    }
    
    selected_internal_cats = [category_map.get(cat, cat.lower()) for cat in st.session_state.category_filter]
    categories = get_asset_categories(latest_entry['assets'])
    
    # Filter assets by category
    filtered_symbols = []
    for cat in selected_internal_cats:
        if cat in categories:
            filtered_symbols.extend(categories[cat])
    
    if filtered_symbols:
        assets_df = assets_df[assets_df['symbol'].isin(filtered_symbols)]
    
    # Apply volume filter
    assets_df = assets_df[assets_df['total_dollar_volume'] >= st.session_state.global_min_volume]
    
    if len(assets_df) == 0:
        st.warning("No assets match the current filters. Try adjusting the filters.")
        return
    
    # Navigation based on view mode
    view_mode = st.session_state.view_mode
    
    if "Overview" in view_mode:
        show_enhanced_market_overview(latest_entry, volume_history)
        
    elif "Volume Analysis" in view_mode:
        show_advanced_volume_analysis(latest_entry, volume_history, assets_df)
        
    elif "Price Analysis" in view_mode:
        show_fixed_price_analysis(assets_df)
        
    elif "Top Movers" in view_mode:
        create_top_movers_display(assets_df, 24)
        
    elif "Advanced" in view_mode:
        show_new_features_analysis(latest_entry, assets_df, volume_history)
        
    elif "Reports" in view_mode:
        st.header("📋 Reports & Export")
        
        # Generate report
        report = generate_market_report(latest_entry, assets_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Market Report")
            st.text_area("Report Content", report, height=400)
            
            # Download report
            st.download_button(
                label="📥 Download Report (TXT)",
                data=report,
                file_name=f"market_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )
        
        with col2:
            st.subheader("📊 Data Export")
            
            # CSV export
            csv_data = export_to_csv(assets_df)
            st.download_button(
                label="📥 Download Data (CSV)",
                data=csv_data,
                file_name=f"market_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
            # JSON export
            json_data = json.dumps(latest_entry, default=json_serialize_fix, indent=2)
            st.download_button(
                label="📥 Download Raw Data (JSON)",
                data=json_data,
                file_name=f"raw_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
    
    # Auto-refresh
    if st.session_state.auto_refresh:
        time.sleep(1800)  # 5 minutes
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption(f"Enhanced Dashboard v6.0 | Data: {len(assets_df)} assets | Quality: {quality_score:.0%}")

# ================================================================================
# RUN APPLICATION
# ================================================================================

if __name__ == "__main__":
    main()

# ================================================================================
# END OF ENHANCED DASHBOARD - COMPLETE
# ================================================================================
"""
"""