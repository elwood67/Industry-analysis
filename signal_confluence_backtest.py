"""
Signal Confluence Backtester
============================
Tests SPY performance when multiple signals fire simultaneously.

Analyzes:
1. Bullish confluence (2, 3, 4, 5, 6+ signals firing together)
2. Bearish confluence (2, 3, 4, 5, 6 signals firing together)
3. Specific signal combinations
4. Tier 1 only confluence
5. Mixed tier confluence

Author: Dave's Trading Lab
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from itertools import combinations
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(r'C:\Users\davet\Documents\GitHub\Industry-analysis\Data\stock_scores')

# All signals from the dashboard
SIGNALS = {
    # ===========================================
    # 🏆 TIER 1 - TOP BULLISH SIGNALS (80%+ Win Rate)
    # ===========================================
    'net_score_turns_positive': {
        'name': 'Net Score Turns Positive',
        'type': 'bullish',
        'tier': 1,
        'threshold': 0,
        'direction': 'crosses_above',
        'metric': 'market_net'
    },
    'avg_bearish_extreme': {
        'name': 'Avg Bearish > 55',
        'type': 'bullish',
        'tier': 1,
        'threshold': 55,
        'direction': 'above',
        'metric': 'market_bear'
    },
    'bull_roc_washout': {
        'name': 'Bull ROC 5d < -10',
        'type': 'bullish',
        'tier': 1,
        'threshold': -10,
        'direction': 'below',
        'metric': 'bull_roc_5d'
    },
    'real_estate_extreme_bearish': {
        'name': 'Real Estate Extremely Bearish',
        'type': 'bullish',
        'tier': 1,
        'threshold': 30,
        'direction': 'below',
        'metric': 'real_estate_breadth'
    },
    'utilities_extreme_bearish': {
        'name': 'Utilities Extremely Bearish',
        'type': 'bullish',
        'tier': 1,
        'threshold': 30,
        'direction': 'below',
        'metric': 'utilities_breadth'
    },
    'sharp_rotation_to_value': {
        'name': 'Sharp Rotation to Value',
        'type': 'bullish',
        'tier': 1,
        'threshold': -5,
        'direction': 'below',
        'metric': 'growth_value_diff'
    },
    'breadth_crosses_above_50': {
        'name': 'Breadth Crosses Above 50%',
        'type': 'bullish',
        'tier': 1,
        'threshold': 50,
        'direction': 'crosses_above',
        'metric': 'market_breadth'
    },
    'bull_bear_convergence': {
        'name': 'Bull/Bear Convergence',
        'type': 'bullish',
        'tier': 1,
        'threshold': 5,
        'direction': 'below',
        'metric': 'bull_bear_spread'
    },
    
    # ===========================================
    # 🥈 TIER 2 - BULLISH SIGNALS (60-80% Win Rate)
    # ===========================================
    'breadth_very_low': {
        'name': 'Very Low Breadth',
        'type': 'bullish',
        'tier': 2,
        'threshold': 20,
        'direction': 'below',
        'metric': 'market_breadth'
    },
    'zscore_extreme_oversold': {
        'name': 'Extremely Oversold (Z-Score)',
        'type': 'bullish',
        'tier': 2,
        'threshold': -2,
        'direction': 'below',
        'metric': 'market_zscore'
    },
    'loss_streak_long': {
        'name': 'Long Loss Streak',
        'type': 'bullish',
        'tier': 2,
        'threshold': -5,
        'direction': 'below',
        'metric': 'market_streak'
    },
    'zscore_oversold': {
        'name': 'Oversold (Z-Score)',
        'type': 'bullish',
        'tier': 2,
        'threshold': -1,
        'direction': 'below',
        'metric': 'market_zscore'
    },
    'breadth_low': {
        'name': 'Low Breadth (20-40%)',
        'type': 'bullish',
        'tier': 2,
        'threshold': 40,
        'direction': 'below',
        'metric': 'market_breadth'
    },
    'momentum_strong_negative': {
        'name': 'Strong Negative Momentum',
        'type': 'bullish',
        'tier': 2,
        'threshold': -10,
        'direction': 'below',
        'metric': 'net_roc_5d'
    },
    
    # ===========================================
    # 🔴 BEARISH/CAUTION SIGNALS
    # ===========================================
    'breadth_very_high': {
        'name': 'Very High Breadth (>80%)',
        'type': 'bearish',
        'tier': 1,
        'threshold': 80,
        'direction': 'above',
        'metric': 'market_breadth'
    },
    'breadth_high': {
        'name': 'Breadth > 70%',
        'type': 'bearish',
        'tier': 1,
        'threshold': 70,
        'direction': 'above',
        'metric': 'market_breadth'
    },
    'net_score_above_20': {
        'name': 'Net Score > 20',
        'type': 'bearish',
        'tier': 1,
        'threshold': 20,
        'direction': 'above',
        'metric': 'market_net'
    },
    'zscore_overbought': {
        'name': 'Overbought (Z-Score)',
        'type': 'bearish',
        'tier': 2,
        'threshold': 1,
        'direction': 'above',
        'metric': 'market_zscore'
    },
    'win_streak_long': {
        'name': 'Long Win Streak',
        'type': 'bearish',
        'tier': 2,
        'threshold': 5,
        'direction': 'above',
        'metric': 'market_streak'
    },
    'financial_services_extreme_bullish': {
        'name': 'Financials Extremely Bullish',
        'type': 'bearish',
        'tier': 2,
        'threshold': 70,
        'direction': 'above',
        'metric': 'financials_breadth'
    },
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and process the stock scoring data."""
    parquet_path = DATA_DIR / 'historical_data.parquet.gzip'
    
    if not parquet_path.exists():
        print(f"ERROR: Data file not found: {parquet_path}")
        return None
    
    df = pd.read_parquet(parquet_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df['net_score'] = df['bullish_score'] - df['bearish_score']
    
    return df


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
    market_agg['bull_roc_5d'] = market_agg['market_bull'].diff(5)
    
    # Bull/Bear Spread
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
    
    # Sector breadths
    growth_sectors = ['Technology', 'Communication Services', 'Consumer Cyclical']
    value_sectors = ['Utilities', 'Consumer Defensive', 'Energy', 'Financial Services']
    
    sector_breadth = df.groupby(['date', 'sector']).apply(
        lambda x: (x['bullish_score'] > x['bearish_score']).mean() * 100
    ).reset_index()
    sector_breadth.columns = ['date', 'sector', 'sector_breadth']
    
    sector_pivot = sector_breadth.pivot(index='date', columns='sector', values='sector_breadth').reset_index()
    
    sector_mapping = {
        'Real Estate': 'real_estate_breadth',
        'Utilities': 'utilities_breadth',
        'Financial Services': 'financials_breadth',
        'Technology': 'technology_breadth',
    }
    
    for sector, col_name in sector_mapping.items():
        if sector in sector_pivot.columns:
            market_agg = market_agg.merge(
                sector_pivot[['date', sector]].rename(columns={sector: col_name}),
                on='date',
                how='left'
            )
        else:
            market_agg[col_name] = 50
    
    # Growth vs Value differential
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
    else:
        market_agg['growth_value_diff'] = 0
    
    return market_agg


def get_spy_data(start_date):
    """Fetch SPY data from Yahoo Finance."""
    print("Fetching SPY data from Yahoo Finance...")
    spy = yf.download('SPY', start=start_date, progress=False)
    spy = spy.reset_index()
    spy.columns = [col[0] if isinstance(col, tuple) else col for col in spy.columns]
    spy['Date'] = pd.to_datetime(spy['Date']).dt.normalize()
    return spy


# =============================================================================
# SIGNAL DETECTION
# =============================================================================

def detect_signals(market_agg):
    """Detect when each signal fires for each day."""
    
    signal_columns = {}
    
    for signal_id, signal in SIGNALS.items():
        metric = signal['metric']
        threshold = signal['threshold']
        direction = signal['direction']
        
        if metric not in market_agg.columns:
            signal_columns[signal_id] = False
            continue
        
        current = market_agg[metric].fillna(0)
        
        if direction == 'below':
            fired = current < threshold
        elif direction == 'above':
            fired = current > threshold
        elif direction == 'crosses_above':
            if metric == 'market_net':
                prev = market_agg['prev_net'].fillna(0)
            elif metric == 'market_breadth':
                prev = market_agg['prev_breadth'].fillna(0)
            else:
                prev = current.shift(1).fillna(0)
            fired = (prev < threshold) & (current >= threshold)
        else:
            fired = False
        
        signal_columns[signal_id] = fired
    
    # Add signal columns to dataframe
    for signal_id, fired in signal_columns.items():
        market_agg[f'sig_{signal_id}'] = fired
    
    return market_agg


def count_signals_by_type(row, signal_type):
    """Count how many signals of a given type are firing."""
    count = 0
    firing_signals = []
    
    for signal_id, signal in SIGNALS.items():
        if signal['type'] == signal_type:
            if row.get(f'sig_{signal_id}', False):
                count += 1
                firing_signals.append(signal_id)
    
    return count, firing_signals


# =============================================================================
# BACKTESTING
# =============================================================================

def backtest_confluence(market_agg, spy_data, forward_days=[5, 10, 15, 20]):
    """
    Backtest signal confluence - what happens when multiple signals fire together.
    """
    
    results = {
        'bullish_confluence': [],
        'bearish_confluence': [],
        'specific_combos': [],
        'tier1_only': [],
    }
    
    # Merge SPY data with market metrics
    merged = market_agg.merge(
        spy_data[['Date', 'Close']].rename(columns={'Date': 'date', 'Close': 'spy_close'}),
        on='date',
        how='left'
    )
    
    # Calculate forward returns
    for days in forward_days:
        merged[f'fwd_ret_{days}d'] = merged['spy_close'].shift(-days) / merged['spy_close'] - 1
    
    # Count signals for each day
    bullish_signals = [s for s, sig in SIGNALS.items() if sig['type'] == 'bullish']
    bearish_signals = [s for s, sig in SIGNALS.items() if sig['type'] == 'bearish']
    tier1_bullish = [s for s, sig in SIGNALS.items() if sig['type'] == 'bullish' and sig['tier'] == 1]
    tier1_bearish = [s for s, sig in SIGNALS.items() if sig['type'] == 'bearish' and sig['tier'] == 1]
    
    # Add counts
    merged['bullish_count'] = merged[[f'sig_{s}' for s in bullish_signals]].sum(axis=1)
    merged['bearish_count'] = merged[[f'sig_{s}' for s in bearish_signals]].sum(axis=1)
    merged['tier1_bullish_count'] = merged[[f'sig_{s}' for s in tier1_bullish]].sum(axis=1)
    merged['tier1_bearish_count'] = merged[[f'sig_{s}' for s in tier1_bearish]].sum(axis=1)
    
    # Track which signals fired
    def get_firing_signals(row, signal_list):
        return [s for s in signal_list if row.get(f'sig_{s}', False)]
    
    merged['bullish_firing'] = merged.apply(lambda r: get_firing_signals(r, bullish_signals), axis=1)
    merged['bearish_firing'] = merged.apply(lambda r: get_firing_signals(r, bearish_signals), axis=1)
    
    print("\n" + "="*80)
    print("📊 SIGNAL CONFLUENCE BACKTEST RESULTS")
    print("="*80)
    
    # ===========================================
    # BULLISH CONFLUENCE
    # ===========================================
    print("\n\n🟢 BULLISH SIGNAL CONFLUENCE (Long SPY)")
    print("-"*60)
    
    bullish_results = []
    
    for n_signals in range(1, 9):
        subset = merged[merged['bullish_count'] >= n_signals].copy()
        
        if len(subset) == 0:
            continue
        
        for days in forward_days:
            col = f'fwd_ret_{days}d'
            valid = subset[subset[col].notna()]
            
            if len(valid) == 0:
                continue
            
            wins = (valid[col] > 0).sum()
            total = len(valid)
            win_rate = wins / total * 100
            avg_ret = valid[col].mean() * 100
            
            bullish_results.append({
                'min_signals': n_signals,
                'forward_days': days,
                'occurrences': total,
                'win_rate': win_rate,
                'avg_return': avg_ret,
                'type': 'bullish'
            })
            
            if days == 10:
                print(f"{n_signals}+ signals: {total:4d} times | WR: {win_rate:5.1f}% | Avg 10d: {avg_ret:+5.2f}%")
    
    # ===========================================
    # BEARISH CONFLUENCE (for SHORTING)
    # ===========================================
    print("\n\n🔴 BEARISH SIGNAL CONFLUENCE (Short SPY / Fade)")
    print("-"*60)
    
    bearish_results = []
    
    for n_signals in range(1, 7):
        subset = merged[merged['bearish_count'] >= n_signals].copy()
        
        if len(subset) == 0:
            continue
        
        for days in forward_days:
            col = f'fwd_ret_{days}d'
            valid = subset[subset[col].notna()]
            
            if len(valid) == 0:
                continue
            
            # For shorting, a WIN is when price goes DOWN
            wins = (valid[col] < 0).sum()
            total = len(valid)
            win_rate = wins / total * 100
            avg_ret = valid[col].mean() * 100  # Negative = good for shorts
            short_return = -avg_ret  # Flip for short perspective
            
            bearish_results.append({
                'min_signals': n_signals,
                'forward_days': days,
                'occurrences': total,
                'win_rate_short': win_rate,
                'avg_return_long': avg_ret,
                'avg_return_short': short_return,
                'type': 'bearish'
            })
            
            if days == 10:
                print(f"{n_signals}+ signals: {total:4d} times | Short WR: {win_rate:5.1f}% | Avg 10d (short): {short_return:+5.2f}%")
    
    # ===========================================
    # TIER 1 ONLY CONFLUENCE
    # ===========================================
    print("\n\n🏆 TIER 1 BULLISH CONFLUENCE ONLY")
    print("-"*60)
    
    tier1_results = []
    
    for n_signals in range(1, 6):
        subset = merged[merged['tier1_bullish_count'] >= n_signals].copy()
        
        if len(subset) == 0:
            continue
        
        for days in forward_days:
            col = f'fwd_ret_{days}d'
            valid = subset[subset[col].notna()]
            
            if len(valid) == 0:
                continue
            
            wins = (valid[col] > 0).sum()
            total = len(valid)
            win_rate = wins / total * 100
            avg_ret = valid[col].mean() * 100
            
            tier1_results.append({
                'min_signals': n_signals,
                'forward_days': days,
                'occurrences': total,
                'win_rate': win_rate,
                'avg_return': avg_ret,
                'type': 'tier1_bullish'
            })
            
            if days == 10:
                print(f"{n_signals}+ T1 signals: {total:4d} times | WR: {win_rate:5.1f}% | Avg 10d: {avg_ret:+5.2f}%")
    
    print("\n\n🏆 TIER 1 BEARISH CONFLUENCE ONLY")
    print("-"*60)
    
    for n_signals in range(1, 4):
        subset = merged[merged['tier1_bearish_count'] >= n_signals].copy()
        
        if len(subset) == 0:
            continue
        
        for days in forward_days:
            col = f'fwd_ret_{days}d'
            valid = subset[subset[col].notna()]
            
            if len(valid) == 0:
                continue
            
            wins = (valid[col] < 0).sum()
            total = len(valid)
            win_rate = wins / total * 100
            avg_ret = valid[col].mean() * 100
            short_return = -avg_ret
            
            tier1_results.append({
                'min_signals': n_signals,
                'forward_days': days,
                'occurrences': total,
                'win_rate_short': win_rate,
                'avg_return_short': short_return,
                'type': 'tier1_bearish'
            })
            
            if days == 10:
                print(f"{n_signals}+ T1 signals: {total:4d} times | Short WR: {win_rate:5.1f}% | Avg 10d (short): {short_return:+5.2f}%")
    
    return merged, bullish_results, bearish_results, tier1_results


def analyze_specific_combinations(merged, forward_days=10):
    """
    Analyze specific signal combinations to find the best ones.
    """
    
    print("\n\n" + "="*80)
    print("🔬 SPECIFIC SIGNAL COMBINATION ANALYSIS")
    print("="*80)
    
    bullish_signals = [s for s, sig in SIGNALS.items() if sig['type'] == 'bullish']
    bearish_signals = [s for s, sig in SIGNALS.items() if sig['type'] == 'bearish']
    
    col = f'fwd_ret_{forward_days}d'
    
    # ===========================================
    # BEST BULLISH PAIRS
    # ===========================================
    print("\n\n🟢 TOP BULLISH SIGNAL PAIRS (10d forward)")
    print("-"*70)
    
    pair_results = []
    
    for combo in combinations(bullish_signals, 2):
        sig1, sig2 = combo
        mask = merged[f'sig_{sig1}'] & merged[f'sig_{sig2}']
        subset = merged[mask & merged[col].notna()]
        
        if len(subset) < 5:  # Need at least 5 occurrences
            continue
        
        wins = (subset[col] > 0).sum()
        total = len(subset)
        win_rate = wins / total * 100
        avg_ret = subset[col].mean() * 100
        
        pair_results.append({
            'signals': f"{SIGNALS[sig1]['name']} + {SIGNALS[sig2]['name']}",
            'signal_ids': combo,
            'occurrences': total,
            'win_rate': win_rate,
            'avg_return': avg_ret
        })
    
    # Sort by win rate, then avg return
    pair_results.sort(key=lambda x: (x['win_rate'], x['avg_return']), reverse=True)
    
    print(f"{'Signal Combination':<55} | Count | WR    | Avg Ret")
    print("-"*70)
    for r in pair_results[:15]:
        print(f"{r['signals'][:55]:<55} | {r['occurrences']:5d} | {r['win_rate']:5.1f}% | {r['avg_return']:+5.2f}%")
    
    # ===========================================
    # BEST BULLISH TRIPLES
    # ===========================================
    print("\n\n🟢 TOP BULLISH SIGNAL TRIPLES (10d forward)")
    print("-"*80)
    
    triple_results = []
    
    for combo in combinations(bullish_signals, 3):
        sig1, sig2, sig3 = combo
        mask = merged[f'sig_{sig1}'] & merged[f'sig_{sig2}'] & merged[f'sig_{sig3}']
        subset = merged[mask & merged[col].notna()]
        
        if len(subset) < 3:  # Need at least 3 occurrences
            continue
        
        wins = (subset[col] > 0).sum()
        total = len(subset)
        win_rate = wins / total * 100
        avg_ret = subset[col].mean() * 100
        
        triple_results.append({
            'signals': f"{SIGNALS[sig1]['name'][:20]} + {SIGNALS[sig2]['name'][:20]} + {SIGNALS[sig3]['name'][:20]}",
            'signal_ids': combo,
            'occurrences': total,
            'win_rate': win_rate,
            'avg_return': avg_ret
        })
    
    triple_results.sort(key=lambda x: (x['win_rate'], x['avg_return']), reverse=True)
    
    print(f"{'Signal Combination':<65} | Count | WR    | Avg Ret")
    print("-"*80)
    for r in triple_results[:15]:
        print(f"{r['signals'][:65]:<65} | {r['occurrences']:5d} | {r['win_rate']:5.1f}% | {r['avg_return']:+5.2f}%")
    
    # ===========================================
    # BEST BEARISH PAIRS (for shorting)
    # ===========================================
    print("\n\n🔴 TOP BEARISH SIGNAL PAIRS - FOR SHORTING (10d forward)")
    print("-"*70)
    
    bear_pair_results = []
    
    for combo in combinations(bearish_signals, 2):
        sig1, sig2 = combo
        mask = merged[f'sig_{sig1}'] & merged[f'sig_{sig2}']
        subset = merged[mask & merged[col].notna()]
        
        if len(subset) < 3:
            continue
        
        # For shorting, WIN = price goes DOWN
        wins = (subset[col] < 0).sum()
        total = len(subset)
        win_rate = wins / total * 100
        avg_ret = subset[col].mean() * 100
        short_return = -avg_ret
        
        bear_pair_results.append({
            'signals': f"{SIGNALS[sig1]['name']} + {SIGNALS[sig2]['name']}",
            'signal_ids': combo,
            'occurrences': total,
            'win_rate_short': win_rate,
            'avg_return_long': avg_ret,
            'avg_return_short': short_return
        })
    
    bear_pair_results.sort(key=lambda x: (x['win_rate_short'], x['avg_return_short']), reverse=True)
    
    print(f"{'Signal Combination':<55} | Count | Short WR | Short Ret")
    print("-"*70)
    for r in bear_pair_results[:10]:
        print(f"{r['signals'][:55]:<55} | {r['occurrences']:5d} | {r['win_rate_short']:6.1f}%  | {r['avg_return_short']:+5.2f}%")
    
    # ===========================================
    # ALL BEARISH SIGNALS FIRING
    # ===========================================
    print("\n\n🔴 ALL BEARISH SIGNALS COMBINATIONS")
    print("-"*70)
    
    for n in range(2, len(bearish_signals) + 1):
        for combo in combinations(bearish_signals, n):
            mask = pd.Series(True, index=merged.index)
            for sig in combo:
                mask = mask & merged[f'sig_{sig}']
            
            subset = merged[mask & merged[col].notna()]
            
            if len(subset) < 2:
                continue
            
            wins = (subset[col] < 0).sum()
            total = len(subset)
            win_rate = wins / total * 100
            avg_ret = subset[col].mean() * 100
            short_return = -avg_ret
            
            signal_names = " + ".join([SIGNALS[s]['name'][:15] for s in combo])
            print(f"{n} signals: {signal_names[:50]:<50} | {total:3d}x | Short WR: {win_rate:5.1f}% | {short_return:+5.2f}%")
    
    return pair_results, triple_results, bear_pair_results


def find_best_entry_dates(merged, min_bullish=4, min_bearish=3):
    """Find the actual dates when high confluence occurred."""
    
    print("\n\n" + "="*80)
    print("📅 HIGH CONFLUENCE DATES")
    print("="*80)
    
    # High bullish confluence dates
    print(f"\n🟢 Dates with {min_bullish}+ Bullish Signals (Last 20):")
    print("-"*60)
    
    high_bullish = merged[merged['bullish_count'] >= min_bullish][['date', 'bullish_count', 'bullish_firing', 'spy_close', 'fwd_ret_10d']].copy()
    high_bullish = high_bullish.sort_values('date', ascending=False).head(20)
    
    for _, row in high_bullish.iterrows():
        signals = ", ".join([SIGNALS[s]['name'][:20] for s in row['bullish_firing']])
        fwd = row['fwd_ret_10d'] * 100 if pd.notna(row['fwd_ret_10d']) else np.nan
        result = f"{fwd:+.2f}%" if pd.notna(fwd) else "pending"
        print(f"{row['date'].strftime('%Y-%m-%d')} | {row['bullish_count']:.0f} signals | SPY: ${row['spy_close']:.2f} | 10d: {result}")
    
    # High bearish confluence dates
    print(f"\n🔴 Dates with {min_bearish}+ Bearish Signals (Last 20):")
    print("-"*60)
    
    high_bearish = merged[merged['bearish_count'] >= min_bearish][['date', 'bearish_count', 'bearish_firing', 'spy_close', 'fwd_ret_10d']].copy()
    high_bearish = high_bearish.sort_values('date', ascending=False).head(20)
    
    for _, row in high_bearish.iterrows():
        signals = ", ".join([SIGNALS[s]['name'][:20] for s in row['bearish_firing']])
        fwd = row['fwd_ret_10d'] * 100 if pd.notna(row['fwd_ret_10d']) else np.nan
        short_result = f"{-fwd:+.2f}%" if pd.notna(fwd) else "pending"
        print(f"{row['date'].strftime('%Y-%m-%d')} | {row['bearish_count']:.0f} signals | SPY: ${row['spy_close']:.2f} | Short 10d: {short_result}")
    
    return high_bullish, high_bearish


def export_results(merged, bullish_results, bearish_results, pair_results, output_path):
    """Export all results to Excel."""
    
    print(f"\n\nExporting results to {output_path}...")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Bullish confluence summary
        if bullish_results:
            df_bull = pd.DataFrame(bullish_results)
            df_bull.to_excel(writer, sheet_name='Bullish Confluence', index=False)
        
        # Bearish confluence summary
        if bearish_results:
            df_bear = pd.DataFrame(bearish_results)
            df_bear.to_excel(writer, sheet_name='Bearish Confluence', index=False)
        
        # Best pairs
        if pair_results:
            df_pairs = pd.DataFrame(pair_results[:30])
            df_pairs.to_excel(writer, sheet_name='Best Bullish Pairs', index=False)
        
        # Daily signal counts
        daily = merged[['date', 'spy_close', 'bullish_count', 'bearish_count', 
                       'tier1_bullish_count', 'tier1_bearish_count',
                       'fwd_ret_5d', 'fwd_ret_10d', 'fwd_ret_20d']].copy()
        daily = daily.sort_values('date', ascending=False)
        daily.to_excel(writer, sheet_name='Daily Signal Counts', index=False)
    
    print("✅ Export complete!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("🎯 SIGNAL CONFLUENCE BACKTESTER")
    print("="*80)
    print("\nLoading data...")
    
    # Load stock scoring data
    df = load_data()
    if df is None:
        return
    
    print(f"Loaded {len(df):,} records")
    
    # Calculate market metrics
    print("Calculating market metrics...")
    market_agg = calculate_market_metrics(df)
    print(f"Date range: {market_agg['date'].min()} to {market_agg['date'].max()}")
    
    # Detect all signals
    print("Detecting signals...")
    market_agg = detect_signals(market_agg)
    
    # Get SPY data
    spy_data = get_spy_data(market_agg['date'].min())
    print(f"SPY data: {len(spy_data)} days")
    
    # Run confluence backtest
    merged, bullish_results, bearish_results, tier1_results = backtest_confluence(market_agg, spy_data)
    
    # Analyze specific combinations
    pair_results, triple_results, bear_pair_results = analyze_specific_combinations(merged)
    
    # Find high confluence dates
    find_best_entry_dates(merged, min_bullish=4, min_bearish=3)
    
    # Export results
    output_path = DATA_DIR / 'signal_confluence_results.xlsx'
    export_results(merged, bullish_results, bearish_results, pair_results, output_path)
    
    print("\n\n" + "="*80)
    print("✅ BACKTEST COMPLETE!")
    print("="*80)
    
    # Summary
    print("\n📊 KEY TAKEAWAYS:")
    print("-"*40)
    
    # Find best bullish confluence
    best_bull = max([r for r in bullish_results if r['forward_days'] == 10], 
                    key=lambda x: x['win_rate'], default=None)
    if best_bull:
        print(f"🟢 Best Bullish: {best_bull['min_signals']}+ signals = {best_bull['win_rate']:.1f}% WR, {best_bull['avg_return']:+.2f}% avg")
    
    # Find best bearish confluence
    best_bear = max([r for r in bearish_results if r['forward_days'] == 10 and r['occurrences'] >= 5], 
                    key=lambda x: x['win_rate_short'], default=None)
    if best_bear:
        print(f"🔴 Best Bearish: {best_bear['min_signals']}+ signals = {best_bear['win_rate_short']:.1f}% Short WR, {best_bear['avg_return_short']:+.2f}% avg")


if __name__ == "__main__":
    main()