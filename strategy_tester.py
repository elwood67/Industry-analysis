"""
Confluence Signal Strategy Tester
==================================
Deep-dive analysis of trading confluence signals on SPY.

Metrics calculated:
- Win rate, average return, total return
- Max drawdown, max consecutive losses
- Profit factor, Sharpe ratio, Sortino ratio
- Average winner vs average loser
- Risk/reward ratio
- Equity curve visualization
- Trade-by-trade breakdown

Author: Dave's Trading Lab
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(r'C:\Users\davet\Documents\GitHub\Industry-analysis\Data\stock_scores')

# Signal definitions
SIGNALS = {
    # Tier 1 Bullish
    'net_score_turns_positive': {'name': 'Net Score Turns Positive', 'type': 'bullish', 'tier': 1, 'threshold': 0, 'direction': 'crosses_above', 'metric': 'market_net'},
    'avg_bearish_extreme': {'name': 'Avg Bearish > 55', 'type': 'bullish', 'tier': 1, 'threshold': 55, 'direction': 'above', 'metric': 'market_bear'},
    'bull_roc_washout': {'name': 'Bull ROC 5d < -10', 'type': 'bullish', 'tier': 1, 'threshold': -10, 'direction': 'below', 'metric': 'bull_roc_5d'},
    'real_estate_extreme_bearish': {'name': 'Real Estate Extremely Bearish', 'type': 'bullish', 'tier': 1, 'threshold': 30, 'direction': 'below', 'metric': 'real_estate_breadth'},
    'utilities_extreme_bearish': {'name': 'Utilities Extremely Bearish', 'type': 'bullish', 'tier': 1, 'threshold': 30, 'direction': 'below', 'metric': 'utilities_breadth'},
    'sharp_rotation_to_value': {'name': 'Sharp Rotation to Value', 'type': 'bullish', 'tier': 1, 'threshold': -5, 'direction': 'below', 'metric': 'growth_value_diff'},
    'breadth_crosses_above_50': {'name': 'Breadth Crosses Above 50%', 'type': 'bullish', 'tier': 1, 'threshold': 50, 'direction': 'crosses_above', 'metric': 'market_breadth'},
    'bull_bear_convergence': {'name': 'Bull/Bear Convergence', 'type': 'bullish', 'tier': 1, 'threshold': 5, 'direction': 'below', 'metric': 'bull_bear_spread'},
    
    # Tier 2 Bullish
    'breadth_very_low': {'name': 'Very Low Breadth', 'type': 'bullish', 'tier': 2, 'threshold': 20, 'direction': 'below', 'metric': 'market_breadth'},
    'zscore_extreme_oversold': {'name': 'Extremely Oversold (Z-Score)', 'type': 'bullish', 'tier': 2, 'threshold': -2, 'direction': 'below', 'metric': 'market_zscore'},
    'loss_streak_long': {'name': 'Long Loss Streak', 'type': 'bullish', 'tier': 2, 'threshold': -5, 'direction': 'below', 'metric': 'market_streak'},
    'zscore_oversold': {'name': 'Oversold (Z-Score)', 'type': 'bullish', 'tier': 2, 'threshold': -1, 'direction': 'below', 'metric': 'market_zscore'},
    'breadth_low': {'name': 'Low Breadth (20-40%)', 'type': 'bullish', 'tier': 2, 'threshold': 40, 'direction': 'below', 'metric': 'market_breadth'},
    'momentum_strong_negative': {'name': 'Strong Negative Momentum', 'type': 'bullish', 'tier': 2, 'threshold': -10, 'direction': 'below', 'metric': 'net_roc_5d'},
    
    # Bearish
    'breadth_very_high': {'name': 'Very High Breadth (>80%)', 'type': 'bearish', 'tier': 1, 'threshold': 80, 'direction': 'above', 'metric': 'market_breadth'},
    'breadth_crosses_above_70': {'name': 'Breadth > 70%', 'type': 'bearish', 'tier': 1, 'threshold': 70, 'direction': 'above', 'metric': 'market_breadth'},
    'net_score_above_20': {'name': 'Net Score > 20', 'type': 'bearish', 'tier': 1, 'threshold': 20, 'direction': 'above', 'metric': 'market_net'},
    'zscore_overbought': {'name': 'Overbought (Z-Score)', 'type': 'bearish', 'tier': 2, 'threshold': 1, 'direction': 'above', 'metric': 'market_zscore'},
    'win_streak_long': {'name': 'Long Win Streak', 'type': 'bearish', 'tier': 2, 'threshold': 5, 'direction': 'above', 'metric': 'market_streak'},
    'financial_services_extreme_bullish': {'name': 'Financials Extremely Bullish', 'type': 'bearish', 'tier': 2, 'threshold': 70, 'direction': 'above', 'metric': 'financials_breadth'},
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load stock scoring data."""
    parquet_path = DATA_DIR / 'historical_data.parquet.gzip'
    
    if not parquet_path.exists():
        print(f"ERROR: Data file not found: {parquet_path}")
        return None
    
    df = pd.read_parquet(parquet_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df['net_score'] = df['bullish_score'] - df['bearish_score']
    
    return df


def calculate_market_metrics(df):
    """Calculate all market-level metrics."""
    
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
    
    market_agg = market_agg.sort_values('date').reset_index(drop=True)
    
    # Momentum
    market_agg['net_roc_5d'] = market_agg['market_net'].diff(5)
    market_agg['bull_roc_5d'] = market_agg['market_bull'].diff(5)
    market_agg['bull_bear_spread'] = abs(market_agg['market_bull'] - market_agg['market_bear'])
    
    # Z-Score
    market_agg['net_rolling_mean'] = market_agg['market_net'].rolling(20).mean()
    market_agg['net_rolling_std'] = market_agg['market_net'].rolling(20).std()
    market_agg['market_zscore'] = (market_agg['market_net'] - market_agg['net_rolling_mean']) / market_agg['net_rolling_std']
    
    # Streak
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
    
    for sector, col_name in [('Real Estate', 'real_estate_breadth'), ('Utilities', 'utilities_breadth'), 
                              ('Financial Services', 'financials_breadth')]:
        if sector in sector_pivot.columns:
            market_agg = market_agg.merge(sector_pivot[['date', sector]].rename(columns={sector: col_name}), on='date', how='left')
        else:
            market_agg[col_name] = 50
    
    # Growth/Value diff
    growth_cols = [s for s in growth_sectors if s in sector_pivot.columns]
    value_cols = [s for s in value_sectors if s in sector_pivot.columns]
    
    if growth_cols and value_cols:
        sector_pivot['growth_avg'] = sector_pivot[growth_cols].mean(axis=1)
        sector_pivot['value_avg'] = sector_pivot[value_cols].mean(axis=1)
        sector_pivot['growth_value_diff'] = sector_pivot['growth_avg'] - sector_pivot['value_avg']
        market_agg = market_agg.merge(sector_pivot[['date', 'growth_value_diff']], on='date', how='left')
    else:
        market_agg['growth_value_diff'] = 0
    
    return market_agg


def detect_signals(market_agg):
    """Detect when each signal fires."""
    
    for sig_id, sig in SIGNALS.items():
        metric = sig['metric']
        threshold = sig['threshold']
        direction = sig['direction']
        
        if metric not in market_agg.columns:
            market_agg[f'sig_{sig_id}'] = False
            continue
        
        current = market_agg[metric].fillna(0)
        
        if direction == 'below':
            market_agg[f'sig_{sig_id}'] = current < threshold
        elif direction == 'above':
            market_agg[f'sig_{sig_id}'] = current > threshold
        elif direction == 'crosses_above':
            if metric == 'market_net':
                prev = market_agg['prev_net'].fillna(0)
            elif metric == 'market_breadth':
                prev = market_agg['prev_breadth'].fillna(0)
            else:
                prev = current.shift(1).fillna(0)
            market_agg[f'sig_{sig_id}'] = (prev < threshold) & (current >= threshold)
        else:
            market_agg[f'sig_{sig_id}'] = False
    
    # Calculate counts
    tier1_bullish = [s for s, sig in SIGNALS.items() if sig['type'] == 'bullish' and sig.get('tier', 2) == 1]
    bearish_sigs = [s for s, sig in SIGNALS.items() if sig['type'] == 'bearish']
    
    market_agg['tier1_bullish_count'] = market_agg[[f'sig_{s}' for s in tier1_bullish]].sum(axis=1)
    market_agg['bearish_count'] = market_agg[[f'sig_{s}' for s in bearish_sigs]].sum(axis=1)
    
    return market_agg


def get_spy_data(start_date):
    """Fetch SPY data."""
    print("Fetching SPY data...")
    spy = yf.download('SPY', start=start_date, progress=False)
    spy = spy.reset_index()
    spy.columns = [col[0] if isinstance(col, tuple) else col for col in spy.columns]
    spy['Date'] = pd.to_datetime(spy['Date']).dt.normalize()
    return spy


# =============================================================================
# STRATEGY BACKTESTING
# =============================================================================

def backtest_strategy(market_agg, spy_data, signal_column, hold_days=10, signal_name="Signal", is_long=True):
    """
    Comprehensive backtest of a trading strategy.
    
    Parameters:
    - signal_column: Column name with True/False for signal
    - hold_days: How many days to hold after signal
    - is_long: True for long strategy, False for short
    """
    
    # Merge data
    merged = market_agg.merge(
        spy_data[['Date', 'Open', 'High', 'Low', 'Close']].rename(columns={'Date': 'date'}),
        on='date', how='left'
    )
    
    # Calculate forward returns for multiple periods
    for days in [1, 2, 3, 5, 10, 15, 20]:
        merged[f'fwd_ret_{days}d'] = merged['Close'].shift(-days) / merged['Close'] - 1
        merged[f'fwd_high_{days}d'] = merged['High'].rolling(days).max().shift(-days) / merged['Close'] - 1
        merged[f'fwd_low_{days}d'] = merged['Low'].rolling(days).min().shift(-days) / merged['Close'] - 1
    
    # Get signal dates
    signal_dates = merged[merged[signal_column] == True].copy()
    
    if len(signal_dates) == 0:
        print(f"No signals found for {signal_name}")
        return None
    
    # Build trade list
    trades = []
    
    for idx, row in signal_dates.iterrows():
        entry_date = row['date']
        entry_price = row['Close']
        
        if pd.isna(entry_price):
            continue
        
        # Get exit info
        ret_col = f'fwd_ret_{hold_days}d'
        high_col = f'fwd_high_{hold_days}d'
        low_col = f'fwd_low_{hold_days}d'
        
        pnl = row[ret_col] if not pd.isna(row[ret_col]) else None
        max_favorable = row[high_col] if not pd.isna(row[high_col]) else None
        max_adverse = row[low_col] if not pd.isna(row[low_col]) else None
        
        if pnl is None:
            continue
        
        # For short trades, invert the returns
        if not is_long:
            pnl = -pnl
            max_favorable, max_adverse = -max_adverse, -max_favorable
        
        # Calculate drawdown during trade
        if is_long:
            trade_drawdown = max_adverse if max_adverse else 0
        else:
            trade_drawdown = -max_favorable if max_favorable else 0
        
        trades.append({
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': entry_date + timedelta(days=hold_days),
            'pnl_pct': pnl * 100,
            'is_winner': pnl > 0,
            'max_favorable_pct': (max_favorable * 100) if max_favorable else 0,
            'max_adverse_pct': (max_adverse * 100) if max_adverse else 0,
            'trade_drawdown_pct': trade_drawdown * 100,
            'tier1_count': row['tier1_bullish_count'],
        })
    
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        print("No valid trades")
        return None
    
    # Calculate metrics
    metrics = calculate_metrics(trades_df, signal_name, hold_days, is_long)
    
    return trades_df, metrics, merged


def calculate_metrics(trades_df, signal_name, hold_days, is_long):
    """Calculate comprehensive trading metrics."""
    
    total_trades = len(trades_df)
    winners = trades_df[trades_df['is_winner']]
    losers = trades_df[~trades_df['is_winner']]
    
    # Basic stats
    win_rate = len(winners) / total_trades * 100
    avg_return = trades_df['pnl_pct'].mean()
    total_return = trades_df['pnl_pct'].sum()
    
    # Winner/Loser analysis
    avg_winner = winners['pnl_pct'].mean() if len(winners) > 0 else 0
    avg_loser = losers['pnl_pct'].mean() if len(losers) > 0 else 0
    max_winner = trades_df['pnl_pct'].max()
    max_loser = trades_df['pnl_pct'].min()
    
    # Risk/Reward
    risk_reward = abs(avg_winner / avg_loser) if avg_loser != 0 else float('inf')
    
    # Profit Factor
    gross_profit = winners['pnl_pct'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['pnl_pct'].sum()) if len(losers) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Consecutive wins/losses
    trades_df['streak'] = (trades_df['is_winner'] != trades_df['is_winner'].shift()).cumsum()
    win_streaks = trades_df[trades_df['is_winner']].groupby('streak').size()
    loss_streaks = trades_df[~trades_df['is_winner']].groupby('streak').size()
    
    max_consecutive_wins = win_streaks.max() if len(win_streaks) > 0 else 0
    max_consecutive_losses = loss_streaks.max() if len(loss_streaks) > 0 else 0
    
    # Equity curve & drawdown
    trades_df['cumulative_return'] = trades_df['pnl_pct'].cumsum()
    trades_df['equity_peak'] = trades_df['cumulative_return'].cummax()
    trades_df['drawdown'] = trades_df['cumulative_return'] - trades_df['equity_peak']
    max_drawdown = trades_df['drawdown'].min()
    
    # Average trade drawdown
    avg_trade_drawdown = trades_df['trade_drawdown_pct'].mean()
    max_trade_drawdown = trades_df['trade_drawdown_pct'].min()
    
    # Sharpe Ratio (annualized, assuming 252 trading days)
    if trades_df['pnl_pct'].std() > 0:
        trades_per_year = 252 / hold_days
        sharpe = (avg_return * trades_per_year) / (trades_df['pnl_pct'].std() * np.sqrt(trades_per_year))
    else:
        sharpe = 0
    
    # Sortino Ratio (using only downside deviation)
    downside_returns = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct']
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino = (avg_return * trades_per_year) / (downside_returns.std() * np.sqrt(trades_per_year))
    else:
        sortino = float('inf')
    
    # Expectancy
    expectancy = (win_rate/100 * avg_winner) - ((100-win_rate)/100 * abs(avg_loser))
    
    # Recovery factor
    recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
    
    metrics = {
        'Signal Name': signal_name,
        'Direction': 'LONG' if is_long else 'SHORT',
        'Hold Period': f'{hold_days} days',
        'Total Trades': total_trades,
        'Winners': len(winners),
        'Losers': len(losers),
        'Win Rate (%)': round(win_rate, 1),
        'Avg Return (%)': round(avg_return, 2),
        'Total Return (%)': round(total_return, 2),
        'Avg Winner (%)': round(avg_winner, 2),
        'Avg Loser (%)': round(avg_loser, 2),
        'Max Winner (%)': round(max_winner, 2),
        'Max Loser (%)': round(max_loser, 2),
        'Risk/Reward Ratio': round(risk_reward, 2),
        'Profit Factor': round(profit_factor, 2),
        'Expectancy (%)': round(expectancy, 2),
        'Max Consecutive Wins': max_consecutive_wins,
        'Max Consecutive Losses': max_consecutive_losses,
        'Max Strategy Drawdown (%)': round(max_drawdown, 2),
        'Avg Trade Drawdown (%)': round(avg_trade_drawdown, 2),
        'Max Trade Drawdown (%)': round(max_trade_drawdown, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Sortino Ratio': round(sortino, 2) if sortino != float('inf') else 'Inf',
        'Recovery Factor': round(recovery_factor, 2) if recovery_factor != float('inf') else 'Inf',
    }
    
    return metrics


def print_metrics(metrics):
    """Print metrics in a nice format."""
    
    print("\n" + "="*70)
    print(f"📊 STRATEGY ANALYSIS: {metrics['Signal Name']}")
    print(f"   Direction: {metrics['Direction']} | Hold: {metrics['Hold Period']}")
    print("="*70)
    
    print("\n📈 PERFORMANCE OVERVIEW")
    print("-"*40)
    print(f"Total Trades:        {metrics['Total Trades']}")
    print(f"Win Rate:            {metrics['Win Rate (%)']}%")
    print(f"Avg Return/Trade:    {metrics['Avg Return (%)']}%")
    print(f"Total Return:        {metrics['Total Return (%)']}%")
    
    print("\n💰 WIN/LOSS ANALYSIS")
    print("-"*40)
    print(f"Winners:             {metrics['Winners']} trades")
    print(f"Losers:              {metrics['Losers']} trades")
    print(f"Avg Winner:          +{metrics['Avg Winner (%)']}%")
    print(f"Avg Loser:           {metrics['Avg Loser (%)']}%")
    print(f"Max Winner:          +{metrics['Max Winner (%)']}%")
    print(f"Max Loser:           {metrics['Max Loser (%)']}%")
    print(f"Risk/Reward:         {metrics['Risk/Reward Ratio']}:1")
    
    print("\n📉 RISK METRICS")
    print("-"*40)
    print(f"Max Strategy DD:     {metrics['Max Strategy Drawdown (%)']}%")
    print(f"Avg Trade Drawdown:  {metrics['Avg Trade Drawdown (%)']}%")
    print(f"Max Trade Drawdown:  {metrics['Max Trade Drawdown (%)']}%")
    print(f"Max Consec. Losses:  {metrics['Max Consecutive Losses']}")
    
    print("\n🎯 QUALITY METRICS")
    print("-"*40)
    print(f"Profit Factor:       {metrics['Profit Factor']}")
    print(f"Expectancy:          {metrics['Expectancy (%)']}% per trade")
    print(f"Sharpe Ratio:        {metrics['Sharpe Ratio']}")
    print(f"Sortino Ratio:       {metrics['Sortino Ratio']}")
    print(f"Recovery Factor:     {metrics['Recovery Factor']}")


def create_equity_chart(trades_df, metrics, signal_name):
    """Create equity curve and drawdown chart."""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            f'Equity Curve - {signal_name}',
            'Trade Returns Distribution',
            'Strategy Drawdown'
        )
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=list(range(len(trades_df))),
            y=trades_df['cumulative_return'],
            mode='lines',
            name='Cumulative Return',
            line=dict(color='#00ff00', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ),
        row=1, col=1
    )
    
    # Add peak line
    fig.add_trace(
        go.Scatter(
            x=list(range(len(trades_df))),
            y=trades_df['equity_peak'],
            mode='lines',
            name='Equity Peak',
            line=dict(color='#ffaa00', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    # Trade returns as bars
    colors = ['#00ff00' if x > 0 else '#ff4444' for x in trades_df['pnl_pct']]
    fig.add_trace(
        go.Bar(
            x=list(range(len(trades_df))),
            y=trades_df['pnl_pct'],
            name='Trade P&L',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=list(range(len(trades_df))),
            y=trades_df['drawdown'],
            mode='lines',
            name='Drawdown',
            line=dict(color='#ff4444', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.3)'
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        template='plotly_dark',
        showlegend=True,
        title=f"Strategy Analysis: {metrics['Win Rate (%)']}% WR | {metrics['Profit Factor']} PF | {metrics['Max Strategy Drawdown (%)']}% Max DD"
    )
    
    fig.update_xaxes(title_text="Trade #", row=3, col=1)
    fig.update_yaxes(title_text="Cumulative %", row=1, col=1)
    fig.update_yaxes(title_text="Trade P&L %", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=3, col=1)
    
    return fig


def print_trade_list(trades_df):
    """Print detailed trade list."""
    
    print("\n" + "="*70)
    print("📋 TRADE-BY-TRADE BREAKDOWN")
    print("="*70)
    print(f"{'#':<4} {'Entry Date':<12} {'Entry $':<10} {'P&L %':<10} {'Max Fav':<10} {'Max Adv':<10} {'Result':<8}")
    print("-"*70)
    
    for i, (_, trade) in enumerate(trades_df.iterrows(), 1):
        result = "✅ WIN" if trade['is_winner'] else "❌ LOSS"
        print(f"{i:<4} {trade['entry_date'].strftime('%Y-%m-%d'):<12} ${trade['entry_price']:<9.2f} "
              f"{trade['pnl_pct']:>+8.2f}% {trade['max_favorable_pct']:>+8.2f}% {trade['max_adverse_pct']:>+8.2f}% {result:<8}")


def analyze_by_signal_count(trades_df):
    """Analyze performance by number of T1 signals firing."""
    
    print("\n" + "="*70)
    print("📊 PERFORMANCE BY # OF TIER 1 SIGNALS")
    print("="*70)
    print(f"{'T1 Signals':<12} {'Trades':<8} {'Win Rate':<10} {'Avg Ret':<10} {'Total Ret':<12}")
    print("-"*70)
    
    for count in sorted(trades_df['tier1_count'].unique()):
        subset = trades_df[trades_df['tier1_count'] == count]
        if len(subset) > 0:
            wr = (subset['is_winner'].sum() / len(subset)) * 100
            avg_ret = subset['pnl_pct'].mean()
            tot_ret = subset['pnl_pct'].sum()
            print(f"{int(count):<12} {len(subset):<8} {wr:>8.1f}% {avg_ret:>+8.2f}% {tot_ret:>+10.2f}%")


def test_different_hold_periods(market_agg, spy_data, signal_column, signal_name, is_long=True):
    """Test strategy across different hold periods."""
    
    print("\n" + "="*70)
    print("📊 HOLD PERIOD OPTIMIZATION")
    print("="*70)
    print(f"{'Hold Days':<12} {'Trades':<8} {'Win Rate':<10} {'Avg Ret':<10} {'Profit Factor':<14} {'Max DD':<10}")
    print("-"*70)
    
    results = []
    
    for hold_days in [1, 2, 3, 5, 7, 10, 15, 20]:
        result = backtest_strategy(market_agg, spy_data, signal_column, hold_days, signal_name, is_long)
        if result:
            trades_df, metrics, _ = result
            results.append({
                'hold_days': hold_days,
                'trades': metrics['Total Trades'],
                'win_rate': metrics['Win Rate (%)'],
                'avg_return': metrics['Avg Return (%)'],
                'profit_factor': metrics['Profit Factor'],
                'max_dd': metrics['Max Strategy Drawdown (%)']
            })
            print(f"{hold_days:<12} {metrics['Total Trades']:<8} {metrics['Win Rate (%)']:>8.1f}% "
                  f"{metrics['Avg Return (%)']:>+8.2f}% {metrics['Profit Factor']:>12.2f} "
                  f"{metrics['Max Strategy Drawdown (%)']:>+8.2f}%")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("🎯 CONFLUENCE SIGNAL STRATEGY TESTER")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = load_data()
    if df is None:
        return
    
    market_agg = calculate_market_metrics(df)
    market_agg = detect_signals(market_agg)
    spy_data = get_spy_data(market_agg['date'].min())
    
    print(f"Date range: {market_agg['date'].min().date()} to {market_agg['date'].max().date()}")
    print(f"SPY data points: {len(spy_data)}")
    
    # Test 3+ Tier 1 Bullish Signal
    print("\n" + "="*70)
    print("🏆 TESTING: 3+ TIER 1 BULLISH SIGNALS")
    print("="*70)
    
    market_agg['signal_t1_3plus'] = market_agg['tier1_bullish_count'] >= 3
    
    result = backtest_strategy(
        market_agg, spy_data, 
        signal_column='signal_t1_3plus',
        hold_days=10,
        signal_name='3+ Tier 1 Bullish Signals',
        is_long=True
    )
    
    if result:
        trades_df, metrics, merged = result
        
        # Print metrics
        print_metrics(metrics)
        
        # Trade breakdown
        print_trade_list(trades_df)
        
        # Analyze by signal count
        analyze_by_signal_count(trades_df)
        
        # Test different hold periods
        hold_results = test_different_hold_periods(
            market_agg, spy_data, 'signal_t1_3plus', 
            '3+ T1 Bullish', is_long=True
        )
        
        # Create equity chart
        fig = create_equity_chart(trades_df, metrics, '3+ Tier 1 Bullish Signals')
        fig.write_html(DATA_DIR / 'strategy_equity_curve.html')
        print(f"\n📈 Equity curve saved to: {DATA_DIR / 'strategy_equity_curve.html'}")
        
        # Export trades to Excel
        trades_df.to_excel(DATA_DIR / 'strategy_trades.xlsx', index=False)
        print(f"📊 Trade list saved to: {DATA_DIR / 'strategy_trades.xlsx'}")
    
    # Also test 4+ Bearish for shorting
    print("\n\n" + "="*70)
    print("🔴 TESTING: 4+ BEARISH SIGNALS (SHORT)")
    print("="*70)
    
    market_agg['signal_bear_4plus'] = market_agg['bearish_count'] >= 4
    
    result_short = backtest_strategy(
        market_agg, spy_data,
        signal_column='signal_bear_4plus', 
        hold_days=10,
        signal_name='4+ Bearish Signals (SHORT)',
        is_long=False
    )
    
    if result_short:
        trades_df_short, metrics_short, _ = result_short
        print_metrics(metrics_short)
        print_trade_list(trades_df_short)
    
    print("\n\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
