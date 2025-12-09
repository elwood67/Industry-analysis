"""
Industry Analysis Backtest Suite
=================================
Comprehensive backtesting of all analysis types:
- Rotation Stages (Leading/Weakening/Lagging/Improving)
- Relative Strength
- Mean Reversion (Z-Score signals)
- Momentum
- Streaks
- Sector/Industry performance

Tests predictive power for forward returns at industry and market level.

Author: Dave's Trading Analysis Suite
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


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

# Forward return periods to test
FORWARD_PERIODS = [5, 10, 15, 20]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_historical_data():
    """Load the historical score data."""
    print(f"📂 Loading data from: {DATA_DIR}")
    
    parquet_path = DATA_DIR / 'historical_data.parquet.gzip'
    df = pd.read_parquet(parquet_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df['net_score'] = df['bullish_score'] - df['bearish_score']
    
    print(f"✅ Loaded {len(df):,} records")
    print(f"📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"🏭 {df['industry'].nunique()} industries, {df['sector'].nunique()} sectors")
    
    return df


def get_spy_returns():
    """Get SPY daily returns for market-level backtesting."""
    print("\n📈 Fetching SPY data...")
    spy = yf.download('SPY', start='2023-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    spy = spy.reset_index()
    spy.columns = [col[0] if isinstance(col, tuple) else col for col in spy.columns]
    
    # Calculate forward returns for different periods
    for days in FORWARD_PERIODS:
        spy[f'fwd_return_{days}d'] = spy['Close'].pct_change(days).shift(-days) * 100
    
    spy['Date'] = pd.to_datetime(spy['Date']).dt.normalize()
    return spy[['Date', 'Close'] + [f'fwd_return_{d}d' for d in FORWARD_PERIODS]]


def get_sector_etf_returns():
    """Get sector ETF returns for sector-level backtesting."""
    print("\n📈 Fetching sector ETF data...")
    
    sector_etfs = {
        'Technology': 'XLK',
        'Healthcare': 'XLV', 
        'Financial Services': 'XLF',
        'Consumer Cyclical': 'XLY',
        'Consumer Defensive': 'XLP',
        'Industrials': 'XLI',
        'Energy': 'XLE',
        'Utilities': 'XLU',
        'Real Estate': 'XLRE',
        'Basic Materials': 'XLB',
        'Communication Services': 'XLC'
    }
    
    all_data = []
    for sector, etf in sector_etfs.items():
        try:
            data = yf.download(etf, start='2023-01-01', progress=False)
            data = data.reset_index()
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            
            for days in FORWARD_PERIODS:
                data[f'fwd_return_{days}d'] = data['Close'].pct_change(days).shift(-days) * 100
            
            data['sector'] = sector
            data['etf'] = etf
            data['Date'] = pd.to_datetime(data['Date']).dt.normalize()
            all_data.append(data[['Date', 'sector', 'etf', 'Close'] + [f'fwd_return_{d}d' for d in FORWARD_PERIODS]])
        except Exception as e:
            print(f"  Warning: Could not fetch {etf} for {sector}: {e}")
    
    return pd.concat(all_data, ignore_index=True)


# =============================================================================
# ANALYSIS CALCULATIONS (matching industry_score_analyzer.py)
# =============================================================================

def calculate_industry_aggregates(df):
    """Calculate daily aggregates at the industry level."""
    
    # Get most common sector for each industry
    industry_sector_map = df.groupby('industry')['sector'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    ).reset_index()
    industry_sector_map.columns = ['industry', 'sector']
    
    # Aggregate by date and industry
    industry_agg = df.groupby(['date', 'industry']).agg({
        'bullish_score': ['mean', 'median', 'std'],
        'bearish_score': ['mean', 'median', 'std'],
        'net_score': ['mean', 'median'],
        'symbol': 'count'
    }).reset_index()
    
    industry_agg.columns = [
        'date', 'industry',
        'bull_mean', 'bull_median', 'bull_std',
        'bear_mean', 'bear_median', 'bear_std',
        'net_mean', 'net_median', 'stock_count'
    ]
    
    # Add sector
    industry_agg = industry_agg.merge(industry_sector_map, on='industry', how='left')
    
    # Calculate breadth
    breadth = df.groupby(['date', 'industry']).apply(
        lambda x: (x['bullish_score'] > x['bearish_score']).mean() * 100
    ).reset_index()
    breadth.columns = ['date', 'industry', 'breadth']
    
    industry_agg = industry_agg.merge(breadth, on=['date', 'industry'], how='left')
    
    return industry_agg.sort_values(['date', 'industry'])


def calculate_market_aggregates(df):
    """Calculate daily market-wide aggregates."""
    
    market_agg = df.groupby('date').agg({
        'bullish_score': 'mean',
        'bearish_score': 'mean',
        'net_score': 'mean',
        'symbol': 'count'
    }).reset_index()
    
    market_agg.columns = ['date', 'market_bull', 'market_bear', 'market_net', 'total_stocks']
    
    # Market breadth
    breadth = df.groupby('date').apply(
        lambda x: (x['bullish_score'] > x['bearish_score']).mean() * 100
    ).reset_index()
    breadth.columns = ['date', 'market_breadth']
    
    market_agg = market_agg.merge(breadth, on='date')
    
    return market_agg.sort_values('date')


def calculate_momentum(industry_agg, periods=[5, 10, 20]):
    """Calculate momentum (rate of change) for industry scores."""
    
    result = industry_agg.copy()
    
    for industry in result['industry'].unique():
        mask = result['industry'] == industry
        industry_data = result.loc[mask].sort_values('date')
        
        for period in periods:
            result.loc[mask, f'net_roc_{period}d'] = industry_data['net_mean'].diff(period).values
            result.loc[mask, f'breadth_roc_{period}d'] = industry_data['breadth'].diff(period).values
    
    return result


def calculate_relative_strength(industry_agg, market_agg):
    """Calculate relative strength vs market."""
    
    result = industry_agg.merge(market_agg[['date', 'market_net']], on='date', how='left')
    
    # Daily RS
    result['daily_rs'] = result['net_mean'] - result['market_net']
    
    # Cumulative RS by industry
    result['cumulative_rs'] = result.groupby('industry')['daily_rs'].cumsum()
    
    return result


def calculate_zscore(industry_agg, window=20):
    """Calculate z-score for mean reversion signals."""
    
    result = industry_agg.copy()
    
    for industry in result['industry'].unique():
        mask = result['industry'] == industry
        industry_data = result.loc[mask].sort_values('date')
        
        rolling_mean = industry_data['net_mean'].rolling(window=window).mean()
        rolling_std = industry_data['net_mean'].rolling(window=window).std()
        
        zscore = (industry_data['net_mean'] - rolling_mean) / rolling_std
        result.loc[mask, 'zscore'] = zscore.values
    
    # Classify signals
    result['mr_signal'] = 'Neutral'
    result.loc[result['zscore'] > 2, 'mr_signal'] = 'Overbought'
    result.loc[result['zscore'] > 1, 'mr_signal'] = result.loc[result['zscore'] > 1, 'mr_signal'].replace('Neutral', 'Mildly Overbought')
    result.loc[result['zscore'] < -2, 'mr_signal'] = 'Oversold'
    result.loc[result['zscore'] < -1, 'mr_signal'] = result.loc[result['zscore'] < -1, 'mr_signal'].replace('Neutral', 'Mildly Oversold')
    
    return result


def calculate_rotation_stages(industry_agg):
    """Classify industries into rotation stages."""
    
    result = industry_agg.copy()
    
    # Need momentum and RS
    if 'net_roc_5d' not in result.columns or 'cumulative_rs' not in result.columns:
        return result
    
    # Get latest data for each date
    def classify_rotation(row):
        mom = row.get('net_roc_5d', 0)
        rs = row.get('cumulative_rs', 0)
        
        if pd.isna(mom) or pd.isna(rs):
            return 'Unknown'
        
        if mom > 0 and rs > 0:
            return 'Leading'
        elif mom < 0 and rs > 0:
            return 'Weakening'
        elif mom < 0 and rs < 0:
            return 'Lagging'
        elif mom > 0 and rs < 0:
            return 'Improving'
        else:
            return 'Neutral'
    
    result['rotation_stage'] = result.apply(classify_rotation, axis=1)
    
    return result


def calculate_streaks(industry_agg):
    """Calculate win/loss streaks."""
    
    result = industry_agg.copy()
    
    for industry in result['industry'].unique():
        mask = result['industry'] == industry
        industry_data = result.loc[mask].sort_values('date').copy()
        
        # Daily change direction
        industry_data['daily_change'] = industry_data['net_mean'].diff()
        industry_data['is_up'] = industry_data['daily_change'] > 0
        
        # Calculate streak
        streak = 0
        streaks = []
        prev_up = None
        
        for is_up in industry_data['is_up']:
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
        
        result.loc[mask, 'current_streak'] = streaks
    
    return result


# =============================================================================
# BACKTESTING FUNCTIONS
# =============================================================================

def backtest_rotation_stages(industry_agg, sector_returns):
    """Backtest predictive power of rotation stages."""
    
    print("\n" + "=" * 70)
    print("   🔄 ROTATION STAGES BACKTEST")
    print("=" * 70)
    
    # Merge with sector returns
    merged = industry_agg.merge(
        sector_returns,
        left_on=['date', 'sector'],
        right_on=['Date', 'sector'],
        how='inner'
    )
    
    results = []
    
    for stage in ['Leading', 'Weakening', 'Lagging', 'Improving']:
        stage_data = merged[merged['rotation_stage'] == stage]
        
        if len(stage_data) < 10:
            continue
        
        for days in FORWARD_PERIODS:
            col = f'fwd_return_{days}d'
            returns = stage_data[col].dropna()
            
            if len(returns) < 10:
                continue
            
            avg_return = returns.mean()
            win_rate = (returns > 0).mean() * 100
            
            results.append({
                'stage': stage,
                'period': f'{days}d',
                'occurrences': len(returns),
                'avg_return': avg_return,
                'win_rate': win_rate,
                'std': returns.std()
            })
    
    results_df = pd.DataFrame(results)
    
    print("\nRotation Stage Performance (Sector ETF Forward Returns):")
    print("-" * 70)
    
    for stage in ['Leading', 'Improving', 'Weakening', 'Lagging']:
        stage_results = results_df[results_df['stage'] == stage]
        if stage_results.empty:
            continue
        
        print(f"\n{stage}:")
        for _, row in stage_results.iterrows():
            emoji = "✅" if row['win_rate'] > 55 else "⚠️" if row['win_rate'] > 45 else "❌"
            print(f"  {row['period']:>4}: {row['occurrences']:>5} signals | "
                  f"Avg: {row['avg_return']:>+6.2f}% | Win Rate: {row['win_rate']:>5.1f}% {emoji}")
    
    return results_df


def backtest_mean_reversion(industry_agg, sector_returns):
    """Backtest mean reversion (z-score) signals."""
    
    print("\n" + "=" * 70)
    print("   📉 MEAN REVERSION (Z-SCORE) BACKTEST")
    print("=" * 70)
    
    # Merge with sector returns
    merged = industry_agg.merge(
        sector_returns,
        left_on=['date', 'sector'],
        right_on=['Date', 'sector'],
        how='inner'
    )
    
    results = []
    
    # Z-score buckets
    zscore_buckets = [
        ('Extremely Oversold (Z < -2)', merged['zscore'] < -2),
        ('Oversold (-2 < Z < -1)', (merged['zscore'] >= -2) & (merged['zscore'] < -1)),
        ('Neutral (-1 < Z < 1)', (merged['zscore'] >= -1) & (merged['zscore'] <= 1)),
        ('Overbought (1 < Z < 2)', (merged['zscore'] > 1) & (merged['zscore'] <= 2)),
        ('Extremely Overbought (Z > 2)', merged['zscore'] > 2),
    ]
    
    for bucket_name, mask in zscore_buckets:
        bucket_data = merged[mask]
        
        if len(bucket_data) < 10:
            continue
        
        for days in FORWARD_PERIODS:
            col = f'fwd_return_{days}d'
            returns = bucket_data[col].dropna()
            
            if len(returns) < 10:
                continue
            
            avg_return = returns.mean()
            win_rate = (returns > 0).mean() * 100
            
            results.append({
                'bucket': bucket_name,
                'period': f'{days}d',
                'occurrences': len(returns),
                'avg_return': avg_return,
                'win_rate': win_rate
            })
    
    results_df = pd.DataFrame(results)
    
    print("\nZ-Score Bucket Performance (Sector ETF Forward Returns):")
    print("-" * 70)
    
    for bucket in ['Extremely Oversold (Z < -2)', 'Oversold (-2 < Z < -1)', 
                   'Neutral (-1 < Z < 1)', 'Overbought (1 < Z < 2)', 
                   'Extremely Overbought (Z > 2)']:
        bucket_results = results_df[results_df['bucket'] == bucket]
        if bucket_results.empty:
            continue
        
        print(f"\n{bucket}:")
        for _, row in bucket_results.iterrows():
            emoji = "✅" if row['win_rate'] > 55 else "⚠️" if row['win_rate'] > 45 else "❌"
            print(f"  {row['period']:>4}: {row['occurrences']:>5} signals | "
                  f"Avg: {row['avg_return']:>+6.2f}% | Win Rate: {row['win_rate']:>5.1f}% {emoji}")
    
    return results_df


def backtest_momentum(industry_agg, sector_returns):
    """Backtest momentum signals."""
    
    print("\n" + "=" * 70)
    print("   📈 MOMENTUM BACKTEST")
    print("=" * 70)
    
    # Merge with sector returns
    merged = industry_agg.merge(
        sector_returns,
        left_on=['date', 'sector'],
        right_on=['Date', 'sector'],
        how='inner'
    )
    
    results = []
    
    # Momentum buckets based on 5d ROC
    if 'net_roc_5d' not in merged.columns:
        print("No momentum data available")
        return pd.DataFrame()
    
    momentum_buckets = [
        ('Strong Positive (ROC > 10)', merged['net_roc_5d'] > 10),
        ('Positive (5 < ROC < 10)', (merged['net_roc_5d'] > 5) & (merged['net_roc_5d'] <= 10)),
        ('Mild Positive (0 < ROC < 5)', (merged['net_roc_5d'] > 0) & (merged['net_roc_5d'] <= 5)),
        ('Mild Negative (-5 < ROC < 0)', (merged['net_roc_5d'] >= -5) & (merged['net_roc_5d'] < 0)),
        ('Negative (-10 < ROC < -5)', (merged['net_roc_5d'] >= -10) & (merged['net_roc_5d'] < -5)),
        ('Strong Negative (ROC < -10)', merged['net_roc_5d'] < -10),
    ]
    
    for bucket_name, mask in momentum_buckets:
        bucket_data = merged[mask]
        
        if len(bucket_data) < 10:
            continue
        
        for days in FORWARD_PERIODS:
            col = f'fwd_return_{days}d'
            returns = bucket_data[col].dropna()
            
            if len(returns) < 10:
                continue
            
            avg_return = returns.mean()
            win_rate = (returns > 0).mean() * 100
            
            results.append({
                'bucket': bucket_name,
                'period': f'{days}d',
                'occurrences': len(returns),
                'avg_return': avg_return,
                'win_rate': win_rate
            })
    
    results_df = pd.DataFrame(results)
    
    print("\nMomentum Bucket Performance (Sector ETF Forward Returns):")
    print("-" * 70)
    
    for bucket in ['Strong Positive (ROC > 10)', 'Positive (5 < ROC < 10)',
                   'Mild Positive (0 < ROC < 5)', 'Mild Negative (-5 < ROC < 0)',
                   'Negative (-10 < ROC < -5)', 'Strong Negative (ROC < -10)']:
        bucket_results = results_df[results_df['bucket'] == bucket]
        if bucket_results.empty:
            continue
        
        print(f"\n{bucket}:")
        for _, row in bucket_results.iterrows():
            emoji = "✅" if row['win_rate'] > 55 else "⚠️" if row['win_rate'] > 45 else "❌"
            print(f"  {row['period']:>4}: {row['occurrences']:>5} signals | "
                  f"Avg: {row['avg_return']:>+6.2f}% | Win Rate: {row['win_rate']:>5.1f}% {emoji}")
    
    return results_df


def backtest_relative_strength(industry_agg, sector_returns):
    """Backtest relative strength signals."""
    
    print("\n" + "=" * 70)
    print("   💪 RELATIVE STRENGTH BACKTEST")
    print("=" * 70)
    
    # Merge with sector returns
    merged = industry_agg.merge(
        sector_returns,
        left_on=['date', 'sector'],
        right_on=['Date', 'sector'],
        how='inner'
    )
    
    results = []
    
    if 'cumulative_rs' not in merged.columns:
        print("No relative strength data available")
        return pd.DataFrame()
    
    # RS quintiles
    merged['rs_quintile'] = pd.qcut(merged['cumulative_rs'].rank(method='first'), 5, labels=['Bottom 20%', 'Q2', 'Q3', 'Q4', 'Top 20%'])
    
    for quintile in ['Top 20%', 'Q4', 'Q3', 'Q2', 'Bottom 20%']:
        quintile_data = merged[merged['rs_quintile'] == quintile]
        
        if len(quintile_data) < 10:
            continue
        
        for days in FORWARD_PERIODS:
            col = f'fwd_return_{days}d'
            returns = quintile_data[col].dropna()
            
            if len(returns) < 10:
                continue
            
            avg_return = returns.mean()
            win_rate = (returns > 0).mean() * 100
            
            results.append({
                'quintile': quintile,
                'period': f'{days}d',
                'occurrences': len(returns),
                'avg_return': avg_return,
                'win_rate': win_rate
            })
    
    results_df = pd.DataFrame(results)
    
    print("\nRelative Strength Quintile Performance (Sector ETF Forward Returns):")
    print("-" * 70)
    
    for quintile in ['Top 20%', 'Q4', 'Q3', 'Q2', 'Bottom 20%']:
        quintile_results = results_df[results_df['quintile'] == quintile]
        if quintile_results.empty:
            continue
        
        print(f"\n{quintile} RS:")
        for _, row in quintile_results.iterrows():
            emoji = "✅" if row['win_rate'] > 55 else "⚠️" if row['win_rate'] > 45 else "❌"
            print(f"  {row['period']:>4}: {row['occurrences']:>5} signals | "
                  f"Avg: {row['avg_return']:>+6.2f}% | Win Rate: {row['win_rate']:>5.1f}% {emoji}")
    
    return results_df


def backtest_streaks(industry_agg, sector_returns):
    """Backtest streak signals."""
    
    print("\n" + "=" * 70)
    print("   🔥 STREAK BACKTEST")
    print("=" * 70)
    
    # Merge with sector returns
    merged = industry_agg.merge(
        sector_returns,
        left_on=['date', 'sector'],
        right_on=['Date', 'sector'],
        how='inner'
    )
    
    results = []
    
    if 'current_streak' not in merged.columns:
        print("No streak data available")
        return pd.DataFrame()
    
    # Streak buckets
    streak_buckets = [
        ('Long Win Streak (5+ days)', merged['current_streak'] >= 5),
        ('Win Streak (3-4 days)', (merged['current_streak'] >= 3) & (merged['current_streak'] < 5)),
        ('Short Win (1-2 days)', (merged['current_streak'] >= 1) & (merged['current_streak'] < 3)),
        ('Short Loss (-1 to -2 days)', (merged['current_streak'] <= -1) & (merged['current_streak'] > -3)),
        ('Loss Streak (-3 to -4 days)', (merged['current_streak'] <= -3) & (merged['current_streak'] > -5)),
        ('Long Loss Streak (5+ days)', merged['current_streak'] <= -5),
    ]
    
    for bucket_name, mask in streak_buckets:
        bucket_data = merged[mask]
        
        if len(bucket_data) < 10:
            continue
        
        for days in FORWARD_PERIODS:
            col = f'fwd_return_{days}d'
            returns = bucket_data[col].dropna()
            
            if len(returns) < 10:
                continue
            
            avg_return = returns.mean()
            win_rate = (returns > 0).mean() * 100
            
            results.append({
                'bucket': bucket_name,
                'period': f'{days}d',
                'occurrences': len(returns),
                'avg_return': avg_return,
                'win_rate': win_rate
            })
    
    results_df = pd.DataFrame(results)
    
    print("\nStreak Performance (Sector ETF Forward Returns):")
    print("-" * 70)
    
    for bucket in ['Long Win Streak (5+ days)', 'Win Streak (3-4 days)', 'Short Win (1-2 days)',
                   'Short Loss (-1 to -2 days)', 'Loss Streak (-3 to -4 days)', 'Long Loss Streak (5+ days)']:
        bucket_results = results_df[results_df['bucket'] == bucket]
        if bucket_results.empty:
            continue
        
        print(f"\n{bucket}:")
        for _, row in bucket_results.iterrows():
            emoji = "✅" if row['win_rate'] > 55 else "⚠️" if row['win_rate'] > 45 else "❌"
            print(f"  {row['period']:>4}: {row['occurrences']:>5} signals | "
                  f"Avg: {row['avg_return']:>+6.2f}% | Win Rate: {row['win_rate']:>5.1f}% {emoji}")
    
    return results_df


def backtest_breadth_levels(industry_agg, sector_returns):
    """Backtest industry breadth levels."""
    
    print("\n" + "=" * 70)
    print("   📊 BREADTH LEVELS BACKTEST")
    print("=" * 70)
    
    # Merge with sector returns
    merged = industry_agg.merge(
        sector_returns,
        left_on=['date', 'sector'],
        right_on=['Date', 'sector'],
        how='inner'
    )
    
    results = []
    
    # Breadth buckets
    breadth_buckets = [
        ('Very High Breadth (>80%)', merged['breadth'] > 80),
        ('High Breadth (60-80%)', (merged['breadth'] > 60) & (merged['breadth'] <= 80)),
        ('Neutral Breadth (40-60%)', (merged['breadth'] >= 40) & (merged['breadth'] <= 60)),
        ('Low Breadth (20-40%)', (merged['breadth'] >= 20) & (merged['breadth'] < 40)),
        ('Very Low Breadth (<20%)', merged['breadth'] < 20),
    ]
    
    for bucket_name, mask in breadth_buckets:
        bucket_data = merged[mask]
        
        if len(bucket_data) < 10:
            continue
        
        for days in FORWARD_PERIODS:
            col = f'fwd_return_{days}d'
            returns = bucket_data[col].dropna()
            
            if len(returns) < 10:
                continue
            
            avg_return = returns.mean()
            win_rate = (returns > 0).mean() * 100
            
            results.append({
                'bucket': bucket_name,
                'period': f'{days}d',
                'occurrences': len(returns),
                'avg_return': avg_return,
                'win_rate': win_rate
            })
    
    results_df = pd.DataFrame(results)
    
    print("\nBreadth Level Performance (Sector ETF Forward Returns):")
    print("-" * 70)
    
    for bucket in ['Very High Breadth (>80%)', 'High Breadth (60-80%)', 
                   'Neutral Breadth (40-60%)', 'Low Breadth (20-40%)', 
                   'Very Low Breadth (<20%)']:
        bucket_results = results_df[results_df['bucket'] == bucket]
        if bucket_results.empty:
            continue
        
        print(f"\n{bucket}:")
        for _, row in bucket_results.iterrows():
            emoji = "✅" if row['win_rate'] > 55 else "⚠️" if row['win_rate'] > 45 else "❌"
            print(f"  {row['period']:>4}: {row['occurrences']:>5} signals | "
                  f"Avg: {row['avg_return']:>+6.2f}% | Win Rate: {row['win_rate']:>5.1f}% {emoji}")
    
    return results_df


def generate_summary_report(all_results):
    """Generate a summary of the most predictive signals."""
    
    print("\n" + "=" * 70)
    print("   🏆 TOP PREDICTIVE SIGNALS SUMMARY")
    print("=" * 70)
    
    # Combine all results
    combined = []
    
    for analysis_type, df in all_results.items():
        if df.empty:
            continue
        df = df.copy()
        df['analysis_type'] = analysis_type
        combined.append(df)
    
    if not combined:
        print("No results to summarize")
        return
    
    all_df = pd.concat(combined, ignore_index=True)
    
    # Rename columns for consistency
    if 'bucket' in all_df.columns:
        all_df['signal'] = all_df['bucket']
    elif 'stage' in all_df.columns:
        all_df['signal'] = all_df['stage']
    elif 'quintile' in all_df.columns:
        all_df['signal'] = all_df['quintile']
    
    # Filter for 10-day results with enough occurrences
    filtered = all_df[(all_df['period'] == '10d') & (all_df['occurrences'] >= 50)]
    
    if filtered.empty:
        filtered = all_df[all_df['occurrences'] >= 20]
    
    # Top bullish signals (highest win rate)
    print("\n🟢 TOP BULLISH SIGNALS (Highest Win Rate):")
    print("-" * 70)
    top_bullish = filtered.nlargest(10, 'win_rate')
    
    for _, row in top_bullish.iterrows():
        signal_name = row.get('signal', row.get('bucket', row.get('stage', row.get('quintile', 'Unknown'))))
        print(f"  {row['analysis_type']:<20} | {signal_name:<35} | "
              f"WR: {row['win_rate']:.1f}% | Avg: {row['avg_return']:+.2f}% | n={row['occurrences']}")
    
    # Top bearish/fade signals (lowest win rate)
    print("\n🔴 TOP BEARISH/FADE SIGNALS (Lowest Win Rate):")
    print("-" * 70)
    top_bearish = filtered.nsmallest(10, 'win_rate')
    
    for _, row in top_bearish.iterrows():
        signal_name = row.get('signal', row.get('bucket', row.get('stage', row.get('quintile', 'Unknown'))))
        print(f"  {row['analysis_type']:<20} | {signal_name:<35} | "
              f"WR: {row['win_rate']:.1f}% | Avg: {row['avg_return']:+.2f}% | n={row['occurrences']}")
    
    # Highest average return
    print("\n💰 HIGHEST AVERAGE RETURNS:")
    print("-" * 70)
    top_returns = filtered.nlargest(10, 'avg_return')
    
    for _, row in top_returns.iterrows():
        signal_name = row.get('signal', row.get('bucket', row.get('stage', row.get('quintile', 'Unknown'))))
        print(f"  {row['analysis_type']:<20} | {signal_name:<35} | "
              f"Avg: {row['avg_return']:+.2f}% | WR: {row['win_rate']:.1f}% | n={row['occurrences']}")


def export_results(all_results, filename='industry_backtest_results.xlsx'):
    """Export all results to Excel."""
    
    print(f"\n📁 Exporting results to {filename}...")
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for analysis_type, df in all_results.items():
            if not df.empty:
                sheet_name = analysis_type[:31]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"✅ Results saved to {filename}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("   📊 INDUSTRY ANALYSIS BACKTEST SUITE")
    print("=" * 70)
    print()
    
    # Load data
    df = load_historical_data()
    
    # Calculate all metrics
    print("\n⚙️ Calculating industry aggregates...")
    industry_agg = calculate_industry_aggregates(df)
    
    print("⚙️ Calculating market aggregates...")
    market_agg = calculate_market_aggregates(df)
    
    print("⚙️ Calculating momentum...")
    industry_agg = calculate_momentum(industry_agg)
    
    print("⚙️ Calculating relative strength...")
    industry_agg = calculate_relative_strength(industry_agg, market_agg)
    
    print("⚙️ Calculating z-scores...")
    industry_agg = calculate_zscore(industry_agg)
    
    print("⚙️ Calculating rotation stages...")
    industry_agg = calculate_rotation_stages(industry_agg)
    
    print("⚙️ Calculating streaks...")
    industry_agg = calculate_streaks(industry_agg)
    
    # Get return data
    spy_returns = get_spy_returns()
    sector_returns = get_sector_etf_returns()
    
    # Run all backtests
    all_results = {}
    
    all_results['Rotation'] = backtest_rotation_stages(industry_agg, sector_returns)
    all_results['Mean_Reversion'] = backtest_mean_reversion(industry_agg, sector_returns)
    all_results['Momentum'] = backtest_momentum(industry_agg, sector_returns)
    all_results['Relative_Strength'] = backtest_relative_strength(industry_agg, sector_returns)
    all_results['Streaks'] = backtest_streaks(industry_agg, sector_returns)
    all_results['Breadth'] = backtest_breadth_levels(industry_agg, sector_returns)
    
    # Generate summary
    generate_summary_report(all_results)
    
    # Export to Excel
    export_results(all_results)
    
    print("\n" + "=" * 70)
    print("   ✅ BACKTEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
