"""
Signal Date Finder & Plotter
=============================
Finds dates when specific signals fired and plots them on SPY chart.

Author: Dave's Trading Analysis Suite
"""

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta

# Configuration - adjust path as needed
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


def load_and_calculate_market_data():
    """Load historical data and calculate market-level aggregates."""
    print(f"📂 Loading data from: {DATA_DIR}")
    
    parquet_path = DATA_DIR / 'historical_data.parquet.gzip'
    df = pd.read_parquet(parquet_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df['net_score'] = df['bullish_score'] - df['bearish_score']
    
    print(f"✅ Loaded {len(df):,} records")
    
    # Calculate daily market aggregates
    market_agg = df.groupby('date').agg({
        'bullish_score': 'mean',
        'bearish_score': 'mean',
        'net_score': 'mean',
        'symbol': 'count'
    }).reset_index()
    
    market_agg.columns = ['date', 'avg_bull', 'avg_bear', 'avg_net', 'stock_count']
    
    # Calculate breadth
    breadth = df.groupby('date').apply(
        lambda x: (x['bullish_score'] > x['bearish_score']).mean() * 100
    ).reset_index()
    breadth.columns = ['date', 'breadth']
    
    market_agg = market_agg.merge(breadth, on='date')
    market_agg = market_agg.sort_values('date').reset_index(drop=True)
    
    return market_agg


def find_net_score_turns_positive(market_agg):
    """Find dates when net score crossed from negative to positive."""
    market_agg = market_agg.copy()
    market_agg['prev_net'] = market_agg['avg_net'].shift(1)
    
    # Signal: previous day negative, current day positive
    signals = market_agg[
        (market_agg['prev_net'] < 0) & 
        (market_agg['avg_net'] > 0)
    ].copy()
    
    return signals['date'].tolist()


def find_breadth_crosses_above_50(market_agg):
    """Find dates when breadth crossed above 50%."""
    market_agg = market_agg.copy()
    market_agg['prev_breadth'] = market_agg['breadth'].shift(1)
    
    signals = market_agg[
        (market_agg['prev_breadth'] < 50) & 
        (market_agg['breadth'] >= 50)
    ].copy()
    
    return signals['date'].tolist()


def find_extreme_bearish(market_agg, threshold=55):
    """Find dates when average bearish score exceeded threshold."""
    signals = market_agg[market_agg['avg_bear'] > threshold].copy()
    return signals['date'].tolist()


def find_bull_roc_drop(market_agg, threshold=-10, period=5):
    """Find dates when bullish score dropped more than threshold over period."""
    market_agg = market_agg.copy()
    market_agg['bull_roc'] = market_agg['avg_bull'].diff(period)
    
    signals = market_agg[market_agg['bull_roc'] < threshold].copy()
    return signals['date'].tolist()


def get_spy_data(start_date, end_date):
    """Fetch SPY data from Yahoo Finance."""
    print(f"📈 Fetching SPY data from {start_date} to {end_date}...")
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    spy = spy.reset_index()
    spy.columns = [col[0] if isinstance(col, tuple) else col for col in spy.columns]
    return spy


def calculate_forward_returns(spy_data, signal_dates, days=10):
    """Calculate forward returns after each signal."""
    results = []
    
    for signal_date in signal_dates:
        signal_date = pd.to_datetime(signal_date)
        
        # Find the signal date in SPY data
        mask = spy_data['Date'] >= signal_date
        if not mask.any():
            continue
            
        idx = spy_data[mask].index[0]
        
        if idx + days >= len(spy_data):
            continue
        
        entry_price = spy_data.loc[idx, 'Close']
        exit_price = spy_data.loc[idx + days, 'Close']
        forward_return = ((exit_price - entry_price) / entry_price) * 100
        
        results.append({
            'date': signal_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'forward_return': forward_return,
            'win': forward_return > 0
        })
    
    return pd.DataFrame(results)


def plot_signals_on_spy(spy_data, signal_dates, signal_name, market_agg=None):
    """Create interactive chart with signals marked on SPY."""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'SPY with {signal_name} Signals', 'Net Score')
    )
    
    # SPY candlestick
    fig.add_trace(
        go.Candlestick(
            x=spy_data['Date'],
            open=spy_data['Open'],
            high=spy_data['High'],
            low=spy_data['Low'],
            close=spy_data['Close'],
            name='SPY',
            increasing_line_color='#00cc66',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # Add signal markers
    signal_dates_dt = [pd.to_datetime(d) for d in signal_dates]
    signal_spy = spy_data[spy_data['Date'].isin(signal_dates_dt)]
    
    if not signal_spy.empty:
        fig.add_trace(
            go.Scatter(
                x=signal_spy['Date'],
                y=signal_spy['Low'] * 0.995,  # Slightly below the candle
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='#00ff00',
                    line=dict(color='white', width=1)
                ),
                name=f'{signal_name} Signal',
                hovertemplate='%{x}<br>Signal Fired<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add net score in lower panel if available
    if market_agg is not None:
        fig.add_trace(
            go.Scatter(
                x=market_agg['date'],
                y=market_agg['avg_net'],
                mode='lines',
                name='Net Score',
                line=dict(color='#00aaff', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 170, 255, 0.2)'
            ),
            row=2, col=1
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=2, col=1)
        
        # Mark signals on net score too
        signal_net = market_agg[market_agg['date'].isin(signal_dates_dt)]
        if not signal_net.empty:
            fig.add_trace(
                go.Scatter(
                    x=signal_net['date'],
                    y=signal_net['avg_net'],
                    mode='markers',
                    marker=dict(symbol='circle', size=10, color='#00ff00'),
                    name='Signal Points',
                    showlegend=False
                ),
                row=2, col=1
            )
    
    # Layout
    fig.update_layout(
        title=f"📊 {signal_name} - {len(signal_dates)} Occurrences",
        template='plotly_dark',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="SPY Price", row=1, col=1)
    fig.update_yaxes(title_text="Net Score", row=2, col=1)
    
    return fig


def main():
    print("=" * 70)
    print("   📊 SIGNAL DATE FINDER & PLOTTER")
    print("=" * 70)
    print()
    
    # Load data
    market_agg = load_and_calculate_market_data()
    
    # Define available signals
    signals = {
        '1': ('Net Score Turns Positive', find_net_score_turns_positive),
        '2': ('Breadth Crosses Above 50%', find_breadth_crosses_above_50),
        '3': ('Extreme Bearish (>55)', lambda x: find_extreme_bearish(x, 55)),
        '4': ('Bull ROC Drop (<-10 in 5d)', lambda x: find_bull_roc_drop(x, -10, 5)),
    }
    
    print("\nAvailable signals:")
    for key, (name, _) in signals.items():
        print(f"  {key}. {name}")
    
    print()
    choice = input("Select signal (1-4) or press Enter for #1: ").strip() or '1'
    
    if choice not in signals:
        print("Invalid choice, using #1")
        choice = '1'
    
    signal_name, signal_func = signals[choice]
    
    # Find signal dates
    print(f"\n🔍 Finding {signal_name} signals...")
    signal_dates = signal_func(market_agg)
    
    print(f"\n✅ Found {len(signal_dates)} occurrences")
    print("\n" + "=" * 70)
    print(f"   {signal_name.upper()} - ALL DATES")
    print("=" * 70)
    
    # Get SPY data
    start_date = market_agg['date'].min() - timedelta(days=30)
    end_date = market_agg['date'].max() + timedelta(days=15)
    spy_data = get_spy_data(start_date, end_date)
    
    # Calculate forward returns
    returns_df = calculate_forward_returns(spy_data, signal_dates, days=10)
    
    # Print results
    print(f"\n{'Date':<15} {'SPY Entry':>12} {'SPY 10d':>12} {'Return':>10} {'Result':>8}")
    print("-" * 60)
    
    for _, row in returns_df.iterrows():
        result = "✅ WIN" if row['win'] else "❌ LOSS"
        print(f"{row['date'].strftime('%Y-%m-%d'):<15} ${row['entry_price']:>10.2f} ${row['exit_price']:>10.2f} {row['forward_return']:>+9.2f}% {result:>8}")
    
    print("-" * 60)
    
    # Summary stats
    if len(returns_df) > 0:
        win_rate = returns_df['win'].mean() * 100
        avg_return = returns_df['forward_return'].mean()
        total_return = returns_df['forward_return'].sum()
        winners = returns_df[returns_df['win']]
        losers = returns_df[~returns_df['win']]
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total Signals: {len(returns_df)}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Avg 10-day Return: {avg_return:+.2f}%")
        print(f"   Total Return (all signals): {total_return:+.2f}%")
        if len(winners) > 0:
            print(f"   Avg Winner: +{winners['forward_return'].mean():.2f}%")
        if len(losers) > 0:
            print(f"   Avg Loser: {losers['forward_return'].mean():.2f}%")
    
    # Create and show plot
    print("\n📈 Generating chart...")
    fig = plot_signals_on_spy(spy_data, signal_dates, signal_name, market_agg)
    
    # Save to HTML
    output_path = Path("signal_chart.html")
    fig.write_html(str(output_path))
    print(f"\n✅ Chart saved to: {output_path.absolute()}")
    
    # Try to open in browser
    try:
        import webbrowser
        webbrowser.open(str(output_path.absolute()))
        print("📊 Opening chart in browser...")
    except:
        print("   Open the HTML file manually to view the chart")
    
    print("\n" + "=" * 70)
    print("   DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
