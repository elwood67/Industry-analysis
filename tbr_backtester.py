"""
TBR (Time-Based Range) Backtester v2
=====================================
Backtests the 9:00 AM and 3:00 PM EST hourly candle retracement strategy.
Strategy: Mark the hourly candle range, wait for first close outside, 
then look for price to retrace into the range at 0.3/0.5/0.7 fib levels.

v2 Changes:
- TBR candle size analysis (quartiles + ATR relative)
- Retest hour-of-day analysis
- Closure speed analysis (immediate vs delayed)
- Candle color (bullish/bearish TBR candle) analysis
- "Best Filter Finder" tab with combo scoring
- JSON export for Claude analysis
- Bug fixes (20h trend avg rev, R-multiple outlier capping)
- 3R success tracking
- Median R-Multiple alongside average

Author: Elwood's Place
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ── Asset Configuration ──────────────────────────────────────────────────────
ASSET_MAP = {
    "NQ (QQQ proxy)": "QQQ",
    "ES (SPY proxy)": "SPY",
    "Gold (GLD)": "GLD",
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "AAPL": "AAPL",
    "TSLA": "TSLA",
    "NVDA": "NVDA",
    "AMZN": "AMZN",
    "META": "META",
    "DIA (Dow proxy)": "DIA",
    "IWM (Russell proxy)": "IWM",
    "USO (Oil proxy)": "USO",
    "SLV (Silver proxy)": "SLV",
}


def fetch_hourly_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1h", progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = df.index.tz_convert("US/Eastern")
    return df


def identify_tbr_setups(df: pd.DataFrame, tbr_hour: int = 9) -> list[dict]:
    setups = []
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    dates = df['date'].unique()
    
    # ATR for range context
    df['tr'] = np.maximum(df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    df['atr_20'] = df['tr'].rolling(20).mean()
    
    for i, date in enumerate(dates):
        day_data = df[df['date'] == date]
        tbr_candle = day_data[day_data['hour'] == tbr_hour]
        if tbr_candle.empty:
            continue
        
        tbr_candle = tbr_candle.iloc[0]
        tbr_high = float(tbr_candle['High'])
        tbr_low = float(tbr_candle['Low'])
        tbr_open = float(tbr_candle['Open'])
        tbr_close = float(tbr_candle['Close'])
        tbr_range = tbr_high - tbr_low
        tbr_volume = float(tbr_candle['Volume']) if 'Volume' in tbr_candle.index else 0
        
        if tbr_range <= 0:
            continue
        
        atr_val = float(tbr_candle.get('atr_20', tbr_range))
        if pd.isna(atr_val) or atr_val <= 0:
            atr_val = tbr_range
        range_vs_atr = tbr_range / atr_val
        
        fib_03_from_high = tbr_high - 0.3 * tbr_range
        fib_05 = tbr_high - 0.5 * tbr_range
        fib_07_from_high = tbr_high - 0.7 * tbr_range
        
        tbr_time = tbr_candle.name
        future_mask = (df.index > tbr_time) & (df.index <= tbr_time + timedelta(hours=30))
        future_candles = df[future_mask]
        if future_candles.empty:
            continue
        
        first_close_above = None
        first_close_below = None
        closure_candle_idx = None
        direction = None
        closure_hour = None
        
        for j, (idx, candle) in enumerate(future_candles.iterrows()):
            if candle['Close'] > tbr_high:
                first_close_above = idx
                closure_candle_idx = j
                direction = 'bullish'
                closure_hour = idx.hour
                break
            elif candle['Close'] < tbr_low:
                first_close_below = idx
                closure_candle_idx = j
                direction = 'bearish'
                closure_hour = idx.hour
                break
        
        if direction is None:
            continue
        
        closure_time = first_close_above or first_close_below
        candles_to_closure = closure_candle_idx + 1
        
        post_closure = future_candles.iloc[closure_candle_idx + 1:]
        if post_closure.empty:
            continue
        if i + 1 < len(dates):
            extended_mask = (df.index > closure_time) & (df.index <= tbr_time + timedelta(hours=48))
            post_closure = df[extended_mask]
        if post_closure.empty:
            continue
        
        retest_found = False
        retest_time = None
        retest_hour = None
        deepest_fib_hit = None
        fib_03_hit = False
        fib_05_hit = False
        fib_07_hit = False
        swept = False
        reversal_size = 0.0
        reversal_pct = 0.0
        max_favorable = 0.0
        max_adverse = 0.0
        bars_to_retest = 0
        
        if direction == 'bullish':
            for k, (idx, candle) in enumerate(post_closure.iterrows()):
                if candle['Low'] <= tbr_high:
                    if not retest_found:
                        retest_found = True
                        retest_time = idx
                        retest_hour = idx.hour
                        bars_to_retest = k + 1
                    if candle['Low'] <= fib_03_from_high: fib_03_hit = True
                    if candle['Low'] <= fib_05: fib_05_hit = True
                    if candle['Low'] <= fib_07_from_high: fib_07_hit = True
                    if candle['Low'] <= tbr_low: swept = True
                    if fib_07_hit: deepest_fib_hit = 0.7
                    elif fib_05_hit: deepest_fib_hit = 0.5
                    elif fib_03_hit: deepest_fib_hit = 0.3
                    else: deepest_fib_hit = 0.0
            
            if retest_found and retest_time is not None:
                post_retest = post_closure[post_closure.index >= retest_time]
                if len(post_retest) > 1:
                    retest_low = post_retest.iloc[0]['Low']
                    subsequent = post_retest.iloc[1:min(len(post_retest), 13)]
                    if not subsequent.empty:
                        max_high_after = subsequent['High'].max()
                        min_low_after = subsequent['Low'].min()
                        max_favorable = float(max_high_after - retest_low)
                        max_adverse = float(retest_low - min_low_after)
                        reversal_size = float(max_high_after - retest_low)
                        if retest_low > 0:
                            reversal_pct = float((reversal_size / retest_low) * 100)
        
        elif direction == 'bearish':
            for k, (idx, candle) in enumerate(post_closure.iterrows()):
                if candle['High'] >= tbr_low:
                    if not retest_found:
                        retest_found = True
                        retest_time = idx
                        retest_hour = idx.hour
                        bars_to_retest = k + 1
                    fib_03_from_low = tbr_low + 0.3 * tbr_range
                    fib_05_from_low = tbr_low + 0.5 * tbr_range
                    fib_07_from_low = tbr_low + 0.7 * tbr_range
                    if candle['High'] >= fib_03_from_low: fib_03_hit = True
                    if candle['High'] >= fib_05_from_low: fib_05_hit = True
                    if candle['High'] >= fib_07_from_low: fib_07_hit = True
                    if candle['High'] >= tbr_high: swept = True
                    if fib_07_hit: deepest_fib_hit = 0.7
                    elif fib_05_hit: deepest_fib_hit = 0.5
                    elif fib_03_hit: deepest_fib_hit = 0.3
                    else: deepest_fib_hit = 0.0
            
            if retest_found and retest_time is not None:
                post_retest = post_closure[post_closure.index >= retest_time]
                if len(post_retest) > 1:
                    retest_high = post_retest.iloc[0]['High']
                    subsequent = post_retest.iloc[1:min(len(post_retest), 13)]
                    if not subsequent.empty:
                        min_low_after = subsequent['Low'].min()
                        max_high_after = subsequent['High'].max()
                        max_favorable = float(retest_high - min_low_after)
                        max_adverse = float(max_high_after - retest_high)
                        reversal_size = float(retest_high - min_low_after)
                        if retest_high > 0:
                            reversal_pct = float((reversal_size / retest_high) * 100)
        
        success_1r = bool(max_favorable >= tbr_range) if retest_found else False
        success_half_r = bool(max_favorable >= (tbr_range * 0.5)) if retest_found else False
        success_2r = bool(max_favorable >= (tbr_range * 2)) if retest_found else False
        success_3r = bool(max_favorable >= (tbr_range * 3)) if retest_found else False
        
        prior_mask = (df.index < tbr_time) & (df.index >= tbr_time - timedelta(hours=50))
        prior_candles = df[prior_mask]
        trend_5h = "neutral"
        trend_20h = "neutral"
        if len(prior_candles) >= 5:
            last5 = prior_candles.tail(5)
            trend_5h = "up" if last5['Close'].iloc[-1] > last5['Close'].iloc[0] else "down"
        if len(prior_candles) >= 20:
            last20 = prior_candles.tail(20)
            trend_20h = "up" if last20['Close'].iloc[-1] > last20['Close'].iloc[0] else "down"
        
        with_trend_5h = (direction == "bullish" and trend_5h == "up") or (direction == "bearish" and trend_5h == "down")
        with_trend_20h = (direction == "bullish" and trend_20h == "up") or (direction == "bearish" and trend_20h == "down")
        
        tbr_candle_bullish = tbr_close > tbr_open
        tbr_candle_color = "green" if tbr_candle_bullish else "red"
        color_matches_direction = (tbr_candle_bullish and direction == "bullish") or (not tbr_candle_bullish and direction == "bearish")
        
        raw_r = float(max_favorable / max_adverse) if max_adverse > 0 else 0.0
        capped_r = min(raw_r, 50.0)
        
        setup = {
            'date': pd.Timestamp(date), 'tbr_time': tbr_time, 'tbr_hour': tbr_hour,
            'tbr_high': tbr_high, 'tbr_low': tbr_low, 'tbr_range': float(tbr_range),
            'tbr_range_pct': float((tbr_range / tbr_low) * 100) if tbr_low > 0 else 0.0,
            'range_vs_atr': float(range_vs_atr),
            'tbr_candle_bullish': bool(tbr_candle_bullish), 'tbr_candle_color': tbr_candle_color,
            'color_matches_direction': bool(color_matches_direction), 'tbr_volume': float(tbr_volume),
            'direction': direction, 'closure_time': closure_time, 'closure_hour': closure_hour,
            'candles_to_closure': candles_to_closure, 'immediate_closure': candles_to_closure == 1,
            'retest_found': bool(retest_found), 'retest_time': retest_time,
            'retest_hour': retest_hour, 'bars_to_retest': bars_to_retest,
            'deepest_fib_hit': deepest_fib_hit,
            'fib_03_hit': bool(fib_03_hit), 'fib_05_hit': bool(fib_05_hit), 'fib_07_hit': bool(fib_07_hit),
            'swept': bool(swept), 'reversal_size': reversal_size, 'reversal_pct': reversal_pct,
            'max_favorable': max_favorable, 'max_adverse': max_adverse,
            'success_half_r': success_half_r, 'success_1r': success_1r,
            'success_2r': success_2r, 'success_3r': success_3r,
            'trend_5h': trend_5h, 'trend_20h': trend_20h,
            'with_trend_5h': bool(with_trend_5h), 'with_trend_20h': bool(with_trend_20h),
            'r_multiple': capped_r,
        }
        setups.append(setup)
    
    return setups


def run_backtest(tickers, tbr_hours=[9, 15], period="2y"):
    all_setups = []
    for ticker_name, ticker_symbol in ASSET_MAP.items():
        if ticker_symbol not in tickers:
            continue
        print(f"  Fetching {ticker_name} ({ticker_symbol})...")
        df = fetch_hourly_data(ticker_symbol, period=period)
        if df.empty:
            print(f"    No data for {ticker_symbol}")
            continue
        for hour in tbr_hours:
            print(f"    Analyzing {hour}:00 candle...")
            setups = identify_tbr_setups(df, tbr_hour=hour)
            for s in setups:
                s['ticker'] = ticker_name
                s['symbol'] = ticker_symbol
            all_setups.extend(setups)
    if not all_setups:
        return pd.DataFrame()
    results = pd.DataFrame(all_setups)
    for ticker in results['ticker'].unique():
        mask = results['ticker'] == ticker
        try:
            results.loc[mask, 'range_size_quartile'] = pd.qcut(
                results.loc[mask, 'tbr_range_pct'], q=4, labels=['Q1_small', 'Q2', 'Q3', 'Q4_large'])
        except ValueError:
            results.loc[mask, 'range_size_quartile'] = 'Q2'
    return results


def generate_summary_stats(results):
    if results.empty:
        return {}
    stats = {}
    total = len(results)
    retested = results['retest_found'].sum()
    stats['total_setups'] = int(total)
    stats['retested'] = int(retested)
    stats['retest_pct'] = round(float(retested / total * 100), 2) if total > 0 else 0
    
    r = results[results['retest_found'] == True].copy()
    if r.empty:
        return stats
    
    stats['success_half_r_pct'] = round(float(r['success_half_r'].mean() * 100), 2)
    stats['success_1r_pct'] = round(float(r['success_1r'].mean() * 100), 2)
    stats['success_2r_pct'] = round(float(r['success_2r'].mean() * 100), 2)
    stats['success_3r_pct'] = round(float(r['success_3r'].mean() * 100), 2)
    stats['avg_reversal_pct'] = round(float(r['reversal_pct'].mean()), 4)
    stats['median_reversal_pct'] = round(float(r['reversal_pct'].median()), 4)
    stats['avg_r_multiple'] = round(float(r['r_multiple'].mean()), 2)
    stats['median_r_multiple'] = round(float(r['r_multiple'].median()), 2)
    
    for fib in [0.3, 0.5, 0.7]:
        subset = r[r['deepest_fib_hit'] == fib]
        if len(subset) > 0:
            stats[f'fib_{fib}_count'] = int(len(subset))
            stats[f'fib_{fib}_success_1r'] = round(float(subset['success_1r'].mean() * 100), 2)
            stats[f'fib_{fib}_avg_reversal_pct'] = round(float(subset['reversal_pct'].mean()), 4)
            stats[f'fib_{fib}_avg_r_multiple'] = round(float(subset['r_multiple'].mean()), 2)
    
    swept = r[r['swept'] == True]
    not_swept = r[r['swept'] == False]
    stats['swept_count'] = int(len(swept))
    stats['not_swept_count'] = int(len(not_swept))
    if len(swept) > 0:
        stats['swept_success_1r'] = round(float(swept['success_1r'].mean() * 100), 2)
        stats['swept_avg_reversal_pct'] = round(float(swept['reversal_pct'].mean()), 4)
    if len(not_swept) > 0:
        stats['not_swept_success_1r'] = round(float(not_swept['success_1r'].mean() * 100), 2)
    
    for period_label in ['5h', '20h']:
        key = f'with_trend_{period_label}'
        with_t = r[r[key] == True]
        against_t = r[r[key] == False]
        if len(with_t) > 0:
            stats[f'with_trend_{period_label}_count'] = int(len(with_t))
            stats[f'with_trend_{period_label}_success'] = round(float(with_t['success_1r'].mean() * 100), 2)
            stats[f'with_trend_{period_label}_avg_rev'] = round(float(with_t['reversal_pct'].mean()), 4)
            stats[f'with_trend_{period_label}_avg_r'] = round(float(with_t['r_multiple'].mean()), 2)
        if len(against_t) > 0:
            stats[f'against_trend_{period_label}_count'] = int(len(against_t))
            stats[f'against_trend_{period_label}_success'] = round(float(against_t['success_1r'].mean() * 100), 2)
            stats[f'against_trend_{period_label}_avg_rev'] = round(float(against_t['reversal_pct'].mean()), 4)
            stats[f'against_trend_{period_label}_avg_r'] = round(float(against_t['r_multiple'].mean()), 2)
    
    for d in ['bullish', 'bearish']:
        subset = r[r['direction'] == d]
        if len(subset) > 0:
            stats[f'{d}_count'] = int(len(subset))
            stats[f'{d}_success_1r'] = round(float(subset['success_1r'].mean() * 100), 2)
            stats[f'{d}_avg_reversal_pct'] = round(float(subset['reversal_pct'].mean()), 4)
    
    for h in [9, 15]:
        subset = r[r['tbr_hour'] == h]
        if len(subset) > 0:
            stats[f'hour_{h}_count'] = int(len(subset))
            stats[f'hour_{h}_success_1r'] = round(float(subset['success_1r'].mean() * 100), 2)
            stats[f'hour_{h}_avg_reversal_pct'] = round(float(subset['reversal_pct'].mean()), 4)
    
    stats['avg_bars_to_retest'] = round(float(r['bars_to_retest'].mean()), 2)
    stats['median_bars_to_retest'] = round(float(r['bars_to_retest'].median()), 2)
    
    r_copy = r.copy()
    r_copy['dow'] = r_copy['date'].dt.dayofweek
    for d_num, d_name in {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri'}.items():
        subset = r_copy[r_copy['dow'] == d_num]
        if len(subset) > 0:
            stats[f'dow_{d_name}_count'] = int(len(subset))
            stats[f'dow_{d_name}_success_1r'] = round(float(subset['success_1r'].mean() * 100), 2)
    
    for color_match, label in [(True, 'color_matches'), (False, 'color_opposes')]:
        subset = r[r['color_matches_direction'] == color_match]
        if len(subset) > 0:
            stats[f'{label}_count'] = int(len(subset))
            stats[f'{label}_success_1r'] = round(float(subset['success_1r'].mean() * 100), 2)
            stats[f'{label}_avg_r'] = round(float(subset['r_multiple'].mean()), 2)
    
    for imm, label in [(True, 'immediate_closure'), (False, 'delayed_closure')]:
        subset = r[r['immediate_closure'] == imm]
        if len(subset) > 0:
            stats[f'{label}_count'] = int(len(subset))
            stats[f'{label}_success_1r'] = round(float(subset['success_1r'].mean() * 100), 2)
            stats[f'{label}_avg_r'] = round(float(subset['r_multiple'].mean()), 2)
    
    if 'range_size_quartile' in r.columns:
        for q in ['Q1_small', 'Q2', 'Q3', 'Q4_large']:
            subset = r[r['range_size_quartile'] == q]
            if len(subset) > 0:
                stats[f'range_{q}_count'] = int(len(subset))
                stats[f'range_{q}_success_1r'] = round(float(subset['success_1r'].mean() * 100), 2)
                stats[f'range_{q}_avg_r'] = round(float(subset['r_multiple'].mean()), 2)
                stats[f'range_{q}_avg_rev'] = round(float(subset['reversal_pct'].mean()), 4)
    
    return stats


def build_json_export(results, stats):
    if results.empty:
        return json.dumps({"error": "No results"})
    
    r = results[results['retest_found'] == True].copy()
    
    export = {
        "meta": {
            "generated": datetime.now().isoformat(),
            "assets_tested": list(results['ticker'].unique()),
            "tbr_hours": sorted(results['tbr_hour'].unique().tolist()),
            "total_setups": int(len(results)),
            "total_retested": int(len(r)),
        },
        "overall_stats": stats,
        "per_asset": {},
        "filter_combos": [],
    }
    
    for ticker in results['ticker'].unique():
        t_all = results[results['ticker'] == ticker]
        t = r[r['ticker'] == ticker]
        if len(t) == 0:
            continue
        export['per_asset'][ticker] = {
            "setups": int(len(t_all)), "retested": int(len(t)),
            "retest_pct": round(float(len(t)/len(t_all)*100), 2),
            "success_1r_pct": round(float(t['success_1r'].mean()*100), 2),
            "success_2r_pct": round(float(t['success_2r'].mean()*100), 2),
            "avg_reversal_pct": round(float(t['reversal_pct'].mean()), 4),
            "avg_r_multiple": round(float(t['r_multiple'].mean()), 2),
            "median_r_multiple": round(float(t['r_multiple'].median()), 2),
        }
    
    # Build filter dimensions
    dims = {
        'session': {'9am': r['tbr_hour'] == 9, '3pm': r['tbr_hour'] == 15},
        'direction': {'bullish': r['direction'] == 'bullish', 'bearish': r['direction'] == 'bearish'},
        'trend_5h': {'with': r['with_trend_5h'] == True, 'against': r['with_trend_5h'] == False},
        'trend_20h': {'with': r['with_trend_20h'] == True, 'against': r['with_trend_20h'] == False},
        'swept': {'held': r['swept'] == False, 'swept': r['swept'] == True},
        'candle_color_match': {'matches': r['color_matches_direction'] == True, 'opposes': r['color_matches_direction'] == False},
        'closure_speed': {'immediate': r['immediate_closure'] == True, 'delayed': r['immediate_closure'] == False},
        'fib': {'0.3': r['deepest_fib_hit'] == 0.3, '0.5': r['deepest_fib_hit'] == 0.5, '0.7': r['deepest_fib_hit'] == 0.7},
    }
    if 'range_size_quartile' in r.columns:
        dims['range_size'] = {q: r['range_size_quartile'] == q for q in ['Q1_small','Q2','Q3','Q4_large']}
    
    def add_combo(filters_dict, mask):
        subset = r[mask]
        if len(subset) >= 8:
            export['filter_combos'].append({
                'filters': filters_dict,
                'count': int(len(subset)),
                'success_1r': round(float(subset['success_1r'].mean()*100), 2),
                'success_2r': round(float(subset['success_2r'].mean()*100), 2),
                'success_3r': round(float(subset['success_3r'].mean()*100), 2),
                'avg_r': round(float(subset['r_multiple'].mean()), 2),
                'median_r': round(float(subset['r_multiple'].median()), 2),
                'avg_rev_pct': round(float(subset['reversal_pct'].mean()), 4),
            })
    
    # Single filters
    for dn, dfs in dims.items():
        for fn, fm in dfs.items():
            add_combo({dn: fn}, fm)
    
    # 2-filter combos
    combo_pairs = [
        ('session','trend_5h'),('session','fib'),('session','direction'),('session','swept'),
        ('session','closure_speed'),('session','candle_color_match'),
        ('trend_5h','fib'),('trend_5h','direction'),('trend_5h','swept'),('trend_5h','candle_color_match'),
        ('trend_5h','closure_speed'),
        ('direction','fib'),('direction','swept'),('direction','candle_color_match'),('direction','closure_speed'),
        ('fib','swept'),('fib','candle_color_match'),('fib','closure_speed'),
        ('closure_speed','swept'),('closure_speed','candle_color_match'),
        ('swept','candle_color_match'),
    ]
    if 'range_size' in dims:
        combo_pairs += [('range_size','session'),('range_size','trend_5h'),('range_size','fib'),
                        ('range_size','direction'),('range_size','swept')]
    
    for da, db in combo_pairs:
        if da not in dims or db not in dims: continue
        for fa, ma in dims[da].items():
            for fb, mb in dims[db].items():
                add_combo({da: fa, db: fb}, ma & mb)
    
    # 3-filter combos
    triples = [
        ('session','trend_5h','fib'),('session','trend_5h','direction'),
        ('session','trend_5h','swept'),('session','trend_5h','candle_color_match'),
        ('session','trend_5h','closure_speed'),
        ('session','fib','direction'),('session','fib','swept'),
        ('session','direction','swept'),('session','direction','candle_color_match'),
        ('trend_5h','fib','direction'),('trend_5h','fib','swept'),
        ('trend_5h','direction','swept'),('trend_5h','direction','candle_color_match'),
        ('trend_5h','direction','closure_speed'),
        ('trend_5h','fib','candle_color_match'),
        ('direction','fib','swept'),('direction','fib','candle_color_match'),
    ]
    for da, db, dc in triples:
        if da not in dims or db not in dims or dc not in dims: continue
        for fa, ma in dims[da].items():
            for fb, mb in dims[db].items():
                for fc, mc in dims[dc].items():
                    add_combo({da: fa, db: fb, dc: fc}, ma & mb & mc)
    
    # 4-filter combos (the elite setups)
    quads = [
        ('session','trend_5h','fib','direction'),
        ('session','trend_5h','fib','swept'),
        ('session','trend_5h','direction','swept'),
        ('session','trend_5h','direction','candle_color_match'),
        ('session','trend_5h','fib','candle_color_match'),
        ('trend_5h','fib','direction','swept'),
        ('trend_5h','fib','direction','candle_color_match'),
    ]
    for da, db, dc, dd in quads:
        if da not in dims or db not in dims or dc not in dims or dd not in dims: continue
        for fa, ma in dims[da].items():
            for fb, mb in dims[db].items():
                for fc, mc in dims[dc].items():
                    for fd, md in dims[dd].items():
                        add_combo({da: fa, db: fb, dc: fc, dd: fd}, ma & mb & mc & md)
    
    # Score and sort
    for c in export['filter_combos']:
        c['score'] = round(c['success_1r'] * np.log1p(c['count']) * (c['median_r'] + 1) / 100, 2)
    export['filter_combos'].sort(key=lambda x: x['score'], reverse=True)
    export['top_25_by_score'] = export['filter_combos'][:25]
    
    high_wr = sorted([c for c in export['filter_combos'] if c['count'] >= 15],
                     key=lambda x: x['success_1r'], reverse=True)
    export['top_25_by_winrate'] = high_wr[:25]
    
    high_r = sorted([c for c in export['filter_combos'] if c['count'] >= 15],
                    key=lambda x: x['median_r'], reverse=True)
    export['top_25_by_r_multiple'] = high_r[:25]
    
    # Retest hour breakdown
    if 'retest_hour' in r.columns:
        retest_by_hour = {}
        for hour in sorted(r['retest_hour'].dropna().unique()):
            subset = r[r['retest_hour'] == hour]
            if len(subset) >= 5:
                retest_by_hour[str(int(hour))] = {
                    'count': int(len(subset)), 
                    'success_1r': round(float(subset['success_1r'].mean()*100), 2),
                    'avg_r': round(float(subset['r_multiple'].mean()), 2),
                }
        export['retest_by_hour'] = retest_by_hour
    
    return json.dumps(export, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import streamlit as st
    import plotly.express as px
    
    st.set_page_config(page_title="TBR Backtester v2", page_icon="🕐", layout="wide")
    st.title("🕐 Time-Based Range (TBR) Backtester v2")
    st.markdown("""**Strategy:** Use the 9:00 AM and/or 3:00 PM EST hourly candle as a POI.  
    After the first hourly close outside the range, wait for price to retrace into the range 
    at the 0.3 / 0.5 / 0.7 fib levels, then trade the reversal.""")
    
    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Backtest Settings")
    selected_assets = st.sidebar.multiselect("Select Assets", options=list(ASSET_MAP.keys()),
        default=["NQ (QQQ proxy)", "ES (SPY proxy)", "BTC-USD", "Gold (GLD)"])
    tbr_hours = st.sidebar.multiselect("TBR Candle Hours (EST)", options=[9, 15], default=[9, 15],
        format_func=lambda x: "9:00 AM" if x == 9 else "3:00 PM")
    period = st.sidebar.selectbox("Lookback Period", options=["6mo", "1y", "2y"], index=2,
        format_func=lambda x: {"6mo": "6 Months", "1y": "1 Year", "2y": "2 Years (Max)"}[x])
    run_button = st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True)
    
    if run_button and selected_assets and tbr_hours:
        tickers = [ASSET_MAP[a] for a in selected_assets]
        with st.spinner("Fetching data and running backtest..."):
            results = run_backtest(tickers, tbr_hours, period)
        if results.empty:
            st.error("No setups found.")
            return
        st.session_state['results'] = results
        st.session_state['stats'] = generate_summary_stats(results)
    
    if 'results' not in st.session_state:
        st.info("👈 Select assets and click **Run Backtest** to begin.")
        return
    
    results = st.session_state['results']
    stats = st.session_state['stats']
    retested = results[results['retest_found'] == True]
    
    # ── Export buttons ───────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.header("📥 Export")
    json_data = build_json_export(results, stats)
    st.sidebar.download_button("📋 Download JSON (for Claude)", data=json_data,
        file_name="tbr_backtest_results.json", mime="application/json", use_container_width=True)
    st.sidebar.download_button("📊 Download CSV (raw data)", data=results.to_csv(index=False),
        file_name="tbr_backtest_raw.csv", mime="text/csv", use_container_width=True)
    
    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📊 Overview", "🎯 Fib Analysis", "📈 Trend Context", "🔄 Sweep Analysis",
        "⏰ Session & Timing", "📅 Day of Week", "🕯️ Candle Properties",
        "📐 Range Size", "🏆 Best Filters", "📋 Raw Data"])
    
    # ── TAB 1: OVERVIEW ─────────────────────────────────────────────────────
    with tab1:
        st.header("Overall TBR Performance")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Setups", stats.get('total_setups',0))
        c2.metric("Retested", f"{stats.get('retested',0)} ({stats.get('retest_pct',0):.1f}%)")
        c3.metric("1R Success Rate", f"{stats.get('success_1r_pct',0):.1f}%")
        c4.metric("Avg R-Multiple", f"{stats.get('avg_r_multiple',0):.2f}")
        c5,c6,c7,c8 = st.columns(4)
        c5.metric("0.5R Success", f"{stats.get('success_half_r_pct',0):.1f}%")
        c6.metric("2R Success", f"{stats.get('success_2r_pct',0):.1f}%")
        c7.metric("3R Success", f"{stats.get('success_3r_pct',0):.1f}%")
        c8.metric("Median R-Multiple", f"{stats.get('median_r_multiple',0):.2f}")
        c9,c10,c11,c12 = st.columns(4)
        c9.metric("Avg Reversal %", f"{stats.get('avg_reversal_pct',0):.3f}%")
        c10.metric("Median Reversal %", f"{stats.get('median_reversal_pct',0):.3f}%")
        c11.metric("Avg Bars to Retest", f"{stats.get('avg_bars_to_retest',0):.1f}")
        c12.metric("Median Bars to Retest", f"{stats.get('median_bars_to_retest',0):.1f}")
        
        st.subheader("Per-Asset Performance")
        ad = []
        for tk in results['ticker'].unique():
            t = results[results['ticker']==tk]; tr = t[t['retest_found']==True]
            ad.append({'Asset':tk,'Setups':len(t),'Retested':len(tr),
                'Retest %':f"{len(tr)/len(t)*100:.1f}%" if len(t)>0 else "0%",
                '1R Win %':f"{tr['success_1r'].mean()*100:.1f}%" if len(tr)>0 else "N/A",
                '2R Win %':f"{tr['success_2r'].mean()*100:.1f}%" if len(tr)>0 else "N/A",
                'Avg Rev %':f"{tr['reversal_pct'].mean():.3f}%" if len(tr)>0 else "N/A",
                'Avg R':f"{tr['r_multiple'].mean():.2f}" if len(tr)>0 else "N/A",
                'Med R':f"{tr['r_multiple'].median():.2f}" if len(tr)>0 else "N/A"})
        st.dataframe(pd.DataFrame(ad), use_container_width=True, hide_index=True)
        
        st.subheader("Bullish vs Bearish")
        ca,cb = st.columns(2)
        with ca:
            st.markdown("**Bullish (Close Above Range)**")
            st.metric("Count", stats.get('bullish_count',0))
            st.metric("1R Success", f"{stats.get('bullish_success_1r',0):.1f}%")
        with cb:
            st.markdown("**Bearish (Close Below Range)**")
            st.metric("Count", stats.get('bearish_count',0))
            st.metric("1R Success", f"{stats.get('bearish_success_1r',0):.1f}%")
        
        if len(retested)>0:
            st.subheader("R-Multiple Distribution")
            fig = px.histogram(retested[retested['r_multiple']<=20], x='r_multiple', nbins=50,
                title="Distribution of R-Multiples (capped at 20 for display)",
                color_discrete_sequence=['#00d4aa'])
            fig.add_vline(x=1.0, line_dash="dash", line_color="yellow", annotation_text="1R")
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # ── TAB 2: FIB ANALYSIS ─────────────────────────────────────────────────
    with tab2:
        st.header("🎯 Fib Level Analysis")
        fd = []
        for fib in [0.3,0.5,0.7]:
            cnt = stats.get(f'fib_{fib}_count',0)
            if cnt>0:
                fd.append({'Fib Level':str(fib),'Count':cnt,
                    '1R Win Rate':stats.get(f'fib_{fib}_success_1r',0),
                    'Avg Reversal %':stats.get(f'fib_{fib}_avg_reversal_pct',0),
                    'Avg R-Multiple':stats.get(f'fib_{fib}_avg_r_multiple',0)})
        if fd:
            fdf = pd.DataFrame(fd)
            c1,c2 = st.columns(2)
            with c1:
                fig = px.bar(fdf, x='Fib Level', y='1R Win Rate', title="1R Win Rate by Fib",
                    color='Fib Level', color_discrete_map={'0.3':'#4ecdc4','0.5':'#ffe66d','0.7':'#ff6b6b'},
                    text='1R Win Rate')
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(template="plotly_dark", height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(fdf, x='Fib Level', y='Avg R-Multiple', title="Avg R-Multiple by Fib",
                    color='Fib Level', color_discrete_map={'0.3':'#4ecdc4','0.5':'#ffe66d','0.7':'#ff6b6b'},
                    text='Avg R-Multiple')
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig.update_layout(template="plotly_dark", height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            dd = fdf.copy()
            dd['1R Win Rate']=dd['1R Win Rate'].apply(lambda x:f"{x:.1f}%")
            dd['Avg Reversal %']=dd['Avg Reversal %'].apply(lambda x:f"{x:.4f}%")
            dd['Avg R-Multiple']=dd['Avg R-Multiple'].apply(lambda x:f"{x:.2f}")
            st.dataframe(dd, use_container_width=True, hide_index=True)
            
            st.subheader("Fib Performance by Asset")
            afd = []
            for tk in retested['ticker'].unique():
                t = retested[retested['ticker']==tk]
                for fib in [0.3,0.5,0.7]:
                    s = t[t['deepest_fib_hit']==fib]
                    if len(s)>0:
                        afd.append({'Asset':tk,'Fib':str(fib),'Count':len(s),
                            '1R Win %':f"{s['success_1r'].mean()*100:.1f}%",
                            'Avg Rev %':f"{s['reversal_pct'].mean():.4f}%",
                            'R-Mult':f"{s['r_multiple'].mean():.2f}"})
            if afd: st.dataframe(pd.DataFrame(afd), use_container_width=True, hide_index=True)
    
    # ── TAB 3: TREND CONTEXT ────────────────────────────────────────────────
    with tab3:
        st.header("📈 Trend Alignment Analysis")
        td = []
        for key, label, desc in [
            ("with_trend_5h","With 5h Trend","Trade direction matches last 5 candles"),
            ("against_trend_5h","Against 5h Trend","Trade direction opposes last 5 candles"),
            ("with_trend_20h","With 20h Trend","Trade direction matches last 20 candles"),
            ("against_trend_20h","Against 20h Trend","Trade direction opposes last 20 candles")]:
            cnt = stats.get(f'{key}_count',0)
            if cnt>0:
                td.append({'Context':label,'Description':desc,'Count':cnt,
                    '1R Success %':stats.get(f'{key}_success',0),
                    'Avg Rev %':stats.get(f'{key}_avg_rev',0),
                    'Avg R-Mult':stats.get(f'{key}_avg_r',0)})
        if td:
            tdf = pd.DataFrame(td)
            fig = px.bar(tdf, x='Context', y='1R Success %', title="With vs Against Trend",
                color='Context', color_discrete_sequence=['#00d4aa','#ff6b6b','#4ecdc4','#ff9999'],
                text='1R Success %')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(template="plotly_dark", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            dd = tdf.copy()
            dd['1R Success %']=dd['1R Success %'].apply(lambda x:f"{x:.1f}%")
            dd['Avg Rev %']=dd['Avg Rev %'].apply(lambda x:f"{x:.4f}%")
            dd['Avg R-Mult']=dd['Avg R-Mult'].apply(lambda x:f"{x:.2f}")
            st.dataframe(dd, use_container_width=True, hide_index=True)
        
        st.subheader("Trend + Fib Level Combo")
        cd = []
        for tv,tl in [(True,"With Trend"),(False,"Against Trend")]:
            for fib in [0.3,0.5,0.7]:
                s = retested[(retested['with_trend_5h']==tv)&(retested['deepest_fib_hit']==fib)]
                if len(s)>=3:
                    cd.append({'Trend':tl,'Fib':str(fib),'Count':len(s),
                        '1R Win %':f"{s['success_1r'].mean()*100:.1f}%",
                        'Avg R':f"{s['r_multiple'].mean():.2f}",
                        'Avg Rev %':f"{s['reversal_pct'].mean():.4f}%"})
        if cd: st.dataframe(pd.DataFrame(cd), use_container_width=True, hide_index=True)
    
    # ── TAB 4: SWEEP ANALYSIS ───────────────────────────────────────────────
    with tab4:
        st.header("🔄 Sweep Analysis")
        sd = []
        for label, mv in [("Range Held",False),("Range Swept",True)]:
            s = retested[retested['swept']==mv]
            if len(s)>0:
                sd.append({'Outcome':label,'Count':len(s),
                    '1R Win %':s['success_1r'].mean()*100,'2R Win %':s['success_2r'].mean()*100,
                    'Avg Rev %':s['reversal_pct'].mean(),'Avg R':s['r_multiple'].mean(),
                    'Med R':s['r_multiple'].median()})
        if sd:
            sdf = pd.DataFrame(sd)
            c1,c2 = st.columns(2)
            with c1:
                fig = px.bar(sdf, x='Outcome', y='1R Win %', title="Win Rate: Held vs Swept",
                    color='Outcome', color_discrete_map={'Range Held':'#00d4aa','Range Swept':'#ff6b6b'},
                    text='1R Win %')
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(template="plotly_dark", height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(sdf, x='Outcome', y='Med R', title="Median R: Held vs Swept",
                    color='Outcome', color_discrete_map={'Range Held':'#00d4aa','Range Swept':'#ff6b6b'},
                    text='Med R')
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig.update_layout(template="plotly_dark", height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            dd = sdf.copy()
            dd['1R Win %']=dd['1R Win %'].apply(lambda x:f"{x:.1f}%")
            dd['2R Win %']=dd['2R Win %'].apply(lambda x:f"{x:.1f}%")
            dd['Avg Rev %']=dd['Avg Rev %'].apply(lambda x:f"{x:.4f}%")
            dd['Avg R']=dd['Avg R'].apply(lambda x:f"{x:.2f}")
            dd['Med R']=dd['Med R'].apply(lambda x:f"{x:.2f}")
            st.dataframe(dd, use_container_width=True, hide_index=True)
    
    # ── TAB 5: SESSION & TIMING ─────────────────────────────────────────────
    with tab5:
        st.header("⏰ Session & Timing Analysis")
        sd = []
        for h,l in [(9,"9:00 AM EST"),(15,"3:00 PM EST")]:
            cnt = stats.get(f'hour_{h}_count',0)
            if cnt>0: sd.append({'Session':l,'Setups':cnt,'1R Win %':stats.get(f'hour_{h}_success_1r',0),
                'Avg Rev %':stats.get(f'hour_{h}_avg_reversal_pct',0)})
        if sd:
            sdf = pd.DataFrame(sd)
            fig = px.bar(sdf, x='Session', y='1R Win %', title="1R Win Rate by Session",
                color='Session', color_discrete_map={'9:00 AM EST':'#4ecdc4','3:00 PM EST':'#ffe66d'},
                text='1R Win %')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(template="plotly_dark", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Retest Hour-of-Day Performance")
        if 'retest_hour' in retested.columns:
            hd = []
            for hour in sorted(retested['retest_hour'].dropna().unique()):
                s = retested[retested['retest_hour']==hour]
                if len(s)>=5:
                    hd.append({'Hour (EST)':f"{int(hour)}:00",'Count':len(s),
                        '1R Win %':s['success_1r'].mean()*100,'Avg R':s['r_multiple'].mean()})
            if hd:
                hdf = pd.DataFrame(hd)
                fig = px.bar(hdf, x='Hour (EST)', y='1R Win %', title="Win Rate by Retest Hour",
                    color='1R Win %', color_continuous_scale=['#ff6b6b','#ffe66d','#00d4aa'],
                    text='1R Win %')
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
                dd = hdf.copy()
                dd['1R Win %']=dd['1R Win %'].apply(lambda x:f"{x:.1f}%")
                dd['Avg R']=dd['Avg R'].apply(lambda x:f"{x:.2f}")
                st.dataframe(dd, use_container_width=True, hide_index=True)
        
        st.subheader("Bars to Retest Distribution")
        if len(retested)>0:
            fig = px.histogram(retested, x='bars_to_retest', nbins=30,
                title="How Many Hourly Candles Before Retest?",
                labels={'bars_to_retest':'Hourly Candles After Closure'},
                color_discrete_sequence=['#4ecdc4'])
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
            c1,c2 = st.columns(2)
            c1.metric("Avg Bars", f"{stats.get('avg_bars_to_retest',0):.1f}")
            c2.metric("Median Bars", f"{stats.get('median_bars_to_retest',0):.1f}")
        
        st.subheader("Closure Speed: Immediate vs Delayed")
        cd = []
        for imm,l in [(True,"Immediate (next candle)"),(False,"Delayed (2+ candles)")]:
            s = retested[retested['immediate_closure']==imm]
            if len(s)>0: cd.append({'Closure':l,'Count':len(s),
                '1R Win %':f"{s['success_1r'].mean()*100:.1f}%",
                'Avg R':f"{s['r_multiple'].mean():.2f}",'Med R':f"{s['r_multiple'].median():.2f}"})
        if cd: st.dataframe(pd.DataFrame(cd), use_container_width=True, hide_index=True)
    
    # ── TAB 6: DAY OF WEEK ──────────────────────────────────────────────────
    with tab6:
        st.header("📅 Day of Week Analysis")
        dd_data = []
        for dn in ['Mon','Tue','Wed','Thu','Fri']:
            cnt = stats.get(f'dow_{dn}_count',0)
            if cnt>0: dd_data.append({'Day':dn,'Count':cnt,'1R Win %':stats.get(f'dow_{dn}_success_1r',0)})
        if dd_data:
            ddf = pd.DataFrame(dd_data)
            fig = px.bar(ddf, x='Day', y='1R Win %', title="1R Win Rate by Day of Week",
                color='1R Win %', color_continuous_scale=['#ff6b6b','#ffe66d','#00d4aa'], text='1R Win %')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
            dd = ddf.copy()
            dd['1R Win %']=dd['1R Win %'].apply(lambda x:f"{x:.1f}%")
            st.dataframe(dd, use_container_width=True, hide_index=True)
    
    # ── TAB 7: CANDLE PROPERTIES ────────────────────────────────────────────
    with tab7:
        st.header("🕯️ TBR Candle Properties")
        st.subheader("Does Candle Color Matter?")
        st.markdown("*If the TBR candle is green and direction is bullish, does it perform differently?*")
        cd = []
        for m,l in [(True,"Color Matches Direction"),(False,"Color Opposes Direction")]:
            s = retested[retested['color_matches_direction']==m]
            if len(s)>0: cd.append({'Scenario':l,'Count':len(s),
                '1R Win %':s['success_1r'].mean()*100,'2R Win %':s['success_2r'].mean()*100,
                'Avg R':s['r_multiple'].mean(),'Med R':s['r_multiple'].median()})
        if cd:
            cdf = pd.DataFrame(cd)
            fig = px.bar(cdf, x='Scenario', y='1R Win %', title="Win Rate by Candle Color Match",
                color='Scenario', color_discrete_map={
                    'Color Matches Direction':'#00d4aa','Color Opposes Direction':'#ff6b6b'},
                text='1R Win %')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(template="plotly_dark", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            dd = cdf.copy()
            dd['1R Win %']=dd['1R Win %'].apply(lambda x:f"{x:.1f}%")
            dd['2R Win %']=dd['2R Win %'].apply(lambda x:f"{x:.1f}%")
            dd['Avg R']=dd['Avg R'].apply(lambda x:f"{x:.2f}")
            dd['Med R']=dd['Med R'].apply(lambda x:f"{x:.2f}")
            st.dataframe(dd, use_container_width=True, hide_index=True)
        
        st.subheader("Candle Color × Direction Matrix")
        md = []
        for color in ['green','red']:
            for direction in ['bullish','bearish']:
                s = retested[(retested['tbr_candle_color']==color)&(retested['direction']==direction)]
                if len(s)>=5:
                    md.append({'TBR Candle':color.capitalize(),'Direction':direction.capitalize(),
                        'Count':len(s),'1R Win %':f"{s['success_1r'].mean()*100:.1f}%",
                        'Avg R':f"{s['r_multiple'].mean():.2f}"})
        if md: st.dataframe(pd.DataFrame(md), use_container_width=True, hide_index=True)
    
    # ── TAB 8: RANGE SIZE ───────────────────────────────────────────────────
    with tab8:
        st.header("📐 Range Size Analysis")
        if 'range_size_quartile' in retested.columns:
            rd = []
            for q in ['Q1_small','Q2','Q3','Q4_large']:
                s = retested[retested['range_size_quartile']==q]
                if len(s)>0:
                    rd.append({'Quartile':q,'Count':len(s),'1R Win %':s['success_1r'].mean()*100,
                        '2R Win %':s['success_2r'].mean()*100,'Avg Rev %':s['reversal_pct'].mean(),
                        'Avg R':s['r_multiple'].mean(),'Med R':s['r_multiple'].median(),
                        'Avg Range %':s['tbr_range_pct'].mean()})
            if rd:
                rdf = pd.DataFrame(rd)
                c1,c2 = st.columns(2)
                with c1:
                    fig = px.bar(rdf, x='Quartile', y='1R Win %', title="Win Rate by Range Size",
                        color='Quartile', text='1R Win %')
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(template="plotly_dark", height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.bar(rdf, x='Quartile', y='Med R', title="Median R by Range Size",
                        color='Quartile', text='Med R')
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    fig.update_layout(template="plotly_dark", height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                dd = rdf.copy()
                dd['1R Win %']=dd['1R Win %'].apply(lambda x:f"{x:.1f}%")
                dd['2R Win %']=dd['2R Win %'].apply(lambda x:f"{x:.1f}%")
                dd['Avg Rev %']=dd['Avg Rev %'].apply(lambda x:f"{x:.4f}%")
                dd['Avg R']=dd['Avg R'].apply(lambda x:f"{x:.2f}")
                dd['Med R']=dd['Med R'].apply(lambda x:f"{x:.2f}")
                dd['Avg Range %']=dd['Avg Range %'].apply(lambda x:f"{x:.3f}%")
                st.dataframe(dd, use_container_width=True, hide_index=True)
        
        st.subheader("Range Size vs ATR")
        if 'range_vs_atr' in retested.columns:
            ad = []
            for l,lo,hi in [("<0.5x ATR",0,0.5),("0.5-1x ATR",0.5,1.0),("1-1.5x ATR",1.0,1.5),
                            ("1.5-2x ATR",1.5,2.0),(">2x ATR",2.0,100)]:
                s = retested[(retested['range_vs_atr']>=lo)&(retested['range_vs_atr']<hi)]
                if len(s)>=5:
                    ad.append({'Range vs ATR':l,'Count':len(s),
                        '1R Win %':f"{s['success_1r'].mean()*100:.1f}%",
                        'Avg R':f"{s['r_multiple'].mean():.2f}"})
            if ad: st.dataframe(pd.DataFrame(ad), use_container_width=True, hide_index=True)
    
    # ── TAB 9: BEST FILTERS ────────────────────────────────────────────────
    with tab9:
        st.header("🏆 Best Filter Combinations")
        st.markdown("""*Systematically tests all filter combinations to find setups with the highest edge.  
        Score = Win Rate × log(Count) × (Median R + 1) — balances win rate, sample size, and reward.*""")
        
        json_parsed = json.loads(build_json_export(results, stats))
        combos = json_parsed.get('filter_combos', [])
        
        if combos:
            def combo_table(title, data, max_rows=25):
                st.subheader(title)
                rows = []
                for c in data[:max_rows]:
                    filters_str = " + ".join([f"{k}={v}" for k,v in c['filters'].items()])
                    rows.append({'Filters':filters_str,'Count':c['count'],
                        '1R Win %':f"{c['success_1r']:.1f}%",'2R Win %':f"{c['success_2r']:.1f}%",
                        '3R Win %':f"{c['success_3r']:.1f}%",
                        'Med R':f"{c['median_r']:.2f}",'Score':c['score']})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            
            combo_table("Top 25 by Composite Score", combos)
            
            high_wr = sorted([c for c in combos if c['count']>=15],
                           key=lambda x: x['success_1r'], reverse=True)
            combo_table("Highest Win Rate (min 15 samples)", high_wr)
            
            high_r = sorted([c for c in combos if c['count']>=15],
                          key=lambda x: x['median_r'], reverse=True)
            combo_table("Best Median R-Multiple (min 15 samples)", high_r)
        
        st.markdown("---")
        st.markdown("💡 **Download the JSON file from the sidebar and paste it into Claude to get AI-powered filter recommendations!**")
    
    # ── TAB 10: RAW DATA ────────────────────────────────────────────────────
    with tab10:
        st.header("📋 Raw Setup Data")
        c1,c2,c3,c4 = st.columns(4)
        with c1: fa = st.selectbox("Asset", ["All"]+list(results['ticker'].unique()))
        with c2: fd = st.selectbox("Direction", ["All","bullish","bearish"])
        with c3: fr = st.selectbox("Retest", ["All","Retested","Not Retested"])
        with c4: fs = st.selectbox("Sweep", ["All","Held","Swept"])
        
        d = results.copy()
        if fa!="All": d = d[d['ticker']==fa]
        if fd!="All": d = d[d['direction']==fd]
        if fr=="Retested": d = d[d['retest_found']==True]
        elif fr=="Not Retested": d = d[d['retest_found']==False]
        if fs=="Held": d = d[d['swept']==False]
        elif fs=="Swept": d = d[d['swept']==True]
        
        cols = ['date','ticker','tbr_hour','direction','tbr_range_pct','tbr_candle_color',
                'color_matches_direction','immediate_closure','retest_found','retest_hour',
                'deepest_fib_hit','swept','reversal_pct','r_multiple','success_1r','success_2r',
                'success_3r','with_trend_5h','with_trend_20h','bars_to_retest','range_size_quartile']
        cols = [c for c in cols if c in d.columns]
        st.dataframe(d[cols].sort_values('date',ascending=False), use_container_width=True,
            hide_index=True, height=600)
        st.download_button("📥 Download Filtered CSV", data=d.to_csv(index=False),
            file_name="tbr_filtered.csv", mime="text/csv")


if __name__ == "__main__":
    main()