"""
ZenTrades VCP Scanner Monitor - Enhanced Edition
Tracks alerts AND checks if breakouts continued using Coinbase API
"""

import streamlit as st
import pandas as pd
import json
import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from zoneinfo import ZoneInfo

# Timezone - Denver (Mountain Time)
DENVER_TZ = ZoneInfo('America/Denver')

def to_denver(dt):
    """Convert datetime to Denver time"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo('UTC'))
    return dt.astimezone(DENVER_TZ)

def denver_now():
    return datetime.now(DENVER_TZ)

# Config
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
ALERTS_FILE = DATA_DIR / "vcp_alerts.json"
PRICE_CACHE_FILE = DATA_DIR / "price_cache.json"
OUTCOMES_FILE = DATA_DIR / "trade_outcomes.json"  # Stores locked-in trade results

st.set_page_config(
    page_title="VCP Alert Monitor",
    page_icon="🎯",
    layout="wide"
)

# Hyperliquid symbol mappings - they use different names for some assets
# Format: "ZenTrades symbol" -> "Hyperliquid symbol"
HYPERLIQUID_SYMBOL_MAP = {
    'KSHIB': 'kSHIB',
    'KPEPE': 'kPEPE', 
    'KBONK': 'kBONK',
    'KLUNC': 'kLUNC',
    'KFLOKI': 'kFLOKI',
    'KSHIB/USDC': 'kSHIB',
    'KPEPE/USDC': 'kPEPE',
    'KBONK/USDC': 'kBONK', 
    'KLUNC/USDC': 'kLUNC',
    'KFLOKI/USDC': 'kFLOKI',
}

def normalize_symbol_for_hyperliquid(symbol):
    """Convert ZenTrades symbol to Hyperliquid format"""
    # Check direct mapping first
    if symbol in HYPERLIQUID_SYMBOL_MAP:
        return HYPERLIQUID_SYMBOL_MAP[symbol]
    
    # Extract base (e.g., "KSHIB/USDC" -> "KSHIB")
    base = symbol.split('/')[0] if '/' in symbol else symbol
    
    if base in HYPERLIQUID_SYMBOL_MAP:
        return HYPERLIQUID_SYMBOL_MAP[base]
    
    return base


# Price fetching functions
def get_hyperliquid_prices():
    """Fetch all mid prices from Hyperliquid - primary source"""
    try:
        url = "https://api.hyperliquid.xyz/info"
        payload = {"type": "allMids"}
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Convert to our symbol format (add /USDC suffix)
            prices = {}
            for symbol, price in data.items():
                # Hyperliquid returns symbols like "BTC", "ETH", "kSHIB", etc.
                # Store multiple variations for matching
                prices[f"{symbol}/USDC"] = float(price)
                prices[symbol] = float(price)
                # Also store uppercase version for matching
                prices[f"{symbol.upper()}/USDC"] = float(price)
                prices[symbol.upper()] = float(price)
                # Also store lowercase
                prices[symbol.lower()] = float(price)
                prices[f"{symbol.lower()}/USDC"] = float(price)
            return prices
        else:
            st.sidebar.warning(f"Hyperliquid API returned status {response.status_code}")
    except Exception as e:
        st.sidebar.warning(f"Hyperliquid API error: {e}")
    return {}


def get_hyperliquid_candles(symbol, start_time, end_time=None, interval="1m"):
    """
    Fetch historical candles from Hyperliquid for backtesting
    Returns list of candles with high/low to check if TP/Stop was hit
    """
    try:
        # Convert symbol format using mapping
        coin = normalize_symbol_for_hyperliquid(symbol)
        
        # Convert timestamps to milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000) if end_time else int(datetime.now().timestamp() * 1000)
        
        url = "https://api.hyperliquid.xyz/info"
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms
            }
        }
        
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            candles = response.json()
            # Each candle has: T (end time), o, h, l, c, v, t (start time)
            return candles
    except Exception as e:
        pass
    return []


def backtest_trade(symbol, entry_price, stop_price, tp_price, entry_time):
    """
    Backtest a single trade using historical candles
    Returns: (outcome, exit_time, exit_price, final_r)
    - outcome: 'win', 'loss', or 'open'
    """
    if not stop_price or not tp_price:
        return ('open', None, None, None)
    
    # Fetch candles from entry time to now
    candles = get_hyperliquid_candles(symbol, entry_time)
    
    if not candles:
        return ('open', None, None, None)
    
    # Sort candles by time
    candles_sorted = sorted(candles, key=lambda x: x.get('t', 0))
    
    for candle in candles_sorted:
        candle_time = candle.get('t', 0)  # Start time in ms
        high = float(candle.get('h', 0))
        low = float(candle.get('l', 0))
        
        # Check if stop was hit (low touched or went below stop)
        stop_hit = low <= stop_price
        # Check if TP was hit (high touched or went above TP)
        tp_hit = high >= tp_price
        
        if stop_hit and tp_hit:
            # Both hit in same candle - need to determine which first
            # Conservative approach: assume stop hit first if open was closer to stop
            candle_open = float(candle.get('o', entry_price))
            if abs(candle_open - stop_price) < abs(candle_open - tp_price):
                exit_time = datetime.fromtimestamp(candle_time / 1000, tz=ZoneInfo('UTC'))
                return ('loss', exit_time, stop_price, -1.0)
            else:
                exit_time = datetime.fromtimestamp(candle_time / 1000, tz=ZoneInfo('UTC'))
                return ('win', exit_time, tp_price, 2.0)
        elif stop_hit:
            exit_time = datetime.fromtimestamp(candle_time / 1000, tz=ZoneInfo('UTC'))
            return ('loss', exit_time, stop_price, -1.0)
        elif tp_hit:
            exit_time = datetime.fromtimestamp(candle_time / 1000, tz=ZoneInfo('UTC'))
            return ('win', exit_time, tp_price, 2.0)
    
    # Neither hit yet - trade still open
    return ('open', None, None, None)


def backtest_trade_extended(symbol, entry_price, stop_price, entry_time):
    """
    Extended backtest tracking multiple TP levels (2R, 4R, 6R, 8R)
    Also simulates the scaled exit strategy:
    - 50% out at 2R, move stop to breakeven
    - 20% out at 4R
    - 15% out at 6R
    - 15% out at 8R
    
    Returns dict with:
    - outcome: 'loss', 'tp1', 'tp2', 'tp3', 'tp4', 'open'
    - tp1_hit, tp2_hit, tp3_hit, tp4_hit: bool
    - scaled_r: R-multiple using the scaling strategy
    - max_r_reached: highest R reached before reversal
    """
    if not stop_price or not entry_price:
        return {
            'outcome': 'open',
            'tp1_hit': False, 'tp2_hit': False, 'tp3_hit': False, 'tp4_hit': False,
            'scaled_r': None, 'max_r_reached': None,
            'exit_time': None
        }
    
    risk = entry_price - stop_price
    tp1_price = entry_price + (2 * risk)  # 2R
    tp2_price = entry_price + (4 * risk)  # 4R
    tp3_price = entry_price + (6 * risk)  # 6R
    tp4_price = entry_price + (8 * risk)  # 8R
    
    # Fetch candles from entry time to now
    candles = get_hyperliquid_candles(symbol, entry_time)
    
    if not candles:
        return {
            'outcome': 'open',
            'tp1_hit': False, 'tp2_hit': False, 'tp3_hit': False, 'tp4_hit': False,
            'scaled_r': None, 'max_r_reached': None,
            'exit_time': None
        }
    
    # Sort candles by time
    candles_sorted = sorted(candles, key=lambda x: x.get('t', 0))
    
    # Track state
    tp1_hit = False
    tp2_hit = False
    tp3_hit = False
    tp4_hit = False
    current_stop = stop_price  # Will move to breakeven after TP1
    max_r_reached = 0
    exit_time = None
    
    # Track remaining position: starts at 100%
    # After TP1 (2R): 50% remaining, 50% took +2R = 1.0R locked
    # After TP2 (4R): 30% remaining, 20% took +4R = 1.0 + 0.8 = 1.8R locked
    # After TP3 (6R): 15% remaining, 15% took +6R = 1.8 + 0.9 = 2.7R locked
    # After TP4 (8R): 0% remaining, 15% took +8R = 2.7 + 1.2 = 3.9R total
    
    for candle in candles_sorted:
        candle_time = candle.get('t', 0)
        high = float(candle.get('h', 0))
        low = float(candle.get('l', 0))
        
        # Calculate current R based on high
        current_r = (high - entry_price) / risk if risk > 0 else 0
        max_r_reached = max(max_r_reached, current_r)
        
        # Check TP levels (in order)
        if not tp1_hit and high >= tp1_price:
            tp1_hit = True
            current_stop = entry_price  # Move stop to breakeven
        
        if tp1_hit and not tp2_hit and high >= tp2_price:
            tp2_hit = True
        
        if tp2_hit and not tp3_hit and high >= tp3_price:
            tp3_hit = True
        
        if tp3_hit and not tp4_hit and high >= tp4_price:
            tp4_hit = True
            exit_time = datetime.fromtimestamp(candle_time / 1000, tz=ZoneInfo('UTC'))
            # All TPs hit - full exit
            # Scaled R: 50% at 2R + 20% at 4R + 15% at 6R + 15% at 8R = 1.0 + 0.8 + 0.9 + 1.2 = 3.9R
            return {
                'outcome': 'tp4',
                'tp1_hit': True, 'tp2_hit': True, 'tp3_hit': True, 'tp4_hit': True,
                'scaled_r': 3.9, 'max_r_reached': max_r_reached,
                'exit_time': exit_time
            }
        
        # Check if stop hit
        if low <= current_stop:
            exit_time = datetime.fromtimestamp(candle_time / 1000, tz=ZoneInfo('UTC'))
            
            if not tp1_hit:
                # Full loss - stopped out before any TP
                return {
                    'outcome': 'loss',
                    'tp1_hit': False, 'tp2_hit': False, 'tp3_hit': False, 'tp4_hit': False,
                    'scaled_r': -1.0, 'max_r_reached': max_r_reached,
                    'exit_time': exit_time
                }
            elif tp1_hit and not tp2_hit:
                # Stopped at breakeven after TP1
                # 50% took profit at 2R, 50% breakeven = 1.0R
                return {
                    'outcome': 'tp1',
                    'tp1_hit': True, 'tp2_hit': False, 'tp3_hit': False, 'tp4_hit': False,
                    'scaled_r': 1.0, 'max_r_reached': max_r_reached,
                    'exit_time': exit_time
                }
            elif tp2_hit and not tp3_hit:
                # Stopped at breakeven after TP2
                # 50% at 2R + 20% at 4R + 30% breakeven = 1.0 + 0.8 + 0 = 1.8R
                return {
                    'outcome': 'tp2',
                    'tp1_hit': True, 'tp2_hit': True, 'tp3_hit': False, 'tp4_hit': False,
                    'scaled_r': 1.8, 'max_r_reached': max_r_reached,
                    'exit_time': exit_time
                }
            elif tp3_hit and not tp4_hit:
                # Stopped at breakeven after TP3
                # 50% at 2R + 20% at 4R + 15% at 6R + 15% breakeven = 1.0 + 0.8 + 0.9 + 0 = 2.7R
                return {
                    'outcome': 'tp3',
                    'tp1_hit': True, 'tp2_hit': True, 'tp3_hit': True, 'tp4_hit': False,
                    'scaled_r': 2.7, 'max_r_reached': max_r_reached,
                    'exit_time': exit_time
                }
    
    # Trade still open - calculate current scaled R based on what's been hit
    if tp3_hit:
        # 50% at 2R + 20% at 4R + 15% at 6R + 15% still open = 2.7R locked + open
        scaled_r = 2.7
        outcome = 'open_tp3'
    elif tp2_hit:
        # 50% at 2R + 20% at 4R + 30% still open = 1.8R locked + open
        scaled_r = 1.8
        outcome = 'open_tp2'
    elif tp1_hit:
        # 50% at 2R + 50% still open at breakeven stop = 1.0R locked + open
        scaled_r = 1.0
        outcome = 'open_tp1'
    else:
        scaled_r = 0
        outcome = 'open'
    
    return {
        'outcome': outcome,
        'tp1_hit': tp1_hit, 'tp2_hit': tp2_hit, 'tp3_hit': tp3_hit, 'tp4_hit': tp4_hit,
        'scaled_r': scaled_r, 'max_r_reached': max_r_reached,
        'exit_time': exit_time
    }


def get_coinbase_price(symbol):
    """Fetch current price from Coinbase (fallback)"""
    try:
        base = symbol.split('/')[0]
        url = f"https://api.coinbase.com/v2/prices/{base}-USD/spot"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return float(data['data']['amount'])
    except:
        pass
    return None


def get_coingecko_price(symbol):
    """Fetch current price from CoinGecko (second fallback)"""
    try:
        base = symbol.split('/')[0].lower()
        
        # Common symbol mappings to CoinGecko IDs
        symbol_map = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'sol': 'solana',
            'avax': 'avalanche-2',
            'matic': 'matic-network',
            'link': 'chainlink',
            'uni': 'uniswap',
            'aave': 'aave',
            'crv': 'curve-dao-token',
            'doge': 'dogecoin',
            'shib': 'shiba-inu',
            'ltc': 'litecoin',
            'bch': 'bitcoin-cash',
            'xlm': 'stellar',
            'xrp': 'ripple',
            'ada': 'cardano',
            'dot': 'polkadot',
            'atom': 'cosmos',
            'near': 'near',
            'ftm': 'fantom',
            'arb': 'arbitrum',
            'op': 'optimism',
            'sui': 'sui',
            'apt': 'aptos',
            'sei': 'sei-network',
            'inj': 'injective-protocol',
            'tia': 'celestia',
            'jup': 'jupiter-exchange-solana',
            'wif': 'dogwifcoin',
            'bonk': 'bonk',
            'pepe': 'pepe',
            'floki': 'floki',
            'bnb': 'binancecoin',
            'fil': 'filecoin',
            'render': 'render-token',
            'rndr': 'render-token',
            'grt': 'the-graph',
            'mkr': 'maker',
            'ldo': 'lido-dao',
            'rpl': 'rocket-pool',
            'pendle': 'pendle',
            'gmx': 'gmx',
            'dydx': 'dydx-chain',
            'snx': 'havven',
            'comp': 'compound-governance-token',
            'zro': 'layerzero',
            'strk': 'starknet',
            'wld': 'worldcoin-wld',
            'blur': 'blur',
            'meme': 'memecoin',
            'ondo': 'ondo-finance',
            'hbar': 'hedera-hashgraph',
            'vet': 'vechain',
            'algo': 'algorand',
            'egld': 'elrond-erd-2',
            'flow': 'flow',
            'mina': 'mina-protocol',
            'kava': 'kava',
            'rose': 'oasis-network',
            'zec': 'zcash',
            'xtz': 'tezos',
            'eos': 'eos',
            'xmr': 'monero',
            'fartcoin': 'fartcoin',
            'brett': 'brett',
            'hype': 'hyperliquid',
            'virtual': 'virtual-protocol',
            'paxg': 'pax-gold',
            'pengu': 'pudgy-penguins',
            'pump': 'pump',
            'saga': 'saga-2',
            # Newer tokens
            'vine': 'vine',
            'purr': 'purr-2',
            'trx': 'tron',
            'trump': 'maga',
            'popcat': 'popcat',
            'moodeng': 'moo-deng',
            'act': 'act-i-the-ai-prophecy',
            'goat': 'goatseus-maximus',
            'ai16z': 'ai16z',
            'zerebro': 'zerebro',
            'grass': 'grass',
            'tnsr': 'tensor',
            'jto': 'jito-governance-token',
            'pyth': 'pyth-network',
            'ena': 'ethena',
            'eigen': 'eigenlayer',
            'mnt': 'mantle',
            'metis': 'metis-token',
            'aster': 'aster-defi',
            'xpl': 'xpla',
            'stable': 'stable-protocol',
            'zora': 'zora',
            'mon': 'mon-protocol',
            'bera': 'berachain',
            'hemi': 'hemi',
        }
        
        coin_id = symbol_map.get(base, base)
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if coin_id in data and 'usd' in data[coin_id]:
                return float(data[coin_id]['usd'])
    except:
        pass
    return None


def get_price_fallback(symbol):
    """Try Coinbase, then CoinGecko as fallbacks"""
    price = get_coinbase_price(symbol)
    if price is None:
        price = get_coingecko_price(symbol)
    return price


def load_price_cache():
    """Load cached prices"""
    if PRICE_CACHE_FILE.exists():
        with open(PRICE_CACHE_FILE, 'r') as f:
            cache = json.load(f)
            # Check if cache is less than 5 minutes old
            if cache.get('timestamp'):
                cache_time = datetime.fromisoformat(cache['timestamp'])
                if datetime.now() - cache_time < timedelta(minutes=5):
                    return cache.get('prices', {})
    return {}


def save_price_cache(prices):
    """Save prices to cache"""
    cache = {
        'timestamp': datetime.now().isoformat(),
        'prices': prices
    }
    with open(PRICE_CACHE_FILE, 'w') as f:
        json.dump(cache, f)


def fetch_prices_for_symbols(symbols):
    """Fetch prices - Hyperliquid first, then fallback to other sources"""
    prices = load_price_cache()
    
    # Check if cache is fresh (under 1 min for Hyperliquid data)
    if PRICE_CACHE_FILE.exists():
        cache_age = time.time() - PRICE_CACHE_FILE.stat().st_mtime
        if cache_age < 60:  # Cache is fresh
            return prices
    
    # First, get all Hyperliquid prices in one call
    hl_prices = get_hyperliquid_prices()
    
    # Match symbols to Hyperliquid prices
    symbols_needing_fallback = []
    for symbol in symbols:
        base = symbol.split('/')[0] if '/' in symbol else symbol
        base_upper = base.upper()
        base_lower = base.lower()
        
        # Try multiple matching strategies
        found = False
        for try_key in [symbol, base, base_upper, base_lower, 
                        f"{base}/USDC", f"{base_upper}/USDC", f"{base_lower}/USDC"]:
            if try_key in hl_prices:
                prices[symbol] = hl_prices[try_key]
                found = True
                break
        
        if not found:
            # Try mapped symbol (e.g., KSHIB -> kSHIB)
            mapped = normalize_symbol_for_hyperliquid(symbol)
            for try_key in [mapped, f"{mapped}/USDC", mapped.upper(), mapped.lower()]:
                if try_key in hl_prices:
                    prices[symbol] = hl_prices[try_key]
                    found = True
                    break
        
        if not found:
            symbols_needing_fallback.append(symbol)
    
    # For symbols not on Hyperliquid, try fallbacks in parallel
    if symbols_needing_fallback:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(get_price_fallback, s): s for s in symbols_needing_fallback}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    price = future.result()
                    if price:
                        prices[symbol] = price
                except:
                    pass
    
    save_price_cache(prices)
    return prices


def load_alerts():
    """Load alerts from JSON file"""
    if ALERTS_FILE.exists():
        with open(ALERTS_FILE, 'r') as f:
            return json.load(f)
    return []


def load_outcomes():
    """Load locked-in trade outcomes"""
    if OUTCOMES_FILE.exists():
        with open(OUTCOMES_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_outcomes(outcomes):
    """Save trade outcomes - keyed by message_id"""
    with open(OUTCOMES_FILE, 'w') as f:
        json.dump(outcomes, f, indent=2)


def save_alerts(alerts):
    """Save alerts to JSON file"""
    with open(ALERTS_FILE, 'w') as f:
        json.dump(alerts, f, indent=2, default=str)


# Sidebar
st.sidebar.title("🎯 VCP Monitor")
st.sidebar.markdown("---")

# Auto-refresh option
auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False)
if auto_refresh:
    st.sidebar.info("Page will refresh every 60 seconds")
    time.sleep(0.1)  # Small delay
    st.rerun()

# Manual refresh
if st.sidebar.button("🔄 Refresh Prices"):
    # Clear price cache
    if PRICE_CACHE_FILE.exists():
        os.remove(PRICE_CACHE_FILE)
    st.rerun()

# Load data
alerts = load_alerts()
df = pd.DataFrame(alerts) if alerts else pd.DataFrame()

# Sidebar stats
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Database Stats")
st.sidebar.metric("Total Alerts Saved", len(alerts))
if alerts:
    # Show date range
    timestamps = [a.get('timestamp', '') for a in alerts if a.get('timestamp')]
    if timestamps:
        oldest = min(timestamps)[:10]
        newest = max(timestamps)[:10]
        st.sidebar.caption(f"From: {oldest}")
        st.sidebar.caption(f"To: {newest}")

st.sidebar.markdown("---")

# Main content
st.title("🎯 ZenTrades VCP Scanner Monitor")

if df.empty:
    st.info("""
    **No alerts recorded yet!**
    
    Run the collector first:
    ```bash
    python tg_collector.py --channel "ZenTrades VCP Scanner" --limit 500
    ```
    
    Or for continuous collection:
    ```bash
    python auto_collector.py
    ```
    """)

else:
    # Convert timestamp to Denver time
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['timestamp_denver'] = df['timestamp'].dt.tz_convert('America/Denver')
    df['date'] = df['timestamp_denver'].dt.date
    df['hour'] = df['timestamp_denver'].dt.hour
    df['day_of_week'] = df['timestamp_denver'].dt.day_name()
    
    # Ensure breakout_price exists - use last_vector_high as fallback
    if 'breakout_price' not in df.columns:
        df['breakout_price'] = None
    df['breakout_price'] = df.apply(
        lambda row: row.get('breakout_price') if pd.notna(row.get('breakout_price')) else row.get('last_vector_high'), 
        axis=1
    )
    
    # Calculate time since alert (using Denver time)
    now_denver = denver_now()
    df['hours_ago'] = (now_denver - df['timestamp_denver']).dt.total_seconds() / 3600
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Performance Tracker", "📈 Frequency", "🪙 Symbols", "⏰ Timing", "📋 Raw Data"])
    
    with tab1:
        st.subheader("🎯 Breakout Performance Tracker")
        st.markdown("See how alerts performed after the breakout signal")
        
        # First row of filters
        col1, col2, col3 = st.columns(3)
        with col1:
            hours_filter = st.selectbox("Show alerts from last:", 
                                        ["4 hours", "12 hours", "24 hours", "48 hours", "All"],
                                        index=2)
        with col2:
            min_volume = st.number_input("Min 24h Volume ($)", value=0, step=100000)
        with col3:
            timeframe_filter = st.multiselect("Timeframe", 
                                              df['timeframe'].dropna().unique().tolist(),
                                              default=df['timeframe'].dropna().unique().tolist())
        
        # Second row of filters
        col4, col5 = st.columns(2)
        with col4:
            # Check if we have Conservative/Risky data
            has_dual_setups = 'conservative_stop' in df.columns and df['conservative_stop'].notna().any()
            if has_dual_setups:
                trade_type = st.selectbox("Risk Level", 
                                         ["Conservative", "Risky"],
                                         index=1)  # Default to Risky
            else:
                trade_type = "Risky"  # Default
        
        with col5:
            # Alert type filter (BO vs CHEAT)
            has_alert_types = 'alert_type' in df.columns and df['alert_type'].notna().any()
            if has_alert_types:
                alert_types = df['alert_type'].dropna().unique().tolist()
                alert_type_filter = st.multiselect("Alert Type", 
                                                   alert_types,
                                                   default=alert_types,
                                                   format_func=lambda x: "🎯 Breakout (BO)" if x == "BO" else "⚡ Cheat (Early)")
            else:
                alert_type_filter = None
        
        # Apply filters
        filtered_df = df.copy()
        
        if hours_filter != "All":
            hours = int(hours_filter.split()[0])
            filtered_df = filtered_df[filtered_df['hours_ago'] <= hours]
        
        if min_volume > 0:
            filtered_df = filtered_df[filtered_df['volume_24h'] >= min_volume]
        
        if timeframe_filter:
            filtered_df = filtered_df[filtered_df['timeframe'].isin(timeframe_filter)]
        
        if alert_type_filter and 'alert_type' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['alert_type'].isin(alert_type_filter)]
        
        if len(filtered_df) > 0:
            # Get unique symbols and fetch current prices
            symbols = filtered_df['symbol'].dropna().unique().tolist()
            
            with st.spinner("Fetching prices from Hyperliquid..."):
                current_prices = fetch_prices_for_symbols(symbols)
            
            # Debug: Show which symbols couldn't get prices
            missing_prices = [s for s in symbols if s not in current_prices or current_prices.get(s) is None]
            if missing_prices:
                st.sidebar.warning(f"No price data for: {', '.join(missing_prices)}")
            
            # Load existing outcomes (locked-in results from backtesting)
            outcomes = load_outcomes()
            outcomes_updated = False
            
            # Progress bar for backtesting
            progress_text = "Backtesting trades..."
            progress_bar = st.progress(0, text=progress_text)
            total_trades = len(filtered_df)
            
            # Calculate performance
            perf_data = []
            perf_data_risky = []  # For comparison mode
            
            for idx, (_, row) in enumerate(filtered_df.iterrows()):
                # Update progress
                progress_bar.progress((idx + 1) / total_trades, text=f"Backtesting {idx + 1}/{total_trades}...")
                
                symbol = row.get('symbol')
                message_id = str(row.get('message_id', ''))
                
                # Get entry price
                entry_price = row.get('entry_price') if pd.notna(row.get('entry_price')) else None
                if not entry_price:
                    bp = row.get('breakout_price')
                    lvh = row.get('last_vector_high')
                    entry_price = bp if pd.notna(bp) else (lvh if pd.notna(lvh) else None)
                
                # Determine stop price based on trade type
                if trade_type == "Conservative" or trade_type == "Legacy":
                    cons_stop = row.get('conservative_stop') if pd.notna(row.get('conservative_stop')) else None
                    stop_price = cons_stop if cons_stop else (row.get('stop_price') if pd.notna(row.get('stop_price')) else None)
                    tp_2r = row.get('conservative_2r') if pd.notna(row.get('conservative_2r')) else None
                elif trade_type == "Risky":
                    risky_stop = row.get('risky_stop') if pd.notna(row.get('risky_stop')) else None
                    stop_price = risky_stop if risky_stop else (row.get('stop_price') if pd.notna(row.get('stop_price')) else None)
                    tp_2r = row.get('risky_2r') if pd.notna(row.get('risky_2r')) else None
                else:  # Both (Compare)
                    # We'll calculate both
                    cons_stop = row.get('conservative_stop') if pd.notna(row.get('conservative_stop')) else None
                    risky_stop = row.get('risky_stop') if pd.notna(row.get('risky_stop')) else None
                    stop_price = cons_stop if cons_stop else (row.get('stop_price') if pd.notna(row.get('stop_price')) else None)
                    tp_2r = row.get('conservative_2r') if pd.notna(row.get('conservative_2r')) else None
                
                current_price = current_prices.get(symbol)
                
                if symbol and entry_price:
                    # Calculate Take Profit (2x risk) if not provided
                    if tp_2r:
                        tp_price = tp_2r
                    elif stop_price and entry_price:
                        risk = entry_price - stop_price
                        tp_price = entry_price + (2 * risk)
                    else:
                        tp_price = None
                    
                    # Unique key for this trade type
                    outcome_key = f"{message_id}_{trade_type.lower()}"
                    
                    # Check if we have a locked-in outcome for this trade
                    locked_outcome = outcomes.get(outcome_key) or outcomes.get(message_id)  # Legacy fallback
                    
                    if locked_outcome and locked_outcome.get('trade_type', 'legacy') == trade_type.lower():
                        # Use the locked-in result
                        outcome = locked_outcome['outcome']
                        status = locked_outcome['status']
                        r_multiple = locked_outcome.get('final_r', None)
                    else:
                        # No locked outcome - run backtest using historical candles
                        entry_time = row['timestamp'].to_pydatetime()
                        if entry_time.tzinfo is None:
                            entry_time = entry_time.replace(tzinfo=ZoneInfo('UTC'))
                        
                        bt_outcome, exit_time, exit_price, final_r = backtest_trade(
                            symbol, entry_price, stop_price, tp_price, entry_time
                        )
                        
                        if bt_outcome == 'win':
                            status = "🟢 TP Hit (2R)"
                            outcome = "win"
                            r_multiple = 2.0
                            # Lock in this outcome
                            outcomes[outcome_key] = {
                                'outcome': 'win',
                                'status': status,
                                'final_r': 2.0,
                                'exit_price': exit_price,
                                'exit_time': exit_time.isoformat() if exit_time else None,
                                'locked_at': denver_now().isoformat(),
                                'trade_type': trade_type.lower()
                            }
                            outcomes_updated = True
                        elif bt_outcome == 'loss':
                            status = "🔴 Stopped Out"
                            outcome = "loss"
                            r_multiple = -1.0
                            # Lock in this outcome
                            outcomes[outcome_key] = {
                                'outcome': 'loss',
                                'status': status,
                                'final_r': -1.0,
                                'exit_price': exit_price,
                                'exit_time': exit_time.isoformat() if exit_time else None,
                                'locked_at': denver_now().isoformat(),
                                'trade_type': trade_type.lower()
                            }
                            outcomes_updated = True
                        else:
                            # Still open - calculate current R-multiple
                            outcome = "open"
                            if current_price:
                                pct_change = ((current_price - entry_price) / entry_price) * 100
                                if stop_price and entry_price != stop_price:
                                    risk_pct = abs((entry_price - stop_price) / entry_price) * 100
                                    r_multiple = pct_change / risk_pct if risk_pct > 0 else None
                                else:
                                    r_multiple = None
                                status = "🟡 In Profit" if pct_change > 0 else "🟠 Underwater"
                            else:
                                r_multiple = None
                                status = "❓ No Price"
                    
                    # Calculate current pct change for display
                    pct_change = ((current_price - entry_price) / entry_price) * 100 if current_price else None
                    
                    # Format stop % for display
                    stop_pct_display = None
                    if trade_type == "Conservative":
                        stop_pct_display = row.get('conservative_stop_pct')
                    elif trade_type == "Risky":
                        stop_pct_display = row.get('risky_stop_pct')
                    else:
                        stop_pct_display = row.get('conservative_stop_pct')
                    
                    # Get alert type
                    alert_type_val = row.get('alert_type', 'BO')
                    alert_type_display = "🎯 BO" if alert_type_val == "BO" else "⚡ CHEAT"
                    
                    perf_data.append({
                        'Type': alert_type_display,
                        'Symbol': symbol,
                        'TF': row.get('timeframe', '?'),
                        'Alert Time': row['timestamp_denver'].strftime('%m/%d %H:%M'),
                        'Entry $': f"${entry_price:.6f}" if entry_price < 1 else f"${entry_price:.2f}",
                        'Current $': f"${current_price:.6f}" if current_price and current_price < 1 else (f"${current_price:.2f}" if current_price else "N/A"),
                        'Stop $': f"${stop_price:.6f}" if stop_price and stop_price < 1 else (f"${stop_price:.2f}" if stop_price else "N/A"),
                        'Stop %': f"{stop_pct_display:.2f}%" if pd.notna(stop_pct_display) else "N/A",
                        'TP (2R)': f"${tp_price:.6f}" if tp_price and tp_price < 1 else (f"${tp_price:.2f}" if tp_price else "N/A"),
                        'Change %': pct_change,
                        'R-Mult': r_multiple,
                        'Status': status,
                        'Outcome': outcome,
                        'Volume': row.get('volume_24h', 0),
                        'alert_type_raw': alert_type_val,  # For filtering
                    })
            
            # Clear progress bar
            progress_bar.empty()
            
            # Save updated outcomes if any changed
            if outcomes_updated:
                save_outcomes(outcomes)
            
            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                
                # Summary metrics based on 2R strategy
                # Wins = TP hit, Losses = Stopped out, Open = still in trade
                wins = len(perf_df[perf_df['Outcome'] == 'win'])
                losses = len(perf_df[perf_df['Outcome'] == 'loss'])
                open_trades = len(perf_df[perf_df['Outcome'] == 'open'])
                closed_trades = wins + losses
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    win_rate = (wins / closed_trades) * 100 if closed_trades > 0 else 0
                    trade_label = f"Win Rate ({trade_type})" if trade_type not in ["Legacy", "Both (Compare)"] else "Win Rate (2R)"
                    st.metric(trade_label, f"{win_rate:.1f}%", f"{wins}W / {losses}L")
                
                with col2:
                    # Expected value per trade in R
                    # Win gives +2R, Loss gives -1R
                    if closed_trades > 0:
                        ev = ((wins * 2) - (losses * 1)) / closed_trades
                        st.metric("Avg R per Trade", f"{ev:+.2f}R")
                    else:
                        st.metric("Avg R per Trade", "N/A")
                
                with col3:
                    st.metric("Open Trades", f"{open_trades}", f"of {len(perf_df)} total")
                
                with col4:
                    # Best R-multiple
                    valid_r = perf_df[perf_df['R-Mult'].notna()]['R-Mult']
                    if len(valid_r) > 0:
                        best_r = valid_r.max()
                        best_r_symbol = perf_df.loc[valid_r.idxmax(), 'Symbol']
                        st.metric("Best Trade", f"{best_r:+.2f}R", best_r_symbol)
                    else:
                        st.metric("Best Trade", "N/A")
                
                st.markdown("---")
                
                # Show trade type info
                st.caption(f"📊 Showing **{trade_type}** risk level")
                
                # Export button for TradingView watchlist
                col1, col2 = st.columns([3, 1])
                with col2:
                    # Get symbols in current table order
                    symbols_list = perf_df['Symbol'].tolist()
                    # Convert to MEXC format
                    tv_symbols = []
                    for sym in symbols_list:
                        base = sym.replace('/USDC', '').replace('/USDT', '')
                        tv_symbols.append(f"MEXC:{base}USDT.P")
                    
                    watchlist_content = ",".join(tv_symbols)
                    
                    st.download_button(
                        label="📺 Export to TradingView",
                        data=watchlist_content,
                        file_name="vcp_alerts_watchlist.txt",
                        mime="text/plain",
                        help="Export current alerts to TradingView watchlist (MEXC format)"
                    )
                
                # Performance table
                display_cols = ['Type', 'Symbol', 'TF', 'Alert Time', 'Entry $', 'Current $', 'Stop $', 'Stop %', 'TP (2R)', 'Change %', 'R-Mult', 'Status']
                
                st.dataframe(
                    perf_df[display_cols].style.applymap(
                        lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else ('color: red' if isinstance(x, (int, float)) and x < 0 else ''),
                        subset=['Change %', 'R-Mult']
                    ).format({'Change %': '{:.2f}%', 'R-Mult': '{:+.2f}R'}),
                    use_container_width=True,
                    height=400
                )
                
                # Show symbols with missing prices in sidebar instead of main area
                symbols_with_prices = {p['Symbol'] for p in perf_data}
                all_filtered_symbols = set(filtered_df['symbol'].dropna().unique())
                missing_symbols = all_filtered_symbols - symbols_with_prices
                
                if missing_symbols:
                    st.sidebar.markdown("---")
                    st.sidebar.warning(f"**No price data for:** {', '.join(sorted(missing_symbols)[:20])}")
                    if len(missing_symbols) > 20:
                        st.sidebar.caption(f"...and {len(missing_symbols) - 20} more")
                    st.sidebar.caption("These tokens may not be on Hyperliquid/Coinbase/CoinGecko")
                
                # Performance distribution chart
                fig = px.histogram(perf_df, x='Change %', nbins=20,
                                   title='Performance Distribution',
                                   color_discrete_sequence=['#00d4aa'])
                fig.add_vline(x=0, line_dash="dash", line_color="white")
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
                
                # ============================================================
                # EXTENDED TP ANALYSIS - Track 2R, 4R, 6R, 8R with scaled exits
                # ============================================================
                st.markdown("---")
                st.subheader("🎯 Extended TP Analysis (2R → 4R → 6R → 8R)")
                st.markdown("""
                **Scaled Exit Strategy:**
                - 50% out at 2R (TP1), move stop to breakeven
                - 20% out at 4R (TP2)
                - 15% out at 6R (TP3)
                - 15% out at 8R (TP4)
                """)
                
                # Load or calculate extended outcomes
                extended_outcomes_file = DATA_DIR / "extended_outcomes.json"
                if extended_outcomes_file.exists():
                    with open(extended_outcomes_file, 'r') as f:
                        extended_outcomes = json.load(f)
                else:
                    extended_outcomes = {}
                
                extended_updated = False
                
                # Run extended backtest for all closed trades
                extended_data = []
                progress_bar2 = st.progress(0, text="Running extended analysis...")
                
                for idx, (_, row) in enumerate(filtered_df.iterrows()):
                    progress_bar2.progress((idx + 1) / len(filtered_df), 
                                          text=f"Extended analysis {idx + 1}/{len(filtered_df)}...")
                    
                    symbol = row.get('symbol')
                    message_id = str(row.get('message_id', ''))
                    
                    # Get entry price
                    entry_price = row.get('entry_price') if pd.notna(row.get('entry_price')) else None
                    
                    # Get stop based on selected trade type
                    if trade_type == "Conservative":
                        stop_price = row.get('conservative_stop') if pd.notna(row.get('conservative_stop')) else None
                    else:  # Risky
                        stop_price = row.get('risky_stop') if pd.notna(row.get('risky_stop')) else None
                    
                    if not symbol or not entry_price or not stop_price:
                        continue
                    
                    # Get alert type for display
                    alert_type_val = row.get('alert_type', 'BO')
                    
                    # Unique key for extended outcome (includes trade type)
                    ext_key = f"{message_id}_{trade_type.lower()}_ext"
                    
                    # Check if we have cached extended outcome
                    if ext_key in extended_outcomes:
                        ext_result = extended_outcomes[ext_key]
                    else:
                        # Run extended backtest
                        entry_time = row['timestamp'].to_pydatetime()
                        if entry_time.tzinfo is None:
                            entry_time = entry_time.replace(tzinfo=ZoneInfo('UTC'))
                        
                        ext_result = backtest_trade_extended(
                            symbol, entry_price, stop_price, entry_time
                        )
                        
                        # Cache if trade is closed
                        if ext_result['outcome'] not in ['open', 'open_tp1', 'open_tp2', 'open_tp3']:
                            extended_outcomes[ext_key] = ext_result
                            if ext_result.get('exit_time'):
                                ext_result['exit_time'] = ext_result['exit_time'].isoformat() if hasattr(ext_result['exit_time'], 'isoformat') else ext_result['exit_time']
                            extended_updated = True
                    
                    risk = entry_price - stop_price
                    extended_data.append({
                        'Type': '🎯 BO' if alert_type_val == 'BO' else '⚡ CHEAT',
                        'Symbol': symbol,
                        'Entry': entry_price,
                        'TP1 (2R)': entry_price + 2*risk,
                        'TP2 (4R)': entry_price + 4*risk,
                        'TP3 (6R)': entry_price + 6*risk,
                        'TP4 (8R)': entry_price + 8*risk,
                        'Outcome': ext_result['outcome'],
                        'TP1 Hit': ext_result['tp1_hit'],
                        'TP2 Hit': ext_result['tp2_hit'],
                        'TP3 Hit': ext_result['tp3_hit'],
                        'TP4 Hit': ext_result.get('tp4_hit', False),
                        'Scaled R': ext_result['scaled_r'],
                        'Max R': ext_result['max_r_reached'],
                        'message_id': message_id,
                        'alert_type': alert_type_val
                    })
                
                progress_bar2.empty()
                
                # Save extended outcomes if updated
                if extended_updated:
                    with open(extended_outcomes_file, 'w') as f:
                        json.dump(extended_outcomes, f, indent=2, default=str)
                
                if extended_data:
                    ext_df = pd.DataFrame(extended_data)
                    
                    # Filter to closed trades only for stats
                    closed_ext = ext_df[~ext_df['Outcome'].isin(['open', 'open_tp1', 'open_tp2', 'open_tp3'])]
                    
                    if len(closed_ext) > 0:
                        # Summary metrics - 4 TP levels now
                        col1, col2, col3, col4 = st.columns(4)
                        
                        # TP hit rates
                        tp1_hits = closed_ext['TP1 Hit'].sum()
                        tp2_hits = closed_ext['TP2 Hit'].sum()
                        tp3_hits = closed_ext['TP3 Hit'].sum()
                        tp4_hits = closed_ext['TP4 Hit'].sum()
                        total_closed = len(closed_ext)
                        
                        with col1:
                            tp1_rate = (tp1_hits / total_closed) * 100 if total_closed > 0 else 0
                            st.metric("TP1 (2R) Hit Rate", f"{tp1_rate:.1f}%", f"{tp1_hits}/{total_closed} trades")
                        
                        with col2:
                            tp2_rate = (tp2_hits / total_closed) * 100 if total_closed > 0 else 0
                            st.metric("TP2 (4R) Hit Rate", f"{tp2_rate:.1f}%", f"{tp2_hits}/{total_closed} trades")
                        
                        with col3:
                            tp3_rate = (tp3_hits / total_closed) * 100 if total_closed > 0 else 0
                            st.metric("TP3 (6R) Hit Rate", f"{tp3_rate:.1f}%", f"{tp3_hits}/{total_closed} trades")
                        
                        with col4:
                            tp4_rate = (tp4_hits / total_closed) * 100 if total_closed > 0 else 0
                            st.metric("TP4 (8R) Hit Rate", f"{tp4_rate:.1f}%", f"{tp4_hits}/{total_closed} trades")
                        
                        # Second row - Avg Scaled R
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            valid_scaled = closed_ext[closed_ext['Scaled R'].notna()]['Scaled R']
                            if len(valid_scaled) > 0:
                                avg_scaled_r = valid_scaled.mean()
                                st.metric("Avg Scaled R", f"{avg_scaled_r:+.2f}R", "Your Strategy")
                            else:
                                st.metric("Avg Scaled R", "N/A")
                        
                        # Compare strategies
                        st.markdown("---")
                        st.subheader("📊 Strategy Comparison")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Simple 2R strategy totals
                            simple_wins = wins
                            simple_losses = losses
                            simple_total_r = (simple_wins * 2) - (simple_losses * 1)
                            simple_avg_r = simple_total_r / closed_trades if closed_trades > 0 else 0
                            
                            st.markdown("### Simple 2R Strategy")
                            st.markdown(f"**Total R:** {simple_total_r:+.1f}R")
                            st.markdown(f"**Avg R/Trade:** {simple_avg_r:+.2f}R")
                            st.markdown(f"**Win Rate:** {(simple_wins/closed_trades*100) if closed_trades > 0 else 0:.1f}%")
                        
                        with col2:
                            # Scaled strategy totals
                            scaled_total_r = closed_ext['Scaled R'].sum()
                            scaled_avg_r = closed_ext['Scaled R'].mean()
                            # "Win" is any trade that hit at least TP1
                            scaled_wins = closed_ext['TP1 Hit'].sum()
                            
                            st.markdown("### Scaled Exit Strategy (2R/4R/6R/8R)")
                            st.markdown(f"**Total R:** {scaled_total_r:+.1f}R")
                            st.markdown(f"**Avg R/Trade:** {scaled_avg_r:+.2f}R")
                            st.markdown(f"**TP1+ Rate:** {(scaled_wins/total_closed*100) if total_closed > 0 else 0:.1f}%")
                        
                        # Outcome distribution chart
                        st.markdown("---")
                        outcome_counts = closed_ext['Outcome'].value_counts()
                        outcome_labels = {
                            'loss': '🔴 Stopped Out (-1R)',
                            'tp1': '🟡 TP1 Only (+1.0R)',
                            'tp2': '🟢 TP1+TP2 (+1.8R)',
                            'tp3': '💪 TP1-3 (+2.7R)',
                            'tp4': '🎯 All TPs Hit (+3.9R)'
                        }
                        outcome_counts.index = outcome_counts.index.map(lambda x: outcome_labels.get(x, x))
                        
                        fig = px.pie(values=outcome_counts.values, names=outcome_counts.index,
                                    title='Trade Outcome Distribution (Scaled Strategy)',
                                    color_discrete_sequence=['#ff4444', '#ffaa00', '#44ff44', '#00aaff', '#00ffff'])
                        fig.update_layout(template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Max R distribution - shows how far trades went
                        valid_max_r = closed_ext[closed_ext['Max R'].notna()]
                        if len(valid_max_r) > 0:
                            fig2 = px.histogram(valid_max_r, x='Max R', nbins=20,
                                              title='Maximum R Reached Before Exit/Reversal',
                                              color_discrete_sequence=['#00d4ff'])
                            fig2.add_vline(x=2, line_dash="dash", line_color="green", annotation_text="2R")
                            fig2.add_vline(x=4, line_dash="dash", line_color="yellow", annotation_text="4R")
                            fig2.add_vline(x=6, line_dash="dash", line_color="orange", annotation_text="6R")
                            fig2.add_vline(x=8, line_dash="dash", line_color="cyan", annotation_text="8R")
                            fig2.update_layout(template='plotly_dark')
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Detailed extended table
                        with st.expander("📋 Extended Analysis Details"):
                            display_ext = closed_ext[['Type', 'Symbol', 'Outcome', 'TP1 Hit', 'TP2 Hit', 'TP3 Hit', 'TP4 Hit', 'Scaled R', 'Max R']].copy()
                            display_ext['Scaled R'] = display_ext['Scaled R'].apply(lambda x: f"{x:+.2f}R" if pd.notna(x) else "N/A")
                            display_ext['Max R'] = display_ext['Max R'].apply(lambda x: f"{x:.2f}R" if pd.notna(x) else "N/A")
                            st.dataframe(display_ext, use_container_width=True)
                    
                    else:
                        st.info("No closed trades yet for extended analysis. Wait for trades to hit TP or Stop.")
                
            else:
                st.warning("Could not fetch prices for the filtered alerts")
        else:
            st.info("No alerts match the current filters")
    
    with tab2:
        st.subheader("Alert Frequency Over Time")
        
        # Daily frequency
        daily_counts = df.groupby('date').size().reset_index(name='count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        
        fig = px.bar(daily_counts, x='date', y='count', 
                     title='Daily VCP Alerts',
                     labels={'date': 'Date', 'count': 'Number of Alerts'})
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
        # Hourly timeline - ALL history (shows when market pops)
        st.subheader("📈 Hourly Alert Timeline (All History)")
        st.caption("Track VCP alert spikes - more alerts often signals increased market activity")
        
        # Create hourly bins for entire history
        df_sorted = df.sort_values('timestamp_denver').copy()
        df_sorted['hour_bin'] = df_sorted['timestamp_denver'].dt.floor('H')
        hourly_timeline = df_sorted.groupby('hour_bin').size().reset_index(name='count')
        hourly_timeline = hourly_timeline.sort_values('hour_bin').reset_index(drop=True)
        
        fig2 = px.bar(hourly_timeline, x='hour_bin', y='count',
                      title='VCP Alerts by Hour - Full History',
                      labels={'hour_bin': 'Date/Time (Denver)', 'count': 'Alerts'})
        fig2.update_traces(marker_color='#00d4ff')
        fig2.update_layout(
            template='plotly_dark',
            hovermode='x unified',
            bargap=0.1
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Also show a heatmap of hour vs day
        st.subheader("🔥 Alert Heatmap - Hour vs Day")
        st.caption("See which hours on which days have the most breakouts")
        
        # Create pivot for heatmap
        df['day_name'] = df['timestamp_denver'].dt.day_name()
        heatmap_data = df.groupby(['day_name', 'hour']).size().reset_index(name='count')
        
        # Pivot to create matrix
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_data.pivot(index='day_name', columns='hour', values='count').fillna(0)
        heatmap_pivot = heatmap_pivot.reindex(day_order)
        
        fig3 = px.imshow(heatmap_pivot,
                        labels=dict(x="Hour (Denver)", y="Day", color="Alerts"),
                        title="When Do VCP Breakouts Happen?",
                        color_continuous_scale='YlOrRd')
        fig3.update_layout(template='plotly_dark')
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.subheader("Symbol Analysis")
        
        if 'symbol' in df.columns and df['symbol'].notna().any():
            col1, col2 = st.columns(2)
            
            with col1:
                # Top symbols
                symbol_counts = df['symbol'].value_counts().head(15)
                fig = px.bar(x=symbol_counts.index, y=symbol_counts.values,
                             title='Most Frequent Symbols',
                             labels={'x': 'Symbol', 'y': 'Alert Count'})
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Volume by symbol
                vol_by_symbol = df.groupby('symbol')['volume_24h'].mean().sort_values(ascending=False).head(15)
                fig = px.bar(x=vol_by_symbol.index, y=vol_by_symbol.values,
                             title='Avg 24h Volume by Symbol',
                             labels={'x': 'Symbol', 'y': 'Avg Volume ($)'})
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
            
            # Load outcomes for win/loss analysis
            outcomes = load_outcomes()
            
            if outcomes:
                st.markdown("---")
                st.subheader(f"🏆 Symbol Performance - {trade_type} (Win/Loss)")
                
                # Build symbol performance data - only for the selected trade type
                symbol_performance = {}
                processed_msg_ids = set()  # Track to avoid double counting
                
                for outcome_key, outcome_data in outcomes.items():
                    # Extract message_id from key (handles both old "123" and new "123_conservative" formats)
                    msg_id = outcome_key.split('_')[0] if '_' in outcome_key else outcome_key
                    
                    # Get the trade type from the outcome
                    outcome_trade_type = outcome_data.get('trade_type', 'conservative')
                    
                    # Only process outcomes matching the currently selected trade type in the dashboard
                    # (trade_type variable is set from the Risk Level filter above)
                    if outcome_trade_type != trade_type.lower():
                        continue
                    
                    # Skip if we've already processed this message_id for this trade type
                    unique_key = f"{msg_id}_{outcome_trade_type}"
                    if unique_key in processed_msg_ids:
                        continue
                    processed_msg_ids.add(unique_key)
                    
                    # Find the symbol for this message_id
                    matching = df[df['message_id'].astype(str) == str(msg_id)]
                    if len(matching) > 0:
                        symbol = matching.iloc[0]['symbol']
                        
                        if symbol not in symbol_performance:
                            symbol_performance[symbol] = {
                                'wins': 0, 
                                'losses': 0, 
                                'total_r': 0,
                                'total_pct': 0,
                                'pct_gains': [],
                            }
                        
                        # Get the stop % for this trade to calculate actual % gain/loss
                        stop_pct = None
                        if outcome_trade_type == 'conservative':
                            stop_pct = matching.iloc[0].get('conservative_stop_pct')
                        elif outcome_trade_type == 'risky':
                            stop_pct = matching.iloc[0].get('risky_stop_pct')
                        
                        if outcome_data.get('outcome') == 'win':
                            symbol_performance[symbol]['wins'] += 1
                            symbol_performance[symbol]['total_r'] += 2.0
                            # Win at 2R = 2x the stop %
                            if stop_pct and pd.notna(stop_pct):
                                pct_gain = abs(stop_pct) * 2
                                symbol_performance[symbol]['total_pct'] += pct_gain
                                symbol_performance[symbol]['pct_gains'].append(pct_gain)
                        elif outcome_data.get('outcome') == 'loss':
                            symbol_performance[symbol]['losses'] += 1
                            symbol_performance[symbol]['total_r'] -= 1.0
                            # Loss = stop %
                            if stop_pct and pd.notna(stop_pct):
                                pct_loss = -abs(stop_pct)
                                symbol_performance[symbol]['total_pct'] += pct_loss
                                symbol_performance[symbol]['pct_gains'].append(pct_loss)
                
                if symbol_performance:
                    # Convert to dataframe (no more aggregation needed - already by symbol)
                    perf_df = pd.DataFrame([
                        {
                            'Symbol': sym,
                            'Wins': data['wins'],
                            'Losses': data['losses'],
                            'Total Trades': data['wins'] + data['losses'],
                            'Win Rate': (data['wins'] / (data['wins'] + data['losses']) * 100) if (data['wins'] + data['losses']) > 0 else 0,
                            'Total R': data['total_r'],
                            'Total %': data['total_pct'],
                            'Avg % per Trade': data['total_pct'] / len(data['pct_gains']) if data['pct_gains'] else 0,
                            'Best Trade %': max(data['pct_gains']) if data['pct_gains'] else 0,
                            'Worst Trade %': min(data['pct_gains']) if data['pct_gains'] else 0,
                        }
                        for sym, data in symbol_performance.items()
                    ])
                    
                    # Filter to symbols with at least 2 trades for meaningful stats
                    perf_df_filtered = perf_df[perf_df['Total Trades'] >= 2].copy()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Most wins
                        top_winners = perf_df.nlargest(10, 'Wins')
                        if len(top_winners) > 0:
                            fig = px.bar(top_winners, x='Symbol', y='Wins',
                                        title='🥇 Symbols with Most Wins',
                                        color='Wins',
                                        color_continuous_scale='Greens')
                            fig.update_layout(template='plotly_dark', showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Most losses
                        top_losers = perf_df.nlargest(10, 'Losses')
                        if len(top_losers) > 0:
                            fig = px.bar(top_losers, x='Symbol', y='Losses',
                                        title='💀 Symbols with Most Losses',
                                        color='Losses',
                                        color_continuous_scale='Reds')
                            fig.update_layout(template='plotly_dark', showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Best win rate (min 2 trades)
                        if len(perf_df_filtered) > 0:
                            best_wr = perf_df_filtered.nlargest(10, 'Win Rate')
                            fig = px.bar(best_wr, x='Symbol', y='Win Rate',
                                        title='📈 Best Win Rate (min 2 trades)',
                                        color='Win Rate',
                                        color_continuous_scale='Greens',
                                        text=best_wr['Total Trades'].apply(lambda x: f'{x} trades'))
                            fig.update_traces(textposition='outside')
                            fig.update_layout(template='plotly_dark', showlegend=False, yaxis_title='Win Rate %')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Best total R
                        best_r = perf_df.nlargest(10, 'Total R')
                        if len(best_r) > 0:
                            fig = px.bar(best_r, x='Symbol', y='Total R',
                                        title='💰 Most Profitable Symbols (Total R)',
                                        color='Total R',
                                        color_continuous_scale='Greens')
                            fig.update_layout(template='plotly_dark', showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Worst performers
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Worst win rate (min 2 trades)
                        if len(perf_df_filtered) > 0:
                            worst_wr = perf_df_filtered.nsmallest(10, 'Win Rate')
                            fig = px.bar(worst_wr, x='Symbol', y='Win Rate',
                                        title='📉 Worst Win Rate (min 2 trades)',
                                        color='Win Rate',
                                        color_continuous_scale='Reds_r',
                                        text=worst_wr['Total Trades'].apply(lambda x: f'{x} trades'))
                            fig.update_traces(textposition='outside')
                            fig.update_layout(template='plotly_dark', showlegend=False, yaxis_title='Win Rate %')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Worst total R
                        worst_r = perf_df.nsmallest(10, 'Total R')
                        if len(worst_r) > 0:
                            fig = px.bar(worst_r, x='Symbol', y='Total R',
                                        title='🔻 Least Profitable Symbols (Total R)',
                                        color='Total R',
                                        color_continuous_scale='Reds_r')
                            fig.update_layout(template='plotly_dark', showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # NEW: Percentage gain/loss charts
                    st.markdown("---")
                    st.subheader("📈 Percentage Gain/Loss Analysis")
                    
                    # Filter to symbols with % data
                    perf_with_pct = perf_df[perf_df['Total %'] != 0].copy()
                    
                    if len(perf_with_pct) > 0:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Highest % gainers
                            top_pct = perf_with_pct.nlargest(10, 'Total %')
                            if len(top_pct) > 0:
                                fig = px.bar(top_pct, x='Symbol', y='Total %',
                                            title='🚀 Highest Total % Gain',
                                            color='Total %',
                                            color_continuous_scale='Greens',
                                            text=top_pct['Total %'].apply(lambda x: f'{x:+.2f}%'))
                                fig.update_traces(textposition='outside')
                                fig.update_layout(template='plotly_dark', showlegend=False, yaxis_title='Total %')
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Biggest % losers
                            worst_pct = perf_with_pct.nsmallest(10, 'Total %')
                            if len(worst_pct) > 0:
                                fig = px.bar(worst_pct, x='Symbol', y='Total %',
                                            title='💸 Biggest Total % Loss',
                                            color='Total %',
                                            color_continuous_scale='Reds_r',
                                            text=worst_pct['Total %'].apply(lambda x: f'{x:+.2f}%'))
                                fig.update_traces(textposition='outside')
                                fig.update_layout(template='plotly_dark', showlegend=False, yaxis_title='Total %')
                                st.plotly_chart(fig, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Best single trade %
                            best_single = perf_with_pct[perf_with_pct['Best Trade %'] > 0].nlargest(10, 'Best Trade %')
                            if len(best_single) > 0:
                                fig = px.bar(best_single, x='Symbol', y='Best Trade %',
                                            title='🎯 Best Single Trade %',
                                            color='Best Trade %',
                                            color_continuous_scale='Greens',
                                            text=best_single['Best Trade %'].apply(lambda x: f'+{x:.2f}%'))
                                fig.update_traces(textposition='outside')
                                fig.update_layout(template='plotly_dark', showlegend=False, yaxis_title='%')
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Worst single trade %
                            worst_single = perf_with_pct[perf_with_pct['Worst Trade %'] < 0].nsmallest(10, 'Worst Trade %')
                            if len(worst_single) > 0:
                                fig = px.bar(worst_single, x='Symbol', y='Worst Trade %',
                                            title='😵 Worst Single Trade %',
                                            color='Worst Trade %',
                                            color_continuous_scale='Reds_r',
                                            text=worst_single['Worst Trade %'].apply(lambda x: f'{x:.2f}%'))
                                fig.update_traces(textposition='outside')
                                fig.update_layout(template='plotly_dark', showlegend=False, yaxis_title='%')
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Full performance table
                    st.markdown("---")
                    st.subheader("📊 Full Symbol Performance Table")
                    
                    # Sort options and export
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        sort_by = st.selectbox("Sort by:", 
                                              ["Total R", "Total %", "Win Rate", "Total Trades", "Avg % per Trade"],
                                              index=0)
                    with col2:
                        sort_order = st.radio("Order:", ["Best First", "Worst First"], horizontal=True)
                    
                    ascending = sort_order == "Worst First"
                    sorted_perf = perf_df.sort_values(sort_by, ascending=ascending).copy()
                    
                    # TradingView Watchlist Export
                    with col3:
                        # Create watchlist content for MEXC
                        symbols_for_watchlist = sorted_perf['Symbol'].tolist()
                        # Convert from "TOKEN/USDC" to "MEXC:TOKENUSDT.P" format
                        tv_symbols = []
                        for sym in symbols_for_watchlist:
                            base = sym.replace('/USDC', '').replace('/USDT', '')
                            tv_symbols.append(f"MEXC:{base}USDT.P")
                        
                        watchlist_content = ",".join(tv_symbols)
                        
                        st.download_button(
                            label="📺 Export to TradingView",
                            data=watchlist_content,
                            file_name="vcp_performance_watchlist.txt",
                            mime="text/plain",
                            help="Import this file into TradingView as a watchlist"
                        )
                    
                    # Format display columns
                    display_perf = sorted_perf.copy()
                    display_perf['Win Rate'] = display_perf['Win Rate'].apply(lambda x: f"{x:.1f}%")
                    display_perf['Total R'] = display_perf['Total R'].apply(lambda x: f"{x:+.1f}R")
                    display_perf['Total %'] = display_perf['Total %'].apply(lambda x: f"{x:+.2f}%")
                    display_perf['Avg % per Trade'] = display_perf['Avg % per Trade'].apply(lambda x: f"{x:+.2f}%")
                    display_perf['Best Trade %'] = display_perf['Best Trade %'].apply(lambda x: f"{x:+.2f}%")
                    display_perf['Worst Trade %'] = display_perf['Worst Trade %'].apply(lambda x: f"{x:+.2f}%")
                    
                    st.dataframe(display_perf[['Symbol', 'Wins', 'Losses', 'Total Trades', 'Win Rate', 'Total R', 'Total %', 'Avg % per Trade']], 
                                use_container_width=True, height=400)
                    
                    # Instructions for TradingView import
                    with st.expander("📖 How to import watchlist into TradingView"):
                        st.markdown("""
                        1. Click the **Export to TradingView** button above
                        2. Open TradingView and go to your **Watchlist** panel
                        3. Click the **⋮** menu (three dots) at the top of the watchlist
                        4. Select **Import list...**
                        5. Choose the downloaded `vcp_performance_watchlist.txt` file
                        6. Your symbols will be added sorted by performance!
                        
                        **Note:** Symbols are formatted for MEXC exchange (MEXC:TOKENUSDT.P)
                        """)
                else:
                    st.info("No completed trades yet - check back after some trades hit TP or Stop!")
            else:
                st.info("No trade outcomes recorded yet. Run the tracker to collect win/loss data!")
    
    with tab4:
        st.subheader("Timing Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hourly = df.groupby('hour').size()
            fig = px.bar(x=hourly.index, y=hourly.values,
                         title='Alerts by Hour (Denver Time)',
                         labels={'x': 'Hour', 'y': 'Count'})
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_counts = df['day_of_week'].value_counts().reindex(dow_order).dropna()
            
            fig = px.bar(x=dow_counts.index, y=dow_counts.values,
                         title='Alerts by Day of Week',
                         labels={'x': 'Day', 'y': 'Count'})
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        
        # Timeframe distribution
        if 'timeframe' in df.columns:
            tf_counts = df['timeframe'].value_counts()
            fig = px.pie(values=tf_counts.values, names=tf_counts.index,
                        title='Alerts by Timeframe')
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("Raw Alert Data")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            if 'symbol' in df.columns:
                symbols = ['All'] + sorted(df['symbol'].dropna().unique().tolist())
                selected_symbol = st.selectbox("Filter by Symbol", symbols)
        with col2:
            if 'timeframe' in df.columns:
                timeframes = ['All'] + sorted(df['timeframe'].dropna().unique().tolist())
                selected_tf = st.selectbox("Filter by Timeframe", timeframes)
        
        filtered_df = df.copy()
        if 'symbol' in df.columns and selected_symbol != 'All':
            filtered_df = filtered_df[filtered_df['symbol'] == selected_symbol]
        if 'timeframe' in df.columns and selected_tf != 'All':
            filtered_df = filtered_df[filtered_df['timeframe'] == selected_tf]
        
        # Display
        display_cols = ['timestamp_denver', 'symbol', 'timeframe', 'volume_24h', 'breakout_price', 'stop_price', 'stop_pct', 'liquidity']
        display_cols = [c for c in display_cols if c in filtered_df.columns]
        
        st.dataframe(
            filtered_df[display_cols].sort_values('timestamp_denver', ascending=False),
            use_container_width=True,
            height=400
        )
        
        # Export
        if st.button("Export to CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"vcp_alerts_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# Sidebar stats
if not df.empty:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Quick Stats")
    st.sidebar.write(f"**Total alerts:** {len(df)}")
    st.sidebar.write(f"**Latest:** {df['timestamp_denver'].max().strftime('%m/%d %H:%M')} MT")
    st.sidebar.write(f"**Unique symbols:** {df['symbol'].nunique()}")
    
    # Recent alerts
    st.sidebar.markdown("---")
    st.sidebar.subheader("🆕 Recent Alerts")
    recent = df.nlargest(5, 'timestamp_denver')[['symbol', 'timeframe', 'timestamp_denver']]
    for _, row in recent.iterrows():
        st.sidebar.write(f"**{row['symbol']}** ({row['timeframe']}) - {row['timestamp_denver'].strftime('%H:%M')} MT")

# Clear data option
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear All Data", type="secondary"):
    if ALERTS_FILE.exists():
        os.remove(ALERTS_FILE)
    if PRICE_CACHE_FILE.exists():
        os.remove(PRICE_CACHE_FILE)
    st.rerun()