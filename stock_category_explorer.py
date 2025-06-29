import streamlit as st
import pandas as pd
import json
import os
import logging
from io import BytesIO
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load and parse the JSON data file."""
    base_path = "Data/stock_scores"
    latest_file = os.path.join(base_path, "market_analysis_latest.json")
    
    try:
        with open(latest_file, 'r') as file:
            data = json.load(file)
        return pd.DataFrame(data['stocks'])
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def load_sectors_data():
    """Load the sectors data to get exchange information."""
    sectors_file = "Data/industry_classification/stock_sectors.csv"
    try:
        return pd.read_csv(sectors_file)
    except Exception as e:
        logger.warning(f"Could not load sectors file: {str(e)}")
        return None

def categorize_market_cap(cap):
    """Categorize market cap value into size category."""
    if cap >= 200:
        return "Mega Cap"
    elif cap >= 10:
        return "Large Cap"
    elif cap >= 2:
        return "Mid Cap"
    elif cap >= 0.3:
        return "Small Cap"
    elif cap >= 0.05:
        return "Micro Cap"
    else:
        return "Nano Cap"

def create_tradingview_watchlist(df, df_sectors=None):
    """
    Create TradingView-compatible watchlist from filtered stocks.
    
    Args:
        df: DataFrame with filtered stocks
        df_sectors: DataFrame with sector/exchange data (optional)
    
    Returns:
        String containing TradingView formatted symbols
    """
    if df_sectors is not None:
        # Merge to get exchange information
        df_merged = pd.merge(df, df_sectors[['symbol', 'exchange']], on='symbol', how='left')
    else:
        # If no sectors data, assume all are NASDAQ (fallback)
        df_merged = df.copy()
        df_merged['exchange'] = 'NMS'
    
    # Define exchange mappings for TradingView
    nasdaq_exchanges = ['NMS', 'NGM', 'NCM']
    nyse_exchanges = ['NYQ', 'ASE']
    
    tv_symbols = []
    
    for _, row in df_merged.iterrows():
        exchange = row.get('exchange', 'NMS')  # Default to NASDAQ if no exchange info
        symbol = row['symbol']
        
        if exchange in nasdaq_exchanges:
            tv_symbols.append(f"NASDAQ:{symbol}")
        elif exchange in nyse_exchanges:
            tv_symbols.append(f"NYSE:{symbol}")
        else:
            # For unknown exchanges, just use the symbol
            tv_symbols.append(symbol)
    
    return '\n'.join(tv_symbols)

def create_filename_safe_string(text):
    """Convert text to filename-safe string."""
    import re
    # Replace spaces and special characters with underscores
    safe_text = re.sub(r'[^\w\-_\.]', '_', text)
    # Remove multiple consecutive underscores
    safe_text = re.sub(r'_+', '_', safe_text)
    # Remove leading/trailing underscores
    safe_text = safe_text.strip('_')
    return safe_text

def main():
    st.title("Stock Category Explorer")
    
    try:
        # Load data
        df = load_data()
        df_sectors = load_sectors_data()
        
        # Add market cap categories
        df['market_cap_category'] = df['market_cap_B'].apply(categorize_market_cap)
        
        # Sidebar controls
        st.sidebar.header("Filters")
        
        # Market cap filter
        cap_categories = [
            "Mega Cap", "Large Cap", "Mid Cap",
            "Small Cap", "Micro Cap", "Nano Cap"
        ]
        selected_caps = st.sidebar.multiselect(
            "Select Market Cap Categories",
            options=cap_categories
        )
        
        # Category type selection (Sector or Industry)
        category_type = st.sidebar.radio(
            "Select Category Type",
            ["Sector", "Industry"]
        )
        
        # Get unique categories based on selection
        categories = sorted(df[category_type.lower()].unique())
        
        # Category selection - single select for focused lists
        selected_category = st.sidebar.selectbox(
            f"Select {category_type}",
            options=[""] + categories,
            help=f"Choose one {category_type.lower()} to create a focused TradingView watchlist"
        )
        
        # Score filtering options
        st.sidebar.markdown("### Score Filters (Optional)")
        
        # Bullish score filter
        use_bullish_filter = st.sidebar.checkbox("Filter by Bullish Score")
        if use_bullish_filter:
            bullish_range = st.sidebar.slider(
                "Bullish Score Range",
                min_value=0,
                max_value=100,
                value=(0, 100),
                step=1
            )
        
        # Bearish score filter
        use_bearish_filter = st.sidebar.checkbox("Filter by Bearish Score")
        if use_bearish_filter:
            bearish_range = st.sidebar.slider(
                "Bearish Score Range",
                min_value=0,
                max_value=100,
                value=(0, 100),
                step=1
            )
        
        # Net score filter
        use_net_filter = st.sidebar.checkbox("Filter by Net Score")
        if use_net_filter:
            net_range = st.sidebar.slider(
                "Net Score Range",
                min_value=-100,
                max_value=100,
                value=(-100, 100),
                step=1
            )
        
        # Filter data based on selections
        filtered_df = df.copy()
        
        if selected_caps:
            filtered_df = filtered_df[filtered_df['market_cap_category'].isin(selected_caps)]
            
        if selected_category:
            filtered_df = filtered_df[filtered_df[category_type.lower()] == selected_category]
        
        # Apply score filters
        if use_bullish_filter:
            filtered_df = filtered_df[
                (filtered_df['bullish_score'] >= bullish_range[0]) & 
                (filtered_df['bullish_score'] <= bullish_range[1])
            ]
        
        if use_bearish_filter:
            filtered_df = filtered_df[
                (filtered_df['bearish_score'] >= bearish_range[0]) & 
                (filtered_df['bearish_score'] <= bearish_range[1])
            ]
        
        if use_net_filter:
            filtered_df = filtered_df[
                (filtered_df['net_score'] >= net_range[0]) & 
                (filtered_df['net_score'] <= net_range[1])
            ]
        
        # Display summary statistics
        st.sidebar.markdown("### Summary Statistics")
        st.sidebar.markdown(f"Total Stocks: {len(filtered_df)}")
        if selected_category:
            st.sidebar.markdown(f"Selected {category_type}: {selected_category}")
        
        if not (selected_caps or selected_category):
            st.warning("👈 Use the filters in the sidebar to select stocks by market cap and sector/industry!")
            return
            
        # Prepare display columns
        display_columns = [
            'symbol', 
            category_type.lower(), 
            'market_cap_category',
            'market_cap_B',
            'bullish_score',
            'bearish_score',
            'net_score'
        ]
        
        # Display results
        st.markdown(f"### Filtered Stock List")
        st.markdown(f"Showing {len(filtered_df)} stocks matching your criteria")
        
        if filtered_df.empty:
            st.info("No stocks match your current filter criteria. Try adjusting the filters.")
            return
        
        # Format market cap for display
        display_df = filtered_df.copy()
        display_df['market_cap_B'] = display_df['market_cap_B'].round(2)
        
        # Display the dataframe
        st.dataframe(
            display_df[display_columns].sort_values('market_cap_B', ascending=False),
            use_container_width=True
        )
        
        # Download buttons section
        st.markdown("### Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download with industry name
            if selected_category:
                safe_category = create_filename_safe_string(selected_category)
                csv_filename = f"{safe_category}_{category_type.lower()}_stocks.csv"
            else:
                csv_filename = "filtered_stocks.csv"
            
            csv = filtered_df[display_columns].to_csv(index=False)
            st.download_button(
                label="📊 Download CSV",
                data=csv,
                file_name=csv_filename,
                mime="text/csv",
                help="Download the filtered stocks as a CSV file"
            )
        
        with col2:
            # TradingView watchlist download with industry name
            if df_sectors is not None:
                tv_watchlist = create_tradingview_watchlist(filtered_df, df_sectors)
                
                if selected_category:
                    safe_category = create_filename_safe_string(selected_category)
                    tv_filename = f"{safe_category}_{category_type.lower()}_watchlist.txt"
                    button_label = f"📈 Download {selected_category} Watchlist"
                else:
                    tv_filename = "tradingview_watchlist.txt"
                    button_label = "📈 Download TradingView Watchlist"
                
                st.download_button(
                    label=button_label,
                    data=tv_watchlist,
                    file_name=tv_filename,
                    mime="text/plain",
                    help=f"Download {selected_category if selected_category else 'filtered stocks'} as TradingView-compatible watchlist"
                )
            else:
                st.info("💡 TradingView watchlist requires exchange data. Please ensure stock_sectors.csv is available in Data/industry_classification/")
        
        # Show preview of TradingView format
        if df_sectors is not None and len(filtered_df) > 0:
            with st.expander("🔍 Preview TradingView Watchlist Format"):
                tv_preview = create_tradingview_watchlist(filtered_df.head(10), df_sectors)
                if selected_category:
                    st.markdown(f"**{selected_category} {category_type} Watchlist Preview:**")
                st.text(f"First 10 symbols:\n{tv_preview}")
                if len(filtered_df) > 10:
                    st.caption(f"... and {len(filtered_df) - 10} more symbols")
                
                st.caption("💡 Copy and paste this into TradingView's watchlist or use the download button above")
        
        # Additional info about TradingView format
        with st.expander("ℹ️ About TradingView Watchlist Format"):
            st.markdown("""
            **TradingView Watchlist Format:**
            - Each symbol is prefixed with its exchange (e.g., `NASDAQ:AAPL`, `NYSE:JPM`)
            - One symbol per line
            - Compatible with TradingView's import watchlist feature
            
            **How to use:**
            1. Download the TradingView watchlist file
            2. In TradingView, go to your watchlist
            3. Click the settings gear icon
            4. Select "Import list"
            5. Upload the downloaded file or copy/paste the contents
            
            **Exchange Mappings:**
            - NASDAQ exchanges: NMS, NGM, NCM → `NASDAQ:`
            - NYSE exchanges: NYQ, ASE → `NYSE:`
            - Other exchanges: Symbol only (no prefix)
            """)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"Application error: {str(e)}")
        
        # Show some debugging info
        with st.expander("🔧 Debug Information"):
            st.code(f"Error details: {str(e)}")
            st.markdown("**Expected file structure:**")
            st.code("""
Data/
├── stock_scores/
│   └── market_analysis_latest.json
└── industry_classification/
    └── stock_sectors.csv
            """)

if __name__ == "__main__":
    main()