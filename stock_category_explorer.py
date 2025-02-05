import streamlit as st
import pandas as pd
import json
import os
import logging
from io import BytesIO

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

def to_excel_download_link(df, filename="filtered_stocks.csv", text="Download CSV"):
    """Convert dataframe to downloadable CSV link."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return f'<a href="{href}" download="{filename}">{text}</a>'

def main():
    st.title("Stock Category Explorer")
    
    try:
        # Load data
        df = load_data()
        
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
        
        # Category selection
        selected_categories = st.sidebar.multiselect(
            f"Select {category_type}s",
            options=categories
        )
        
        # Filter data based on selections
        filtered_df = df.copy()
        
        if selected_caps:
            filtered_df = filtered_df[filtered_df['market_cap_category'].isin(selected_caps)]
            
        if selected_categories:
            filtered_df = filtered_df[filtered_df[category_type.lower()].isin(selected_categories)]
        
        # Display summary statistics
        st.sidebar.markdown("### Summary Statistics")
        st.sidebar.markdown(f"Total Stocks: {len(filtered_df)}")
        st.sidebar.markdown(f"Total {category_type}s: {len(filtered_df[category_type.lower()].unique())}")
        
        if not (selected_caps or selected_categories):
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
        
        # Format market cap for display
        filtered_df['market_cap_B'] = filtered_df['market_cap_B'].round(2)
        
        # Display the dataframe
        st.dataframe(
            filtered_df[display_columns].sort_values('market_cap_B', ascending=False),
            use_container_width=True
        )
        
        # Add download button
        if not filtered_df.empty:
            csv = filtered_df[display_columns].to_csv(index=False)
            st.download_button(
                label="Download Filtered Stocks CSV",
                data=csv,
                file_name="filtered_stocks.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()