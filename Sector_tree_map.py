import streamlit as st
import pandas as pd
import plotly.express as px
import os

def load_data():
    file_path = r"data\industry_classification\stock_sectors.csv"
    df = pd.read_csv(file_path)
    return df

def generate_statistics(df):
    # Sector Statistics
    sector_stats = df.groupby('sector').agg({
        'symbol': 'count',
        'market_cap_B': ['sum', 'mean', 'median', 'min', 'max']
    }).round(2)
    sector_stats.columns = ['Number of Companies', 'Total Market Cap ($B)', 
                          'Average Market Cap ($B)', 'Median Market Cap ($B)',
                          'Smallest Company ($B)', 'Largest Company ($B)']
    
    # Industry Statistics
    industry_stats = df.groupby(['sector', 'industry']).agg({
        'symbol': 'count',
        'market_cap_B': ['sum', 'mean', 'median', 'min', 'max']
    }).round(2)
    industry_stats.columns = ['Number of Companies', 'Total Market Cap ($B)', 
                            'Average Market Cap ($B)', 'Median Market Cap ($B)',
                            'Smallest Company ($B)', 'Largest Company ($B)']
    
    return sector_stats, industry_stats

def export_statistics(df, sector_stats, industry_stats):
    # Create output directory if it doesn't exist
    output_dir = "market_analysis_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Export sector statistics
    sector_stats.to_csv(f"{output_dir}/sector_statistics.csv")
    
    # Export industry statistics
    industry_stats.to_csv(f"{output_dir}/industry_statistics.csv")
    
    # Create a detailed report with sector percentages
    total_market_cap = df['market_cap_B'].sum()
    sector_percentages = (sector_stats['Total Market Cap ($B)'] / total_market_cap * 100).round(2)
    sector_stats['Market Share (%)'] = sector_percentages
    
    # Sort everything by market cap for better readability
    sector_stats = sector_stats.sort_values('Total Market Cap ($B)', ascending=False)
    industry_stats = industry_stats.sort_values('Total Market Cap ($B)', ascending=False)
    
    # Export the full report
    with pd.ExcelWriter(f"{output_dir}/market_structure_report.xlsx") as writer:
        sector_stats.to_excel(writer, sheet_name='Sector Statistics')
        industry_stats.to_excel(writer, sheet_name='Industry Statistics')
        
        # Add top companies sheet
        top_companies = df.sort_values('market_cap_B', ascending=False)[
            ['symbol', 'sector', 'industry', 'market_cap_B']
        ].head(100)
        top_companies.to_excel(writer, sheet_name='Top 100 Companies')
    
    return output_dir

def create_treemap(df):
    # Add company counts to sector and industry names
    sector_counts = df.groupby('sector').size()
    industry_counts = df.groupby(['sector', 'industry']).size().reset_index(name='count')
    
    df_viz = df.copy()
    df_viz['sector'] = df_viz['sector'].apply(lambda x: f"{x} ({sector_counts[x]} companies)")
    industry_lookup = industry_counts.set_index('industry')['count']
    df_viz['industry'] = df_viz['industry'].apply(lambda x: f"{x} ({industry_lookup[x]} companies)")
    
    fig = px.treemap(
        df_viz,
        path=[px.Constant("Market"), 'sector', 'industry'],
        values='market_cap_B',
        title='Market Structure by Sector and Industry',
        color='sector',
        color_discrete_sequence=px.colors.qualitative.Set3,
        hover_data=['market_cap_B']
    )
    
    fig.update_traces(
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>Market Cap: $%{value:.2f}B<extra></extra>"
    )
    
    return fig

def main():
    st.set_page_config(layout="wide")
    
    st.title("Market Structure Analysis")
    st.write("Analyze market sectors and industries by market capitalization")
    
    # Load data
    df = load_data()
    
    # Generate statistics
    sector_stats, industry_stats = generate_statistics(df)
    
    # Add export button at the top
    if st.button('📊 Export Full Market Analysis Report'):
        output_dir = export_statistics(df, sector_stats, industry_stats)
        st.success(f'''Reports exported to "{output_dir}" folder:
        - sector_statistics.csv
        - industry_statistics.csv
        - market_structure_report.xlsx (includes all statistics and top 100 companies)''')
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Treemap Visualization", "Sector Statistics", "Industry Analysis"])
    
    with tab1:
        fig = create_treemap(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Sector Statistics")
        st.dataframe(sector_stats)
        
        # Add market share pie chart
        market_share = (sector_stats['Total Market Cap ($B)'] / sector_stats['Total Market Cap ($B)'].sum() * 100).round(2)
        fig_pie = px.pie(values=market_share, names=market_share.index, 
                        title="Market Share by Sector (%)")
        st.plotly_chart(fig_pie)
            
    with tab3:
        st.subheader("Industry Analysis")
        
        selected_sector = st.selectbox(
            "Select a sector to analyze",
            options=sorted(df['sector'].unique())
        )
        
        sector_data = df[df['sector'] == selected_sector]
        
        # Industry statistics for selected sector
        industry_data = industry_stats.loc[selected_sector]
        st.write(f"Industry Statistics for {selected_sector}")
        st.dataframe(industry_data)
        
        # Distribution plot
        st.write(f"Market Cap Distribution in {selected_sector}")
        fig_bar = px.bar(industry_data, 
                        y='Total Market Cap ($B)',
                        title=f"Industry Size in {selected_sector}")
        st.plotly_chart(fig_bar)
        
        if st.checkbox("Show companies in this sector"):
            st.dataframe(sector_data[['symbol', 'industry', 'market_cap_B']]
                        .sort_values('market_cap_B', ascending=False))

if __name__ == "__main__":
    main()