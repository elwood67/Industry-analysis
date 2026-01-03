"""
2025 Industry & Sector Performance Report
==========================================
Comprehensive analysis of market performance at both sector AND industry level.

Metrics:
- YTD Performance ranking
- Breadth analysis (% of stocks bullish)
- Consistency/Stability scores
- Rotation timeline (when each led/lagged)
- Monthly breakdown
- Top/Bottom performers within each sector
- Current momentum heading into 2026

Author: Dave's Trading Lab
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(r'C:\Users\davet\Documents\GitHub\Industry-analysis\Data\stock_scores')
OUTPUT_DIR = DATA_DIR  # Save reports here

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load the historical stock scoring data."""
    parquet_path = DATA_DIR / 'historical_data.parquet.gzip'
    
    if not parquet_path.exists():
        print(f"ERROR: Data file not found: {parquet_path}")
        return None
    
    print("Loading data...")
    df = pd.read_parquet(parquet_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df['net_score'] = df['bullish_score'] - df['bearish_score']
    df['is_bullish'] = df['bullish_score'] > df['bearish_score']
    
    print(f"Loaded {len(df):,} records")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Unique stocks: {df['symbol'].nunique():,}")
    print(f"Sectors: {df['sector'].nunique()}")
    print(f"Industries: {df['industry'].nunique()}")
    
    return df


def filter_2025_data(df):
    """Filter data to 2025 only."""
    df_2025 = df[df['date'].dt.year == 2025].copy()
    print(f"\n2025 data: {len(df_2025):,} records")
    print(f"2025 date range: {df_2025['date'].min().date()} to {df_2025['date'].max().date()}")
    return df_2025


# =============================================================================
# SECTOR ANALYSIS
# =============================================================================

def analyze_sectors(df):
    """Comprehensive sector-level analysis."""
    
    print("\n" + "="*80)
    print("📊 SECTOR ANALYSIS - 2025")
    print("="*80)
    
    # Daily sector aggregates
    sector_daily = df.groupby(['date', 'sector']).agg({
        'bullish_score': 'mean',
        'bearish_score': 'mean',
        'net_score': 'mean',
        'is_bullish': 'mean',  # This gives us breadth (% bullish)
        'symbol': 'count'
    }).reset_index()
    
    sector_daily.columns = ['date', 'sector', 'avg_bull', 'avg_bear', 'avg_net', 'breadth', 'stock_count']
    sector_daily['breadth_pct'] = sector_daily['breadth'] * 100
    
    # Overall 2025 metrics by sector
    sector_summary = df.groupby('sector').agg({
        'bullish_score': 'mean',
        'bearish_score': 'mean',
        'net_score': 'mean',
        'is_bullish': 'mean',
        'symbol': 'nunique'
    }).reset_index()
    
    sector_summary.columns = ['sector', 'avg_bull', 'avg_bear', 'avg_net', 'avg_breadth', 'num_stocks']
    sector_summary['avg_breadth_pct'] = sector_summary['avg_breadth'] * 100
    sector_summary = sector_summary.sort_values('avg_net', ascending=False)
    
    # Calculate consistency (std dev of daily net scores - lower = more consistent)
    sector_consistency = sector_daily.groupby('sector')['avg_net'].std().reset_index()
    sector_consistency.columns = ['sector', 'volatility']
    sector_summary = sector_summary.merge(sector_consistency, on='sector')
    
    # Calculate trend (end of year vs start of year)
    first_month = sector_daily[sector_daily['date'] <= sector_daily['date'].min() + timedelta(days=30)]
    last_month = sector_daily[sector_daily['date'] >= sector_daily['date'].max() - timedelta(days=30)]
    
    first_avg = first_month.groupby('sector')['avg_net'].mean().reset_index()
    first_avg.columns = ['sector', 'jan_net']
    last_avg = last_month.groupby('sector')['avg_net'].mean().reset_index()
    last_avg.columns = ['sector', 'dec_net']
    
    sector_summary = sector_summary.merge(first_avg, on='sector')
    sector_summary = sector_summary.merge(last_avg, on='sector')
    sector_summary['ytd_change'] = sector_summary['dec_net'] - sector_summary['jan_net']
    
    # Print sector summary
    print("\n📈 SECTOR RANKINGS (by Avg Net Score)")
    print("-"*80)
    print(f"{'Rank':<5} {'Sector':<25} {'Avg Net':<10} {'Breadth':<10} {'Volatility':<12} {'YTD Δ':<10}")
    print("-"*80)
    
    for i, row in sector_summary.iterrows():
        rank = sector_summary.index.get_loc(i) + 1
        trend = "📈" if row['ytd_change'] > 2 else "📉" if row['ytd_change'] < -2 else "➡️"
        print(f"{rank:<5} {row['sector']:<25} {row['avg_net']:>+7.2f}   {row['avg_breadth_pct']:>6.1f}%   "
              f"{row['volatility']:>8.2f}     {row['ytd_change']:>+6.2f} {trend}")
    
    return sector_daily, sector_summary


# =============================================================================
# INDUSTRY ANALYSIS
# =============================================================================

def analyze_industries(df):
    """Comprehensive industry-level analysis."""
    
    print("\n" + "="*80)
    print("🏭 INDUSTRY ANALYSIS - 2025")
    print("="*80)
    
    # Daily industry aggregates
    industry_daily = df.groupby(['date', 'sector', 'industry']).agg({
        'bullish_score': 'mean',
        'bearish_score': 'mean',
        'net_score': 'mean',
        'is_bullish': 'mean',
        'symbol': 'count'
    }).reset_index()
    
    industry_daily.columns = ['date', 'sector', 'industry', 'avg_bull', 'avg_bear', 'avg_net', 'breadth', 'stock_count']
    industry_daily['breadth_pct'] = industry_daily['breadth'] * 100
    
    # Overall 2025 metrics by industry - GROUP BY BOTH SECTOR AND INDUSTRY
    industry_summary = df.groupby(['sector', 'industry']).agg({
        'bullish_score': 'mean',
        'bearish_score': 'mean',
        'net_score': 'mean',
        'is_bullish': 'mean',
        'symbol': 'nunique'
    }).reset_index()
    
    industry_summary.columns = ['sector', 'industry', 'avg_bull', 'avg_bear', 'avg_net', 'avg_breadth', 'num_stocks']
    industry_summary['avg_breadth_pct'] = industry_summary['avg_breadth'] * 100
    
    # FILTER OUT INDUSTRIES WITH NO STOCKS
    industry_summary = industry_summary[industry_summary['num_stocks'] > 0].copy()
    
    # Consistency - calculate per sector-industry pair
    industry_consistency = industry_daily.groupby(['sector', 'industry'])['avg_net'].std().reset_index()
    industry_consistency.columns = ['sector', 'industry', 'volatility']
    industry_summary = industry_summary.merge(industry_consistency, on=['sector', 'industry'], how='left')
    
    # YTD change
    first_month = industry_daily[industry_daily['date'] <= industry_daily['date'].min() + timedelta(days=30)]
    last_month = industry_daily[industry_daily['date'] >= industry_daily['date'].max() - timedelta(days=30)]
    
    first_avg = first_month.groupby(['sector', 'industry'])['avg_net'].mean().reset_index()
    first_avg.columns = ['sector', 'industry', 'jan_net']
    last_avg = last_month.groupby(['sector', 'industry'])['avg_net'].mean().reset_index()
    last_avg.columns = ['sector', 'industry', 'dec_net']
    
    industry_summary = industry_summary.merge(first_avg, on=['sector', 'industry'], how='left')
    industry_summary = industry_summary.merge(last_avg, on=['sector', 'industry'], how='left')
    industry_summary['ytd_change'] = industry_summary['dec_net'] - industry_summary['jan_net']
    
    # Sort by net score
    industry_summary = industry_summary.sort_values('avg_net', ascending=False)
    
    # Top 20 Industries
    print("\n🏆 TOP 20 INDUSTRIES (by Avg Net Score)")
    print("-"*100)
    print(f"{'Rank':<5} {'Industry':<35} {'Sector':<20} {'Net':<8} {'Breadth':<10} {'Stocks':<8} {'YTD Δ':<8}")
    print("-"*100)
    
    for i, (_, row) in enumerate(industry_summary.head(20).iterrows(), 1):
        trend = "📈" if row['ytd_change'] > 2 else "📉" if row['ytd_change'] < -2 else "➡️"
        ytd_str = f"{row['ytd_change']:>+5.1f}" if pd.notna(row['ytd_change']) else "  N/A"
        print(f"{i:<5} {row['industry'][:34]:<35} {row['sector'][:19]:<20} {row['avg_net']:>+5.1f}   "
              f"{row['avg_breadth_pct']:>6.1f}%   {row['num_stocks']:>5}   {ytd_str} {trend}")
    
    # Bottom 20 Industries
    print("\n📉 BOTTOM 20 INDUSTRIES (by Avg Net Score)")
    print("-"*100)
    print(f"{'Rank':<5} {'Industry':<35} {'Sector':<20} {'Net':<8} {'Breadth':<10} {'Stocks':<8} {'YTD Δ':<8}")
    print("-"*100)
    
    for i, (_, row) in enumerate(industry_summary.tail(20).iloc[::-1].iterrows(), 1):
        trend = "📈" if row['ytd_change'] > 2 else "📉" if row['ytd_change'] < -2 else "➡️"
        ytd_str = f"{row['ytd_change']:>+5.1f}" if pd.notna(row['ytd_change']) else "  N/A"
        print(f"{i:<5} {row['industry'][:34]:<35} {row['sector'][:19]:<20} {row['avg_net']:>+5.1f}   "
              f"{row['avg_breadth_pct']:>6.1f}%   {row['num_stocks']:>5}   {ytd_str} {trend}")
    
    return industry_daily, industry_summary


# =============================================================================
# MONTHLY ANALYSIS
# =============================================================================

def analyze_monthly(df):
    """Monthly performance breakdown."""
    
    print("\n" + "="*80)
    print("📅 MONTHLY BREAKDOWN - 2025")
    print("="*80)
    
    df['month'] = df['date'].dt.to_period('M')
    
    # Sector by month
    sector_monthly = df.groupby(['month', 'sector']).agg({
        'net_score': 'mean',
        'is_bullish': 'mean'
    }).reset_index()
    
    sector_monthly.columns = ['month', 'sector', 'avg_net', 'breadth']
    sector_monthly['breadth_pct'] = sector_monthly['breadth'] * 100
    
    # Pivot for display
    sector_pivot = sector_monthly.pivot(index='sector', columns='month', values='avg_net')
    
    print("\n📊 SECTOR NET SCORES BY MONTH")
    print("-"*120)
    
    # Get month columns
    months = sorted(sector_monthly['month'].unique())
    month_labels = [m.strftime('%b') for m in months]
    
    header = f"{'Sector':<25} " + " ".join([f"{m:>8}" for m in month_labels])
    print(header)
    print("-"*120)
    
    for sector in sector_pivot.index:
        row_data = sector_pivot.loc[sector]
        values = " ".join([f"{v:>+8.1f}" if pd.notna(v) else f"{'N/A':>8}" for v in row_data])
        print(f"{sector:<25} {values}")
    
    # Find best/worst month for each sector
    print("\n🏆 BEST & WORST MONTHS BY SECTOR")
    print("-"*80)
    print(f"{'Sector':<25} {'Best Month':<15} {'Score':<10} {'Worst Month':<15} {'Score':<10}")
    print("-"*80)
    
    for sector in sector_pivot.index:
        row = sector_pivot.loc[sector].dropna()
        if len(row) > 0:
            best_month = row.idxmax()
            worst_month = row.idxmin()
            print(f"{sector:<25} {best_month.strftime('%B'):<15} {row[best_month]:>+7.1f}   "
                  f"{worst_month.strftime('%B'):<15} {row[worst_month]:>+7.1f}")
    
    return sector_monthly


# =============================================================================
# ROTATION ANALYSIS
# =============================================================================

def analyze_rotation(sector_daily):
    """Analyze sector rotation throughout the year."""
    
    print("\n" + "="*80)
    print("🔄 SECTOR ROTATION ANALYSIS - 2025")
    print("="*80)
    
    # Find which sector was #1 each week
    sector_daily['week'] = sector_daily['date'].dt.to_period('W')
    
    weekly_avg = sector_daily.groupby(['week', 'sector'])['avg_net'].mean().reset_index()
    
    # Find leader each week
    weekly_leader = weekly_avg.loc[weekly_avg.groupby('week')['avg_net'].idxmax()]
    
    # Count weeks each sector led
    leader_counts = weekly_leader['sector'].value_counts()
    
    print("\n👑 WEEKS AS MARKET LEADER")
    print("-"*50)
    for sector, count in leader_counts.items():
        bar = "█" * count
        print(f"{sector:<25} {count:>3} weeks  {bar}")
    
    # Find which sector was last each week
    weekly_laggard = weekly_avg.loc[weekly_avg.groupby('week')['avg_net'].idxmin()]
    laggard_counts = weekly_laggard['sector'].value_counts()
    
    print("\n📉 WEEKS AS MARKET LAGGARD")
    print("-"*50)
    for sector, count in laggard_counts.items():
        bar = "█" * count
        print(f"{sector:<25} {count:>3} weeks  {bar}")
    
    return weekly_leader, weekly_laggard


# =============================================================================
# CURRENT MOMENTUM (Heading into 2026)
# =============================================================================

def analyze_current_momentum(df, sector_daily, industry_daily):
    """Analyze current momentum heading into 2026."""
    
    print("\n" + "="*80)
    print("🚀 CURRENT MOMENTUM (Heading into 2026)")
    print("="*80)
    
    # Last 20 trading days
    recent_cutoff = df['date'].max() - timedelta(days=30)
    
    # Sector momentum
    recent_sector = sector_daily[sector_daily['date'] >= recent_cutoff]
    sector_momentum = recent_sector.groupby('sector').agg({
        'avg_net': 'mean',
        'breadth_pct': 'mean'
    }).reset_index()
    sector_momentum.columns = ['sector', 'recent_net', 'recent_breadth']
    sector_momentum = sector_momentum.sort_values('recent_net', ascending=False)
    
    print("\n📈 SECTOR MOMENTUM (Last 30 Days)")
    print("-"*60)
    print(f"{'Rank':<5} {'Sector':<25} {'Net Score':<12} {'Breadth':<10}")
    print("-"*60)
    
    for i, (_, row) in enumerate(sector_momentum.iterrows(), 1):
        momentum = "🔥" if row['recent_net'] > 5 else "❄️" if row['recent_net'] < -5 else "➡️"
        print(f"{i:<5} {row['sector']:<25} {row['recent_net']:>+8.2f}     {row['recent_breadth']:>6.1f}%  {momentum}")
    
    # Industry momentum - filter to industries with actual data
    recent_industry = industry_daily[industry_daily['date'] >= recent_cutoff]
    industry_momentum = recent_industry.groupby(['sector', 'industry']).agg({
        'avg_net': 'mean',
        'breadth_pct': 'mean',
        'stock_count': 'mean'
    }).reset_index()
    industry_momentum.columns = ['sector', 'industry', 'recent_net', 'recent_breadth', 'avg_stocks']
    
    # Filter to industries with stocks
    industry_momentum = industry_momentum[industry_momentum['avg_stocks'] > 0]
    industry_momentum = industry_momentum.sort_values('recent_net', ascending=False)
    
    print("\n🔥 HOTTEST INDUSTRIES RIGHT NOW (Top 15)")
    print("-"*80)
    for i, (_, row) in enumerate(industry_momentum.head(15).iterrows(), 1):
        print(f"{i:>2}. {row['industry'][:35]:<36} ({row['sector'][:15]:<15}) Net: {row['recent_net']:>+6.1f}")
    
    print("\n❄️ COLDEST INDUSTRIES RIGHT NOW (Bottom 15)")
    print("-"*80)
    for i, (_, row) in enumerate(industry_momentum.tail(15).iloc[::-1].iterrows(), 1):
        print(f"{i:>2}. {row['industry'][:35]:<36} ({row['sector'][:15]:<15}) Net: {row['recent_net']:>+6.1f}")
    
    return sector_momentum, industry_momentum


# =============================================================================
# CONSISTENCY ANALYSIS
# =============================================================================

def analyze_consistency(industry_summary):
    """Find most and least consistent industries."""
    
    print("\n" + "="*80)
    print("📊 CONSISTENCY ANALYSIS")
    print("="*80)
    
    # Filter to valid industries only
    valid_industries = industry_summary[
        (industry_summary['num_stocks'] > 0) & 
        (industry_summary['volatility'].notna())
    ].copy()
    
    # Most consistent (low volatility + positive net)
    consistent = valid_industries[valid_industries['avg_net'] > 0].nsmallest(15, 'volatility')
    
    print("\n🎯 MOST CONSISTENT BULLISH INDUSTRIES (Low volatility + Positive)")
    print("-"*90)
    print(f"{'Industry':<40} {'Sector':<20} {'Net':<8} {'Volatility':<12}")
    print("-"*90)
    
    for _, row in consistent.iterrows():
        print(f"{row['industry'][:39]:<40} {row['sector'][:19]:<20} {row['avg_net']:>+5.1f}   {row['volatility']:>8.2f}")
    
    # Most volatile
    volatile = valid_industries.nlargest(15, 'volatility')
    
    print("\n🎢 MOST VOLATILE INDUSTRIES")
    print("-"*90)
    print(f"{'Industry':<40} {'Sector':<20} {'Net':<8} {'Volatility':<12}")
    print("-"*90)
    
    for _, row in volatile.iterrows():
        print(f"{row['industry'][:39]:<40} {row['sector'][:19]:<20} {row['avg_net']:>+5.1f}   {row['volatility']:>8.2f}")


# =============================================================================
# SECTOR DEEP DIVE
# =============================================================================

def sector_deep_dive(df, industry_summary, sector_name):
    """Deep dive into a specific sector's industries."""
    
    print(f"\n" + "="*80)
    print(f"🔍 DEEP DIVE: {sector_name.upper()}")
    print("="*80)
    
    # Filter to only industries in this sector with stocks
    sector_industries = industry_summary[
        (industry_summary['sector'] == sector_name) & 
        (industry_summary['num_stocks'] > 0)
    ].sort_values('avg_net', ascending=False)
    
    print(f"\nIndustries in {sector_name}: {len(sector_industries)}")
    print("-"*80)
    print(f"{'Rank':<5} {'Industry':<40} {'Net':<8} {'Breadth':<10} {'Stocks':<8}")
    print("-"*80)
    
    for i, (_, row) in enumerate(sector_industries.iterrows(), 1):
        print(f"{i:<5} {row['industry'][:39]:<40} {row['avg_net']:>+5.1f}   {row['avg_breadth_pct']:>6.1f}%   {row['num_stocks']:>5}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(sector_daily, industry_summary, sector_summary):
    """Create interactive visualizations."""
    
    print("\n" + "="*80)
    print("📈 CREATING VISUALIZATIONS")
    print("="*80)
    
    # Filter industry_summary to only include industries with stocks
    valid_industries = industry_summary[industry_summary['num_stocks'] > 0].copy()
    
    # 1. Sector performance heatmap over time
    pivot = sector_daily.pivot(index='date', columns='sector', values='avg_net')
    
    fig1 = go.Figure(data=go.Heatmap(
        z=pivot.values.T,
        x=pivot.index,
        y=pivot.columns,
        colorscale='RdYlGn',
        zmid=0
    ))
    
    fig1.update_layout(
        title='Sector Net Scores Throughout 2025',
        template='plotly_dark',
        height=500
    )
    
    fig1.write_html(OUTPUT_DIR / '2025_sector_heatmap.html')
    print(f"✅ Saved: 2025_sector_heatmap.html")
    
    # 2. Industry treemap - only with valid industries
    fig2 = px.treemap(
        valid_industries,
        path=['sector', 'industry'],
        values='num_stocks',
        color='avg_net',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        title='2025 Industry Performance Treemap'
    )
    
    fig2.update_layout(template='plotly_dark', height=700)
    fig2.write_html(OUTPUT_DIR / '2025_industry_treemap.html')
    print(f"✅ Saved: 2025_industry_treemap.html")
    
    # 3. Sector line chart over time
    fig3 = go.Figure()
    
    for sector in sector_daily['sector'].unique():
        sector_data = sector_daily[sector_daily['sector'] == sector]
        fig3.add_trace(go.Scatter(
            x=sector_data['date'],
            y=sector_data['avg_net'],
            mode='lines',
            name=sector
        ))
    
    fig3.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    fig3.update_layout(
        title='Sector Net Scores Over Time - 2025',
        template='plotly_dark',
        height=600,
        xaxis_title='Date',
        yaxis_title='Net Score'
    )
    
    fig3.write_html(OUTPUT_DIR / '2025_sector_trends.html')
    print(f"✅ Saved: 2025_sector_trends.html")
    
    # 4. Top vs Bottom industries comparison
    top_10 = valid_industries.head(10)
    bottom_10 = valid_industries.tail(10)
    
    fig4 = go.Figure()
    
    fig4.add_trace(go.Bar(
        y=top_10['industry'],
        x=top_10['avg_net'],
        orientation='h',
        name='Top 10',
        marker_color='green'
    ))
    
    fig4.add_trace(go.Bar(
        y=bottom_10['industry'],
        x=bottom_10['avg_net'],
        orientation='h',
        name='Bottom 10',
        marker_color='red'
    ))
    
    fig4.update_layout(
        title='Top 10 vs Bottom 10 Industries - 2025',
        template='plotly_dark',
        height=600,
        xaxis_title='Average Net Score',
        barmode='group'
    )
    
    fig4.write_html(OUTPUT_DIR / '2025_top_bottom_industries.html')
    print(f"✅ Saved: 2025_top_bottom_industries.html")


# =============================================================================
# EXPORT TO EXCEL
# =============================================================================

def export_to_excel(sector_summary, industry_summary, sector_monthly):
    """Export all data to Excel."""
    
    print("\n" + "="*80)
    print("📊 EXPORTING TO EXCEL")
    print("="*80)
    
    output_path = OUTPUT_DIR / '2025_Performance_Report.xlsx'
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sector summary
        sector_summary.to_excel(writer, sheet_name='Sector Summary', index=False)
        
        # Industry summary
        industry_summary.to_excel(writer, sheet_name='Industry Summary', index=False)
        
        # Top 50 industries
        industry_summary.head(50).to_excel(writer, sheet_name='Top 50 Industries', index=False)
        
        # Bottom 50 industries
        industry_summary.tail(50).to_excel(writer, sheet_name='Bottom 50 Industries', index=False)
    
    print(f"✅ Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("📊 2025 INDUSTRY & SECTOR PERFORMANCE REPORT")
    print("="*80)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Filter to 2025
    df_2025 = filter_2025_data(df)
    
    if len(df_2025) == 0:
        print("No 2025 data found!")
        return
    
    # Run analyses
    sector_daily, sector_summary = analyze_sectors(df_2025)
    industry_daily, industry_summary = analyze_industries(df_2025)
    sector_monthly = analyze_monthly(df_2025)
    analyze_rotation(sector_daily)
    sector_momentum, industry_momentum = analyze_current_momentum(df_2025, sector_daily, industry_daily)
    analyze_consistency(industry_summary)
    
    # Deep dive into key sectors
    for sector in ['Technology', 'Financial Services', 'Healthcare']:
        if sector in industry_summary['sector'].values:
            sector_deep_dive(df_2025, industry_summary, sector)
    
    # Create visualizations
    create_visualizations(sector_daily, industry_summary, sector_summary)
    
    # Export to Excel
    export_to_excel(sector_summary, industry_summary, sector_monthly)
    
    print("\n" + "="*80)
    print("✅ REPORT COMPLETE!")
    print("="*80)
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print("- 2025_Performance_Report.xlsx")
    print("- 2025_sector_heatmap.html")
    print("- 2025_industry_treemap.html")
    print("- 2025_sector_trends.html")
    print("- 2025_top_bottom_industries.html")


if __name__ == "__main__":
    main()