import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openpyxl
from io import BytesIO

# Page config
st.set_page_config(
    page_title="ZARA Global Sales Analytics 2021-2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.8rem; font-weight: 700; color: #1f1f1f; margin-bottom: 0.5rem; 
                  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .sub-header {font-size: 1.3rem; color: #555; margin-bottom: 2rem; font-weight: 500;}
    .kpi-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.8rem; border-radius: 15px; color: white; text-align: center;
                box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3); transition: transform 0.2s;}
    .kpi-card:hover {transform: translateY(-5px); box-shadow: 0 12px 24px rgba(102, 126, 234, 0.4);}
    .kpi-value {font-size: 2.2rem; font-weight: bold; margin: 0.5rem 0;}
    .kpi-label {font-size: 1rem; opacity: 0.95; font-weight: 500;}
    .insight-box {background: linear-gradient(to right, #f8f9fa 0%, #e9ecef 100%); 
                   padding: 1.5rem; border-left: 5px solid #667eea; 
                   margin: 1.5rem 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}
    .sidebar-section {background-color: #f8f9fa; padding: 1rem; border-radius: 10px; 
                       margin-bottom: 1rem; border: 1px solid #e9ecef;}
    .filter-header {color: #667eea; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;}
    div[data-testid="stSidebar"] {background: linear-gradient(180deg, #667eea15 0%, #764ba215 100%);}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {background-color: #f8f9fa; border-radius: 8px 8px 0 0; 
                                    padding: 12px 24px; font-weight: 600;}
    .stTabs [aria-selected="true"] {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                      color: white;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_validate_data():
    """Load and validate the Excel data with comprehensive checks."""
    try:
        df = pd.read_excel('ZARA_Global_Sales_Dataset_2021-2025.xlsx', sheet_name='Data')
        
        required_cols = [
            'Product Name', 'Country', 'Gender', 'Product Category', 'Year',
            'Sales_EUR', 'Sales_Local', 'Currency', 'Units_Sold',
            'Avg_Selling_Price_Local', 'Channel', 'Region', 'Source_Level', 'Method_Note'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
        
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
        df['Sales_EUR'] = pd.to_numeric(df['Sales_EUR'], errors='coerce')
        df['Sales_Local'] = pd.to_numeric(df['Sales_Local'], errors='coerce')
        df['Units_Sold'] = pd.to_numeric(df['Units_Sold'], errors='coerce')
        df['Avg_Selling_Price_Local'] = pd.to_numeric(df['Avg_Selling_Price_Local'], errors='coerce')
        
        df['ASP_EUR'] = np.where(df['Units_Sold'] > 0, df['Sales_EUR'] / df['Units_Sold'], 0)
        df['Sales_per_Unit_EUR'] = df['ASP_EUR']
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return df
    
    except FileNotFoundError:
        st.error("File 'ZARA_Global_Sales_Dataset_2021-2025.xlsx' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def compute_yoy_metrics(df, group_by_cols):
    """Compute YoY growth metrics for any grouping level."""
    if df.empty:
        return pd.DataFrame()
    
    agg_dict = {'Sales_EUR': 'sum', 'Units_Sold': 'sum', 'ASP_EUR': 'mean'}
    
    grouped = df.groupby(group_by_cols + ['Year']).agg(agg_dict).reset_index()
    grouped = grouped.sort_values(group_by_cols + ['Year'])
    
    for col in ['Sales_EUR', 'Units_Sold', 'ASP_EUR']:
        grouped[f'{col}_prev'] = grouped.groupby(group_by_cols)[col].shift(1)
        grouped[f'YoY_{col}_%'] = np.where(
            grouped[f'{col}_prev'] > 0,
            ((grouped[col] - grouped[f'{col}_prev']) / grouped[f'{col}_prev']) * 100,
            np.nan
        )
    
    return grouped

@st.cache_data
def compute_price_volume_decomp(df, group_by_cols):
    """Decompose sales change into price and volume effects."""
    if df.empty:
        return pd.DataFrame()
    
    agg = df.groupby(group_by_cols + ['Year']).agg({
        'Sales_EUR': 'sum',
        'Units_Sold': 'sum',
        'ASP_EUR': 'mean'
    }).reset_index()
    
    agg = agg.sort_values(group_by_cols + ['Year'])
    
    agg['Units_prev'] = agg.groupby(group_by_cols)['Units_Sold'].shift(1)
    agg['ASP_prev'] = agg.groupby(group_by_cols)['ASP_EUR'].shift(1)
    agg['Sales_prev'] = agg.groupby(group_by_cols)['Sales_EUR'].shift(1)
    
    agg['Volume_Effect'] = (agg['Units_Sold'] - agg['Units_prev']) * agg['ASP_prev']
    agg['Price_Effect'] = (agg['ASP_EUR'] - agg['ASP_prev']) * agg['Units_prev']
    agg['Total_Change'] = agg['Sales_EUR'] - agg['Sales_prev']
    
    agg = agg.dropna(subset=['Volume_Effect', 'Price_Effect'])
    
    return agg

def format_number(num, prefix='', suffix=''):
    """Format large numbers with M/B suffixes."""
    if pd.isna(num):
        return 'N/A'
    if abs(num) >= 1e9:
        return f"{prefix}{num/1e9:.2f}B{suffix}"
    elif abs(num) >= 1e6:
        return f"{prefix}{num/1e6:.1f}M{suffix}"
    elif abs(num) >= 1e3:
        return f"{prefix}{num/1e3:.1f}K{suffix}"
    else:
        return f"{prefix}{num:.0f}{suffix}"

def create_kpi_card(label, value, delta=None):
    """Create an enhanced KPI card component."""
    delta_html = ""
    if delta is not None:
        color = "#10b981" if delta >= 0 else "#ef4444"
        arrow = "↑" if delta >= 0 else "↓"
        delta_html = f"<div style='color: {color}; font-size: 1rem; margin-top: 0.5rem;'>{arrow} {abs(delta):.1f}%</div>"
    
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """

# Load data
df = load_and_validate_data()

if df is not None:
    # ENHANCED SIDEBAR
    with st.sidebar:
        st.markdown("### 🎯 ANALYTICS CONTROL CENTER")
        st.markdown("---")
        
        # Time Period Section
        with st.expander("📅 TIME PERIOD", expanded=True):
            min_year = int(df['Year'].min())
            max_year = int(df['Year'].max())
            year_range = st.slider(
                "Select Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                help="Drag to select the time period for analysis"
            )
        
        # Geographic Filters Section
        with st.expander("🌍 GEOGRAPHIC FILTERS", expanded=True):
            all_regions = sorted(df['Region'].dropna().unique())
            selected_regions = st.multiselect(
                "Regions",
                options=all_regions,
                default=all_regions,
                help="Select one or more regions"
            )
            
            filtered_countries = df[df['Region'].isin(selected_regions)]['Country'].dropna().unique()
            selected_countries = st.multiselect(
                "Countries",
                options=sorted(filtered_countries),
                default=[],
                help="Optional: Filter specific countries"
            )
        
        # Product Filters Section
        with st.expander("👔 PRODUCT FILTERS", expanded=True):
            all_categories = sorted(df['Product Category'].dropna().unique())
            selected_categories = st.multiselect(
                "Categories",
                options=all_categories,
                default=all_categories
            )
            
            all_genders = sorted(df['Gender'].dropna().unique())
            selected_genders = st.multiselect(
                "Gender",
                options=all_genders,
                default=all_genders
            )
        
        # Channel Filters Section
        with st.expander("🛒 CHANNEL FILTERS", expanded=True):
            all_channels = sorted(df['Channel'].dropna().unique())
            selected_channels = st.multiselect(
                "Sales Channels",
                options=all_channels,
                default=all_channels
            )
        
        # Advanced Options Section
        with st.expander("⚙️ ADVANCED OPTIONS", expanded=False):
            show_source_filter = st.checkbox("Filter by Data Source", value=False)
            selected_sources = []
            if show_source_filter:
                all_sources = sorted(df['Source_Level'].dropna().unique())
                selected_sources = st.multiselect(
                    "Source Level",
                    options=all_sources,
                    default=all_sources
                )
            
            top_n = st.slider("Top N Items", 5, 30, 10, help="Number of top items to display in rankings")
            
            metric_options = {
                'Sales (EUR)': 'Sales_EUR',
                'Units Sold': 'Units_Sold',
                'ASP (EUR)': 'ASP_EUR'
            }
            selected_metric_label = st.selectbox("Primary Metric", list(metric_options.keys()))
            selected_metric = metric_options[selected_metric_label]
        
        st.markdown("---")
        
        # Reset Button
        if st.button("🔄 RESET ALL FILTERS", use_container_width=True):
            st.rerun()
        
        # Info Box
        st.markdown("""
        <div style='background-color: #667eea15; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
            <small><b>💡 Quick Tips:</b><br>
            • Use filters to focus analysis<br>
            • Click chart legends to show/hide data<br>
            • Hover over charts for details<br>
            • Export data from Explorer tab</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Apply filters
    filtered_df = df[
        (df['Year'] >= year_range[0]) &
        (df['Year'] <= year_range[1]) &
        (df['Region'].isin(selected_regions)) &
        (df['Gender'].isin(selected_genders)) &
        (df['Product Category'].isin(selected_categories)) &
        (df['Channel'].isin(selected_channels))
    ]
    
    if selected_countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]
    
    if show_source_filter and selected_sources:
        filtered_df = filtered_df[filtered_df['Source_Level'].isin(selected_sources)]
    
    # Main header
    st.markdown('<div class="main-header">ZARA GLOBAL SALES ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Executive Intelligence Dashboard | Powered by Advanced Analytics</div>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Executive Overview",
        "🌍 Geo Intelligence",
        "👔 Category & Assortment",
        "🛒 Channel & Omnichannel",
        "💰 Pricing & Volume Drivers",
        "🔍 Explorer",
        "📋 Data Quality"
    ])
    
    # TAB 1: ENHANCED Executive Overview
    with tab1:
        st.header("Executive Overview")
        
        # KPIs
        total_sales = filtered_df['Sales_EUR'].sum()
        total_units = filtered_df['Units_Sold'].sum()
        avg_asp = filtered_df['ASP_EUR'].mean()
        
        channel_split = filtered_df.groupby('Channel')['Sales_EUR'].sum()
        online_share = 0
        if 'Online' in channel_split.index and channel_split.sum() > 0:
            online_share = (channel_split['Online'] / channel_split.sum()) * 100
        
        if year_range[1] > year_range[0]:
            current_year_df = filtered_df[filtered_df['Year'] == year_range[1]]
            prev_year_df = filtered_df[filtered_df['Year'] == year_range[1] - 1]
            
            current_sales = current_year_df['Sales_EUR'].sum()
            prev_sales = prev_year_df['Sales_EUR'].sum()
            
            yoy_growth = ((current_sales - prev_sales) / prev_sales * 100) if prev_sales > 0 else 0
        else:
            yoy_growth = None
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(create_kpi_card("Total Sales", format_number(total_sales, '€'), yoy_growth), unsafe_allow_html=True)
        with col2:
            st.markdown(create_kpi_card("Units Sold", format_number(total_units)), unsafe_allow_html=True)
        with col3:
            st.markdown(create_kpi_card("Avg ASP", format_number(avg_asp, '€')), unsafe_allow_html=True)
        with col4:
            st.markdown(create_kpi_card("YoY Growth", f"{yoy_growth:.1f}%" if yoy_growth is not None else "N/A"), unsafe_allow_html=True)
        with col5:
            st.markdown(create_kpi_card("Online Share", f"{online_share:.1f}%"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced Business insights
        growth_status = "📈 Strong Expansion" if yoy_growth and yoy_growth > 10 else "📊 Moderate Growth" if yoy_growth and yoy_growth > 0 else "⚠️ Contraction"
        online_status = "🚀 Digital Leader" if online_share > 40 else "💪 Growing Digital" if online_share > 20 else "🏪 Store-Focused"
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>💡 Executive Summary:</strong><br><br>
        <strong>Financial Performance:</strong> Generated <strong>{format_number(total_sales, '€')}</strong> in revenue with <strong>{yoy_growth:.1f}% YoY growth</strong> ({growth_status})<br>
        <strong>Volume Metrics:</strong> Sold <strong>{format_number(total_units)}</strong> units at an average price of <strong>€{avg_asp:.2f}</strong><br>
        <strong>Digital Transformation:</strong> Online channel represents <strong>{online_share:.1f}%</strong> of total sales ({online_status})<br>
        <strong>Strategic Focus:</strong> {'Invest in digital acceleration and omnichannel integration' if online_share < 30 else 'Optimize omnichannel experience and scale digital operations'}
        </div>
        """, unsafe_allow_html=True)
        
        # IMPROVED: Sales & Units Trend with Toggle
        st.subheader("📊 Performance Trends: Sales & Units")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            trend_view = st.radio("View Mode:", ['Dual Axis', 'Separate Metrics'], key='trend_view')
            show_yoy = st.checkbox("Show YoY %", value=False, key='show_yoy_trend')
        
        trend_df = filtered_df.groupby('Year').agg({
            'Sales_EUR': 'sum',
            'Units_Sold': 'sum'
        }).reset_index()
        
        with col1:
            if trend_view == 'Dual Axis':
                fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig_trend.add_trace(
                    go.Scatter(
                        x=trend_df['Year'],
                        y=trend_df['Sales_EUR'],
                        name='Sales (EUR)',
                        mode='lines+markers',
                        line=dict(color='#667eea', width=3),
                        marker=dict(size=10),
                        hovertemplate='<b>Year: %{x}</b><br>Sales: €%{y:,.0f}<extra></extra>'
                    ),
                    secondary_y=False
                )
                
                fig_trend.add_trace(
                    go.Scatter(
                        x=trend_df['Year'],
                        y=trend_df['Units_Sold'],
                        name='Units Sold',
                        mode='lines+markers',
                        line=dict(color='#764ba2', width=3, dash='dash'),
                        marker=dict(size=10, symbol='diamond'),
                        hovertemplate='<b>Year: %{x}</b><br>Units: %{y:,.0f}<extra></extra>'
                    ),
                    secondary_y=True
                )
                
                fig_trend.update_xaxes(title_text="Year", showgrid=True, gridcolor='#f0f0f0')
                fig_trend.update_yaxes(title_text="<b>Sales (EUR)</b>", secondary_y=False, showgrid=True, gridcolor='#f0f0f0')
                fig_trend.update_yaxes(title_text="<b>Units Sold</b>", secondary_y=True)
                
                fig_trend.update_layout(
                    title="Sales & Units Performance Trend",
                    hovermode='x unified',
                    height=450,
                    plot_bgcolor='white',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                fig_trend = go.Figure()
                
                fig_trend.add_trace(go.Bar(
                    x=trend_df['Year'],
                    y=trend_df['Sales_EUR'],
                    name='Sales (EUR)',
                    marker_color='#667eea',
                    hovertemplate='<b>Year: %{x}</b><br>Sales: €%{y:,.0f}<extra></extra>'
                ))
                
                fig_trend.add_trace(go.Bar(
                    x=trend_df['Year'],
                    y=trend_df['Units_Sold'],
                    name='Units Sold',
                    marker_color='#764ba2',
                    yaxis='y2',
                    hovertemplate='<b>Year: %{x}</b><br>Units: %{y:,.0f}<extra></extra>'
                ))
                
                fig_trend.update_layout(
                    title="Sales & Units by Year",
                    xaxis=dict(title='Year'),
                    yaxis=dict(title='Sales (EUR)', side='left'),
                    yaxis2=dict(title='Units Sold', overlaying='y', side='right'),
                    barmode='group',
                    height=450,
                    plot_bgcolor='white',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
        
        # IMPROVED: Regional Sales Contribution
        st.subheader("🌍 Regional Sales Contribution Over Time")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            region_view = st.radio("View Type:", ['Stacked Area', 'Stacked Bar', '100% Stacked'], key='region_view')
        
        region_trend = filtered_df.groupby(['Year', 'Region'])['Sales_EUR'].sum().reset_index()
        
        with col1:
            if region_view == 'Stacked Area':
                fig_region = px.area(
                    region_trend,
                    x='Year',
                    y='Sales_EUR',
                    color='Region',
                    title="Regional Sales Contribution (Stacked Area)",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_region.update_traces(
                    hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Sales: €%{y:,.0f}<extra></extra>'
                )
            elif region_view == 'Stacked Bar':
                fig_region = px.bar(
                    region_trend,
                    x='Year',
                    y='Sales_EUR',
                    color='Region',
                    title="Regional Sales Contribution (Stacked Bar)",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_region.update_traces(
                    hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Sales: €%{y:,.0f}<extra></extra>'
                )
            else:  # 100% Stacked
                region_pct = region_trend.groupby('Year').apply(
                    lambda x: x.assign(Sales_Pct=(x['Sales_EUR'] / x['Sales_EUR'].sum()) * 100)
                ).reset_index(drop=True)
                
                fig_region = px.bar(
                    region_pct,
                    x='Year',
                    y='Sales_Pct',
                    color='Region',
                    title="Regional Sales Mix (100% Stacked)",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    labels={'Sales_Pct': 'Percentage of Sales'}
                )
                fig_region.update_traces(
                    hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Share: %{y:.1f}%<extra></extra>'
                )
                fig_region.update_layout(barmode='stack')
            
            fig_region.update_layout(
                height=450,
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_region, use_container_width=True)
        
        # IMPROVED: Waterfall Chart with Better Spacing
        if year_range[1] > year_range[0]:
            st.subheader(f"📊 YoY Change Decomposition: {year_range[1]-1} → {year_range[1]}")
            
            current_year_data = filtered_df[filtered_df['Year'] == year_range[1]].groupby('Region')['Sales_EUR'].sum()
            prev_year_data = filtered_df[filtered_df['Year'] == year_range[1] - 1].groupby('Region')['Sales_EUR'].sum()
            
            changes = (current_year_data - prev_year_data).sort_values(ascending=False)
            
            # Create waterfall with proper spacing
            x_labels = list(changes.index) + ['NET CHANGE']
            y_values = list(changes.values) + [changes.sum()]
            measures = ['relative'] * len(changes) + ['total']
            
            # Color code: green for positive, red for negative
            colors = ['#10b981' if v > 0 else '#ef4444' for v in changes.values] + ['#667eea']
            
            fig_waterfall = go.Figure(go.Waterfall(
                x=x_labels,
                y=y_values,
                measure=measures,
                text=[format_number(v, '€') for v in y_values],
                textposition='outside',
                connector={"line": {"color": "#6b7280", "width": 2, "dash": "dot"}},
                increasing={"marker": {"color": "#10b981"}},
                decreasing={"marker": {"color": "#ef4444"}},
                totals={"marker": {"color": "#667eea"}}
            ))
            
            fig_waterfall.update_layout(
                title=f"Regional Contribution to YoY Sales Change",
                xaxis_title="Region",
                yaxis_title="Sales Change (EUR)",
                height=500,
                plot_bgcolor='white',
                showlegend=False,
                xaxis=dict(showgrid=False, tickangle=-45),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0', zeroline=True, zerolinecolor='#000', zerolinewidth=2)
            )
            
            st.plotly_chart(fig_waterfall, use_container_width=True)
            
            # Add interpretation
            top_contributor = changes.idxmax()
            top_value = changes.max()
            bottom_contributor = changes.idxmin()
            bottom_value = changes.min()
            
            st.markdown(f"""
            <div class="insight-box">
            <strong>🎯 Key Drivers:</strong><br>
            • <strong>Top Growth Driver:</strong> {top_contributor} contributed <strong>{format_number(top_value, '€')}</strong> to growth<br>
            • <strong>Biggest Drag:</strong> {bottom_contributor} reduced sales by <strong>{format_number(abs(bottom_value), '€')}</strong><br>
            • <strong>Net Impact:</strong> Overall YoY change of <strong>{format_number(changes.sum(), '€')}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Pareto Chart
        st.subheader("📊 Top Countries: Cumulative Sales Contribution (Pareto Analysis)")
        
        country_sales = filtered_df.groupby('Country')['Sales_EUR'].sum().sort_values(ascending=False).head(20)
        country_sales_cumsum = country_sales.cumsum() / country_sales.sum() * 100
        
        fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_pareto.add_trace(
            go.Bar(
                x=country_sales.index,
                y=country_sales.values,
                name='Sales',
                marker_color='#667eea',
                hovertemplate='<b>%{x}</b><br>Sales: €%{y:,.0f}<extra></extra>'
            ),
            secondary_y=False
        )
        
        fig_pareto.add_trace(
            go.Scatter(
                x=country_sales_cumsum.index,
                y=country_sales_cumsum.values,
                name='Cumulative %',
                line=dict(color='#ef4444', width=3),
                mode='lines+markers',
                marker=dict(size=8),
                hovertemplate='<b>%{x}</b><br>Cumulative: %{y:.1f}%<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Add 80% reference line
        fig_pareto.add_hline(y=80, line_dash="dash", line_color="green", secondary_y=True, 
                            annotation_text="80% Target", annotation_position="right")
        
        fig_pareto.update_xaxes(title_text="Country", tickangle=-45)
        fig_pareto.update_yaxes(title_text="<b>Sales (EUR)</b>", secondary_y=False)
        fig_pareto.update_yaxes(title_text="<b>Cumulative %</b>", secondary_y=True, range=[0, 105])
        
        fig_pareto.update_layout(
            title="Top 20 Countries: 80/20 Rule Analysis",
            height=500,
            plot_bgcolor='white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_pareto, use_container_width=True)
        
        # Calculate 80% threshold
        countries_to_80 = (country_sales_cumsum <= 80).sum() + 1
        pct_countries_to_80 = (countries_to_80 / len(filtered_df['Country'].unique())) * 100
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>📊 80/20 Analysis:</strong><br>
        • Just <strong>{countries_to_80} countries</strong> ({pct_countries_to_80:.1f}% of total markets) generate <strong>80% of sales</strong><br>
        • These high-impact markets should receive priority in resource allocation<br>
        • Remaining markets may benefit from simplified operational models
        </div>
        """, unsafe_allow_html=True)
    
    # TAB 2: Geo Intelligence with IMPROVED Growth Quadrant
    with tab2:
        st.header("Geographic Intelligence")
        
        # World map
        st.subheader("🗺️ Global Sales Heatmap")
        
        map_year = st.select_slider("Select Year for Map", options=sorted(filtered_df['Year'].unique()), key='map_year')
        
        map_data = filtered_df[filtered_df['Year'] == map_year].groupby('Country').agg({
            'Sales_EUR': 'sum',
            'Units_Sold': 'sum'
        }).reset_index()
        
        fig_map = px.choropleth(
            map_data,
            locations='Country',
            locationmode='country names',
            color='Sales_EUR',
            hover_name='Country',
            hover_data={'Sales_EUR': ':,.0f', 'Units_Sold': ':,.0f'},
            title=f"Global Sales Distribution ({map_year})",
            color_continuous_scale='Viridis'
        )
        fig_map.update_layout(height=500, geo=dict(showframe=False, showcoastlines=True))
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Heatmap: Country x Category
        st.subheader("🔥 Country × Category Sales Intensity")
        
        heatmap_data = filtered_df.groupby(['Country', 'Product Category'])['Sales_EUR'].sum().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='Country', columns='Product Category', values='Sales_EUR').fillna(0)
        
        top_countries = filtered_df.groupby('Country')['Sales_EUR'].sum().nlargest(15).index
        heatmap_pivot_top = heatmap_pivot.loc[heatmap_pivot.index.isin(top_countries)]
        
        fig_heatmap = px.imshow(
            heatmap_pivot_top,
            labels=dict(x="Category", y="Country", color="Sales (EUR)"),
            title=f"Sales Intensity: Top 15 Countries × Categories",
            aspect='auto',
            color_continuous_scale='RdYlGn'
        )
        fig_heatmap.update_layout(height=500)
        fig_heatmap.update_xaxes(tickangle=-45)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Country ranking with drill-down
        st.subheader("🏆 Country Performance Ranking")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            country_rank = filtered_df.groupby('Country')[selected_metric].sum().sort_values(ascending=False).head(top_n)
            
            fig_country_rank = px.bar(
                x=country_rank.values,
                y=country_rank.index,
                orientation='h',
                title=f"Top {top_n} Countries",
                labels={'x': selected_metric_label, 'y': 'Country'},
                color=country_rank.values,
                color_continuous_scale='Blues'
            )
            fig_country_rank.update_layout(height=600, showlegend=False)
            fig_country_rank.update_traces(hovertemplate='<b>%{y}</b><br>Value: %{x:,.0f}<extra></extra>')
            st.plotly_chart(fig_country_rank, use_container_width=True)
        
        with col2:
            selected_country = st.selectbox("Drill into Country Details:", options=country_rank.index, key='drill_country')
            
            if selected_country:
                country_cat_data = filtered_df[filtered_df['Country'] == selected_country].groupby('Product Category')[selected_metric].sum().sort_values(ascending=False)
                
                fig_country_cat = px.bar(
                    x=country_cat_data.values,
                    y=country_cat_data.index,
                    orientation='h',
                    title=f"Category Breakdown: {selected_country}",
                    labels={'x': selected_metric_label, 'y': 'Category'},
                    color=country_cat_data.values,
                    color_continuous_scale='Purples'
                )
                fig_country_cat.update_layout(height=600, showlegend=False)
                fig_country_cat.update_traces(hovertemplate='<b>%{y}</b><br>Value: %{x:,.0f}<extra></extra>')
                st.plotly_chart(fig_country_cat, use_container_width=True)
        
        # IMPROVED: Growth Quadrant Analysis
        if year_range[1] > year_range[0]:
            st.subheader("📊 Growth Quadrant Analysis: Strategic Portfolio View")
            
            yoy_country = compute_yoy_metrics(filtered_df, ['Country'])
            latest_yoy = yoy_country[yoy_country['Year'] == year_range[1]].copy()
            
            # Add quadrant classification
            median_growth = latest_yoy['YoY_Sales_EUR_%'].median()
            median_size = latest_yoy['Sales_EUR'].median()
            
            def classify_quadrant(row):
                if row['YoY_Sales_EUR_%'] >= median_growth and row['Sales_EUR'] >= median_size:
                    return '⭐ Stars'
                elif row['YoY_Sales_EUR_%'] >= median_growth and row['Sales_EUR'] < median_size:
                    return '🌟 Rising Stars'
                elif row['YoY_Sales_EUR_%'] < median_growth and row['Sales_EUR'] >= median_size:
                    return '💰 Cash Cows'
                else:
                    return '🐕 Dogs'
            
            latest_yoy['Quadrant'] = latest_yoy.apply(classify_quadrant, axis=1)
            
            # Color mapping
            color_map = {
                '⭐ Stars': '#10b981',
                '🌟 Rising Stars': '#3b82f6',
                '💰 Cash Cows': '#f59e0b',
                '🐕 Dogs': '#ef4444'
            }
            
            fig_quadrant = px.scatter(
                latest_yoy,
                x='YoY_Sales_EUR_%',
                y='Sales_EUR',
                size='Sales_EUR',
                hover_name='Country',
                title=f"Strategic Country Portfolio Matrix ({year_range[1]})",
                labels={'YoY_Sales_EUR_%': 'YoY Growth Rate (%)', 'Sales_EUR': 'Market Size (EUR)'},
                color='Quadrant',
                color_discrete_map=color_map,
                size_max=60
            )
            
            # Add quadrant lines
            fig_quadrant.add_hline(y=median_size, line_dash="dash", line_color="gray", line_width=2,
                                  annotation_text="Median Size", annotation_position="right")
            fig_quadrant.add_vline(x=median_growth, line_dash="dash", line_color="gray", line_width=2,
                                  annotation_text="Median Growth", annotation_position="top")
            
            # Add quadrant labels
            max_sales = latest_yoy['Sales_EUR'].max()
            min_growth = latest_yoy['YoY_Sales_EUR_%'].min()
            max_growth = latest_yoy['YoY_Sales_EUR_%'].max()
            
            fig_quadrant.add_annotation(x=max_growth*0.7, y=max_sales*0.9, text="⭐<br><b>STARS</b><br>Invest & Grow", 
                           showarrow=False, font=dict(size=14, color='#10b981'), bgcolor='rgba(16, 185, 129, 0.1)', borderpad=10)
            fig_quadrant.add_annotation(x=max_growth*0.7, y=max_sales*0.2, text="🌟<br><b>RISING STARS</b><br>Nurture", 
                           showarrow=False, font=dict(size=14, color='#3b82f6'), bgcolor='rgba(59, 130, 246, 0.1)', borderpad=10)
            fig_quadrant.add_annotation(x=min_growth*0.7, y=max_sales*0.9, text="💰<br><b>CASH COWS</b><br>Harvest", 
                           showarrow=False, font=dict(size=14, color='#f59e0b'), bgcolor='rgba(245, 158, 11, 0.1)', borderpad=10)
            fig_quadrant.add_annotation(x=min_growth*0.7, y=max_sales*0.2, text="🐕<br><b>DOGS</b><br>Divest/Exit", 
                           showarrow=False, font=dict(size=14, color='#ef4444'), bgcolor='rgba(239, 68, 68, 0.1)', borderpad=10)
            
            fig_quadrant.update_traces(hovertemplate='<b>%{hovertext}</b><br>Growth: %{x:.1f}%<br>Sales: €%{y:,.0f}<extra></extra>')
            fig_quadrant.update_layout(height=600, plot_bgcolor='white',
                                      xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                                      yaxis=dict(showgrid=True, gridcolor='#f0f0f0'))
            
            st.plotly_chart(fig_quadrant, use_container_width=True)
            
            # Strategic recommendations by quadrant
            quad_counts = latest_yoy['Quadrant'].value_counts()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                stars = latest_yoy[latest_yoy['Quadrant'] == '⭐ Stars']
                st.metric("⭐ Stars", len(stars), f"{stars['Sales_EUR'].sum()/1e6:.0f}M EUR")
            with col2:
                rising = latest_yoy[latest_yoy['Quadrant'] == '🌟 Rising Stars']
                st.metric("🌟 Rising Stars", len(rising), f"{rising['Sales_EUR'].sum()/1e6:.0f}M EUR")
            with col3:
                cows = latest_yoy[latest_yoy['Quadrant'] == '💰 Cash Cows']
                st.metric("💰 Cash Cows", len(cows), f"{cows['Sales_EUR'].sum()/1e6:.0f}M EUR")
            with col4:
                dogs = latest_yoy[latest_yoy['Quadrant'] == '🐕 Dogs']
                st.metric("🐕 Dogs", len(dogs), f"{dogs['Sales_EUR'].sum()/1e6:.0f}M EUR")
            
            st.markdown("""
            <div class="insight-box">
            <strong>🎯 Strategic Recommendations:</strong><br>
            • <strong>⭐ Stars:</strong> Double down on investment - these are winning markets with scale<br>
            • <strong>🌟 Rising Stars:</strong> Strategic bets - invest in infrastructure before competitors<br>
            • <strong>💰 Cash Cows:</strong> Optimize for profitability - extract value efficiently<br>
            • <strong>🐕 Dogs:</strong> Evaluate exit strategies - consider divestment or turnaround plans
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 3: IMPROVED Category & Assortment
    with tab3:
        st.header("Category & Assortment Intelligence")
        
        # Treemap
        st.subheader("🗂️ Category → Product Sales Hierarchy")
        
        treemap_data = filtered_df.groupby(['Product Category', 'Product Name'])['Sales_EUR'].sum().reset_index()
        treemap_data = treemap_data.nlargest(100, 'Sales_EUR')
        
        fig_treemap = px.treemap(
            treemap_data,
            path=['Product Category', 'Product Name'],
            values='Sales_EUR',
            title='Sales Distribution: Category → Product (Top 100 Products)',
            color='Sales_EUR',
            color_continuous_scale='Viridis',
            hover_data={'Sales_EUR': ':,.0f'}
        )
        fig_treemap.update_traces(hovertemplate='<b>%{label}</b><br>Sales: €%{value:,.0f}<extra></extra>')
        fig_treemap.update_layout(height=600)
        st.plotly_chart(fig_treemap, use_container_width=True)
        
        # IMPROVED: Category Performance Trends
        st.subheader("📈 Category Performance Trends")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            cat_trend_view = st.radio("Display Style:", ['Line Chart', 'Area Chart', 'Faceted View'], key='cat_trend_view')
            top_categories = st.slider("Number of Categories", 3, 10, 5, key='top_cat')
        
        cat_trend = filtered_df.groupby(['Year', 'Product Category'])['Sales_EUR'].sum().reset_index()
        
        # Get top N categories by total sales
        top_cats = filtered_df.groupby('Product Category')['Sales_EUR'].sum().nlargest(top_categories).index
        cat_trend_filtered = cat_trend[cat_trend['Product Category'].isin(top_cats)]
        
        with col1:
            if cat_trend_view == 'Line Chart':
                fig_cat_trend = px.line(
                    cat_trend_filtered,
                    x='Year',
                    y='Sales_EUR',
                    color='Product Category',
                    title=f'Sales Trend: Top {top_categories} Categories',
                    markers=True,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_cat_trend.update_traces(line=dict(width=3), marker=dict(size=10),
                                           hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Sales: €%{y:,.0f}<extra></extra>')
                
            elif cat_trend_view == 'Area Chart':
                fig_cat_trend = px.area(
                    cat_trend_filtered,
                    x='Year',
                    y='Sales_EUR',
                    color='Product Category',
                    title=f'Sales Trend: Top {top_categories} Categories (Stacked)',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_cat_trend.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Sales: €%{y:,.0f}<extra></extra>')
                
            else:  # Faceted View
                fig_cat_trend = px.line(
                    cat_trend_filtered,
                    x='Year',
                    y='Sales_EUR',
                    facet_col='Product Category',
                    facet_col_wrap=3,
                    title=f'Sales Trend: Top {top_categories} Categories (Individual Panels)',
                    markers=True,
                    color_discrete_sequence=['#667eea']
                )
                fig_cat_trend.update_traces(line=dict(width=3), marker=dict(size=8),
                                           hovertemplate='Year: %{x}<br>Sales: €%{y:,.0f}<extra></extra>')
                fig_cat_trend.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            
            fig_cat_trend.update_layout(
                height=500,
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_cat_trend, use_container_width=True)
        
        # IMPROVED: Category Mix Evolution
        st.subheader("🥧 Category Mix Evolution")

        col1, col2 = st.columns([3, 1])
        with col2:
            mix_view = st.radio("View Type:", ['Percentage', 'Absolute Values', 'Trend Lines'], key='mix_view')

    cat_mix = filtered_df.groupby(['Year', 'Product Category'])['Sales_EUR'].sum().reset_index()
    
    with col1:
        if mix_view == 'Percentage':
            cat_mix_pct = cat_mix.groupby('Year').apply(
                lambda x: x.assign(Sales_Pct=(x['Sales_EUR'] / x['Sales_EUR'].sum()) * 100)
            ).reset_index(drop=True)
            
            fig_cat_mix = px.bar(
                cat_mix_pct,
                x='Year',
                y='Sales_Pct',
                color='Product Category',
                title='Category Mix Over Time (100% Stacked)',
                labels={'Sales_Pct': 'Percentage of Total Sales (%)'},
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_cat_mix.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Share: %{y:.1f}%<extra></extra>')
            fig_cat_mix.update_layout(barmode='stack')
            
        elif mix_view == 'Absolute Values':
            fig_cat_mix = px.bar(
                cat_mix,
                x='Year',
                y='Sales_EUR',
                color='Product Category',
                title='Category Sales Over Time (Stacked)',
                labels={'Sales_EUR': 'Sales (EUR)'},
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_cat_mix.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Sales: €%{y:,.0f}<extra></extra>')
            fig_cat_mix.update_layout(barmode='stack')
            
        else:  # Trend Lines
            # Calculate percentage for each category
            cat_mix_pct = cat_mix.groupby('Year').apply(
                lambda x: x.assign(Sales_Pct=(x['Sales_EUR'] / x['Sales_EUR'].sum()) * 100)
            ).reset_index(drop=True)
            
            fig_cat_mix = px.line(
                cat_mix_pct,
                x='Year',
                y='Sales_Pct',
                color='Product Category',
                title='Category Share Trends',
                labels={'Sales_Pct': 'Share of Total Sales (%)'},
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_cat_mix.update_traces(line=dict(width=3), marker=dict(size=10),
                                     hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Share: %{y:.1f}%<extra></extra>')
        
        fig_cat_mix.update_layout(
            height=500,
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_cat_mix, use_container_width=True)
    
    # Identify mix shifts
    if year_range[1] > year_range[0]:
        first_year = cat_mix[cat_mix['Year'] == year_range[0]].set_index('Product Category')['Sales_EUR']
        last_year = cat_mix[cat_mix['Year'] == year_range[1]].set_index('Product Category')['Sales_EUR']
        
        first_year_pct = (first_year / first_year.sum()) * 100
        last_year_pct = (last_year / last_year.sum()) * 100
        
        mix_change = (last_year_pct - first_year_pct).sort_values(ascending=False)
        
        gainers = mix_change[mix_change > 0].head(3)
        losers = mix_change[mix_change < 0].tail(3)
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>📊 Mix Shift Analysis ({year_range[0]} → {year_range[1]}):</strong><br><br>
        <strong>Share Gainers:</strong><br>
        {'<br>'.join([f"• {cat}: +{val:.1f}pp" for cat, val in gainers.items()])}<br><br>
        <strong>Share Losers:</strong><br>
        {'<br>'.join([f"• {cat}: {val:.1f}pp" for cat, val in losers.items()])}
        </div>
        """, unsafe_allow_html=True)
    
    # ASP by category
    st.subheader("💰 Average Selling Price by Category")
    
    asp_cat = filtered_df.groupby('Product Category').agg({
        'ASP_EUR': ['mean', 'median', 'std', 'min', 'max']
    }).reset_index()
    asp_cat.columns = ['Product Category', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
    asp_cat = asp_cat.sort_values('Mean', ascending=False)
    
    fig_asp = go.Figure()
    
    fig_asp.add_trace(go.Bar(
        x=asp_cat['Product Category'],
        y=asp_cat['Mean'],
        name='Mean ASP',
        marker_color='#667eea',
        error_y=dict(type='data', array=asp_cat['Std Dev'], color='#764ba2'),
        hovertemplate='<b>%{x}</b><br>Mean: €%{y:.2f}<br>Std Dev: €%{error_y.array:.2f}<extra></extra>'
    ))
    
    fig_asp.update_layout(
        title='Average Selling Price by Category (with Standard Deviation)',
        xaxis_title='Product Category',
        yaxis_title='ASP (EUR)',
        height=500,
        plot_bgcolor='white',
        xaxis=dict(tickangle=-45, showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
    )
    
    st.plotly_chart(fig_asp, use_container_width=True)
    
    # Show detailed table
    with st.expander("📊 View Detailed ASP Statistics"):
        st.dataframe(asp_cat.style.format({
            'Mean': '€{:.2f}',
            'Median': '€{:.2f}',
            'Std Dev': '€{:.2f}',
            'Min': '€{:.2f}',
            'Max': '€{:.2f}'
        }), use_container_width=True)

# TAB 4: COMPLETELY REDESIGNED Channel & Omnichannel
with tab4:
    st.header("Channel & Omnichannel Intelligence")
    
    # Channel Overview Metrics
    st.subheader("📊 Channel Performance Overview")
    
    channel_metrics = filtered_df.groupby('Channel').agg({
        'Sales_EUR': 'sum',
        'Units_Sold': 'sum',
        'ASP_EUR': 'mean'
    }).reset_index()
    
    col1, col2, col3 = st.columns(3)
    
    for idx, row in channel_metrics.iterrows():
        with [col1, col2, col3][idx % 3]:
            pct_of_total = (row['Sales_EUR'] / channel_metrics['Sales_EUR'].sum()) * 100
            st.metric(
                f"{row['Channel']} Channel",
                format_number(row['Sales_EUR'], '€'),
                f"{pct_of_total:.1f}% of total"
            )
    
    st.markdown("---")
    
    # IMPROVED: Channel Mix Evolution
    st.subheader("📈 Channel Mix Evolution")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        channel_viz = st.radio("Visualization:", ['Stacked Area', 'Line Chart', 'Bar Chart'], key='channel_viz')
        show_percentage = st.checkbox("Show as Percentage", value=True, key='channel_pct')
    
    channel_year = filtered_df.groupby(['Year', 'Channel'])['Sales_EUR'].sum().reset_index()
    
    with col1:
        if show_percentage:
            channel_year_pct = channel_year.groupby('Year').apply(
                lambda x: x.assign(Sales_Pct=(x['Sales_EUR'] / x['Sales_EUR'].sum()) * 100)
            ).reset_index(drop=True)
            
            if channel_viz == 'Stacked Area':
                fig_channel = px.area(
                    channel_year_pct,
                    x='Year',
                    y='Sales_Pct',
                    color='Channel',
                    title='Channel Mix Evolution (%)',
                    color_discrete_map={'Online': '#667eea', 'Store': '#764ba2'},
                    labels={'Sales_Pct': 'Share of Total Sales (%)'}
                )
            elif channel_viz == 'Line Chart':
                fig_channel = px.line(
                    channel_year_pct,
                    x='Year',
                    y='Sales_Pct',
                    color='Channel',
                    title='Channel Share Trends (%)',
                    markers=True,
                    color_discrete_map={'Online': '#667eea', 'Store': '#764ba2'},
                    labels={'Sales_Pct': 'Share of Total Sales (%)'}
                )
                fig_channel.update_traces(line=dict(width=4), marker=dict(size=12))
            else:  # Bar Chart
                fig_channel = px.bar(
                    channel_year_pct,
                    x='Year',
                    y='Sales_Pct',
                    color='Channel',
                    title='Channel Mix by Year (%)',
                    color_discrete_map={'Online': '#667eea', 'Store': '#764ba2'},
                    labels={'Sales_Pct': 'Share of Total Sales (%)'}
                )
                fig_channel.update_layout(barmode='stack')
            
            fig_channel.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Share: %{y:.1f}%<extra></extra>')
        else:
            if channel_viz == 'Stacked Area':
                fig_channel = px.area(
                    channel_year,
                    x='Year',
                    y='Sales_EUR',
                    color='Channel',
                    title='Channel Sales Evolution (EUR)',
                    color_discrete_map={'Online': '#667eea', 'Store': '#764ba2'}
                )
            elif channel_viz == 'Line Chart':
                fig_channel = px.line(
                    channel_year,
                    x='Year',
                    y='Sales_EUR',
                    color='Channel',
                    title='Channel Sales Trends (EUR)',
                    markers=True,
                    color_discrete_map={'Online': '#667eea', 'Store': '#764ba2'}
                )
                fig_channel.update_traces(line=dict(width=4), marker=dict(size=12))
            else:  # Bar Chart
                fig_channel = px.bar(
                    channel_year,
                    x='Year',
                    y='Sales_EUR',
                    color='Channel',
                    title='Channel Sales by Year (EUR)',
                    color_discrete_map={'Online': '#667eea', 'Store': '#764ba2'}
                )
                fig_channel.update_layout(barmode='group')
            
            fig_channel.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Sales: €%{y:,.0f}<extra></extra>')
        
        fig_channel.update_layout(
            height=450,
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_channel, use_container_width=True)
    
    # Calculate digital transformation rate
    if year_range[1] > year_range[0]:
        online_first = channel_year[(channel_year['Year'] == year_range[0]) & (channel_year['Channel'] == 'Online')]['Sales_EUR'].sum()
        online_last = channel_year[(channel_year['Year'] == year_range[1]) & (channel_year['Channel'] == 'Online')]['Sales_EUR'].sum()
        total_first = channel_year[channel_year['Year'] == year_range[0]]['Sales_EUR'].sum()
        total_last = channel_year[channel_year['Year'] == year_range[1]]['Sales_EUR'].sum()
        
        online_share_first = (online_first / total_first * 100) if total_first > 0 else 0
        online_share_last = (online_last / total_last * 100) if total_last > 0 else 0
        digital_shift = online_share_last - online_share_first
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>🚀 Digital Transformation Progress:</strong><br>
        • Online share increased from <strong>{online_share_first:.1f}%</strong> ({year_range[0]}) to <strong>{online_share_last:.1f}%</strong> ({year_range[1]})<br>
        • Net digital shift: <strong>+{digital_shift:.1f} percentage points</strong><br>
        • {'Accelerated digital adoption - outpacing industry' if digital_shift > 10 else 'Steady digital growth trajectory' if digital_shift > 5 else 'Moderate digital transformation pace'}
        </div>
        """, unsafe_allow_html=True)
    
    # IMPROVED: Online Penetration by Country
    st.subheader("🌐 Online Penetration by Market")
    
    col1, col2 = st.columns([2, 1])
    
    country_channel = filtered_df.groupby(['Country', 'Channel'])['Sales_EUR'].sum().reset_index()
    country_total = country_channel.groupby('Country')['Sales_EUR'].sum().reset_index()
    country_online = country_channel[country_channel['Channel'] == 'Online'].copy()
    
    country_online = country_online.merge(country_total, on='Country', suffixes=('_online', '_total'))
    country_online['Online_Share_%'] = (country_online['Sales_EUR_online'] / country_online['Sales_EUR_total']) * 100
    country_online = country_online.sort_values('Online_Share_%', ascending=False)
    
    with col1:
        top_online = country_online.head(top_n)
        
        fig_online_share = px.bar(
            top_online,
            y='Country',
            x='Online_Share_%',
            orientation='h',
            title=f'Top {top_n} Markets by Online Penetration',
            labels={'Online_Share_%': 'Online Share (%)'},
            color='Online_Share_%',
            color_continuous_scale='Blues',
            text='Online_Share_%'
        )
        fig_online_share.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Online Share: %{x:.1f}%<extra></extra>'
        )
        fig_online_share.update_layout(height=500, showlegend=False, plot_bgcolor='white')
        st.plotly_chart(fig_online_share, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Digital Maturity")
        
        high_digital = len(country_online[country_online['Online_Share_%'] > 40])
        medium_digital = len(country_online[(country_online['Online_Share_%'] >= 20) & (country_online['Online_Share_%'] <= 40)])
        low_digital = len(country_online[country_online['Online_Share_%'] < 20])
        
        maturity_data = pd.DataFrame({
            'Maturity': ['High (>40%)', 'Medium (20-40%)', 'Low (<20%)'],
            'Count': [high_digital, medium_digital, low_digital]
        })
        
        fig_maturity = px.pie(
            maturity_data,
            values='Count',
            names='Maturity',
            title='Digital Maturity Distribution',
            color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444']
        )
        fig_maturity.update_traces(textposition='inside', textinfo='percent+label')
        fig_maturity.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_maturity, use_container_width=True)
    
    # IMPROVED: Channel Performance by Category
    st.subheader("📦 Channel Performance by Category")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        cat_channel_view = st.radio("View:", ['Grouped', 'Stacked', 'Heatmap'], key='cat_channel_view')
    
    cat_channel = filtered_df.groupby(['Product Category', 'Channel'])['Sales_EUR'].sum().reset_index()
    
    with col1:
        if cat_channel_view == 'Heatmap':
            cat_channel_pivot = cat_channel.pivot(index='Product Category', columns='Channel', values='Sales_EUR').fillna(0)
            
            fig_cat_channel = px.imshow(
                cat_channel_pivot,
                labels=dict(x="Channel", y="Category", color="Sales (EUR)"),
                title='Channel × Category Sales Heatmap',
                aspect='auto',
                color_continuous_scale='Viridis'
            )
            fig_cat_channel.update_xaxes(side="top")
        else:
            fig_cat_channel = px.bar(
                cat_channel,
                x='Product Category',
                y='Sales_EUR',
                color='Channel',
                title=f'Sales by Category & Channel ({cat_channel_view})',
                barmode='group' if cat_channel_view == 'Grouped' else 'stack',
                labels={'Sales_EUR': 'Sales (EUR)'},
                color_discrete_map={'Online': '#667eea', 'Store': '#764ba2'}
            )
            fig_cat_channel.update_traces(hovertemplate='<b>%{fullData.name}</b><br>%{x}<br>Sales: €%{y:,.0f}<extra></extra>')
            fig_cat_channel.update_xaxes(tickangle=-45)
        
        fig_cat_channel.update_layout(height=500, plot_bgcolor='white')
        st.plotly_chart(fig_cat_channel, use_container_width=True)
    
    # Identify channel-category affinity
    cat_channel_pct = cat_channel.groupby('Product Category').apply(
        lambda x: x.assign(Channel_Pct=(x['Sales_EUR'] / x['Sales_EUR'].sum()) * 100)
    ).reset_index(drop=True)
    
    online_dominant = cat_channel_pct[cat_channel_pct['Channel'] == 'Online'].nlargest(3, 'Channel_Pct')
    store_dominant = cat_channel_pct[cat_channel_pct['Channel'] == 'Store'].nlargest(3, 'Channel_Pct')
    
    st.markdown(f"""
    <div class="insight-box">
    <strong>🎯 Channel Affinity Insights:</strong><br><br>
    <strong>Online-Dominant Categories:</strong><br>
    {'<br>'.join([f"• {row['Product Category']}: {row['Channel_Pct']:.1f}% online" for _, row in online_dominant.iterrows()])}<br><br>
    <strong>Store-Dominant Categories:</strong><br>
    {'<br>'.join([f"• {row['Product Category']}: {row['Channel_Pct']:.1f}% in-store" for _, row in store_dominant.iterrows()])}
    </div>
    """, unsafe_allow_html=True)
    
    # IMPROVED: Omnichannel Diagnostic
    st.subheader("🔄 Omnichannel Diagnostic Dashboard")
    
    if year_range[1] > year_range[0]:
        online_diag = []
        for country in filtered_df['Country'].unique():
            country_data = filtered_df[filtered_df['Country'] == country]
            
            channel_sales = country_data.groupby('Channel')['Sales_EUR'].sum()
            online_share = 0
            if 'Online' in channel_sales.index and channel_sales.sum() > 0:
                online_share = (channel_sales['Online'] / channel_sales.sum()) * 100
            
            current_sales = country_data[country_data['Year'] == year_range[1]]['Sales_EUR'].sum()
            prev_sales = country_data[country_data['Year'] == year_range[1] - 1]['Sales_EUR'].sum()
            yoy = ((current_sales - prev_sales) / prev_sales * 100) if prev_sales > 0 else np.nan
            
            # Calculate online growth separately
            online_current = country_data[(country_data['Year'] == year_range[1]) & (country_data['Channel'] == 'Online')]['Sales_EUR'].sum()
            online_prev = country_data[(country_data['Year'] == year_range[1] - 1) & (country_data['Channel'] == 'Online')]['Sales_EUR'].sum()
            online_growth = ((online_current - online_prev) / online_prev * 100) if online_prev > 0 else np.nan
            
            online_diag.append({
                'Country': country,
                'Online_Share_%': online_share,
                'YoY_Growth_%': yoy,
                'Online_Growth_%': online_growth,
                'Total_Sales': current_sales
            })
        
        online_diag_df = pd.DataFrame(online_diag).dropna()
        
        # Classify into quadrants
        median_online_share = online_diag_df['Online_Share_%'].median()
        median_growth = online_diag_df['YoY_Growth_%'].median()
        
        def classify_omni_quadrant(row):
            if row['Online_Share_%'] >= median_online_share and row['YoY_Growth_%'] >= median_growth:
                return '🚀 Digital Champions'
            elif row['Online_Share_%'] >= median_online_share and row['YoY_Growth_%'] < median_growth:
                return '⚠️ Mature Digital'
            elif row['Online_Share_%'] < median_online_share and row['YoY_Growth_%'] >= median_growth:
                return '📈 Growing Traditional'
            else:
                return '🔴 At Risk'
        
        online_diag_df['Segment'] = online_diag_df.apply(classify_omni_quadrant, axis=1)
        
        color_map_omni = {
            '🚀 Digital Champions': '#10b981',
            '⚠️ Mature Digital': '#f59e0b',
            '📈 Growing Traditional': '#3b82f6',
            '🔴 At Risk': '#ef4444'
        }
        
        fig_omni_diag = px.scatter(
            online_diag_df,
            x='Online_Share_%',
            y='YoY_Growth_%',
            size='Total_Sales',
            hover_name='Country',
            hover_data={'Online_Share_%': ':.1f', 'YoY_Growth_%': ':.1f', 'Online_Growth_%': ':.1f', 'Total_Sales': ':,.0f'},
            title='Omnichannel Strategic Quadrant Analysis',
            labels={'Online_Share_%': 'Online Penetration (%)', 'YoY_Growth_%': 'Total Growth (%)'},
            color='Segment',
            color_discrete_map=color_map_omni,
            size_max=50
        )
        
        # Add quadrant lines
        fig_omni_diag.add_hline(y=median_growth, line_dash="dash", line_color="gray", line_width=2)
        fig_omni_diag.add_vline(x=median_online_share, line_dash="dash", line_color="gray", line_width=2)
        
        # Add quadrant labels
        max_online = online_diag_df['Online_Share_%'].max()
        max_growth = online_diag_df['YoY_Growth_%'].max()
        min_growth = online_diag_df['YoY_Growth_%'].min()
        
        fig_omni_diag.add_annotation(x=max_online*0.8, y=max_growth*0.8, text="🚀 Digital<br>Champions", 
                            showarrow=False, font=dict(size=12, color='#10b981'), bgcolor='rgba(16, 185, 129, 0.15)', borderpad=8)
        fig_omni_diag.add_annotation(x=max_online*0.8, y=min_growth*0.8, text="⚠️ Mature<br>Digital", 
                            showarrow=False, font=dict(size=12, color='#f59e0b'), bgcolor='rgba(245, 158, 11, 0.15)', borderpad=8)
        fig_omni_diag.add_annotation(x=max_online*0.2, y=max_growth*0.8, text="📈 Growing<br>Traditional", 
                            showarrow=False, font=dict(size=12, color='#3b82f6'), bgcolor='rgba(59, 130, 246, 0.15)', borderpad=8)
        fig_omni_diag.add_annotation(x=max_online*0.2, y=min_growth*0.8, text="🔴 At<br>Risk", 
                            showarrow=False, font=dict(size=12, color='#ef4444'), bgcolor='rgba(239, 68, 68, 0.15)', borderpad=8)
        
        fig_omni_diag.update_traces(hovertemplate='<b>%{hovertext}</b><br>Online Share: %{x:.1f}%<br>Total Growth: %{y:.1f}%<br>Online Growth: %{customdata[2]:.1f}%<br>Sales: €%{customdata[3]:,.0f}<extra></extra>')
        fig_omni_diag.update_layout(height=600, plot_bgcolor='white',
                                   xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                                   yaxis=dict(showgrid=True, gridcolor='#f0f0f0'))
        
        st.plotly_chart(fig_omni_diag, use_container_width=True)
        
        # Segment summary
        col1, col2, col3, col4 = st.columns(4)
        
        for col, (segment, color) in zip([col1, col2, col3, col4], color_map_omni.items()):
            segment_data = online_diag_df[online_diag_df['Segment'] == segment]
            with col:
                st.markdown(f"""
                <div style='background-color: {color}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {color};'>
                    <div style='font-size: 0.9rem; font-weight: 600; color: {color};'>{segment}</div>
                    <div style='font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;'>{len(segment_data)}</div>
                    <div style='font-size: 0.8rem; color: #666;'>markets</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>🎯 Omnichannel Strategy Recommendations:</strong><br>
        • <strong>🚀 Digital Champions:</strong> Showcase markets - replicate success patterns globally<br>
        • <strong>⚠️ Mature Digital:</strong> Focus on innovation and customer experience to reignite growth<br>
        • <strong>📈 Growing Traditional:</strong> Accelerate digital transformation while maintaining momentum<br>
        • <strong>🔴 At Risk:</strong> Urgent intervention required - consider market viability assessment
        </div>
        """, unsafe_allow_html=True)

# TAB 5: Pricing & Volume Drivers
with tab5:
    st.header("Pricing & Volume Intelligence")
    
    # Price vs Volume bubble
    st.subheader("💎 Price-Volume-Sales Relationship")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        bubble_dimension = st.radio("Analyze by:", ['Product Category', 'Country', 'Region'], key='bubble_dim')
    
    cat_pv = filtered_df.groupby(bubble_dimension).agg({
        'ASP_EUR': 'mean',
        'Units_Sold': 'sum',
        'Sales_EUR': 'sum'
    }).reset_index()
    
    with col1:
        fig_pv_bubble = px.scatter(
            cat_pv,
            x='ASP_EUR',
            y='Units_Sold',
            size='Sales_EUR',
            hover_name=bubble_dimension,
            title=f'Price vs Volume Analysis by {bubble_dimension}',
            labels={'ASP_EUR': 'Average Selling Price (EUR)', 'Units_Sold': 'Total Units Sold'},
            color='ASP_EUR',
            color_continuous_scale='Plasma',
            size_max=60
        )
        fig_pv_bubble.update_traces(hovertemplate='<b>%{hovertext}</b><br>ASP: €%{x:.2f}<br>Units: %{y:,.0f}<br>Sales: €%{marker.size:,.0f}<extra></extra>')
        fig_pv_bubble.update_layout(height=500, plot_bgcolor='white',
                                   xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                                   yaxis=dict(showgrid=True, gridcolor='#f0f0f0'))
        
        st.plotly_chart(fig_pv_bubble, use_container_width=True)
    
    # Price/Volume decomposition
    st.subheader("📊 Price & Volume Effect Decomposition")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        decomp_level = st.selectbox("Analyze by:", ['Region', 'Country', 'Product Category'], key='decomp_level')
        decomp_view = st.radio("View:", ['Stacked', 'Grouped', 'Separate'], key='decomp_view')
    
    if year_range[1] > year_range[0]:
        decomp_data = compute_price_volume_decomp(filtered_df, [decomp_level])
        decomp_latest = decomp_data[decomp_data['Year'] == year_range[1]].nlargest(top_n, 'Total_Change', keep='all')
        
        with col1:
            if decomp_view == 'Separate':
                fig_decomp = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Volume Effect', 'Price Effect'),
                    shared_yaxes=True
                )
                
                fig_decomp.add_trace(
                    go.Bar(x=decomp_latest['Volume_Effect'], y=decomp_latest[decomp_level], 
                          orientation='h', name='Volume', marker_color='steelblue',
                          hovertemplate='<b>%{y}</b><br>Volume Effect: €%{x:,.0f}<extra></extra>'),
                    row=1, col=1
                )
                
                fig_decomp.add_trace(
                    go.Bar(x=decomp_latest['Price_Effect'], y=decomp_latest[decomp_level], 
                          orientation='h', name='Price', marker_color='coral',
                          hovertemplate='<b>%{y}</b><br>Price Effect: €%{x:,.0f}<extra></extra>'),
                    row=1, col=2
                )
                
                fig_decomp.update_layout(height=500, showlegend=False, plot_bgcolor='white')
            else:
                fig_decomp = go.Figure()
                fig_decomp.add_trace(go.Bar(
                    x=decomp_latest[decomp_level],
                    y=decomp_latest['Volume_Effect'],
                    name='Volume Effect',
                    marker_color='steelblue',
                    hovertemplate='<b>%{x}</b><br>Volume Effect: €%{y:,.0f}<extra></extra>'
                ))
                fig_decomp.add_trace(go.Bar(
                    x=decomp_latest[decomp_level],
                    y=decomp_latest['Price_Effect'],
                    name='Price Effect',
                    marker_color='coral',
                    hovertemplate='<b>%{x}</b><br>Price Effect: €%{y:,.0f}<extra></extra>'
                ))
                
                fig_decomp.update_layout(
                    barmode='stack' if decomp_view == 'Stacked' else 'group',
                    xaxis_title=decomp_level,
                    yaxis_title='Effect (EUR)',
                    height=500,
                    plot_bgcolor='white',
                    xaxis=dict(tickangle=-45)
                )
            
            fig_decomp.update_layout(
                title=f'Sales Change Decomposition: {year_range[1]-1} → {year_range[1]}'
            )
            
            st.plotly_chart(fig_decomp, use_container_width=True)
        
        # Analysis summary
        volume_led = decomp_latest[decomp_latest['Volume_Effect'] > decomp_latest['Price_Effect']]
        price_led = decomp_latest[decomp_latest['Price_Effect'] > decomp_latest['Volume_Effect']]
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>💡 Growth Driver Analysis:</strong><br>
        • <strong>{len(volume_led)}</strong> {decomp_level.lower()}s are <strong>volume-led</strong> (healthy, sustainable growth)<br>
        • <strong>{len(price_led)}</strong> {decomp_level.lower()}s are <strong>price-led</strong> (may face volume pressure)<br>
        • <strong>Recommendation:</strong> {'Focus on volume acceleration in price-led segments' if len(price_led) > len(volume_led) else 'Maintain balanced growth strategy'}
        </div>
        """, unsafe_allow_html=True)
    
    # Promotion pressure proxy
    st.subheader("⚠️ Promotional Pressure Signals")
    
    if year_range[1] > year_range[0]:
        promo_signals = []
        
        for cat in filtered_df['Product Category'].unique():
            cat_data = filtered_df[filtered_df['Product Category'] == cat]
            
            current_asp = cat_data[cat_data['Year'] == year_range[1]]['ASP_EUR'].mean()
            prev_asp = cat_data[cat_data['Year'] == year_range[1] - 1]['ASP_EUR'].mean()
            
            current_units = cat_data[cat_data['Year'] == year_range[1]]['Units_Sold'].sum()
            prev_units = cat_data[cat_data['Year'] == year_range[1] - 1]['Units_Sold'].sum()
            
            asp_change = ((current_asp - prev_asp) / prev_asp * 100) if prev_asp > 0 else 0
            unit_change = ((current_units - prev_units) / prev_units * 100) if prev_units > 0 else 0
            
            if asp_change < -5 and unit_change > 5:
                signal = '🔴 High'
                priority = 1
            elif asp_change < 0 and unit_change > 0:
                signal = '🟡 Moderate'
                priority = 2
            else:
                signal = '🟢 Low'
                priority = 3
            
            promo_signals.append({
                'Category': cat,
                'ASP_Change_%': asp_change,
                'Unit_Change_%': unit_change,
                'Signal': signal,
                'Priority': priority
            })
        
        promo_df = pd.DataFrame(promo_signals).sort_values(['Priority', 'ASP_Change_%'])
        
        fig_promo = px.scatter(
            promo_df,
            x='ASP_Change_%',
            y='Unit_Change_%',
            size=abs(promo_df['ASP_Change_%']),
            color='Signal',
            hover_name='Category',
            title='Promotional Pressure Matrix',
            labels={'ASP_Change_%': 'ASP Change (%)', 'Unit_Change_%': 'Unit Volume Change (%)'},
            color_discrete_map={'🔴 High': '#ef4444', '🟡 Moderate': '#f59e0b', '🟢 Low': '#10b981'}
        )
        
        fig_promo.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
        fig_promo.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1)
        
        fig_promo.update_traces(hovertemplate='<b>%{hovertext}</b><br>ASP Change: %{x:.1f}%<br>Unit Change: %{y:.1f}%<extra></extra>')
        fig_promo.update_layout(height=500, plot_bgcolor='white',
                               xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                               yaxis=dict(showgrid=True, gridcolor='#f0f0f0'))
        
        st.plotly_chart(fig_promo, use_container_width=True)
        
        # Show detailed table
        with st.expander("📋 View Detailed Signals"):
            st.dataframe(
                promo_df[['Category', 'ASP_Change_%', 'Unit_Change_%', 'Signal']].style.format({
                    'ASP_Change_%': '{:.2f}%',
                    'Unit_Change_%': '{:.2f}%'
                }),
                use_container_width=True
            )

# TAB 6: Explorer
with tab6:
    st.header("📊 Data Explorer & Custom Analytics")
    
    st.markdown("### Build Your Custom Pivot Table")
    st.markdown("Create ad-hoc analysis by selecting dimensions and metrics:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        row_dims = st.multiselect(
            "📊 Row Dimensions",
            ['Year', 'Region', 'Country', 'Gender', 'Product Category', 'Channel'],
            default=['Region'],
            help="Select dimensions to display as rows"
        )
    
    with col2:
        col_dims = st.multiselect(
            "📈 Column Dimensions",
            ['Year', 'Region', 'Country', 'Gender', 'Product Category', 'Channel'],
            default=['Year'],
            help="Select dimensions to display as columns"
        )
    
    with col3:
        value_metric = st.selectbox(
            "💰 Value Metric",
            ['Sales_EUR', 'Units_Sold', 'ASP_EUR'],
            key='pivot_metric',
            help="Select the metric to aggregate"
        )
        
        agg_func = st.selectbox(
            "🔢 Aggregation Function",
            ['sum', 'mean', 'median', 'count', 'min', 'max'],
            key='pivot_agg',
            help="How to aggregate the selected metric"
        )
    
    if row_dims and col_dims:
        try:
            pivot_table = filtered_df.pivot_table(
                index=row_dims,
                columns=col_dims,
                values=value_metric,
                aggfunc=agg_func,
                fill_value=0
            )
            
            st.markdown("---")
            st.subheader("📋 Pivot Table Results")
            
            # Format the pivot table
            if value_metric == 'Sales_EUR':
                styled_pivot = pivot_table.style.format('€{:,.0f}')
            elif value_metric == 'Units_Sold':
                styled_pivot = pivot_table.style.format('{:,.0f}')
            else:
                styled_pivot = pivot_table.style.format('€{:.2f}')

            st.dataframe(styled_pivot, use_container_width=True, height=400)

            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", format_number(pivot_table.values.sum()))
            with col2:
                st.metric("Average", format_number(pivot_table.values.mean()))
            with col3:
                st.metric("Minimum", format_number(pivot_table.values.min()))
            with col4:
                st.metric("Maximum", format_number(pivot_table.values.max()))
            
            # Export options
            csv = pivot_table.to_csv()
            st.download_button(
                label="📥 Download Pivot as CSV",
                data=csv,
                file_name=f'zara_pivot_{value_metric}_{agg_func}.csv',
                mime='text/csv',
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Unable to create pivot: {str(e)}")
            st.info("💡 Tip: Try selecting different dimensions or check if your data supports the selected combination.")
    else:
        st.info("👈 Select at least one row and one column dimension to generate the pivot table.")

# TAB 7: Data Quality & Method
with tab7:
    st.header("📋 Data Quality & Methodology")
    
    # Source level reconciliation
    st.subheader("📊 Data Source Reconciliation")
    
    source_recon = df.groupby(['Year', 'Source_Level']).agg({
        'Sales_EUR': 'sum',
        'Units_Sold': 'sum'
    }).reset_index()
    
    source_recon['Sales_Share_%'] = source_recon.groupby('Year')['Sales_EUR'].transform(
        lambda x: (x / x.sum()) * 100
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(source_recon.style.format({
            'Sales_EUR': '€{:,.0f}',
            'Units_Sold': '{:,.0f}',
            'Sales_Share_%': '{:.1f}%'
        }), use_container_width=True)
    
    with col2:
        fig_source = px.bar(
            source_recon,
            x='Year',
            y='Sales_Share_%',
            color='Source_Level',
            title='Data Source Composition by Year',
            barmode='stack',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_source.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Share: %{y:.1f}%<extra></extra>')
        fig_source.update_layout(height=400, plot_bgcolor='white')
        st.plotly_chart(fig_source, use_container_width=True)
    
    # Completeness checks
    st.subheader("✅ Data Completeness Assessment")
    
    completeness = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_%': (df.isnull().sum().values / len(df) * 100).round(2),
        'Data_Type': df.dtypes.values
    })
    completeness = completeness[completeness['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if not completeness.empty:
        fig_completeness = px.bar(
            completeness,
            x='Column',
            y='Missing_%',
            title='Data Completeness by Column',
            color='Missing_%',
            color_continuous_scale='Reds',
            text='Missing_%'
        )
        fig_completeness.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_completeness.update_layout(height=400, showlegend=False, plot_bgcolor='white')
        st.plotly_chart(fig_completeness, use_container_width=True)
        
        st.dataframe(completeness, use_container_width=True)
    else:
        st.success("✅ Excellent! No missing values detected in the dataset.")
    
    # Method notes
    st.subheader("📖 Methodology & Data Source Documentation")
    
    with st.expander("📄 Data Source Definitions", expanded=True):
        method_notes = df[['Source_Level', 'Method_Note']].drop_duplicates()
        for _, row in method_notes.iterrows():
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid #667eea;'>
                <strong style='color: #667eea;'>{row['Source_Level']}</strong><br>
                <span style='color: #555;'>{row['Method_Note']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Guardrails
    st.subheader("⚠️ Data Quality Guardrails")
    
    warnings = []
    recommendations = []
    
    # Check sample size
    if len(filtered_df) < 100:
        warnings.append(("Small Sample Size", f"Current filters return only {len(filtered_df)} records. Results may not be statistically representative."))
        recommendations.append("Consider broadening your filter criteria to include more data.")
    
    # Check ASP outliers
    asp_q1 = filtered_df['ASP_EUR'].quantile(0.25)
    asp_q3 = filtered_df['ASP_EUR'].quantile(0.75)
    asp_iqr = asp_q3 - asp_q1
    asp_outliers = filtered_df[(filtered_df['ASP_EUR'] < (asp_q1 - 3 * asp_iqr)) | 
                                 (filtered_df['ASP_EUR'] > (asp_q3 + 3 * asp_iqr))]
    
    if len(asp_outliers) > len(filtered_df) * 0.05:
        warnings.append(("ASP Outliers Detected", f"{len(asp_outliers)} records ({len(asp_outliers)/len(filtered_df)*100:.1f}%) have extreme ASP values."))
        recommendations.append("Review pricing data for potential data entry errors or unusual transactions.")
    
    # Check for zero sales
    zero_sales = filtered_df[filtered_df['Sales_EUR'] == 0]
    if len(zero_sales) > 0:
        warnings.append(("Zero Sales Records", f"{len(zero_sales)} records have zero sales values."))
        recommendations.append("These may represent returns, discontinued products, or data quality issues.")
    
    # Check year distribution
    year_dist = filtered_df['Year'].value_counts()
    if year_dist.std() / year_dist.mean() > 0.5:
        warnings.append(("Uneven Year Distribution", "Data is unevenly distributed across years."))
        recommendations.append("Consider whether this reflects business reality or data collection issues.")
    
    if warnings:
        for title, message in warnings:
            st.warning(f"**{title}:** {message}")
        
        st.markdown("### 💡 Recommendations")
        for rec in recommendations:
            st.markdown(f"• {rec}")
    else:
        st.success("✅ All data quality checks passed! No significant issues detected with current filter selection.")
    
    # Dataset statistics
    st.subheader("📊 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Filtered Records", f"{len(filtered_df):,}")
    with col3:
        st.metric("Date Range", f"{df['Year'].min()}-{df['Year'].max()}")
    with col4:
        st.metric("Countries", f"{df['Country'].nunique()}")
    st.subheader("🔍 Filtered Data Preview")
    
    preview_rows = st.slider("Number of rows to display", 10, 100, 50, key='preview_rows')
    
    st.dataframe(
    promo_df[['Category', 'ASP_Change_%', 'Unit_Change_%', 'Signal']].style.format({
        'ASP_Change_%': '{:.2f}%',
        'Unit_Change_%': '{:.2f}%'
    }),
    use_container_width=True
)
    
    st.caption(f"Showing first {preview_rows} of {len(filtered_df):,} filtered records")
    
    # Download full filtered dataset
    csv_full = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Filtered Dataset as CSV",
        data=csv_full,
        file_name='zara_filtered_data.csv',
        mime='text/csv',
        use_container_width=True
    )