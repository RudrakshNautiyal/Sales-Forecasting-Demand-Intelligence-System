import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Sales Forecasting Dashboard', layout='wide')

@st.cache_data
def load_data():
    df      = pd.read_csv('Datasets/train_processed.csv', parse_dates=['Order Date'])
    monthly = pd.read_csv('Datasets/monthly_sales.csv', parse_dates=['Month'])
    weekly  = pd.read_csv('Datasets/weekly_sales.csv',  parse_dates=['Week'])
    return df, monthly, weekly

df, monthly, weekly = load_data()

page = st.sidebar.radio('Navigate', ['Sales Overview', 'Forecast Explorer', 'Anomaly Report', 'Product Demand Segments'])

if page == 'Sales Overview':
    st.title('Sales Overview Dashboard')

    region_filter   = st.sidebar.multiselect('Region',   df['Region'].unique(),   default=list(df['Region'].unique()))
    category_filter = st.sidebar.multiselect('Category', df['Category'].unique(), default=list(df['Category'].unique()))

    filtered = df[df['Region'].isin(region_filter) & df['Category'].isin(category_filter)]

    col1, col2, col3 = st.columns(3)
    col1.metric('Total Sales',   f"${filtered['Sales'].sum():,.0f}")
    col2.metric('Total Orders',  f"{filtered['Order ID'].nunique():,}")
    col3.metric('Avg Order Value', f"${filtered['Sales'].mean():,.2f}")

    st.subheader('Total Sales by Year')
    yearly = filtered.groupby('Year')['Sales'].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(yearly.index, yearly.values, color='#4C72B0', width=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))
    for i, (yr, val) in enumerate(yearly.items()):
        ax.text(yr, val+1000, f'${val:,.0f}', ha='center', fontsize=9)
    ax.set_xlabel('Year')
    ax.set_ylabel('Sales ($)')
    st.pyplot(fig)

    st.subheader('Monthly Sales Trend')
    monthly_filtered = filtered.groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum()
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(monthly_filtered.index, monthly_filtered.values, color='#4C72B0', linewidth=1.8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))
    ax2.set_ylabel('Sales ($)')
    st.pyplot(fig2)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Sales by Region')
        region_sales = filtered.groupby('Region')['Sales'].sum().sort_values()
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.barh(region_sales.index, region_sales.values, color='#55A868')
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))
        st.pyplot(fig3)

    with col2:
        st.subheader('Sales by Category')
        cat_sales = filtered.groupby('Category')['Sales'].sum().sort_values()
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.barh(cat_sales.index, cat_sales.values, color='#C44E52')
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))
        st.pyplot(fig4)

elif page == 'Forecast Explorer':
    st.title('Forecast Explorer')

    segment_type = st.selectbox('Select Segment', ['Overall', 'Furniture', 'Technology', 'Office Supplies', 'West', 'East'])
    horizon      = st.slider('Forecast Horizon (months)', 1, 3, 3)

    if segment_type == 'Overall':
        data = monthly.set_index('Month')['Total_Sales']
    elif segment_type in ['Furniture', 'Technology', 'Office Supplies']:
        data = df[df['Category'] == segment_type].groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum()
    else:
        data = df[df['Region'] == segment_type].groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum()

    data = data.reset_index()
    data.columns = ['ds', 'y']

    train = data.iloc[:-3]
    test  = data.iloc[-3:]

    m        = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(train)
    future   = m.make_future_dataframe(periods=horizon, freq='MS')
    forecast = m.predict(future)

    pred = forecast.iloc[-horizon:]['yhat'].values
    mae  = mean_absolute_error(test['y'].values[:horizon], pred[:horizon])
    rmse = np.sqrt(mean_squared_error(test['y'].values[:horizon], pred[:horizon]))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data['ds'], data['y'], label='Actual', color='#4C72B0', linewidth=1.8)
    ax.plot(forecast.iloc[-horizon:]['ds'], pred, label='Forecast', color='#C44E52', marker='o', linewidth=2)
    ax.fill_between(forecast.iloc[-horizon:]['ds'],
                    forecast.iloc[-horizon:]['yhat_lower'],
                    forecast.iloc[-horizon:]['yhat_upper'],
                    alpha=0.2, color='#C44E52')
    ax.set_title(f'{segment_type} — {horizon}-Month Forecast')
    ax.set_ylabel('Sales ($)')
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.metric('MAE',  f"${mae:,.2f}")
    col2.metric('RMSE', f"${rmse:,.2f}")

elif page == 'Anomaly Report':
    st.title('Anomaly Report')

    w = weekly.set_index('Week').copy()

    iso = IsolationForest(contamination=0.05, random_state=42)
    w['iso_flag'] = iso.fit_predict(w[['Total_Sales']]) == -1

    w['rolling_mean'] = w['Total_Sales'].rolling(4).mean()
    w['rolling_std']  = w['Total_Sales'].rolling(4).std()
    w['zscore']       = (w['Total_Sales'] - w['rolling_mean']) / w['rolling_std']
    w['z_flag']       = w['zscore'].abs() > 2

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(w.index, w['Total_Sales'], color='#4C72B0', linewidth=1.2, label='Weekly Sales')
    axes[0].scatter(w[w['iso_flag']].index, w[w['iso_flag']]['Total_Sales'],
                    color='#C44E52', zorder=5, s=60, label='Anomaly')
    axes[0].set_title('Isolation Forest Anomalies')
    axes[0].legend()
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))

    axes[1].plot(w.index, w['Total_Sales'], color='#4C72B0', linewidth=1.2, label='Weekly Sales')
    axes[1].scatter(w[w['z_flag']].index, w[w['z_flag']]['Total_Sales'],
                    color='#55A868', zorder=5, s=60, label='Anomaly')
    axes[1].set_title('Z-Score Anomalies')
    axes[1].legend()
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}K'))

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader('Detected Anomaly Dates')
    col1, col2 = st.columns(2)
    with col1:
        st.write('**Isolation Forest**')
        st.dataframe(w[w['iso_flag']][['Total_Sales']].reset_index().rename(columns={'Week':'Date','Total_Sales':'Sales ($)'}))
    with col2:
        st.write('**Z-Score**')
        st.dataframe(w[w['z_flag']][['Total_Sales','zscore']].reset_index().rename(columns={'Week':'Date','Total_Sales':'Sales ($)'}))

elif page == 'Product Demand Segments':
    st.title('Product Demand Segments')

    total_sales = df.groupby('Sub-Category')['Sales'].sum()
    avg_order   = df.groupby('Sub-Category')['Sales'].mean()
    volatility  = df.groupby(['Sub-Category', pd.Grouper(key='Order Date', freq='MS')])['Sales'].sum().groupby('Sub-Category').std()
    yearly      = df.groupby(['Sub-Category', 'Year'])['Sales'].sum().unstack()
    growth      = yearly.pct_change(axis=1).iloc[:, 1:].mean(axis=1)

    features = pd.DataFrame({
        'Total_Sales' : total_sales,
        'Avg_Order'   : avg_order,
        'Volatility'  : volatility,
        'Growth_Rate' : growth
    }).dropna()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    features['Cluster'] = km.fit_predict(X_scaled)

    cluster_summary = features.groupby('Cluster').mean()
    label_map = {}
    for cluster in features['Cluster'].unique():
        c = cluster_summary.loc[cluster]
        if c['Total_Sales'] > cluster_summary['Total_Sales'].median() and c['Volatility'] < cluster_summary['Volatility'].median():
            label_map[cluster] = 'High Volume, Stable Demand'
        elif c['Growth_Rate'] > cluster_summary['Growth_Rate'].median():
            label_map[cluster] = 'Growing Demand'
        elif c['Total_Sales'] < cluster_summary['Total_Sales'].median() and c['Volatility'] > cluster_summary['Volatility'].median():
            label_map[cluster] = 'Low Volume, High Volatility'
        else:
            label_map[cluster] = 'Declining Demand'

    features['Label'] = features['Cluster'].map(label_map)

    pca              = PCA(n_components=2)
    coords           = pca.fit_transform(X_scaled)
    features['PCA1'] = coords[:, 0]
    features['PCA2'] = coords[:, 1]

    colors = ['#4C72B0', '#C44E52', '#55A868', '#8172B2']
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (label, group) in enumerate(features.groupby('Label')):
        ax.scatter(group['PCA1'], group['PCA2'], label=label, color=colors[i], s=100, zorder=3)
        for subcat, row in group.iterrows():
            ax.annotate(subcat, (row['PCA1'], row['PCA2']), fontsize=7.5,
                        xytext=(4, 4), textcoords='offset points')
    ax.set_title('Product Demand Segmentation (PCA)')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader('Sub-Category Cluster Assignments')
    st.dataframe(features[['Label', 'Total_Sales', 'Growth_Rate', 'Volatility']].sort_values('Label').reset_index())