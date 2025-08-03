import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page setup
st.set_page_config(page_title="NYC 311 Dashboard", page_icon="🏙️", layout="wide")
st.title("🏙️ NYC 311 Call Volume Dashboard")

# Load data
@st.cache_data
def load_data():
    df_train, df_test, predictions, anomalies = None, None, None, None
    
    try:
        df_train = pd.read_csv('data/train.csv')
        df_train['created_date'] = pd.to_datetime(df_train['created_date'])
        st.success("✅ Training data loaded")
    except:
        st.warning("⚠️ Training data not found at data/train.csv")
    
    try:
        df_test = pd.read_csv('data/test.csv')
        df_test['created_date'] = pd.to_datetime(df_test['created_date'])
        st.success("✅ Test data loaded")
    except:
        st.warning("⚠️ Test data not found at data/test.csv")
    
    try:
        predictions = pd.read_csv('submission.csv')
        predictions['date'] = pd.to_datetime(predictions['date'])
        st.success("✅ Predictions loaded")
        
    except:
        st.warning("⚠️ Predictions not found at submission.csv")
    
    try:
        anomalies = pd.read_csv('anomalies.csv')
        anomalies['Date'] = pd.to_datetime(anomalies['Date'])
        st.success("✅ Anomalies loaded")
    except:
        st.warning("⚠️ Anomalies not found at anomalies.csv")
    
    return df_train, df_test, predictions, anomalies

df_train, df_test, predictions, anomalies = load_data()

st.info(" File Status - Make sure you have run the prediction and anomaly scripts first!")

if df_train is not None or df_test is not None:
    available_data = []
    if df_train is not None:
        available_data.append(df_train)
    if df_test is not None:
        available_data.append(df_test)
    
    df_all = pd.concat(available_data) if available_data else pd.DataFrame()
    
    
    st.sidebar.header("Filters")
    
    start_date = st.sidebar.date_input("Start Date", df_all['created_date'].min().date())
    end_date = st.sidebar.date_input("End Date", df_all['created_date'].max().date())
    
    if 'borough' in df_all.columns:
        boroughs = ['All'] + list(df_all['borough'].dropna().unique())
        borough = st.sidebar.selectbox("Borough", boroughs)
    else:
        borough = 'All'
    
    # Filter data
    filtered_data = df_all[
        (df_all['created_date'].dt.date >= start_date) & 
        (df_all['created_date'].dt.date <= end_date)
    ]
    
    if borough != 'All' and 'borough' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['borough'] == borough]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Calls", f"{len(filtered_data):,}")
    with col2:
        daily_avg = len(filtered_data) / max(1, (end_date - start_date).days + 1)
        st.metric("Daily Average", f"{daily_avg:.0f}")
    with col3:
        if predictions is not None and 'Predicted_Total_Calls' in predictions.columns:
            st.metric("Predicted Avg", f"{predictions['Predicted_Total_Calls'].mean():.0f}")
        else:
            st.metric("Predicted Avg", "N/A")
    with col4:
        if anomalies is not None:
            st.metric("Anomalies", len(anomalies))
        else:
            st.metric("Anomalies", "N/A")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Map", "Trends", "Anomalies"])
    
    with tab1:
        st.subheader("NYC Call Distribution")
        
        if 'latitude' in filtered_data.columns and 'longitude' in filtered_data.columns:
            map_data = filtered_data.dropna(subset=['latitude', 'longitude']).sample(n=min(5000, len(filtered_data)))
            if len(map_data) > 0:
                st.map(map_data[['latitude', 'longitude']])
        
        if 'borough' in filtered_data.columns:
            borough_counts = filtered_data['borough'].value_counts()
            fig = px.bar(x=borough_counts.index, y=borough_counts.values, 
                        title="Calls by Borough")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Call Volume Trends")
        
        filtered_data['date'] = filtered_data['created_date'].dt.date
        daily_counts = filtered_data.groupby('date').size().reset_index()
        daily_counts.columns = ['date', 'calls']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_counts['date'], y=daily_counts['calls'], 
                                name='Actual', line=dict(color='blue')))
        
        if predictions is not None and 'Predicted_Total_Calls' in predictions.columns:
            fig.add_trace(go.Scatter(x=predictions['date'], y=predictions['Predicted_Total_Calls'], 
                                     name='Predicted', line=dict(color='red', dash='dash')))
        
        fig.update_layout(title="Daily Call Volume", xaxis_title="Date", yaxis_title="Calls")
        st.plotly_chart(fig, use_container_width=True)
        
        dow_data = filtered_data['created_date'].dt.day_name().value_counts()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_data = dow_data.reindex(dow_order)
        
        fig2 = px.bar(x=dow_data.index, y=dow_data.values, title="Calls by Day of Week")
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.subheader("Detected Anomalies")
        
        if anomalies is not None and len(anomalies) > 0:
            display_anomalies = anomalies.copy()
            display_anomalies['Type'] = display_anomalies['Anomaly_Score'].apply(
                lambda x: '🔴 Spike' if x > 0 else '🔵 Dip'
            )
            st.dataframe(display_anomalies[['Date', 'Type', 'Actual', 'Expected', 'Anomaly_Score', 'Note']])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=anomalies['Date'], y=anomalies['Expected'], 
                                     name='Expected', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=anomalies['Date'], y=anomalies['Actual'], 
                                     mode='markers', name='Actual', 
                                     marker=dict(color='red', size=10)))
            fig.update_layout(title="Anomalies: Expected vs Actual")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomalies detected")

else:
    st.error(" No data files found!")
    st.markdown("""
    **To fix this, make sure you have:**
    
    1. **Run your prediction notebook** to create `submission.csv`
    2. **Run your anomaly detection script** to create `anomalies.csv`  
    3. **Place train.csv and test.csv** in a `data/` folder
    
    **Expected file structure:**
    ```
    NYC 311 Koyiljon/
    ├──  app.py (this file)
    ├──  submission.csv
    ├──  anomalies.csv
    └──  data/
        ├──  train.csv
        └──  test.csv
    ```
    """)
    
    st.subheader(" Demo Dashboard (Sample Data)")
    import numpy as np
    dates = pd.date_range('2024-05-01', '2024-05-30')
    sample_calls = np.random.randint(8000, 12000, len(dates))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=sample_calls, name='Sample Calls'))
    fig.update_layout(title="Sample: Daily Call Volume", xaxis_title="Date", yaxis_title="Calls")
    st.plotly_chart(fig, use_container_width=True)
