import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from prophet import Prophet
import networkx as nx
from eventlet import event

# Generate synthetic data with more complexity
def generate_data():
    schools = [f"School_{i}" for i in range(1, 11)]  # 10 Schools
    time_range = [datetime.now() - timedelta(hours=i) for i in range(24)]  # Past 24 hours
    
    data = []
    for school in schools:
        for time in time_range:
            bandwidth_usage = np.random.normal(loc=50, scale=20)  # Normal distribution for more variation
            bandwidth_usage = max(1, min(150, bandwidth_usage))  # Keep within a realistic range
            data.append([school, time, bandwidth_usage])
    
    # Introduce anomalies manually
    for _ in range(10):  # Increase anomaly occurrences
        school = np.random.choice(schools)
        time = np.random.choice(time_range)
        bandwidth_usage = np.random.choice([1, 160])  # Extreme low or high values
        data.append([school, time, bandwidth_usage])
    
    df = pd.DataFrame(data, columns=["School", "Timestamp", "Bandwidth_Usage"])
    df.to_csv("network_data.csv", index=False)
    return df

# Load or generate data
df = generate_data()

def detect_anomalies(df):
    # Using IQR for anomaly detection
    Q1 = df['Bandwidth_Usage'].quantile(0.25)
    Q3 = df['Bandwidth_Usage'].quantile(0.75)
    IQR = Q3 - Q1
    anomalies = df[(df['Bandwidth_Usage'] < (Q1 - 1.5 * IQR)) | (df['Bandwidth_Usage'] > (Q3 + 1.5 * IQR))]
    return anomalies

def train_prophet(df):
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # Convert to datetime format
    df_numeric = df[["Timestamp", "Bandwidth_Usage"]]  # Select only numeric columns
    df_prophet = df_numeric.groupby("Timestamp").mean().reset_index()  # Average bandwidth per timestamp
    df_prophet.columns = ["ds", "y"]  # Prophet model requires columns: ds (date) and y (value)
    
    model = Prophet()
    model.fit(df_prophet)  # Train Prophet on past data
    
    future = model.make_future_dataframe(periods=5, freq="H")  # Predict next 5 hours
    forecast = model.predict(future)
    
    return forecast

def allocate_bandwidth(demand_predictions, total_bandwidth=500):
    total_demand = sum(demand_predictions.values())  # Calculate total demand
    allocation = {school: (demand / total_demand) * total_bandwidth for school, demand in demand_predictions.items()}
    return allocation

def sdn_load_balancer(network_graph, demand_predictions):
    shortest_paths = {}
    for school in demand_predictions.keys():
        shortest_paths[school] = nx.shortest_path(network_graph, source='Central_Node', target=school, weight='energy_efficiency')
    return shortest_paths

# Create a complex network topology
graph = nx.Graph()

graph.add_edges_from([
    ('Central_Node', 'Node_A', {'energy_efficiency': 1}), ('Central_Node', 'Node_B', {'energy_efficiency': 2}),
    ('Node_A', 'School_1', {'energy_efficiency': 1}), ('Node_A', 'School_2', {'energy_efficiency': 1}), ('Node_A', 'Node_C', {'energy_efficiency': 2}),
    ('Node_B', 'School_3', {'energy_efficiency': 1}), ('Node_B', 'School_4', {'energy_efficiency': 1}), ('Node_B', 'Node_D', {'energy_efficiency': 2}),
    ('Node_C', 'School_5', {'energy_efficiency': 1}), ('Node_C', 'School_6', {'energy_efficiency': 1}),
    ('Node_D', 'School_7', {'energy_efficiency': 1}), ('Node_D', 'School_8', {'energy_efficiency': 1}),
    ('Node_A', 'Node_B', {'energy_efficiency': 1}), ('Node_C', 'Node_D', {'energy_efficiency': 1}),
    ('Node_C', 'School_9', {'energy_efficiency': 1}), ('Node_D', 'School_10', {'energy_efficiency': 1})
])

# Train model and make predictions
forecast = train_prophet(df)
demand_predictions = {f"School_{i}": np.random.randint(20, 100) for i in range(1, 11)}  # Random demand per school
bandwidth_allocation = allocate_bandwidth(demand_predictions)  # Allocate bandwidth
network_routes = sdn_load_balancer(graph, demand_predictions)  # Compute SDN-based routes
anomalies = detect_anomalies(df)  # Detect anomalies

# Streamlit UI
st.set_page_config(layout="wide", page_title="Smart Network Allocation", page_icon="🌐")
st.title("📡 Smart Network Resource Allocation with SDN Load Balancing")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Bandwidth Usage Over Time")
    fig = px.line(df, x="Timestamp", y="Bandwidth_Usage", color="School", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🔍 Predicted Bandwidth Allocations")
    st.dataframe(pd.DataFrame(list(bandwidth_allocation.items()), columns=["School", "Allocated Bandwidth (Mbps)"]))

st.markdown("---")

col3, col4 = st.columns([1, 1])

with col3:
    st.subheader("🔗 SDN-based Load Balancing Routes (Energy Efficient)")
    st.dataframe(pd.DataFrame([(school, ' -> '.join(path)) for school, path in network_routes.items()], columns=["School", "Route"]))

with col4:
    st.subheader("⚠️ Detected Anomalies")
    if anomalies.empty:
        st.success("No anomalies detected")
    else:
        st.dataframe(anomalies)

st.markdown("---")
st.caption("Developed for optimized network performance, intelligent bandwidth allocation, and anomaly detection using machine learning and SDN principles ⚡")
