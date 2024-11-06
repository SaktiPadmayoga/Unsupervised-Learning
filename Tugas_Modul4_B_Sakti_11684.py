import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px
import numpy as np
from sklearn.metrics import pairwise_distances
import plotly.graph_objects as go

# Function to create a 3D scatter plot and determine the cluster of a new point
def scatter(model, model_name, data, new_point, features, color_scale, title):
    clusters = model.fit_predict(data[features])
    data[f"{model_name}_Cluster"] = clusters

    # Determine the cluster for the new point
    if model_name == "KMeans_model":
        new_cluster = model.predict(new_point[features])[0]
    else:
        distances = pairwise_distances(new_point[features], data[features])
        nearest_index = distances.argmin()
        new_cluster = clusters[nearest_index]

    # Create a 3D scatter plot using Plotly Express
    fig = px.scatter_3d(data, x='Avg_Credit_Limit', y='Total_Credit_Cards', z='Total_visits_online',
                        color=f"{model_name}_Cluster", title=title, color_continuous_scale=color_scale)

    # Add the new point to the plot
    fig.add_trace(
        go.Scatter3d(
            x=new_point['Avg_Credit_Limit'],
            y=new_point['Total_Credit_Cards'],
            z=new_point['Total_visits_online'],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='New Point'
        )
    )
    return fig, new_cluster

# Streamlit page configuration
st.set_page_config(
    page_title="11684 - Unsupervised Learning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar file uploader
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.markdown("<h1 style='text-align: center;'>Unsupervised Learning - Sakti</h1>", unsafe_allow_html=True)
    st.dataframe(input_data)

# Directory where the models are stored
model_path = {
    "AGG_model": 'AGG_model.pkl',
    "KMeans_model": 'KMeans_model.pkl',
    "DBSCAN_model": 'DBSCAN_model.pkl'
}

# Load models into a dictionary
models = {}
for model_name, path in model_path.items():
    if os.path.exists(path):
        with open(path, 'rb') as f:
            models[model_name] = pickle.load(f)
    else:
        st.write(f"Model {model_name} not found at path: {path}")

# Sidebar inputs for the new point
avg_CL = st.sidebar.number_input("Average Credit Limit", 0, 100000)
sum_CC = st.sidebar.number_input("Total Credit Cards", 0, 10)
sum_VO = st.sidebar.number_input("Total Visits Online", 0, 16)

if st.sidebar.button("Predict!"):
    # Features used for prediction
    features = ['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_online']
    
    # Creating a new DataFrame for the new point
    new_point = pd.DataFrame({
        'Avg_Credit_Limit': [avg_CL],
        'Total_Credit_Cards': [sum_CC],
        'Total_visits_online': [sum_VO]
    })

    # Define clustering methods and colors for scatter plot
    cluster_method = [
        ("KMeans_model", models.get("KMeans_model"), "KMeans Clustering", px.colors.sequential.Cividis),
        ("AGG_model", models.get("AGG_model"), "Agglomerative Clustering", px.colors.sequential.Mint),
        ("DBSCAN_model", models.get("DBSCAN_model"), "DBSCAN Clustering", px.colors.sequential.Plasma)
    ]

    # Display scatter plots in three columns
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for i, (model_name, model, title, color_scale) in enumerate(cluster_method):
        if model:
            fig, new_cluster = scatter(model, model_name, input_data, new_point, features, color_scale, title)
            with cols[i]:
                st.plotly_chart(fig)
                st.markdown(f"<p style='text-align: center;'>Titik Dari Data yang baru masuk ke dalam cluster: {new_cluster}</p>", unsafe_allow_html=True)
