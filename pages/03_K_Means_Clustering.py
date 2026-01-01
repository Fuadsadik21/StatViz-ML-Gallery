import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import sys
import os

# Allow importing from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import page_config, render_header

# --- SETUP ---
page_config()
render_header()

st.header("K-Means: Voronoi Tessellation & The Elbow")
st.markdown("""
**The Statistical Insight:** K-Means minimizes **Inertia** (sum of squared distances to
the nearest centroid).
The colored background represents **Voronoi Cells**—the region of space "claimed" by each centroid. 
Check the **Elbow Plot** below to statistically determine the optimal 'k'.
""")
def render_kmeans():
    # --- Sidebar Controls ---
    st.sidebar.subheader("⚙️ Data Generation")
    n_samples = st.sidebar.slider("Sample Size", 50, 500, 300, key='k_n')
    cluster_std = st.sidebar.slider("Cluster Separation (Std Dev)", 0.5, 3.0, 1.0, key='k_std')
    
    st.sidebar.subheader("⚙️ Model Parameters")
    k_value = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)

    # --- Data Generation ---
    # We generate 4 "True" centers to see if the user can find them with k=4
    X, y_true = make_blobs(n_samples=n_samples, centers=4, cluster_std=cluster_std, random_state=42)

    # --- Model Training ---
    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    centers = kmeans.cluster_centers_

    # --- Visualization: Voronoi Background ---
    # Create a meshgrid to predict the "territory" of each centroid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # --- Plotting ---
    col1, col2 = st.columns([3, 2])

    with col1:
        fig = go.Figure()

        # 1. The Voronoi Background (Contours)
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, 0.1),
            y=np.arange(y_min, y_max, 0.1),
            z=Z,
            showscale=False,
            colorscale='Viridis',
            opacity=0.4,
            hoverinfo='skip'
        ))

        # 2. The Data Points
        fig.add_trace(go.Scatter(
            x=X[:, 0], y=X[:, 1],
            mode='markers',
            marker=dict(color=labels, colorscale='Viridis', line_width=1, size=8),
            name='Data Points'
        ))

        # 3. The Centroids
        fig.add_trace(go.Scatter(
            x=centers[:, 0], y=centers[:, 1],
            mode='markers',
            marker=dict(color='red', symbol='x', size=12, line_width=2),
            name='Centroids'
        ))

        fig.update_layout(
            title=f"K-Means Clustering (k={k_value})", 
            xaxis_title="Feature 1", 
            yaxis_title="Feature 2",
            height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # --- The Elbow Method (Statistical Diagnostic) ---
        st.subheader("📉 The Elbow Method")
        st.write("calculating inertia for k=1 to 10...")
        
        inertias = []
        k_range = range(1, 11)
        for k in k_range:
            temp_model = KMeans(n_clusters=k, random_state=42, n_init=10)
            temp_model.fit(X)
            inertias.append(temp_model.inertia_)
            
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(k_range), y=inertias,
            mode='lines+markers',
            marker=dict(size=8, color='blue'),
            line=dict(width=3)
        ))
        
        # Highlight current k
        fig_elbow.add_trace(go.Scatter(
            x=[k_value], y=[inertias[k_value-1]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name=f'Current k={k_value}'
        ))

        fig_elbow.update_layout(
            title="Elbow Curve (Inertia vs k)",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia (Sum of Squared Distances)",
            height=400,
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    st.info(f"Current Inertia: {kmeans.inertia_:.2f} (Lower is tighter clusters, but beware of overfitting!)")
render_kmeans()