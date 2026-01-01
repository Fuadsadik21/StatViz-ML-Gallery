import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
import sys
import os

# Allow importing from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import page_config, render_header

# --- SETUP ---
page_config()
render_header()

st.header("PCA: Variance & Projection")
st.markdown("""
**The Statistical Insight:** Principal Component Analysis (PCA) identifies directions (principal components) in the data that capture the most variance.
By projecting data onto these components, we can reduce dimensionality while retaining most of the information.
""")
def render_pca():
    st.info("💡 **Goal:** Reduce the number of variables while keeping the 'information' (variance) intact.")
    
    # --- Sidebar Controls ---
    st.sidebar.subheader("⚙️ Data Generation")
    n_samples = st.sidebar.slider("Number of Points", 100, 1000, 300, key='pca_n')
    noise = st.sidebar.slider("Data Spread (Noise)", 0.1, 2.0, 0.5)

    # --- Data Generation (3D Blob) ---
    # We create a 3D dataset that is mostly flat (aligned with a plane) 
    # to show how PCA finds that plane.
    np.random.seed(42)
    X, _ = make_blobs(n_samples=n_samples, centers=1, cluster_std=noise, n_features=3)
    # Stretch it to create a clear principal component
    X = X @ np.array([[1, 0.5, 0.1], [0.2, 1, 0.5], [0.1, 0.1, 1]])

    # --- PCA Logic ---
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    exp_var = pca.explained_variance_ratio_

    # --- Visualization ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Original 3D Space")
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            mode='markers',
            marker=dict(size=3, color=X[:, 2], colorscale='Viridis', opacity=0.8)
        )])
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=400, template="plotly_white")
        st.plotly_chart(fig_3d, use_container_width=True)

    with col2:
        st.subheader("Projected 2D Space (PC1 vs PC2)")
        fig_2d = go.Figure()
        fig_2d.add_trace(go.Scatter(
            x=X_pca[:, 0], y=X_pca[:, 1],
            mode='markers',
            marker=dict(color=X[:, 2], colorscale='Viridis', opacity=0.8)
        ))
        fig_2d.update_layout(title="Top 2 Principal Components", height=400, template="plotly_white")
        st.plotly_chart(fig_2d, use_container_width=True)

    # --- Statistical Diagnostic: Scree Plot ---
    st.subheader("📉 The Scree Plot (Explained Variance)")
    fig_scree = go.Figure(data=[
        go.Bar(x=['PC1', 'PC2', 'PC3'], y=exp_var, name='Individual'),
        go.Scatter(x=['PC1', 'PC2', 'PC3'], y=np.cumsum(exp_var), name='Cumulative', line=dict(color='red'))
    ])
    fig_scree.update_layout(height=300, template="plotly_white", yaxis_title="Variance Ratio")
    st.plotly_chart(fig_scree, use_container_width=True)
    
    st.success(f"The first two components capture **{sum(exp_var[:2])*100:.1f}%** of the total variance!")
def render_decision_tree():
    st.header("Decision Trees: The Logic of Splitting")
    st.info("💡 **Goal:** Divide the data into smaller groups by asking a series of 'Yes/No' questions based on feature values.")

    # --- Sidebar Controls ---
    st.sidebar.subheader("⚙️ Model Parameters")
    max_depth = st.sidebar.slider("Max Depth (Complexity)", 1, 10, 3)
    criterion = st.sidebar.selectbox("Split Criterion", ["gini", "entropy"])

    # --- Data Generation ---
    X, y = make_blobs(n_samples=300, centers=2, cluster_std=2.5, random_state=42)

    # --- Model Training ---
    clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    clf.fit(X, y)

    # --- Visualization: Decision Boundary ---
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig = go.Figure()
    # Boundary
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, 0.1), y=np.arange(y_min, y_max, 0.1), z=Z,
        showscale=False, colorscale='RdBu', opacity=0.3
    ))
    # Points
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1], mode='markers',
        marker=dict(color=y, colorscale='RdBu', line_width=1), name='Data'
    ))

    fig.update_layout(title=f"Decision Boundary (Depth={max_depth})", height=500, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📝 What is Gini/Entropy?"):
        st.write("These are measures of **Impurity**. A split is chosen if it results in 'cleaner' (more pure) groups of data.")
render_pca()