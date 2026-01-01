import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
import sys
import os

# Allow importing from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import page_config, render_header

# --- SETUP ---
page_config()
render_header()

st.header("Decision Trees: The Logic of Splitting")
st.markdown("""
**The Statistical Insight:** Decision Trees recursively split the feature space into regions based on feature values, aiming to create homogeneous groups.
Each split is chosen to maximize information gain (reduce impurity) using criteria like Gini impurity or Entropy.
""")
def render_decision_tree():
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
    st.header("Decision Trees: The Logic of Splitting")
    st.info("💡 **Goal:** Divide the data into smaller groups by asking a series of 'Yes/No' questions based on feature values.")
render_decision_tree()