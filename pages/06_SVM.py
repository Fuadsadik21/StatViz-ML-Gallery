import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.datasets import make_classification
import sys
import os

# Allow importing from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import page_config, render_header

# --- SETUP ---
page_config()
render_header()

st.header("SVM: Maximizing the Margin")
st.markdown("""
**The Statistical Insight:** Support Vector Machines (SVM) find the hyperplane that maximizes the margin between classes.
The support vectors are the critical data points that define this boundary.
""")
def render_svm():
    st.info("💡 **Goal:** Find the widest possible 'street' that separates two classes.")

    # --- Sidebar Controls ---
    st.sidebar.subheader("⚙️ Model Parameters")
    c_param = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])

    # --- Data Generation ---
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)

    # --- Model Training ---
    model = SVC(kernel=kernel, C=c_param)
    model.fit(X, y)

    # --- Visualization ---
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    # Decision Function for the "Street" margins
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig = go.Figure()
    # Decision Boundary & Margins
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, 0.02), y=np.arange(y_min, y_max, 0.02), z=Z,
        colorscale='RdBu', contours_coloring='lines',
        contours=dict(start=-1, end=1, size=1), line_width=2, showscale=False
    ))
    # Support Vectors
    sv = model.support_vectors_
    fig.add_trace(go.Scatter(
        x=sv[:, 0], y=sv[:, 1], mode='markers',
        marker=dict(size=12, line=dict(color='black', width=2), symbol='circle-open'),
        name='Support Vectors'
    ))
    # Data
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1], mode='markers',
        marker=dict(color=y, colorscale='RdBu'), name='Data'
    ))

    fig.update_layout(title="Decision Boundary & Support Vectors", height=500, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"Number of Support Vectors: {len(model.support_)}")
render_svm()