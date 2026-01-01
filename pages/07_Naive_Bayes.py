import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_blobs
import sys
import os

# Parent directory import fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import page_config, render_header

page_config()
render_header()

st.header("Naive Bayes: Probabilistic Independence")
st.info("💡 **Goal:** Predict classes using Bayes' Theorem, assuming that every feature is independent of the others.")

# --- Sidebar ---
st.sidebar.subheader("⚙️ Model Parameters")
prior_ratio = st.sidebar.slider("Class Prior Weight", 0.1, 0.9, 0.5, help="Simulate imbalanced data by changing the prior probability.")

# --- Data & Model ---
X, y = make_blobs(n_samples=300, centers=2, cluster_std=2.0, random_state=42)
gnb = GaussianNB(priors=[1-prior_ratio, prior_ratio])
gnb.fit(X, y)

# --- Visualization: Probability Heatmap ---
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

fig = go.Figure()
fig.add_trace(go.Heatmap(
    x=np.arange(x_min, x_max, 0.1), y=np.arange(y_min, y_max, 0.1), z=Z,
    colorscale='RdBu', opacity=0.6, showscale=True
))
fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y, colorscale='RdBu', line_width=1)))

fig.update_layout(title="Posterior Probability Surface", height=500, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

with st.expander("📚 Why is it 'Naive'?"):
    st.write("""
    It is 'Naive' because it assumes that the presence of one feature is completely unrelated to the presence of another. 
    Even if this assumption is false (which it usually is in real life), the classifier often performs surprisingly well!
    """)