import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
import sys
import os

# Parent directory import fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import page_config, render_header

page_config()
render_header()

st.header("Random Forest: The Wisdom of Crowds")
st.info("💡 **Goal:** Reduce overfitting (Variance) by averaging the predictions of many 'weak' Decision Trees (Bagging).")

# --- Sidebar Controls ---
st.sidebar.subheader("⚙️ Hyperparameters")
n_estimators = st.sidebar.slider("Number of Trees", 1, 50, 5, help="More trees = Smoother decision boundary")
max_depth = st.sidebar.slider("Max Depth per Tree", 1, 10, 3)
noise_level = st.sidebar.slider("Data Noise", 0.0, 0.5, 0.3)

# --- Data Generation (Moons) ---
X, y = make_moons(n_samples=300, noise=noise_level, random_state=42)

# --- Model Training ---
rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
rf.fit(X, y)

# --- Visualization: The Probability Contour ---
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predict probabilities (the "vote" confidence)
Z = rf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

fig = go.Figure()
# The Probability Surface
fig.add_trace(go.Contour(
    x=np.arange(x_min, x_max, 0.1), y=np.arange(y_min, y_max, 0.1), z=Z,
    colorscale='RdBu', opacity=0.4, showscale=True,
    hoverinfo='skip'
))
# The Data Points
fig.add_trace(go.Scatter(
    x=X[:, 0], y=X[:, 1], mode='markers',
    marker=dict(color=y, colorscale='RdBu', line_width=1, size=8),
    name='Data'
))

fig.update_layout(title=f"Decision Surface ({n_estimators} Trees)", height=500, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# --- Stats Insight: Feature Importance ---
# We simulate a scenario where we have features to "rank"
st.subheader("📊 Feature Importance (The Explainability Bonus)")
st.write("Unlike Neural Nets, Random Forests can easily tell us *which* features mattered most.")

# Dummy plot to illustrate the concept since we only have X/Y coordinates
fig_imp = go.Figure(go.Bar(
    x=['X-Coordinate', 'Y-Coordinate'], 
    y=rf.feature_importances_,
    marker_color=['#1f77b4', '#ff7f0e']
))
fig_imp.update_layout(height=250, template="plotly_white", title="Relative Importance of Features")
st.plotly_chart(fig_imp, use_container_width=True)