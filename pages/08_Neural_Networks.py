import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_circles
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import page_config, render_header

page_config()
render_header()

st.header("Neural Networks: Deep Learning Intuition")
st.info("💡 **Goal:** Mimic brain-like structures to find non-linear patterns. Watch how 'Depth' allows the model to capture complex shapes.")

# --- Sidebar ---
st.sidebar.subheader("🧠 Architecture")
layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 1)
neurons = st.sidebar.slider("Neurons per Layer", 2, 20, 5)
activation = st.sidebar.selectbox("Activation Function", ["tanh", "relu", "logistic"])

# --- Data: Circles (Non-linear) ---
X, y = make_circles(n_samples=400, factor=0.5, noise=0.1, random_state=42)

# --- Model ---
hidden_layer_sizes = tuple([neurons] * layers)
mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=1000, random_state=42)
mlp.fit(X, y)

# --- Visualization ---
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig = go.Figure()
fig.add_trace(go.Contour(
    x=np.arange(x_min, x_max, 0.05), y=np.arange(y_min, y_max, 0.05), z=Z,
    showscale=False, colorscale='Tropic', opacity=0.4
))
fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y, colorscale='Tropic')))

fig.update_layout(title=f"Decision Boundary: {layers} Layer(s), {neurons} Neurons", height=550, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.success(f"Final Accuracy: {mlp.score(X, y):.2%}")