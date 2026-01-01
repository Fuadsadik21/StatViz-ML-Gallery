import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import sys
import os

# Allow importing from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import page_config, render_header

# --- SETUP ---
page_config()
render_header()

# --- THE LOGISTIC REGRESSION MODULE ---
def render_logistic_regression():
    st.header("Logistic Regression: The Bayesian vs. Frequentist View")
    st.markdown("""
    **The Statistical Insight:** Standard Logistic Regression (Frequentist) gives you a single "best fit" S-curve (Maximum Likelihood Estimation).  
    But in reality, there is uncertainty. Bayesian methods would show a *cloud* of possible curves.
    """)

    # --- Sidebar Controls ---
    st.sidebar.subheader("⚙️ Data Generation")
    n_samples = st.sidebar.slider("Sample Size", 50, 500, 100)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 5.0, 1.0)
    
    st.sidebar.subheader("⚙️ Model Parameters")
    # A simple "Bayesian-like" visualization toggle
    show_uncertainty = st.sidebar.checkbox("Show Uncertainty (Bayesian Intuition)", value=False)
    threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5)

    # --- Data Generation (Synthetic) ---
    X, y = make_classification(
        n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1,
        n_samples=n_samples, random_state=42, flip_y=0.05 * noise_level
    )
    
    # Fit Standard Model (Frequentist / MLE)
    model = LogisticRegression()
    model.fit(X, y)
    
    # Generate smooth curve data
    X_test = np.linspace(X.min() - 1, X.max() + 1, 300).reshape(-1, 1)
    y_prob = model.predict_proba(X_test)[:, 1]

    # --- Visualisation ---
    fig = go.Figure()

    # 1. The Data Points
    fig.add_trace(go.Scatter(
        x=X.flatten(), y=y,
        mode='markers',
        name='Observed Data',
        marker=dict(color=y, colorscale='Viridis', line_width=1, opacity=0.6)
    ))

    # 2. The MLE Sigmoid Curve (The Frequentist Fit)
    fig.add_trace(go.Scatter(
        x=X_test.flatten(), y=y_prob,
        mode='lines',
        name='MLE Sigmoid (Frequentist)',
        line=dict(color='black', width=3)
    ))

    # 3. (Optional) Bayesian Uncertainty "Cloud"
    # We simulate this by bootstrapping (fitting models to random subsets)
    if show_uncertainty:
        uncertainty_traces = []
        for i in range(20):
            # Bootstrap sample
            indices = np.random.choice(np.arange(n_samples), size=int(n_samples*0.8), replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            # Fit temp model
            boot_model = LogisticRegression()
            try:
                boot_model.fit(X_boot, y_boot)
                y_boot_prob = boot_model.predict_proba(X_test)[:, 1]
                fig.add_trace(go.Scatter(
                    x=X_test.flatten(), y=y_boot_prob,
                    mode='lines',
                    name='Posterior Sample',
                    line=dict(color='blue', width=1, dash='solid'),
                    opacity=0.1,
                    showlegend=False,
                    hoverinfo='skip'
                ))
            except:
                pass # Skip if bootstrap sample has only 1 class

    # 4. Decision Threshold Line
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Decision Boundary")

    fig.update_layout(
        title="Probabilistic Decision Boundary",
        xaxis_title="Feature Value (X)",
        yaxis_title="Probability P(Y=1|X)",
        template="plotly_white",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Statistical Diagnostics ---
    with st.expander("📊 Statistical Diagnostics (Under the Hood)"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Model Coefficients:**")
            st.write(f"Intercept (β0): {model.intercept_[0]:.4f}")
            st.write(f"Slope (β1): {model.coef_[0][0]:.4f}")
        with col2:
            pred = (model.predict_proba(X)[:, 1] > threshold).astype(int)
            acc = np.mean(pred == y)
            st.metric("Accuracy", f"{acc:.2%}")

# Call the function to render the page content
render_logistic_regression()