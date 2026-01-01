import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import sys
import os

# Allow importing from the parent directory (where utils.py is)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import page_config, render_header

# --- SETUP ---
page_config()
render_header()

st.header("Linear Regression: The Residuals Reality Check")
st.markdown("""
**The Statistical Insight:** In OLS, we assume errors are normally distributed with constant variance (**Homoscedasticity**).
Use the controls to introduce *Heteroscedasticity* and watch the **Residual Plot** form a "cone" shape.
""")


def render_linear_regression():
    st.info("💡 **Goal:** Fit a line that minimizes the squared differences between observed and predicted values.")    # --- Sidebar Controls ---
    st.sidebar.subheader("⚙️ Data Generation")
    n_samples = st.sidebar.slider("Sample Size", 50, 500, 200, key='lin_n')
    noise_base = st.sidebar.slider("Base Noise Level", 0.0, 50.0, 10.0, key='lin_noise')
    
    st.sidebar.subheader("⚙️ Assumption Breakers")
    hetero_factor = st.sidebar.slider("Heteroscedasticity Intensity", 0.0, 5.0, 0.0, help="Increases noise as X increases")
    
    # --- Data Generation ---
    np.random.seed(42)
    X = np.linspace(0, 100, n_samples)
    
    # Truth: y = 3x + 20
    true_y = 3 * X + 20
    
    # Noise: If hetero_factor > 0, noise grows with X
    noise_scale = noise_base * (1 + hetero_factor * (X / 100))
    noise = np.random.normal(0, noise_scale, n_samples)
    
    y = true_y + noise
    
    # Reshape for sklearn
    X_reshaped = X.reshape(-1, 1)
    
    # Fit Model
    model = LinearRegression()
    model.fit(X_reshaped, y)
    y_pred = model.predict(X_reshaped)
    residuals = y - y_pred

    # --- Visualisation ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Plot 1: The Regression Fit
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Data', marker=dict(opacity=0.6, color='#636EFA')))
        fig_fit.add_trace(go.Scatter(x=X, y=y_pred, mode='lines', name='OLS Fit', line=dict(color='red', width=3)))
        fig_fit.add_trace(go.Scatter(x=X, y=true_y, mode='lines', name='True Relationship', line=dict(color='green', dash='dash')))
        
        fig_fit.update_layout(title="OLS Fit vs. Ground Truth", xaxis_title="X", yaxis_title="y", height=400, template="plotly_white")
        st.plotly_chart(fig_fit, use_container_width=True)

    with col2:
        # Plot 2: Residuals vs Fitted (The Diagnostic)
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=y_pred, y=residuals, 
            mode='markers', 
            marker=dict(color='orange', opacity=0.7),
            name='Residuals'
        ))
        
        # Zero line
        fig_res.add_hline(y=0, line_dash="dash", line_color="black")
        
        fig_res.update_layout(
            title="Residuals vs. Fitted",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig_res, use_container_width=True)

    # --- Metrics ---
    r2 = model.score(X_reshaped, y)
    with st.expander("📊 Statistical Diagnostics"):
        c1, c2, c3 = st.columns(3)
        c1.metric("R-Squared", f"{r2:.3f}")
        c2.metric("Intercept (β0)", f"{model.intercept_:.2f}", delta=f"{model.intercept_ - 20:.2f} vs Truth")
        c3.metric("Slope (β1)", f"{model.coef_[0]:.2f}", delta=f"{model.coef_[0] - 3:.2f} vs Truth")
render_linear_regression()