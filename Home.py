import streamlit as st
from utils import page_config, render_header

# 1. Initialize the professional theme and layout
page_config()

# 2. Render the top branding header
render_header()

# 3. Hero Section
st.title("🎓 Machine Learning Statistical Gallery")
st.markdown("""
### Transforming "Black Boxes" into "Glass Boxes"
Welcome to an interactive educational journey through Machine Learning. This gallery is designed for students and data scientists who want to go beyond simply calling `.fit()` and `.predict()`. 

**Goal:** To build a deep, visual intuition of the statistical mechanics that drive modern ML models.
""")

st.divider()

# 4. The Curriculum Overview
st.header("📚 The Curriculum")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 Supervised Learning")
    st.markdown("""
    - **Linear Regression:** Visualizing residuals and the breakdown of homoscedasticity.
    - **Logistic Regression:** A look at Frequentist vs. Bayesian uncertainty.
    - **Decision Trees:** Recursive partitioning and the fine line of overfitting.
    - **Random Forest:** Harnessing the "Wisdom of Crowds" to reduce variance.
    - **SVM:** Maximizing the margin and identifying support vectors.
    - **Naive Bayes:** Understanding probabilistic independence.
    """)

with col2:
    st.subheader("🌀 Unsupervised & Deep Learning")
    st.markdown("""
    - **K-Means Clustering:** Centroid competition and Voronoi territories.
    - **PCA:** Reducing dimensionality while preserving maximum variance.
    - **Neural Networks:** Visualizing how layers warp space to solve non-linear problems.
    """)

st.divider()

# 5. Call to Action
st.info("👈 **Get Started:** Use the sidebar to select an algorithm and start exploring!")

# 6. Footer (Great for your Portfolio)
st.caption("Built by Fuad Sadik • Powered by Streamlit, Plotly, and Scikit-Learn")