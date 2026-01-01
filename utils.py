import streamlit as st

def page_config():
    # Set the config for every page
    st.set_page_config(
        page_title="StatViz ML Gallery",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-title {
            font-size: 42px !important;
            font-weight: 700 !important;
            color: #1E3A8A;
            margin-bottom: 0px;
        }
        .stAlert {
            border-radius: 10px;
            border: none;
            background-color: #F0F4FF;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="main-title">📊 StatViz-ML-Gallery</p>', unsafe_allow_html=True)
        st.markdown("### *A visual, stats-first exploration of machine learning.*")
    with col2:
        st.write("") 
        if st.button("Reset View"):
            st.cache_data.clear()
    st.divider()