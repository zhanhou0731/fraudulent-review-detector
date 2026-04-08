import streamlit as st
import os
from views import home, experiments, api_docs

st.set_page_config(page_title="Fraudulent Review Detection", page_icon="🛡️", layout="wide")

with open(os.path.join("assets", "style.css")) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9566/9566143.png", width=80)
    st.title("Menu")
    
    page = st.radio("Go to", ["🏠 Homepage", "📈 Experiment Results", "🔌 API Documentation"])
    
    st.markdown("---")
    st.caption("FYP Prototype")

if page == "🏠 Homepage":
    home.render_home()

elif page == "📈 Experiment Results":
    st.title("📈 Experiment Results")
    experiments.render_experiments()

elif page == "🔌 API Documentation":
    st.title("🔌 API Documentation")
    api_docs.render_docs()