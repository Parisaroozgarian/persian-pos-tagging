import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

st.set_page_config(
    page_title="Persian PoS Tagging Research",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Persian PoS Tagging — Layer Freezing Research")
st.markdown("""
Welcome! This app explores **partial layer freezing** when fine-tuning distilmBERT
for Persian Part-of-Speech tagging.

Use the sidebar to navigate:
- 📊 **Data Exploration**
- 🤖 **Model Training**
- 📈 **Results Analysis**
- 🗄️ **Database Management**
""")

try:
    import torch
    st.success(f"torch ✅ — {'GPU 🟢' if torch.cuda.is_available() else 'CPU 🟡'}")
except Exception as e:
    st.error(f"torch ❌ {e}")
