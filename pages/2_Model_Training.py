import streamlit as st
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

st.title("🤖 Model Training & Freezing Experiments")
st.info("⚠️ Live training requires more RAM than the free tier provides. Please run training locally or use the pre-loaded results in Results Analysis.")
st.markdown("""
### To run training locally:
```bash
cd ~/Desktop/PersianPosTagging
python3 -m streamlit run app.py
```
""")
