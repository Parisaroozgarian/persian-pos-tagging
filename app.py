import streamlit as st
import sys
import os

st.title("🔬 Persian PoS Tagging - Startup Diagnostics")

# Test all imports one by one
imports_to_test = [
    ("torch", "import torch"),
    ("transformers", "from transformers import AutoModelForTokenClassification"),
    ("datasets", "from datasets import load_dataset"),
    ("seaborn", "import seaborn as sns"),
    ("sklearn", "from sklearn.metrics import accuracy_score"),
    ("plotly", "import plotly.graph_objects as go"),
    ("sqlalchemy", "from sqlalchemy import create_engine"),
    ("conllu", "import conllu"),
]

for name, imp in imports_to_test:
    try:
        exec(imp)
        st.success(f"✅ {name}")
    except Exception as e:
        st.error(f"❌ {name}: {e}")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

src_modules = [
    ("data_loader", "from data_loader import load_persian_ud_dataset"),
    ("model_trainer", "from model_trainer import ModelTrainer"),
    ("database", "from database import get_database_manager"),
    ("evaluation", "from evaluation import compare_model_performances"),
    ("freezing_strategies", "from freezing_strategies import get_available_strategies"),
    ("visualization", "from visualization import create_performance_radar_chart"),
]

st.markdown("---")
st.subheader("Source Module Imports")

for name, imp in src_modules:
    try:
        exec(imp)
        st.success(f"✅ src/{name}")
    except Exception as e:
        st.error(f"❌ src/{name}: {e}")

device = "GPU ✅" if __import__('torch').cuda.is_available() else "CPU ⚠️"
st.info(f"Device: {device}")
