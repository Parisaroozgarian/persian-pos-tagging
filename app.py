import streamlit as st
import os
import sys

st.set_page_config(
    page_title="Persian PoS Tagging - Layer Freezing Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Safe imports — show errors in UI instead of crashing
import_errors = []

try:
    import torch
    TORCH_OK = True
    DEVICE   = "CUDA ✅ GPU" if torch.cuda.is_available() else "CPU ⚠️"
except Exception as e:
    TORCH_OK = False
    DEVICE   = "Unknown"
    import_errors.append(f"torch: {e}")

try:
    from database import get_database_manager
    DATABASE_AVAILABLE = True
except Exception as e:
    DATABASE_AVAILABLE = False
    import_errors.append(f"database: {e}")

if import_errors:
    with st.expander("⚠️ Some modules failed to load", expanded=True):
        for err in import_errors:
            st.error(err)

# Session state
for key, default in [
    ('dataset_loaded', False),
    ('models_trained', False),
    ('training_results', {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Main UI
st.title("🔬 Partial Layer Freezing for Persian PoS Tagging")
st.markdown("### Research Tool for Multilingual Model Fine-tuning")
st.markdown("""
This interactive application explores the impact of partial layer freezing when fine-tuning 
distilled multilingual transformer models for Part-of-Speech tagging in Persian.
""")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## Research Objective
    
    Explore how **partial layer freezing** affects model performance when fine-tuning 
    distilmBERT for Persian Part-of-Speech tagging using the Universal Dependencies dataset.
    
    ### Key Questions:
    - Which layers should be frozen for optimal performance?
    - How does freezing affect training efficiency?
    - What linguistic features are preserved through freezing?
    
    ### Navigation Guide:
    1. **📊 Data Exploration** — Load and explore the Persian UD dataset
    2. **🤖 Model Training** — Configure and train baseline vs. frozen models
    3. **📈 Results Analysis** — Compare performance and analyze findings
    4. **🗄️ Database Management** — View experiment history and manage research data
    """)

with col2:
    st.markdown("### Quick Status")

    if TORCH_OK:
        if "GPU" in DEVICE:
            st.success(f"⚡ {DEVICE}")
        else:
            st.warning(f"🖥️ {DEVICE}")
    else:
        st.error("❌ PyTorch not loaded")

    if st.session_state.dataset_loaded:
        st.success("✅ Dataset Loaded")
    else:
        st.info("📊 Load dataset to begin")

    if st.session_state.models_trained:
        st.success("✅ Models Trained")
    else:
        st.info("🤖 Train models after loading data")

    if DATABASE_AVAILABLE:
        try:
            db_manager = get_database_manager()
            stats = db_manager.get_dataset_stats()
            st.success("🗄️ Database Connected")
            if stats.get('total_experiments', 0) > 0:
                st.metric("Total Experiments", stats['total_experiments'])
        except Exception as e:
            st.warning(f"🗄️ Database error: {e}")
    else:
        st.warning("🗄️ Database Unavailable")

    if st.session_state.dataset_loaded and 'dataset_info' in st.session_state:
        info = st.session_state.dataset_info
        st.metric("Training Samples",   info.get('train_size', 0))
        st.metric("Validation Samples", info.get('val_size', 0))
        st.metric("Unique POS Tags",    info.get('num_labels', 0))

st.markdown("---")

with st.expander("📚 Background & Methodology"):
    st.markdown("""
    ### Partial Layer Freezing in Transformer Models
    
    **Concept**: Freezing specific layers during fine-tuning prevents their weights from updating,
    potentially preserving general linguistic knowledge while allowing task-specific adaptation.
    
    **DistilmBERT Architecture**:
    - 6 transformer layers (vs. 12 in BERT)
    - Each layer contains: Multi-head attention + Feed-forward network
    - Early layers: Basic linguistic features (syntax, morphology)
    - Late layers: Task-specific representations
    
    **Freezing Strategies**:
    - **Early Freezing**: Preserve basic linguistic knowledge
    - **Late Freezing**: Allow task-specific adaptation
    - **Alternating**: Mix frozen and trainable layers
    """)

with st.expander("🎯 Getting Started"):
    st.markdown("""
    1. **Navigate to Data Exploration** 📊 — Load the Persian UD dataset
    2. **Go to Model Training** 🤖 — Configure freezing strategies and train
    3. **Analyze Results** 📈 — Compare model performances
    """)