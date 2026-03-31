import streamlit as st
import sys
import os
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from data_loader import load_persian_ud_dataset, prepare_datasets, get_dataset_statistics
from visualization import create_pos_distribution_chart, create_dataset_overview, create_model_architecture_diagram
from database import get_database_manager

st.set_page_config(page_title="Data Exploration", page_icon="📊", layout="wide")

st.title("📊 Persian UD Dataset Exploration")

st.markdown("""
Explore and prepare the Persian Universal Dependencies dataset for Part-of-Speech tagging experiments.
""")

# Sidebar controls
st.sidebar.header("Dataset Configuration")

# Dataset size selection
dataset_size = st.sidebar.selectbox(
    "Dataset Size",
    options=[None, 500, 1000, 2000, 5000],
    index=2,  # Default to 1000
    help="Select dataset size for faster experimentation. None = full dataset"
)

max_length = st.sidebar.slider(
    "Max Sequence Length",
    min_value=64,
    max_value=256,
    value=128,
    step=32,
    help="Maximum sequence length for tokenization"
)

tokenizer_name = st.sidebar.selectbox(
    "Tokenizer",
    options=[
        "distilbert-base-multilingual-cased",
        "bert-base-multilingual-cased",
        "xlm-roberta-base"
    ],
    index=0,
    help="Choose tokenizer for preprocessing"
)

# Load dataset button
if st.sidebar.button("Load Dataset", type="primary"):
    with st.spinner("Loading Persian UD dataset..."):
        try:
            # Load raw dataset
            dataset_dict = load_persian_ud_dataset(subset_size=dataset_size)
            
            if dataset_dict:
                st.success("✅ Dataset loaded successfully!")
                
                # Prepare datasets for training
                with st.spinner("Preparing datasets and tokenizing..."):
                    prepared_data = prepare_datasets(
                        dataset_dict, 
                        tokenizer_name=tokenizer_name, 
                        max_length=max_length,
                        subset_size=dataset_size
                    )
                
                # Calculate statistics
                stats = get_dataset_statistics(prepared_data)
                
                # Store in session state
                st.session_state.dataset_loaded = True
                st.session_state.prepared_data = prepared_data
                st.session_state.dataset_stats = stats
                st.session_state.dataset_info = {
                    'train_size': stats['train_size'],
                    'val_size': stats['val_size'],
                    'num_labels': stats['num_labels']
                }
                
                # Save to database
                try:
                    db_manager = get_database_manager()
                    dataset_config = {
                        'subset_size': dataset_size,
                        'tokenizer_name': tokenizer_name,
                        'max_length': max_length
                    }
                    dataset_id = db_manager.save_dataset(dataset_config, stats)
                    if dataset_id:
                        st.session_state.dataset_id = dataset_id
                        st.success("✅ Dataset prepared, tokenized, and saved to database!")
                    else:
                        st.success("✅ Dataset prepared and tokenized!")
                        st.warning("⚠️ Could not save to database")
                except Exception as e:
                    st.success("✅ Dataset prepared and tokenized!")
                    st.warning(f"⚠️ Database save failed: {str(e)}")
                
                st.rerun()
            else:
                st.error("❌ Failed to load dataset")
                
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")

# Main content
if not st.session_state.get('dataset_loaded', False):
    st.info("👆 Configure and load the dataset using the sidebar controls")
    
    # Show architecture diagram while waiting
    st.markdown("---")
    st.subheader("DistilmBERT Architecture")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        arch_fig = create_model_architecture_diagram()
        st.plotly_chart(arch_fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        ### DistilmBERT for Multilingual PoS Tagging
        
        **Architecture Features:**
        - **6 Transformer Layers** (vs 12 in BERT)
        - **Multi-head Attention** in each layer
        - **Feed-forward Networks** for transformation
        - **Multilingual Training** on 104 languages
        - **Knowledge Distillation** from BERT-base
        
        **Layer Functions:**
        - **Early Layers (0-1)**: Basic linguistic features (morphology, syntax)
        - **Middle Layers (2-3)**: Intermediate representations
        - **Late Layers (4-5)**: Task-specific and semantic features
        - **Classification Head**: Final PoS tag prediction
        
        **Freezing Impact:**
        Different layers capture different linguistic phenomena, making 
        strategic freezing crucial for preserving learned knowledge while 
        allowing task-specific adaptation.
        """)

else:
    # Display dataset information
    stats = st.session_state.dataset_stats
    prepared_data = st.session_state.prepared_data
    
    # Dataset overview metrics
    st.subheader("📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Samples", f"{stats['train_size']:,}")
    with col2:
        st.metric("Validation Samples", f"{stats['val_size']:,}")
    with col3:
        st.metric("Unique POS Tags", stats['num_labels'])
    with col4:
        st.metric("Avg Sentence Length", f"{stats['avg_sentence_length']:.1f}")
    
    # Dataset statistics
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric("Total Training Tokens", f"{stats['total_train_tokens']:,}")
        st.metric("Max Sentence Length", stats['max_sentence_length'])
    
    with col2:
        st.metric("Total Validation Tokens", f"{stats['total_val_tokens']:,}")
        st.metric("Min Sentence Length", stats['min_sentence_length'])
    
    # Visualizations
    st.markdown("---")
    
    # POS tag distribution
    st.subheader("🏷️ POS Tag Distribution")
    
    tab1, tab2 = st.tabs(["Training Set", "Validation Set"])
    
    with tab1:
        train_dist_fig = create_pos_distribution_chart(
            stats['train_pos_distribution'], 
            "POS Tag Distribution - Training Set"
        )
        if train_dist_fig:
            st.plotly_chart(train_dist_fig, use_container_width=True)
    
    with tab2:
        val_dist_fig = create_pos_distribution_chart(
            stats['val_pos_distribution'], 
            "POS Tag Distribution - Validation Set"
        )
        if val_dist_fig:
            st.plotly_chart(val_dist_fig, use_container_width=True)
    
    # Dataset overview visualization
    st.markdown("---")
    st.subheader("📊 Comprehensive Dataset Analysis")
    
    overview_fig = create_dataset_overview(stats)
    if overview_fig:
        st.plotly_chart(overview_fig, use_container_width=True)
    
    # Label mapping information
    st.markdown("---")
    st.subheader("🔤 Label Mapping")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**POS Tags to IDs:**")
        label_df = pd.DataFrame([
            {"POS Tag": tag, "ID": idx} 
            for tag, idx in prepared_data['label2id'].items()
        ])
        st.dataframe(label_df, use_container_width=True, height=300)
    
    with col2:
        st.markdown("**Sample Sentences:**")
        # Show first few sentences as examples
        sample_sentences = prepared_data['train_sentences'][:5]
        sample_pos = prepared_data['train_pos'][:5]
        
        for i, (sent, pos) in enumerate(zip(sample_sentences, sample_pos)):
            with st.expander(f"Example {i+1}: {' '.join(sent[:8])}..."):
                example_df = pd.DataFrame({
                    "Token": sent,
                    "POS Tag": pos
                })
                st.dataframe(example_df, use_container_width=True)
    
    # Preprocessing details
    st.markdown("---")
    st.subheader("⚙️ Preprocessing Configuration")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown(f"""
        **Tokenizer:** `{tokenizer_name}`  
        **Max Sequence Length:** {max_length}  
        **Dataset Size:** {dataset_size if dataset_size else 'Full dataset'}  
        **Special Tokens:** `[CLS]`, `[SEP]`, `[PAD]`
        """)
    
    with info_col2:
        st.markdown(f"""
        **Label Alignment:** Subword token alignment  
        **Ignored Label ID:** -100 (for special tokens)  
        **Padding Strategy:** Max length  
        **Truncation:** Enabled
        """)
    
    # Export option
    st.markdown("---")
    
    if st.button("✅ Proceed to Model Training", type="primary"):
        st.success("Dataset is ready! Navigate to the 'Model Training' page to continue.")
        st.info("💡 Use the sidebar navigation to go to the Model Training page.")

# Help section
with st.expander("ℹ️ Help & Information"):
    st.markdown("""
    ### About Persian Universal Dependencies
    
    The Persian-Seraji dataset is part of the Universal Dependencies project, providing:
    - **Consistent annotation** across languages
    - **Rich morphological information** for Persian
    - **Standardized POS tagset** following UD guidelines
    
    ### POS Tag Categories
    Common tags in this dataset include:
    - **NOUN**: Nouns and nominal elements
    - **VERB**: Verbs and verbal constructions  
    - **ADJ**: Adjectives and adjectival phrases
    - **ADP**: Adpositions (prepositions/postpositions)
    - **PRON**: Pronouns
    - **DET**: Determiners
    - **CCONJ/SCONJ**: Coordinating/subordinating conjunctions
    
    ### Configuration Tips
    - **Smaller datasets** (500-1000) for quick experimentation
    - **Larger datasets** (5000+) for robust results
    - **Sequence length 128** is usually sufficient for most sentences
    - **DistilmBERT tokenizer** is optimized for multilingual tasks
    """)