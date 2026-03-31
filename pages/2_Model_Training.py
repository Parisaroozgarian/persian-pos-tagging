import streamlit as st
import sys
import os
import torch
import time
import threading
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from model_trainer import ModelTrainer, create_data_loaders
from freezing_strategies import get_available_strategies, visualize_freezing_strategy, get_strategy_recommendations
from visualization import create_real_time_training_plot, update_training_plot
from database import get_database_manager

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

st.title("🤖 Model Training & Freezing Experiments")

st.markdown("""
Configure and train DistilmBERT models with different layer freezing strategies for Persian PoS tagging.
""")

# Check if dataset is loaded
if not st.session_state.get('dataset_loaded', False):
    st.error("❌ Dataset not loaded. Please go to the Data Exploration page first.")
    st.stop()

# Get dataset info
prepared_data = st.session_state.prepared_data
dataset_stats = st.session_state.dataset_stats

# Sidebar configuration
st.sidebar.header("Training Configuration")
st.sidebar.subheader("Hyperparameters")

epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=10, value=3)
batch_size = st.sidebar.selectbox("Batch Size", options=[8, 16, 32], index=1)
learning_rate = st.sidebar.selectbox(
    "Learning Rate",
    options=[1e-5, 2e-5, 3e-5, 5e-5],
    index=1,
    format_func=lambda x: f"{x:.0e}"
)

st.sidebar.subheader("Model Settings")
model_name = st.sidebar.selectbox(
    "Base Model",
    options=[
        "distilbert-base-multilingual-cased",
        "bert-base-multilingual-cased"
    ],
    index=0
)

# Show device info
device = "CUDA ✅ GPU" if torch.cuda.is_available() else "CPU ⚠️"
st.sidebar.info(f"Device: **{device}**")

# Initialize trainer
@st.cache_resource
def get_trainer(model_name):
    return ModelTrainer(model_name)

trainer = get_trainer(model_name)

# Create data loaders
@st.cache_data
def get_data_loaders(batch_size):
    return create_data_loaders(
        prepared_data['train_dataset'],
        prepared_data['val_dataset'],
        batch_size=batch_size
    )

train_dataloader, val_dataloader = get_data_loaders(batch_size)

# Main content - Strategy Selection
st.subheader("🎯 Freezing Strategy Selection")
strategies = get_available_strategies()
selected_strategies = []
strategy_configs = {}

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Available Freezing Strategies")
    for strategy_key, strategy_info in strategies.items():
        if st.checkbox(f"**{strategy_info['name']}**", key=f"select_{strategy_key}"):
            selected_strategies.append(strategy_key)
            st.markdown(f"*{strategy_info['description']}*")
            st.markdown(f"**Rationale:** {strategy_info['rationale']}")
            if strategy_key == 'custom':
                custom_layers = st.multiselect(
                    "Select layers to freeze (0-5 for DistilmBERT):",
                    options=list(range(6)),
                    key=f"custom_layers_{strategy_key}"
                )
                strategy_configs[strategy_key] = {'custom_layers': custom_layers}
            viz_fig = visualize_freezing_strategy(strategy_key)
            st.plotly_chart(viz_fig, use_container_width=True)
            st.markdown("---")

with col2:
    st.markdown("### Strategy Recommendations")
    recommendations = get_strategy_recommendations()
    for strategy_key in selected_strategies:
        if strategy_key in recommendations:
            rec = recommendations[strategy_key]
            with st.expander(f"{strategies[strategy_key]['name']}"):
                st.markdown("**Use when:**")
                for use_case in rec['use_when']:
                    st.markdown(f"- {use_case}")
                st.markdown("**Pros:**")
                for pro in rec['pros']:
                    st.markdown(f"- ✅ {pro}")
                st.markdown("**Cons:**")
                for con in rec['cons']:
                    st.markdown(f"- ⚠️ {con}")

# Training section
st.markdown("---")
st.subheader("🚀 Training Execution")

if not selected_strategies:
    st.info("👆 Select at least one freezing strategy to begin training")
else:
    st.success(f"✅ {len(selected_strategies)} strategy(ies) selected: {', '.join(selected_strategies)}")

    # Session state init
    for key, default in [
        ('training_results', {}),
        ('training_in_progress', False),
        ('live_metrics', {}),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        experiment_name = st.text_input(
            "Experiment Name",
            value=f"Experiment_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )

        if st.button("🎯 Start Training Experiments", type="primary",
                     disabled=st.session_state.training_in_progress):
            try:
                db_manager = get_database_manager()
                experiment_config = {
                    'model_name': model_name,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'strategies': selected_strategies
                }
                dataset_id    = st.session_state.get('dataset_id', None)
                experiment_id = db_manager.create_experiment(
                    name=experiment_name,
                    description=f"Persian PoS tagging with {len(selected_strategies)} freezing strategies",
                    dataset_id=dataset_id,
                    config=experiment_config
                )
                if experiment_id:
                    st.session_state.current_experiment_id = experiment_id
            except Exception:
                pass

            st.session_state.training_in_progress = True
            st.session_state.training_results     = {}
            st.session_state.pop('training_error', None)
            st.rerun()

    # ── Training execution ──────────────────────────────────────────────────
    if st.session_state.training_in_progress:
        st.markdown("### 🔄 Training in Progress")

        # ── Background thread function ──
        def _run_training():
            try:
                training_results = {}
                freezing_info    = {}

                for i, strategy in enumerate(selected_strategies):
                    st.session_state['current_strategy_index'] = i
                    st.session_state['current_strategy']       = strategy

                    model = trainer.create_model(
                        num_labels=prepared_data['num_labels'],
                        label2id=prepared_data['label2id'],
                        id2label=prepared_data['id2label']
                    )
                    custom_layers = strategy_configs.get(strategy, {}).get('custom_layers', None)
                    freezing_info[strategy] = trainer.apply_freezing_strategy(
                        model, strategy, layer_indices=custom_layers
                    )

                    start_time = time.time()
                    results    = trainer.train_model(
                        model=model,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        epochs=epochs,
                        learning_rate=learning_rate,
                    )
                    results['training_time'] = time.time() - start_time
                    training_results[strategy] = results

                # ── Save results ──
                st.session_state.training_results     = training_results
                st.session_state.freezing_info        = freezing_info
                st.session_state.models_trained       = True
                st.session_state.training_in_progress = False

                try:
                    if 'current_experiment_id' in st.session_state:
                        db_manager    = get_database_manager()
                        experiment_id = st.session_state.current_experiment_id
                        for strat in training_results:
                            db_manager.save_training_result(
                                experiment_id=experiment_id,
                                strategy=strat,
                                results=training_results[strat],
                                freezing_info=freezing_info[strat]
                            )
                        db_manager.update_experiment_status(
                            experiment_id, 'completed', datetime.now()
                        )
                except Exception:
                    pass

            except Exception as e:
                st.session_state['training_error']    = str(e)
                st.session_state.training_in_progress = False

        # Start thread only once
        if ('training_thread' not in st.session_state or
                not st.session_state['training_thread'].is_alive()):
            t = threading.Thread(target=_run_training, daemon=True)
            st.session_state['training_thread'] = t
            t.start()

        # ── Live UI placeholders ──
        status_ph   = st.empty()
        progress_ph = st.empty()
        metrics_ph  = st.empty()
        chart_ph    = st.empty()

        # ── ✅ Polling loop — refreshes every 2 seconds ──
        while st.session_state.training_in_progress:
            live        = st.session_state.get('live_metrics', {})
            epoch       = live.get('epoch', 0)
            total_ep    = live.get('total_epochs', epochs)
            batch       = live.get('batch', 0)
            total_bat   = live.get('total_batches', 1)
            cur_loss    = live.get('current_loss')
            history     = live.get('history', {})
            device_str  = live.get('device', '?')
            s_idx       = st.session_state.get('current_strategy_index', 0)
            s_name      = st.session_state.get('current_strategy', '…')

            # Status bar
            status_ph.info(
                f"⚡ Device: **{device_str.upper()}** | "
                f"Strategy **{s_idx+1}/{len(selected_strategies)}**: `{s_name}` | "
                f"Epoch **{epoch}/{total_ep}** | "
                f"Batch **{batch}/{total_bat}** | "
                f"Loss: **{cur_loss if cur_loss is not None else '…'}**"
            )

            # Overall progress bar
            overall = (s_idx + (epoch / total_ep if total_ep else 0)) / len(selected_strategies)
            progress_ph.progress(min(overall, 1.0))

            # Metrics row + live chart
            if history.get('train_loss'):
                with metrics_ph.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Train Loss",   f"{history['train_loss'][-1]:.4f}")
                    c2.metric("Val Accuracy", f"{history['val_accuracy'][-1]:.4f}")
                    c3.metric("Val F1",       f"{history['val_f1'][-1]:.4f}")
                    c4.metric("Val Loss",     f"{history['val_loss'][-1]:.4f}")

                ep_list = list(range(1, len(history['train_loss']) + 1))
                fig = make_subplots(rows=1, cols=2,
                                    subplot_titles=['Loss', 'Accuracy & F1'])
                fig.add_trace(go.Scatter(x=ep_list, y=history['train_loss'],
                                         name='Train Loss', mode='lines+markers'), row=1, col=1)
                fig.add_trace(go.Scatter(x=ep_list, y=history['val_loss'],
                                         name='Val Loss', mode='lines+markers'), row=1, col=1)
                fig.add_trace(go.Scatter(x=ep_list, y=history['val_accuracy'],
                                         name='Accuracy', mode='lines+markers'), row=1, col=2)
                fig.add_trace(go.Scatter(x=ep_list, y=history['val_f1'],
                                         name='F1', mode='lines+markers'), row=1, col=2)
                fig.update_layout(height=300)
                chart_ph.plotly_chart(fig, use_container_width=True)

            time.sleep(2)
            st.rerun()

        # ── Training finished ──
        if st.session_state.get('training_error'):
            st.error(f"❌ Training failed: {st.session_state['training_error']}")
        else:
            st.success("🎉 Training completed and results saved!")
            st.info("Navigate to the Results Analysis page to explore your findings.")
            time.sleep(1)
            st.rerun()

# ── Results summary after training ──────────────────────────────────────────
if st.session_state.get('models_trained', False):
    st.markdown("---")
    st.subheader("✅ Training Completed")

    results = st.session_state.training_results
    st.markdown("### Quick Results Summary")

    results_data = []
    for strategy, result in results.items():
        results_data.append({
            'Strategy':       strategy,
            'Best Accuracy':  f"{result['best_val_accuracy']:.4f}",
            'Training Time':  f"{result['training_time']:.1f}s"
        })

    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    st.success("🔬 Ready for detailed analysis! Go to the Results Analysis page.")

# Help section
with st.expander("ℹ️ Training Help & Tips"):
    st.markdown("""
    ### Training Configuration Guidelines
    
    **Epochs:** 3-5 epochs usually sufficient for fine-tuning.
    
    **Batch Size:** Larger batches (32) for stable training. Smaller (8-16) for limited GPU memory.
    
    **Learning Rate:** 2e-5 is a good starting point for BERT-family models.
    
    ### Strategy Selection Tips
    - **Start with baseline:** Always include 'none' (no freezing) for comparison
    - **Systematic exploration:** Try early, late, and alternating freezing
    - **Custom strategies:** Use domain knowledge to target specific layers
    
    ### Performance Expectations
    - **Baseline accuracy:** 85-95% for PoS tagging
    - **Frozen models:** May be 2-5% lower but much faster
    - **Training time:** Freezing can reduce time by 30-60%
    """)