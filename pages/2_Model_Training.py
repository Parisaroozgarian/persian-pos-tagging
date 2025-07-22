import streamlit as st
import sys
import os
import torch
import time
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

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

# Training hyperparameters
st.sidebar.subheader("Hyperparameters")

epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=10, value=3)
batch_size = st.sidebar.selectbox("Batch Size", options=[8, 16, 32], index=1)
learning_rate = st.sidebar.selectbox(
    "Learning Rate", 
    options=[1e-5, 2e-5, 3e-5, 5e-5], 
    index=1,
    format_func=lambda x: f"{x:.0e}"
)

# Model configuration
st.sidebar.subheader("Model Settings")

model_name = st.sidebar.selectbox(
    "Base Model",
    options=[
        "distilbert-base-multilingual-cased",
        "bert-base-multilingual-cased"
    ],
    index=0
)

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

# Main content
st.subheader("🎯 Freezing Strategy Selection")

# Get available strategies
strategies = get_available_strategies()

# Strategy selection interface
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
            
            # Custom strategy configuration
            if strategy_key == 'custom':
                custom_layers = st.multiselect(
                    "Select layers to freeze (0-5 for DistilmBERT):",
                    options=list(range(6)),
                    key=f"custom_layers_{strategy_key}"
                )
                strategy_configs[strategy_key] = {'custom_layers': custom_layers}
            
            # Show visualization for selected strategy
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
    
    # Training progress tracking
    if 'training_results' not in st.session_state:
        st.session_state.training_results = {}
    if 'training_in_progress' not in st.session_state:
        st.session_state.training_in_progress = False
    
    # Start training button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        experiment_name = st.text_input("Experiment Name", value=f"Experiment_{datetime.now().strftime('%Y%m%d_%H%M')}")
        
        if st.button("🎯 Start Training Experiments", type="primary", disabled=st.session_state.training_in_progress):
            # Create experiment in database
            try:
                db_manager = get_database_manager()
                experiment_config = {
                    'model_name': model_name,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'strategies': selected_strategies
                }
                
                dataset_id = st.session_state.get('dataset_id', None)
                experiment_id = db_manager.create_experiment(
                    name=experiment_name,
                    description=f"Persian PoS tagging with {len(selected_strategies)} freezing strategies",
                    dataset_id=dataset_id,
                    config=experiment_config
                )
                
                if experiment_id:
                    st.session_state.current_experiment_id = experiment_id
                    st.session_state.training_in_progress = True
                    st.rerun()
                else:
                    st.error("Failed to create experiment in database")
            except Exception as e:
                st.error(f"Database error: {str(e)}")
                # Proceed without database
                st.session_state.training_in_progress = True
                st.rerun()
    
    # Training execution
    if st.session_state.training_in_progress:
        st.markdown("### 🔄 Training in Progress")
        
        # Initialize results storage
        training_results = {}
        freezing_info = {}
        
        # Progress tracking
        overall_progress = st.progress(0)
        status_text = st.empty()
        
        # Real-time training plots
        training_plot_container = st.empty()
        metrics_container = st.container()
        
        for i, strategy in enumerate(selected_strategies):
            status_text.write(f"Training strategy {i+1}/{len(selected_strategies)}: {strategy}")
            
            # Create model for this strategy
            model = trainer.create_model(
                num_labels=prepared_data['num_labels'],
                label2id=prepared_data['label2id'],
                id2label=prepared_data['id2label']
            )
            
            # Apply freezing strategy
            custom_layers = strategy_configs.get(strategy, {}).get('custom_layers', None)
            freezing_info[strategy] = trainer.apply_freezing_strategy(
                model, strategy, layer_indices=custom_layers
            )
            
            # Display strategy info
            with metrics_container:
                st.markdown(f"#### {strategies[strategy]['name']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Frozen Layers", len(freezing_info[strategy]['frozen_layers']))
                with col2:
                    st.metric("Trainable Params", f"{freezing_info[strategy]['trainable_params']:,}")
                with col3:
                    st.metric("Frozen %", f"{freezing_info[strategy]['frozen_percentage']:.1f}%")
            
            # Training progress callback
            def progress_callback(epoch, batch_idx, total_batches, current_loss):
                # Update progress within epoch
                epoch_progress = (epoch * total_batches + batch_idx) / (epochs * total_batches)
                strategy_progress = (i + epoch_progress) / len(selected_strategies)
                overall_progress.progress(strategy_progress)
            
            # Train model
            start_time = time.time()
            
            with st.spinner(f"Training {strategy} strategy..."):
                results = trainer.train_model(
                    model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    progress_callback=progress_callback
                )
            
            training_time = time.time() - start_time
            
            # Store results
            training_results[strategy] = results
            training_results[strategy]['training_time'] = training_time
            
            # Update real-time plot
            with training_plot_container:
                if results['history']:
                    # Create training curve for this strategy
                    st.markdown(f"##### Training Progress: {strategy}")
                    
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                    
                    fig = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=['Training Loss', 'Validation Accuracy', 'Validation F1']
                    )
                    
                    epochs_list = list(range(1, len(results['history']['train_loss']) + 1))
                    
                    fig.add_trace(
                        go.Scatter(x=epochs_list, y=results['history']['train_loss'], 
                                 mode='lines+markers', name='Train Loss'),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=epochs_list, y=results['history']['val_accuracy'], 
                                 mode='lines+markers', name='Val Accuracy'),
                        row=1, col=2
                    )
                    fig.add_trace(
                        go.Scatter(x=epochs_list, y=results['history']['val_f1'], 
                                 mode='lines+markers', name='Val F1'),
                        row=1, col=3
                    )
                    
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics for this strategy
            with metrics_container:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Best Accuracy", f"{results['best_val_accuracy']:.4f}")
                with col2:
                    final_f1 = results['history']['val_f1'][-1] if results['history']['val_f1'] else 0
                    st.metric("Final F1", f"{final_f1:.4f}")
                with col3:
                    st.metric("Training Time", f"{training_time:.1f}s")
                with col4:
                    avg_epoch_time = sum(results['history']['epoch_times']) / len(results['history']['epoch_times'])
                    st.metric("Avg Epoch Time", f"{avg_epoch_time:.1f}s")
                
                st.markdown("---")
        
        # Training completed
        overall_progress.progress(1.0)
        status_text.write("✅ All training experiments completed!")
        
        # Store results in session state
        st.session_state.training_results = training_results
        st.session_state.freezing_info = freezing_info
        st.session_state.models_trained = True
        st.session_state.training_in_progress = False
        
        # Save results to database
        try:
            if 'current_experiment_id' in st.session_state:
                db_manager = get_database_manager()
                experiment_id = st.session_state.current_experiment_id
                
                # Save each strategy's results
                for strategy in training_results:
                    db_manager.save_training_result(
                        experiment_id=experiment_id,
                        strategy=strategy,
                        results=training_results[strategy],
                        freezing_info=freezing_info[strategy]
                    )
                
                # Update experiment status
                db_manager.update_experiment_status(experiment_id, 'completed', datetime.now())
                st.success("🎉 Training completed and results saved to database!")
            else:
                st.success("🎉 Training completed successfully!")
        except Exception as e:
            st.success("🎉 Training completed successfully!")
            st.warning(f"⚠️ Database save failed: {str(e)}")
        
        st.info("Navigate to the Results Analysis page to explore your findings.")
        
        # Auto-refresh to update UI
        time.sleep(2)
        st.rerun()

# Display current training status
if st.session_state.get('models_trained', False):
    st.markdown("---")
    st.subheader("✅ Training Completed")
    
    results = st.session_state.training_results
    
    # Quick results summary
    st.markdown("### Quick Results Summary")
    
    results_data = []
    for strategy, result in results.items():
        results_data.append({
            'Strategy': strategy,
            'Best Accuracy': f"{result['best_val_accuracy']:.4f}",
            'Training Time': f"{result['training_time']:.1f}s"
        })
    
    import pandas as pd
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
    st.success("🔬 Ready for detailed analysis! Go to the Results Analysis page.")

# Help section
with st.expander("ℹ️ Training Help & Tips"):
    st.markdown("""
    ### Training Configuration Guidelines
    
    **Epochs:**
    - 3-5 epochs usually sufficient for fine-tuning
    - More epochs may lead to overfitting on small datasets
    
    **Batch Size:**
    - Larger batches (32) for stable training
    - Smaller batches (8-16) for limited GPU memory
    
    **Learning Rate:**
    - 2e-5 is a good starting point for BERT-family models
    - Lower rates (1e-5) for more conservative fine-tuning
    - Higher rates (5e-5) for faster adaptation
    
    ### Strategy Selection Tips
    
    **Start with baseline:** Always include 'none' (no freezing) for comparison
    
    **Systematic exploration:** Try early, late, and alternating freezing
    
    **Custom strategies:** Use domain knowledge to target specific layers
    
    ### Performance Expectations
    
    - **Baseline accuracy:** 85-95% for PoS tagging
    - **Frozen models:** May be 2-5% lower but much faster
    - **Training time:** Freezing can reduce time by 30-60%
    """)
