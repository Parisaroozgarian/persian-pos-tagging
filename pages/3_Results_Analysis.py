import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from evaluation import (
    create_performance_comparison, 
    create_training_curves, 
    create_efficiency_analysis,
    create_detailed_results_table,
    export_results_report
)
from freezing_strategies import analyze_layer_importance, visualize_parameter_reduction
from visualization import create_performance_radar_chart

st.set_page_config(page_title="Results Analysis", page_icon="📈", layout="wide")

st.title("📈 Results Analysis & Insights")

st.markdown("""
Comprehensive analysis of training results and freezing strategy effectiveness.
""")

# Check if training is completed
if not st.session_state.get('models_trained', False):
    st.error("❌ No training results found. Please complete model training first.")
    st.info("💡 Go to the Model Training page to train your models.")
    st.stop()

# Get results from session state
training_results = st.session_state.training_results
freezing_info = st.session_state.freezing_info
dataset_stats = st.session_state.dataset_stats

# Sidebar for analysis options
st.sidebar.header("Analysis Options")

analysis_type = st.sidebar.selectbox(
    "Analysis Focus",
    options=[
        "Overall Performance",
        "Training Efficiency", 
        "Layer Analysis",
        "Detailed Metrics"
    ]
)

export_format = st.sidebar.selectbox(
    "Export Format",
    options=["Markdown Report", "CSV Data", "JSON Results"]
)

# Main analysis content
if analysis_type == "Overall Performance":
    st.subheader("🎯 Overall Performance Comparison")
    
    # Performance comparison chart
    perf_fig = create_performance_comparison(training_results)
    if perf_fig:
        st.plotly_chart(perf_fig, use_container_width=True)
    
    # Results summary table
    st.subheader("📊 Detailed Results Summary")
    results_table = create_detailed_results_table(training_results, freezing_info)
    if not results_table.empty:
        st.dataframe(results_table, use_container_width=True)
        
        # Highlight best performing strategy
        best_strategy = results_table.iloc[0]['Strategy']
        best_accuracy = results_table.iloc[0]['Best Accuracy']
        
        st.success(f"🏆 Best performing strategy: **{best_strategy}** with accuracy of **{best_accuracy:.4f}**")
    
    # Performance insights
    st.subheader("🔍 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Performance Rankings")
        
        # Sort strategies by performance
        sorted_results = sorted(
            training_results.items(), 
            key=lambda x: x[1]['best_val_accuracy'], 
            reverse=True
        )
        
        for i, (strategy, result) in enumerate(sorted_results, 1):
            accuracy = result['best_val_accuracy']
            delta = ""
            
            if i > 1:
                baseline_acc = sorted_results[0][1]['best_val_accuracy']
                diff = accuracy - baseline_acc
                delta = f" ({diff:+.4f})"
            
            st.write(f"{i}. **{strategy}**: {accuracy:.4f}{delta}")
    
    with col2:
        st.markdown("### Strategy Analysis")
        
        # Find most efficient strategy (best accuracy with most frozen parameters)
        efficient_strategies = []
        for strategy in training_results:
            if strategy in freezing_info and freezing_info[strategy]['frozen_percentage'] > 0:
                efficiency_score = (
                    training_results[strategy]['best_val_accuracy'] * 
                    (1 + freezing_info[strategy]['frozen_percentage'] / 100)
                )
                efficient_strategies.append((strategy, efficiency_score))
        
        if efficient_strategies:
            most_efficient = max(efficient_strategies, key=lambda x: x[1])[0]
            st.success(f"🚀 Most efficient: **{most_efficient}**")
        
        # Analyze baseline vs frozen performance
        if 'none' in training_results:
            baseline_acc = training_results['none']['best_val_accuracy']
            
            for strategy in training_results:
                if strategy != 'none':
                    frozen_acc = training_results[strategy]['best_val_accuracy']
                    performance_retention = (frozen_acc / baseline_acc) * 100
                    
                    if performance_retention >= 95:
                        st.info(f"✅ {strategy}: {performance_retention:.1f}% performance retention")
                    elif performance_retention >= 90:
                        st.warning(f"⚠️ {strategy}: {performance_retention:.1f}% performance retention")
                    else:
                        st.error(f"❌ {strategy}: {performance_retention:.1f}% performance retention")

elif analysis_type == "Training Efficiency":
    st.subheader("⚡ Training Efficiency Analysis")
    
    # Training curves
    curves_fig = create_training_curves(training_results)
    if curves_fig:
        st.plotly_chart(curves_fig, use_container_width=True)
    
    # Efficiency analysis
    efficiency_fig = create_efficiency_analysis(training_results, freezing_info)
    if efficiency_fig:
        st.plotly_chart(efficiency_fig, use_container_width=True)
    
    # Parameter reduction visualization
    st.subheader("📉 Parameter Reduction Analysis")
    
    param_fig = visualize_parameter_reduction(list(freezing_info.values()))
    if param_fig:
        st.plotly_chart(param_fig, use_container_width=True)
    
    # Training time comparison
    st.subheader("⏱️ Training Time Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Total Training Time")
        for strategy, result in training_results.items():
            training_time = result.get('training_time', 0)
            st.metric(strategy, f"{training_time:.1f}s")
    
    with col2:
        st.markdown("### Time Reduction")
        if 'none' in training_results:
            baseline_time = training_results['none'].get('training_time', 1)
            
            for strategy in training_results:
                if strategy != 'none':
                    strategy_time = training_results[strategy].get('training_time', 0)
                    time_reduction = ((baseline_time - strategy_time) / baseline_time) * 100
                    st.metric(strategy, f"{time_reduction:.1f}%")
    
    with col3:
        st.markdown("### Efficiency Score")
        # Calculate efficiency as accuracy per second
        for strategy, result in training_results.items():
            accuracy = result['best_val_accuracy']
            time = result.get('training_time', 1)
            efficiency = accuracy / time * 1000  # per 1000 seconds
            st.metric(strategy, f"{efficiency:.2f}")

elif analysis_type == "Layer Analysis":
    st.subheader("🧠 Layer-wise Analysis")
    
    # Layer importance analysis
    layer_analysis = analyze_layer_importance(None, training_results)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Observations")
        if layer_analysis['observations']:
            for obs in layer_analysis['observations']:
                st.info(f"💡 {obs}")
        else:
            st.info("Run different freezing strategies to generate layer insights.")
    
    with col2:
        st.markdown("### Recommendations")
        if layer_analysis['recommendations']:
            for rec in layer_analysis['recommendations']:
                st.success(f"✅ {rec}")
        else:
            st.info("Complete training experiments for personalized recommendations.")
    
    # Freezing strategy effectiveness
    st.subheader("🎯 Freezing Strategy Effectiveness")
    
    if freezing_info:
        # Create comparison table
        strategy_data = []
        
        for strategy in training_results:
            row = {
                'Strategy': strategy,
                'Accuracy': training_results[strategy]['best_val_accuracy']
            }
            
            if strategy in freezing_info:
                info = freezing_info[strategy]
                row.update({
                    'Frozen Layers': ', '.join(map(str, info['frozen_layers'])) if info['frozen_layers'] else 'None',
                    'Frozen Params %': info['frozen_percentage'],
                    'Trainable Params': info['trainable_params']
                })
            else:
                row.update({
                    'Frozen Layers': 'None',
                    'Frozen Params %': 0.0,
                    'Trainable Params': 'All'
                })
            
            strategy_data.append(row)
        
        strategy_df = pd.DataFrame(strategy_data)
        strategy_df = strategy_df.sort_values('Accuracy', ascending=False)
        
        st.dataframe(strategy_df, use_container_width=True)
    
    # Layer contribution analysis
    st.subheader("📊 Layer Contribution Analysis")
    
    # Compare early vs late freezing if both are available
    if 'early' in training_results and 'late' in training_results:
        early_acc = training_results['early']['best_val_accuracy']
        late_acc = training_results['late']['best_val_accuracy']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Early Layer Freezing", f"{early_acc:.4f}")
            if 'none' in training_results:
                baseline = training_results['none']['best_val_accuracy']
                early_retention = (early_acc / baseline) * 100
                st.caption(f"Performance retention: {early_retention:.1f}%")
        
        with col2:
            st.metric("Late Layer Freezing", f"{late_acc:.4f}")
            if 'none' in training_results:
                baseline = training_results['none']['best_val_accuracy']
                late_retention = (late_acc / baseline) * 100
                st.caption(f"Performance retention: {late_retention:.1f}%")
        
        # Analysis
        if early_acc > late_acc:
            st.success("🔍 **Finding**: Early layer freezing outperforms late layer freezing, suggesting that task-specific adaptations in later layers are more critical.")
        else:
            st.success("🔍 **Finding**: Late layer freezing outperforms early layer freezing, suggesting that preserving basic linguistic features is more important.")

elif analysis_type == "Detailed Metrics":
    st.subheader("📋 Detailed Metrics Analysis")
    
    # Comprehensive metrics table
    detailed_table = create_detailed_results_table(training_results, freezing_info)
    if not detailed_table.empty:
        st.dataframe(detailed_table, use_container_width=True)
    
    # Per-strategy detailed analysis
    st.subheader("📈 Strategy-by-Strategy Analysis")
    
    for strategy, results in training_results.items():
        with st.expander(f"📊 {strategy} - Detailed Analysis"):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best Validation Accuracy", f"{results['best_val_accuracy']:.4f}")
                final_f1 = results['history']['val_f1'][-1] if results['history']['val_f1'] else 0
                st.metric("Final F1 Score", f"{final_f1:.4f}")
            
            with col2:
                total_time = results.get('training_time', 0)
                st.metric("Total Training Time", f"{total_time:.1f}s")
                avg_epoch_time = np.mean(results['history']['epoch_times']) if results['history']['epoch_times'] else 0
                st.metric("Average Epoch Time", f"{avg_epoch_time:.1f}s")
            
            with col3:
                if strategy in freezing_info:
                    info = freezing_info[strategy]
                    st.metric("Frozen Parameters", f"{info['frozen_percentage']:.1f}%")
                    st.metric("Trainable Parameters", f"{info['trainable_params']:,}")
            
            # Training history plot for this strategy
            if results['history']:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['Training Loss', 'Validation Loss', 'Validation Accuracy', 'Validation F1']
                )
                
                epochs = list(range(1, len(results['history']['train_loss']) + 1))
                
                fig.add_trace(go.Scatter(x=epochs, y=results['history']['train_loss'], name='Train Loss'), row=1, col=1)
                fig.add_trace(go.Scatter(x=epochs, y=results['history']['val_loss'], name='Val Loss'), row=1, col=2)
                fig.add_trace(go.Scatter(x=epochs, y=results['history']['val_accuracy'], name='Val Acc'), row=2, col=1)
                fig.add_trace(go.Scatter(x=epochs, y=results['history']['val_f1'], name='Val F1'), row=2, col=2)
                
                fig.update_layout(height=400, showlegend=False, title=f"Training Progress: {strategy}")
                st.plotly_chart(fig, use_container_width=True)

# Export section
st.markdown("---")
st.subheader("💾 Export Results")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📄 Generate Report"):
        report = export_results_report(training_results, freezing_info, dataset_stats)
        
        st.download_button(
            label="Download Markdown Report",
            data=report,
            file_name=f"persian_pos_freezing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

with col2:
    if st.button("📊 Export CSV"):
        detailed_table = create_detailed_results_table(training_results, freezing_info)
        if not detailed_table.empty:
            csv = detailed_table.to_csv(index=False)
            
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

with col3:
    if st.button("🔧 Export JSON"):
        import json
        
        export_data = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'dataset_stats': dataset_stats,
                'model_architecture': 'distilbert-base-multilingual-cased'
            },
            'training_results': training_results,
            'freezing_info': freezing_info
        }
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean data for JSON serialization
        clean_data = json.loads(json.dumps(export_data, default=convert_numpy))
        
        st.download_button(
            label="Download JSON Data",
            data=json.dumps(clean_data, indent=2),
            file_name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Research insights and conclusions
st.markdown("---")
st.subheader("🔬 Research Insights & Conclusions")

if training_results:
    # Generate automated insights
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("### 🎯 Key Findings")
        
        # Performance analysis
        best_strategy = max(training_results.items(), key=lambda x: x[1]['best_val_accuracy'])
        st.success(f"🏆 Best strategy: **{best_strategy[0]}** ({best_strategy[1]['best_val_accuracy']:.4f} accuracy)")
        
        # Efficiency analysis
        if freezing_info:
            efficient_strategies = [(s, training_results[s]['best_val_accuracy'], freezing_info.get(s, {}).get('frozen_percentage', 0)) 
                                   for s in training_results if s in freezing_info]
            
            if efficient_strategies:
                # Find strategy with good accuracy and high parameter reduction
                efficiency_scores = [(s, acc * (1 + frozen_pct/100)) for s, acc, frozen_pct in efficient_strategies if frozen_pct > 0]
                if efficiency_scores:
                    most_efficient = max(efficiency_scores, key=lambda x: x[1])
                    st.info(f"⚡ Most efficient: **{most_efficient[0]}**")
        
        # Time savings
        if 'none' in training_results:
            baseline_time = training_results['none'].get('training_time', 0)
            time_savings = []
            
            for strategy in training_results:
                if strategy != 'none':
                    strategy_time = training_results[strategy].get('training_time', 0)
                    if baseline_time > 0:
                        savings = ((baseline_time - strategy_time) / baseline_time) * 100
                        if savings > 0:
                            time_savings.append((strategy, savings))
            
            if time_savings:
                best_time_saver = max(time_savings, key=lambda x: x[1])
                st.info(f"⏱️ Fastest training: **{best_time_saver[0]}** ({best_time_saver[1]:.1f}% time reduction)")
    
    with insights_col2:
        st.markdown("### 💡 Practical Recommendations")
        
        # Generate recommendations based on results
        recommendations = []
        
        # Performance-based recommendations
        sorted_by_performance = sorted(training_results.items(), key=lambda x: x[1]['best_val_accuracy'], reverse=True)
        
        if len(sorted_by_performance) > 1:
            best_acc = sorted_by_performance[0][1]['best_val_accuracy']
            second_best_acc = sorted_by_performance[1][1]['best_val_accuracy']
            
            if abs(best_acc - second_best_acc) < 0.01:  # Very close performance
                recommendations.append("🔄 Multiple strategies show similar performance - consider training time and resource constraints for selection")
        
        # Freezing effectiveness
        if 'none' in training_results:
            baseline_acc = training_results['none']['best_val_accuracy']
            
            good_frozen_strategies = [s for s in training_results 
                                    if s != 'none' and training_results[s]['best_val_accuracy'] >= baseline_acc * 0.95]
            
            if good_frozen_strategies:
                recommendations.append(f"✅ {len(good_frozen_strategies)} freezing strategies maintain >95% performance")
            else:
                recommendations.append("⚠️ Freezing significantly impacts performance - consider full fine-tuning for this task")
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        # Future research directions
        st.markdown("### 🚀 Future Research")
        future_directions = [
            "🔍 Analyze attention patterns in frozen vs unfrozen layers",
            "📊 Test on larger Persian datasets for robustness",
            "🌐 Compare with other multilingual models (XLM-R, mT5)",
            "⚗️ Experiment with gradual unfreezing strategies"
        ]
        
        for direction in future_directions:
            st.markdown(f"- {direction}")

# Help section
with st.expander("ℹ️ Analysis Help & Interpretation"):
    st.markdown("""
    ### Understanding the Results
    
    **Accuracy Metrics:**
    - Values closer to 1.0 (100%) are better
    - Differences of 0.01 (1%) can be significant
    - Compare against baseline (no freezing) for context
    
    **Training Efficiency:**
    - Parameter reduction shows computational savings
    - Time reduction indicates faster training
    - Consider accuracy vs efficiency trade-offs
    
    **Strategy Interpretation:**
    
    **Early Freezing performs well** → Basic linguistic features from pre-training are sufficient
    
    **Late Freezing performs well** → Task-specific high-level adaptations aren't critical
    
    **No clear winner** → Task requires balanced adaptation across all layers
    
    ### Statistical Significance
    - Run multiple experiments with different random seeds for robust conclusions
    - Small differences (<1%) may be due to training variance
    - Consider confidence intervals for production decisions
    """)
