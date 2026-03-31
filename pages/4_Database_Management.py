"""
Database Management page for Persian PoS Tagging research application
"""

import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from database import get_database_manager

st.set_page_config(page_title="Database Management", page_icon="🗄️", layout="wide")

st.title("🗄️ Database Management")

st.markdown("""
Manage experimental data, view historical results, and analyze research trends across multiple experiments.
""")

# Initialize database manager
try:
    db_manager = get_database_manager()
    st.success("✅ Database connection established")
except Exception as e:
    st.error(f"❌ Database connection failed: {str(e)}")
    st.stop()

# Sidebar for database operations
st.sidebar.header("Database Operations")

operation = st.sidebar.selectbox(
    "Select Operation",
    options=[
        "Dashboard",
        "Experiment History", 
        "Data Export",
        "Database Statistics",
        "Cleanup & Maintenance"
    ]
)

# Main content based on selected operation
if operation == "Dashboard":
    st.subheader("📊 Research Dashboard")
    
    # Get database statistics
    stats = db_manager.get_dataset_stats()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Datasets", stats.get('total_datasets', 0))
    with col2:
        st.metric("Total Experiments", stats.get('total_experiments', 0))
    with col3:
        st.metric("Completed Experiments", stats.get('completed_experiments', 0))
    with col4:
        completion_rate = 0
        if stats.get('total_experiments', 0) > 0:
            completion_rate = (stats.get('completed_experiments', 0) / stats.get('total_experiments', 0)) * 100
        st.metric("Completion Rate", f"{completion_rate:.1f}%")
    
    # Recent experiments
    st.markdown("---")
    st.subheader("📈 Recent Experiments")
    
    experiments = db_manager.get_experiments(limit=10)
    
    if experiments:
        # Create experiments dataframe
        exp_data = []
        for exp in experiments:
            exp_data.append({
                'Name': exp['name'],
                'Model': exp['model_name'],
                'Strategies': ', '.join(exp['strategies']) if exp['strategies'] else 'None',
                'Status': exp['status'],
                'Created': exp['created_at'].strftime('%Y-%m-%d %H:%M') if exp['created_at'] else 'Unknown'
            })
        
        exp_df = pd.DataFrame(exp_data)
        st.dataframe(exp_df, width='stretch')
    else:
        st.info("No experiments found in database. Start by running some experiments!")
    
    # Experiment trends
    if experiments:
        st.markdown("---")
        st.subheader("📈 Experiment Trends")
        
        # Group experiments by date
        exp_dates = [exp['created_at'].date() for exp in experiments if exp['created_at']]
        date_counts = pd.Series(exp_dates).value_counts().sort_index()
        
        if not date_counts.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=date_counts.index,
                y=date_counts.values,
                mode='lines+markers',
                name='Experiments per Day'
            ))
            
            fig.update_layout(
                title="Experiment Activity Over Time",
                xaxis_title="Date",
                yaxis_title="Number of Experiments",
                height=400
            )
            
            st.plotly_chart(fig, width='stretch')

elif operation == "Experiment History":
    st.subheader("🔬 Experiment History")
    
    experiments = db_manager.get_experiments(limit=50)
    
    if experiments:
        # Experiment selection
        exp_options = {f"{exp['name']} ({exp['id'][:8]})": exp['id'] for exp in experiments}
        selected_exp_name = st.selectbox("Select Experiment", options=list(exp_options.keys()))
        
        if selected_exp_name:
            selected_exp_id = exp_options[selected_exp_name]
            
            # Get experiment details
            selected_exp = next(exp for exp in experiments if exp['id'] == selected_exp_id)
            
            # Display experiment info
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Experiment Details")
                st.write(f"**Name:** {selected_exp['name']}")
                st.write(f"**Description:** {selected_exp['description'] or 'No description'}")
                st.write(f"**Model:** {selected_exp['model_name']}")
                st.write(f"**Status:** {selected_exp['status']}")
                st.write(f"**Created:** {selected_exp['created_at'].strftime('%Y-%m-%d %H:%M')}")
                if selected_exp['completed_at']:
                    st.write(f"**Completed:** {selected_exp['completed_at'].strftime('%Y-%m-%d %H:%M')}")
            
            with col2:
                st.markdown("### Strategies Tested")
                if selected_exp['strategies']:
                    for strategy in selected_exp['strategies']:
                        st.write(f"• {strategy}")
                else:
                    st.write("No strategies recorded")
            
            # Get and display results
            results = db_manager.get_experiment_results(selected_exp_id)
            
            if results:
                st.markdown("---")
                st.subheader("📊 Results Summary")
                
                # Results table
                results_data = []
                for strategy, result in results.items():
                    results_data.append({
                        'Strategy': strategy,
                        'Best Accuracy': f"{result['best_val_accuracy']:.4f}" if result['best_val_accuracy'] else 'N/A',
                        'Final F1': f"{result['final_f1_score']:.4f}" if result['final_f1_score'] else 'N/A',
                        'Training Time': f"{result['training_time']:.1f}s" if result['training_time'] else 'N/A',
                        'Frozen %': f"{result['freezing_info']['frozen_percentage']:.1f}%" if result['freezing_info']['frozen_percentage'] else '0%'
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, width='stretch')
                
                # Performance comparison chart
                if len(results) > 1:
                    strategies = list(results.keys())
                    accuracies = [results[s]['best_val_accuracy'] for s in strategies if results[s]['best_val_accuracy']]
                    
                    if accuracies:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=strategies[:len(accuracies)],
                            y=accuracies,
                            marker_color='lightblue'
                        ))
                        
                        fig.update_layout(
                            title="Strategy Performance Comparison",
                            xaxis_title="Freezing Strategy",
                            yaxis_title="Best Validation Accuracy",
                            height=400
                        )
                        
                        st.plotly_chart(fig, width='stretch')
            else:
                st.info("No results found for this experiment")
    else:
        st.info("No experiments found in database")

elif operation == "Data Export":
    st.subheader("💾 Data Export")
    
    export_type = st.selectbox(
        "Export Type",
        options=["All Experiments", "Specific Experiment", "Summary Report"]
    )
    
    if export_type == "All Experiments":
        st.markdown("### Export All Experiment Data")
        
        experiments = db_manager.get_experiments(limit=1000)
        
        if experiments:
            # Create comprehensive export data
            all_results = []
            
            for exp in experiments:
                results = db_manager.get_experiment_results(exp['id'])
                
                for strategy, result in results.items():
                    all_results.append({
                        'experiment_id': exp['id'],
                        'experiment_name': exp['name'],
                        'model_name': exp['model_name'],
                        'strategy': strategy,
                        'best_accuracy': result['best_val_accuracy'],
                        'final_f1': result['final_f1_score'],
                        'training_time': result['training_time'],
                        'frozen_percentage': result['freezing_info']['frozen_percentage'],
                        'created_at': exp['created_at']
                    })
            
            if all_results:
                export_df = pd.DataFrame(all_results)
                
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download All Results (CSV)",
                    data=csv_data,
                    file_name=f"all_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                st.dataframe(export_df.head(10), width='stretch')
                st.caption(f"Preview showing first 10 of {len(export_df)} total records")
        else:
            st.info("No experiment data available for export")
    
    elif export_type == "Specific Experiment":
        experiments = db_manager.get_experiments()
        
        if experiments:
            exp_options = {f"{exp['name']} ({exp['id'][:8]})": exp['id'] for exp in experiments}
            selected_exp_name = st.selectbox("Select Experiment to Export", options=list(exp_options.keys()))
            
            if selected_exp_name:
                selected_exp_id = exp_options[selected_exp_name]
                results = db_manager.get_experiment_results(selected_exp_id)
                
                if results:
                    # Create detailed export for single experiment
                    detailed_data = []
                    
                    for strategy, result in results.items():
                        detailed_data.append({
                            'strategy': strategy,
                            'best_accuracy': result['best_val_accuracy'],
                            'final_f1': result['final_f1_score'],
                            'final_val_loss': result['final_val_loss'],
                            'training_time': result['training_time'],
                            'avg_epoch_time': result['avg_epoch_time'],
                            'frozen_layers': str(result['freezing_info']['frozen_layers']),
                            'trainable_params': result['freezing_info']['trainable_params'],
                            'frozen_percentage': result['freezing_info']['frozen_percentage']
                        })
                    
                    detailed_df = pd.DataFrame(detailed_data)
                    
                    csv_data = detailed_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Experiment Results (CSV)",
                        data=csv_data,
                        file_name=f"experiment_{selected_exp_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    st.dataframe(detailed_df, width='stretch')

elif operation == "Database Statistics":
    st.subheader("📈 Database Statistics")
    
    stats = db_manager.get_dataset_stats()
    experiments = db_manager.get_experiments(limit=1000)
    
    # Storage statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Storage Usage")
        st.metric("Total Records", stats.get('total_experiments', 0))
        st.metric("Active Experiments", len([e for e in experiments if e['status'] == 'completed']))
        st.metric("Failed Experiments", len([e for e in experiments if e['status'] == 'failed']))
    
    with col2:
        st.markdown("### Performance Insights")
        
        if experiments:
            # Calculate average performance across all experiments
            all_accuracies = []
            all_times = []
            
            for exp in experiments:
                results = db_manager.get_experiment_results(exp['id'])
                for strategy, result in results.items():
                    if result['best_val_accuracy']:
                        all_accuracies.append(result['best_val_accuracy'])
                    if result['training_time']:
                        all_times.append(result['training_time'])
            
            if all_accuracies:
                st.metric("Avg Accuracy", f"{sum(all_accuracies)/len(all_accuracies):.4f}")
            
            if all_times:
                st.metric("Avg Training Time", f"{sum(all_times)/len(all_times):.1f}s")
    
    # Strategy performance analysis
    if experiments:
        st.markdown("---")
        st.subheader("🎯 Strategy Performance Analysis")
        
        strategy_stats = {}
        
        for exp in experiments:
            results = db_manager.get_experiment_results(exp['id'])
            for strategy, result in results.items():
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {'accuracies': [], 'times': []}
                
                if result['best_val_accuracy']:
                    strategy_stats[strategy]['accuracies'].append(result['best_val_accuracy'])
                if result['training_time']:
                    strategy_stats[strategy]['times'].append(result['training_time'])
        
        if strategy_stats:
            strategy_summary = []
            
            for strategy, data in strategy_stats.items():
                if data['accuracies']:
                    avg_acc = sum(data['accuracies']) / len(data['accuracies'])
                    strategy_summary.append({
                        'Strategy': strategy,
                        'Experiments': len(data['accuracies']),
                        'Avg Accuracy': f"{avg_acc:.4f}",
                        'Best Accuracy': f"{max(data['accuracies']):.4f}",
                        'Avg Time': f"{sum(data['times'])/len(data['times']):.1f}s" if data['times'] else 'N/A'
                    })
            
            if strategy_summary:
                strategy_df = pd.DataFrame(strategy_summary)
                st.dataframe(strategy_df, width='stretch')

elif operation == "Cleanup & Maintenance":
    st.subheader("🧹 Database Cleanup & Maintenance")
    
    st.warning("⚠️ Cleanup operations cannot be undone. Please proceed with caution.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Cleanup Options")
        
        if st.button("🗑️ Clear Failed Experiments", type="secondary"):
            # This would require implementing a cleanup method in the database manager
            st.info("Cleanup functionality would be implemented here")
        
        if st.button("📊 Rebuild Statistics", type="secondary"):
            st.info("Statistics rebuild functionality would be implemented here")
    
    with col2:
        st.markdown("### Backup Options")
        
        if st.button("💾 Create Backup", type="secondary"):
            st.info("Backup functionality would be implemented here")
        
        if st.button("📥 Download Full Export", type="secondary"):
            st.info("Full export functionality would be implemented here")

# Help section
with st.expander("ℹ️ Database Management Help"):
    st.markdown("""
    ### Database Features
    
    **Dashboard**: Overview of all research activity and key metrics
    
    **Experiment History**: Detailed view of individual experiments and their results
    
    **Data Export**: Export experimental data in various formats for external analysis
    
    **Database Statistics**: Performance insights and usage analytics
    
    **Cleanup & Maintenance**: Tools for managing database storage and performance
    
    ### Data Storage
    
    The application automatically saves:
    - Dataset configurations and statistics
    - Experiment parameters and metadata
    - Training results and performance metrics
    - Freezing strategy details and outcomes
    
    ### Export Formats
    
    - **CSV**: Tabular data suitable for spreadsheet analysis
    - **JSON**: Structured data for programmatic access
    - **Markdown**: Human-readable reports
    
    ### Best Practices
    
    - Regularly export important results as backups
    - Use descriptive experiment names for easy identification
    - Monitor database statistics to track research progress
    """)