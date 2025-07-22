import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def get_available_strategies():
    """
    Get list of available freezing strategies
    
    Returns:
        dict: Strategy names and descriptions
    """
    return {
        'none': {
            'name': 'No Freezing (Baseline)',
            'description': 'All layers are trainable. Standard fine-tuning approach.',
            'rationale': 'Establishes baseline performance with full model adaptation.'
        },
        'early': {
            'name': 'Early Layer Freezing',
            'description': 'Freeze the first half of transformer layers.',
            'rationale': 'Preserves basic linguistic features learned during pre-training.'
        },
        'late': {
            'name': 'Late Layer Freezing',
            'description': 'Freeze the last half of transformer layers.',
            'rationale': 'Allows task-specific adaptation while preserving high-level representations.'
        },
        'alternating': {
            'name': 'Alternating Layer Freezing',
            'description': 'Freeze every other layer (0, 2, 4...).',
            'rationale': 'Maintains information flow while reducing trainable parameters.'
        },
        'custom': {
            'name': 'Custom Layer Selection',
            'description': 'Manually select which layers to freeze.',
            'rationale': 'Targeted freezing based on specific hypotheses or insights.'
        }
    }

def visualize_freezing_strategy(strategy, total_layers=6, frozen_layers=None):
    """
    Create visualization of freezing strategy
    
    Args:
        strategy: Strategy name
        total_layers: Total number of layers in model
        frozen_layers: List of frozen layer indices (for custom strategy)
    
    Returns:
        plotly figure: Visualization of layer states
    """
    strategies = get_available_strategies()
    
    # Determine which layers are frozen based on strategy
    if strategy == 'none':
        frozen_indices = []
    elif strategy == 'early':
        frozen_indices = list(range(total_layers // 2))
    elif strategy == 'late':
        frozen_indices = list(range(total_layers // 2, total_layers))
    elif strategy == 'alternating':
        frozen_indices = list(range(0, total_layers, 2))
    elif strategy == 'custom':
        frozen_indices = frozen_layers if frozen_layers else []
    else:
        frozen_indices = []
    
    # Create visualization data
    layers = list(range(total_layers))
    states = ['Frozen' if i in frozen_indices else 'Trainable' for i in layers]
    colors = ['#ff6b6b' if state == 'Frozen' else '#4ecdc4' for state in states]
    
    # Create figure
    fig = go.Figure()
    
    # Add layer bars
    fig.add_trace(go.Bar(
        x=[f'Layer {i}' for i in layers],
        y=[1] * total_layers,
        marker_color=colors,
        text=states,
        textposition='inside',
        textfont=dict(color='white', size=12),
        hovertemplate='<b>%{x}</b><br>State: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Layer Freezing Strategy: {strategies[strategy]['name']}",
        xaxis_title="Transformer Layers",
        yaxis_title="",
        showlegend=False,
        height=300,
        yaxis=dict(showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def visualize_parameter_reduction(freezing_info_list):
    """
    Visualize parameter reduction across different strategies
    
    Args:
        freezing_info_list: List of freezing information dictionaries
    
    Returns:
        plotly figure: Parameter reduction comparison
    """
    if not freezing_info_list:
        return None
    
    strategies = [info['strategy'] for info in freezing_info_list]
    trainable_params = [info['trainable_params'] for info in freezing_info_list]
    total_params = [info['total_params'] for info in freezing_info_list]
    frozen_percentages = [info['frozen_percentage'] for info in freezing_info_list]
    
    # Create subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Trainable vs Total Parameters', 'Frozen Parameter Percentage'],
        specs=[[{"secondary_y": False}, {"type": "bar"}]]
    )
    
    # Bar chart for trainable vs total parameters
    fig.add_trace(
        go.Bar(name='Total Parameters', x=strategies, y=total_params, marker_color='lightblue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Trainable Parameters', x=strategies, y=trainable_params, marker_color='darkblue'),
        row=1, col=1
    )
    
    # Bar chart for frozen percentage
    fig.add_trace(
        go.Bar(x=strategies, y=frozen_percentages, marker_color='orange', showlegend=False),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Parameter Reduction Analysis",
        height=400,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Freezing Strategy", row=1, col=1)
    fig.update_xaxes(title_text="Freezing Strategy", row=1, col=2)
    fig.update_yaxes(title_text="Number of Parameters", row=1, col=1)
    fig.update_yaxes(title_text="Frozen Percentage (%)", row=1, col=2)
    
    return fig

def get_strategy_recommendations():
    """
    Get recommendations for when to use each strategy
    
    Returns:
        dict: Strategy recommendations
    """
    return {
        'none': {
            'use_when': [
                'Establishing baseline performance',
                'Large amounts of task-specific training data available',
                'Computational resources are abundant'
            ],
            'pros': ['Maximum adaptation capability', 'Best performance potential'],
            'cons': ['High computational cost', 'Risk of catastrophic forgetting']
        },
        'early': {
            'use_when': [
                'Target language has similar linguistic structure to pre-training data',
                'Limited computational resources',
                'Want to preserve syntactic knowledge'
            ],
            'pros': ['Preserves basic linguistic features', 'Faster training', 'Reduced overfitting'],
            'cons': ['May limit adaptation capability', 'Potential performance ceiling']
        },
        'late': {
            'use_when': [
                'Task requires significant adaptation',
                'Different domain from pre-training data',
                'Complex task-specific patterns needed'
            ],
            'pros': ['Allows task-specific learning', 'Maintains representation capacity'],
            'cons': ['May lose some general knowledge', 'Higher computational cost than early freezing']
        },
        'alternating': {
            'use_when': [
                'Balanced approach needed',
                'Moderate computational constraints',
                'Experimental exploration'
            ],
            'pros': ['Maintains information flow', 'Balanced parameter reduction'],
            'cons': ['May not be optimal for specific tasks', 'Complex training dynamics']
        },
        'custom': {
            'use_when': [
                'Specific architectural insights available',
                'Layer-wise analysis has been performed',
                'Fine-grained control needed'
            ],
            'pros': ['Targeted optimization', 'Hypothesis-driven approach'],
            'cons': ['Requires domain expertise', 'Risk of suboptimal choices']
        }
    }

def analyze_layer_importance(model, strategy_results):
    """
    Analyze the importance of different layers based on results
    
    Args:
        model: Trained model
        strategy_results: Dictionary of results from different strategies
    
    Returns:
        dict: Layer importance analysis
    """
    analysis = {
        'observations': [],
        'recommendations': []
    }
    
    # Compare performance across strategies
    if 'none' in strategy_results and 'early' in strategy_results:
        baseline_acc = strategy_results['none'].get('best_val_accuracy', 0)
        early_acc = strategy_results['early'].get('best_val_accuracy', 0)
        
        if early_acc >= baseline_acc * 0.95:  # Within 5% of baseline
            analysis['observations'].append(
                "Early layer freezing maintains good performance, suggesting early layers capture sufficient linguistic knowledge."
            )
        else:
            analysis['observations'].append(
                "Early layer freezing significantly reduces performance, indicating task-specific adaptation in early layers is important."
            )
    
    if 'none' in strategy_results and 'late' in strategy_results:
        baseline_acc = strategy_results['none'].get('best_val_accuracy', 0)
        late_acc = strategy_results['late'].get('best_val_accuracy', 0)
        
        if late_acc >= baseline_acc * 0.95:
            analysis['observations'].append(
                "Late layer freezing maintains good performance, suggesting pre-trained high-level representations are suitable."
            )
        else:
            analysis['observations'].append(
                "Late layer freezing reduces performance, indicating need for task-specific high-level adaptations."
            )
    
    # Generate recommendations based on observations
    if len(analysis['observations']) > 0:
        analysis['recommendations'].append(
            "Consider the trade-off between computational efficiency and performance when choosing freezing strategy."
        )
        analysis['recommendations'].append(
            "Experiment with different layer combinations based on the observed patterns."
        )
    
    return analysis
