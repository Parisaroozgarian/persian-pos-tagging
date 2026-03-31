import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_pos_distribution_chart(pos_counts, title="POS Tag Distribution"):
    """
    Create POS tag distribution visualization
    
    Args:
        pos_counts: Dictionary of POS tag counts
        title: Chart title
    
    Returns:
        plotly figure: Bar chart of POS distribution
    """
    if not pos_counts:
        return None
    
    # Sort by frequency
    sorted_tags = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)
    tags, counts = zip(*sorted_tags)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(tags),
        y=list(counts),
        marker_color='steelblue',
        text=list(counts),
        textposition='outside'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="POS Tags",
        yaxis_title="Frequency",
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig

def create_dataset_overview(stats):
    """
    Create dataset overview visualization
    
    Args:
        stats: Dataset statistics dictionary
    
    Returns:
        plotly figure: Dataset overview dashboard
    """
    if not stats:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Dataset Split Sizes',
            'Sentence Length Distribution',
            'Token Distribution by Split',
            'Dataset Metrics'
        ],
        specs=[
            [{"type": "bar"}, {"type": "histogram"}],
            [{"type": "bar"}, {"type": "table"}]
        ]
    )
    
    # Dataset split sizes
    fig.add_trace(
        go.Bar(
            x=['Training', 'Validation'],
            y=[stats.get('train_size', 0), stats.get('val_size', 0)],
            marker_color=['lightblue', 'lightcoral'],
            name='Split Sizes'
        ),
        row=1, col=1
    )
    
    # Sentence length distribution (simulated for demo)
    lengths = np.random.gamma(2, stats.get('avg_sentence_length', 10)/2, 1000)
    fig.add_trace(
        go.Histogram(
            x=lengths,
            nbinsx=20,
            marker_color='lightgreen',
            name='Sentence Lengths'
        ),
        row=1, col=2
    )
    
    # Token distribution by split
    fig.add_trace(
        go.Bar(
            x=['Training Tokens', 'Validation Tokens'],
            y=[stats.get('total_train_tokens', 0), stats.get('total_val_tokens', 0)],
            marker_color=['darkblue', 'darkred'],
            name='Token Counts'
        ),
        row=2, col=1
    )
    
    # Dataset metrics table
    metrics_data = [
        ['Training Samples', stats.get('train_size', 0)],
        ['Validation Samples', stats.get('val_size', 0)],
        ['Unique POS Tags', stats.get('num_labels', 0)],
        ['Avg Sentence Length', f"{stats.get('avg_sentence_length', 0):.1f}"],
        ['Max Sentence Length', stats.get('max_sentence_length', 0)],
        ['Min Sentence Length', stats.get('min_sentence_length', 0)]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Value'], fill_color='lightgray'),
            cells=dict(values=list(zip(*metrics_data)), fill_color='white')
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Dataset Overview",
        height=700,
        showlegend=False
    )
    
    return fig

def create_model_architecture_diagram():
    """
    Create DistilmBERT architecture diagram
    
    Returns:
        plotly figure: Architecture visualization
    """
    # Create a simplified architecture diagram
    layers = [
        'Input Embeddings',
        'Transformer Layer 0',
        'Transformer Layer 1', 
        'Transformer Layer 2',
        'Transformer Layer 3',
        'Transformer Layer 4',
        'Transformer Layer 5',
        'Classification Head'
    ]
    
    # Create coordinates for the diagram
    y_positions = list(range(len(layers)))
    x_positions = [0.5] * len(layers)
    
    fig = go.Figure()
    
    # Add layer boxes
    for i, layer in enumerate(layers):
        color = 'lightblue' if 'Transformer' in layer else 'lightcoral' if 'Input' in layer else 'lightgreen'
        
        fig.add_shape(
            type="rect",
            x0=0.2, y0=i-0.3,
            x1=0.8, y1=i+0.3,
            fillcolor=color,
            line=dict(color="black", width=1)
        )
        
        fig.add_annotation(
            x=0.5, y=i,
            text=layer,
            showarrow=False,
            font=dict(size=12)
        )
        
        # Add arrows between layers
        if i < len(layers) - 1:
            fig.add_annotation(
                x=0.5, y=i+0.35,
                ax=0.5, ay=i+0.65,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="black"
            )
    
    fig.update_layout(
        title="DistilmBERT Architecture for POS Tagging",
        xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
        yaxis=dict(range=[-0.5, len(layers)-0.5], showticklabels=False, showgrid=False),
        height=600,
        width=400,
        plot_bgcolor='white'
    )
    
    return fig

def create_real_time_training_plot():
    """
    Create real-time training progress visualization
    
    Returns:
        plotly figure: Empty plot for real-time updates
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Training Loss', 'Validation Loss', 'Validation Accuracy', 'Learning Rate'],
        vertical_spacing=0.1
    )
    
    # Initialize empty traces
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Train Loss'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Val Loss'), row=1, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Val Accuracy'), row=2, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Learning Rate'), row=2, col=2)
    
    fig.update_layout(
        title="Real-time Training Progress",
        height=500,
        showlegend=False
    )
    
    return fig

def update_training_plot(fig, epoch_data, strategy_name):
    """
    Update training plot with new data
    
    Args:
        fig: Plotly figure to update
        epoch_data: Dictionary with training metrics
        strategy_name: Name of current strategy
    
    Returns:
        plotly figure: Updated figure
    """
    epochs = list(range(1, len(epoch_data['train_loss']) + 1))
    
    # Update traces
    fig.data[0].x = epochs
    fig.data[0].y = epoch_data['train_loss']
    
    fig.data[1].x = epochs
    fig.data[1].y = epoch_data['val_loss']
    
    fig.data[2].x = epochs
    fig.data[2].y = epoch_data['val_accuracy']
    
    # Update title
    fig.update_layout(title=f"Training Progress: {strategy_name}")
    
    return fig

def create_layer_analysis_heatmap(layer_weights, layer_names=None):
    """
    Create heatmap for layer weight analysis
    
    Args:
        layer_weights: 2D array of layer weights
        layer_names: Optional list of layer names
    
    Returns:
        plotly figure: Heatmap visualization
    """
    if layer_names is None:
        layer_names = [f"Layer {i}" for i in range(len(layer_weights))]
    
    fig = go.Figure(data=go.Heatmap(
        z=layer_weights,
        x=list(range(layer_weights.shape[1])),
        y=layer_names,
        colorscale='RdBu',
        colorbar=dict(title="Weight Magnitude")
    ))
    
    fig.update_layout(
        title="Layer Weight Analysis",
        xaxis_title="Weight Index",
        yaxis_title="Layers",
        height=400
    )
    
    return fig

def create_performance_radar_chart(metrics_dict):
    """
    Create radar chart for performance metrics comparison
    
    Args:
        metrics_dict: Dictionary of metrics for different strategies
    
    Returns:
        plotly figure: Radar chart
    """
    if not metrics_dict:
        return None
    
    # Define metrics to include
    metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'Training Speed']
    
    fig = go.Figure()
    
    for strategy, metrics in metrics_dict.items():
        values = [
            metrics.get('accuracy', 0),
            metrics.get('f1', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            1 - metrics.get('training_time_normalized', 0)  # Inverse for speed
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_names,
            fill='toself',
            name=strategy
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Performance Metrics Comparison",
        showlegend=True
    )
    
    return fig