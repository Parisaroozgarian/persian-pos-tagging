import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

def create_performance_comparison(results_dict):
    """
    Create performance comparison visualization
    
    Args:
        results_dict: Dictionary containing results from different strategies
    
    Returns:
        plotly figure: Performance comparison chart
    """
    if not results_dict:
        return None
    
    strategies = list(results_dict.keys())
    accuracies = [results_dict[s].get('best_val_accuracy', 0) for s in strategies]
    f1_scores = [results_dict[s]['history']['val_f1'][-1] if results_dict[s]['history']['val_f1'] else 0 for s in strategies]
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Accuracy',
        x=strategies,
        y=accuracies,
        marker_color='lightblue',
        yaxis='y1'
    ))
    
    fig.add_trace(go.Bar(
        name='F1 Score',
        x=strategies,
        y=f1_scores,
        marker_color='lightcoral',
        yaxis='y1'
    ))
    
    fig.update_layout(
        title='Model Performance Comparison Across Freezing Strategies',
        xaxis_title='Freezing Strategy',
        yaxis_title='Score',
        barmode='group',
        height=500,
        legend=dict(x=0.7, y=1)
    )
    
    # Add performance annotations
    for i, (strategy, acc, f1) in enumerate(zip(strategies, accuracies, f1_scores)):
        fig.add_annotation(
            x=strategy,
            y=max(acc, f1) + 0.01,
            text=f"Acc: {acc:.3f}<br>F1: {f1:.3f}",
            showarrow=False,
            font=dict(size=10)
        )
    
    return fig

def create_training_curves(results_dict):
    """
    Create training curves for all strategies
    
    Args:
        results_dict: Dictionary containing training results
    
    Returns:
        plotly figure: Training curves
    """
    if not results_dict:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Training Loss', 'Validation Loss', 'Validation Accuracy', 'Validation F1'],
        vertical_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set1[:len(results_dict)]
    
    for i, (strategy, results) in enumerate(results_dict.items()):
        history = results['history']
        epochs = list(range(1, len(history['train_loss']) + 1))
        color = colors[i % len(colors)]
        
        # Training loss
        fig.add_trace(
            go.Scatter(
                x=epochs, y=history['train_loss'],
                mode='lines+markers',
                name=f'{strategy} (train loss)',
                line=dict(color=color),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Validation loss
        fig.add_trace(
            go.Scatter(
                x=epochs, y=history['val_loss'],
                mode='lines+markers',
                name=f'{strategy} (val loss)',
                line=dict(color=color, dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Validation accuracy
        fig.add_trace(
            go.Scatter(
                x=epochs, y=history['val_accuracy'],
                mode='lines+markers',
                name=f'{strategy} (val acc)',
                line=dict(color=color, dash='dot'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Validation F1
        fig.add_trace(
            go.Scatter(
                x=epochs, y=history['val_f1'],
                mode='lines+markers',
                name=f'{strategy} (val f1)',
                line=dict(color=color, dash='dashdot'),
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title='Training Progress Across All Strategies',
        height=600,
        showlegend=True
    )
    
    # Update axes labels
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Epoch", row=i, col=j)
    
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=2, col=1)
    fig.update_yaxes(title_text="F1 Score", row=2, col=2)
    
    return fig

def create_efficiency_analysis(results_dict, freezing_info_dict):
    """
    Create efficiency analysis comparing training time and parameter usage
    
    Args:
        results_dict: Training results
        freezing_info_dict: Freezing strategy information
    
    Returns:
        plotly figure: Efficiency analysis
    """
    if not results_dict or not freezing_info_dict:
        return None
    
    strategies = list(results_dict.keys())
    total_times = []
    accuracies = []
    param_reductions = []
    
    for strategy in strategies:
        # Calculate total training time
        epoch_times = results_dict[strategy]['history']['epoch_times']
        total_time = sum(epoch_times)
        total_times.append(total_time)
        
        # Get final accuracy
        accuracies.append(results_dict[strategy].get('best_val_accuracy', 0))
        
        # Get parameter reduction
        if strategy in freezing_info_dict:
            param_reductions.append(freezing_info_dict[strategy]['frozen_percentage'])
        else:
            param_reductions.append(0)
    
    # Create efficiency scatter plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=total_times,
        y=accuracies,
        mode='markers+text',
        text=strategies,
        textposition='top center',
        marker=dict(
            size=[10 + pr/5 for pr in param_reductions],  # Size based on parameter reduction
            color=param_reductions,
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(title="Frozen Parameters (%)")
        ),
        hovertemplate='<b>%{text}</b><br>' +
                      'Training Time: %{x:.1f}s<br>' +
                      'Accuracy: %{y:.3f}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Training Efficiency Analysis',
        xaxis_title='Total Training Time (seconds)',
        yaxis_title='Best Validation Accuracy',
        height=500
    )
    
    return fig

def create_detailed_results_table(results_dict, freezing_info_dict):
    """
    Create detailed results table
    
    Args:
        results_dict: Training results
        freezing_info_dict: Freezing information
    
    Returns:
        pandas DataFrame: Detailed results
    """
    if not results_dict:
        return pd.DataFrame()
    
    data = []
    
    for strategy, results in results_dict.items():
        history = results['history']
        
        row = {
            'Strategy': strategy,
            'Best Accuracy': results.get('best_val_accuracy', 0),
            'Final F1': history['val_f1'][-1] if history['val_f1'] else 0,
            'Final Val Loss': history['val_loss'][-1] if history['val_loss'] else 0,
            'Avg Epoch Time (s)': np.mean(history['epoch_times']) if history['epoch_times'] else 0,
            'Total Time (s)': sum(history['epoch_times']) if history['epoch_times'] else 0
        }
        
        # Add freezing information if available
        if strategy in freezing_info_dict:
            info = freezing_info_dict[strategy]
            row.update({
                'Frozen Layers': ', '.join(map(str, info['frozen_layers'])) if info['frozen_layers'] else 'None',
                'Trainable Params': f"{info['trainable_params']:,}",
                'Frozen %': f"{info['frozen_percentage']:.1f}%"
            })
        else:
            row.update({
                'Frozen Layers': 'N/A',
                'Trainable Params': 'N/A',
                'Frozen %': 'N/A'
            })
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Sort by accuracy descending
    df = df.sort_values('Best Accuracy', ascending=False)
    
    return df

def generate_confusion_matrix(model, dataloader, id2label, tokenizer):
    """
    Generate confusion matrix for model predictions
    
    Args:
        model: Trained model
        dataloader: Data loader for evaluation
        id2label: ID to label mapping
        tokenizer: Tokenizer used
    
    Returns:
        tuple: (confusion matrix, classification report)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            # Flatten and filter out -100 labels
            for pred_seq, label_seq in zip(predictions.cpu().numpy(), labels.cpu().numpy()):
                for pred, label in zip(pred_seq, label_seq):
                    if label != -100:
                        all_predictions.append(pred)
                        all_labels.append(label)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Create classification report
    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(all_labels, all_predictions, target_names=target_names, output_dict=True)
    
    return cm, report

def export_results_report(results_dict, freezing_info_dict, dataset_stats):
    """
    Generate comprehensive results report
    
    Args:
        results_dict: Training results
        freezing_info_dict: Freezing information
        dataset_stats: Dataset statistics
    
    Returns:
        str: Formatted report
    """
    report = []
    report.append("# Persian PoS Tagging - Partial Layer Freezing Results\n")
    
    # Dataset information
    report.append("## Dataset Information")
    report.append(f"- Training samples: {dataset_stats.get('train_size', 'N/A')}")
    report.append(f"- Validation samples: {dataset_stats.get('val_size', 'N/A')}")
    report.append(f"- Number of POS tags: {dataset_stats.get('num_labels', 'N/A')}")
    report.append(f"- Average sentence length: {dataset_stats.get('avg_sentence_length', 'N/A'):.1f} tokens")
    report.append("")
    
    # Results summary
    report.append("## Results Summary")
    if results_dict:
        # Sort strategies by performance
        sorted_strategies = sorted(
            results_dict.items(), 
            key=lambda x: x[1].get('best_val_accuracy', 0), 
            reverse=True
        )
        
        report.append("### Performance Ranking:")
        for i, (strategy, results) in enumerate(sorted_strategies, 1):
            acc = results.get('best_val_accuracy', 0)
            f1 = results['history']['val_f1'][-1] if results['history']['val_f1'] else 0
            time = sum(results['history']['epoch_times']) if results['history']['epoch_times'] else 0
            
            report.append(f"{i}. **{strategy}**: Accuracy = {acc:.4f}, F1 = {f1:.4f}, Time = {time:.1f}s")
        
        report.append("")
        
        # Detailed analysis
        report.append("### Detailed Analysis")
        best_strategy, best_results = sorted_strategies[0]
        report.append(f"- **Best performing strategy**: {best_strategy}")
        report.append(f"- **Best accuracy achieved**: {best_results.get('best_val_accuracy', 0):.4f}")
        
        if len(sorted_strategies) > 1:
            baseline_acc = next((r.get('best_val_accuracy', 0) for s, r in sorted_strategies if s == 'none'), None)
            if baseline_acc:
                for strategy, results in sorted_strategies:
                    if strategy != 'none':
                        acc = results.get('best_val_accuracy', 0)
                        diff = acc - baseline_acc
                        report.append(f"- **{strategy}** vs baseline: {diff:+.4f} ({diff/baseline_acc*100:+.1f}%)")
        
        report.append("")
    
    # Freezing strategy analysis
    if freezing_info_dict:
        report.append("## Freezing Strategy Analysis")
        for strategy, info in freezing_info_dict.items():
            report.append(f"### {strategy}")
            report.append(f"- Frozen layers: {info['frozen_layers']}")
            report.append(f"- Trainable parameters: {info['trainable_params']:,}")
            report.append(f"- Parameter reduction: {info['frozen_percentage']:.1f}%")
            report.append("")
    
    # Conclusions and recommendations
    report.append("## Conclusions and Recommendations")
    report.append("Based on the experimental results:")
    report.append("")
    
    if results_dict:
        # Generate automatic insights
        best_strategy = max(results_dict.items(), key=lambda x: x[1].get('best_val_accuracy', 0))[0]
        
        if best_strategy == 'none':
            report.append("- Full fine-tuning (no freezing) achieved the best performance")
            report.append("- Consider this approach when computational resources allow")
        else:
            report.append(f"- {best_strategy} freezing strategy achieved optimal performance")
            report.append("- This suggests effective parameter efficiency can be achieved")
        
        # Efficiency insights
        if freezing_info_dict:
            efficient_strategies = [
                (s, r, freezing_info_dict.get(s, {})) 
                for s, r in results_dict.items() 
                if s in freezing_info_dict and freezing_info_dict[s]['frozen_percentage'] > 0
            ]
            
            if efficient_strategies:
                best_efficient = max(efficient_strategies, key=lambda x: x[1].get('best_val_accuracy', 0))
                report.append(f"- Most efficient strategy: {best_efficient[0]} (parameter reduction: {best_efficient[2]['frozen_percentage']:.1f}%)")
        
        report.append("")
        report.append("### Future Research Directions:")
        report.append("- Experiment with different layer combinations")
        report.append("- Analyze layer-wise representations and activations")
        report.append("- Test on larger datasets and different Persian text domains")
        report.append("- Compare with other multilingual models")
    
    return "\n".join(report)
