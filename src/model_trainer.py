import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer, 
    AdamW, 
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import time
import copy

class ModelTrainer:
    """Class for training PoS tagging models with freezing capabilities"""
    
    def __init__(self, model_name="distilbert-base-multilingual-cased"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_model(self, num_labels, label2id, id2label):
        """
        Create a fresh model instance
        
        Args:
            num_labels: Number of POS labels
            label2id: Label to ID mapping
            id2label: ID to label mapping
        
        Returns:
            model: Fresh model instance
        """
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label
        )
        return model.to(self.device)
    
    def apply_freezing_strategy(self, model, strategy, layer_indices=None):
        """
        Apply freezing strategy to model layers
        
        Args:
            model: The model to freeze layers in
            strategy: Freezing strategy ('none', 'early', 'late', 'alternating', 'custom')
            layer_indices: For custom strategy, list of layer indices to freeze
        
        Returns:
            dict: Information about frozen layers
        """
        total_layers = len(model.distilbert.transformer.layer)
        frozen_layers = []
        
        # Reset all parameters to trainable first
        for param in model.parameters():
            param.requires_grad = True
        
        if strategy == 'none':
            # No freezing - all layers trainable
            pass
        
        elif strategy == 'early':
            # Freeze first half of layers
            freeze_count = total_layers // 2
            for i in range(freeze_count):
                for param in model.distilbert.transformer.layer[i].parameters():
                    param.requires_grad = False
                frozen_layers.append(i)
        
        elif strategy == 'late':
            # Freeze last half of layers
            freeze_start = total_layers // 2
            for i in range(freeze_start, total_layers):
                for param in model.distilbert.transformer.layer[i].parameters():
                    param.requires_grad = False
                frozen_layers.append(i)
        
        elif strategy == 'alternating':
            # Freeze every other layer
            for i in range(0, total_layers, 2):
                for param in model.distilbert.transformer.layer[i].parameters():
                    param.requires_grad = False
                frozen_layers.append(i)
        
        elif strategy == 'custom' and layer_indices:
            # Freeze specified layers
            for i in layer_indices:
                if 0 <= i < total_layers:
                    for param in model.distilbert.transformer.layer[i].parameters():
                        param.requires_grad = False
                    frozen_layers.append(i)
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        return {
            'strategy': strategy,
            'frozen_layers': frozen_layers,
            'total_layers': total_layers,
            'trainable_params': trainable_params,
            'total_params': total_params,
            'frozen_percentage': (1 - trainable_params / total_params) * 100
        }
    
    def compute_metrics(self, predictions, labels):
        """
        Compute evaluation metrics
        
        Args:
            predictions: Model predictions
            labels: True labels
        
        Returns:
            dict: Computed metrics
        """
        # Flatten and filter out -100 labels
        flat_predictions = []
        flat_labels = []
        
        for pred_seq, label_seq in zip(predictions, labels):
            for pred, label in zip(pred_seq, label_seq):
                if label != -100:
                    flat_predictions.append(pred)
                    flat_labels.append(label)
        
        accuracy = accuracy_score(flat_labels, flat_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            flat_labels, flat_predictions, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train_model(self, model, train_dataloader, val_dataloader, epochs=3, learning_rate=2e-5, progress_callback=None):
        """
        Train the model
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            progress_callback: Optional callback for progress updates
        
        Returns:
            dict: Training history and final model
        """
        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        model.train()
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'epoch_times': []
        }
        
        best_val_accuracy = 0
        best_model_state = None
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            total_train_loss = 0
            model.train()
            
            train_progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Training")
            for batch_idx, batch in enumerate(train_progress):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Update progress bar
                train_progress.set_postfix({'loss': loss.item()})
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(epoch, batch_idx, len(train_dataloader), loss.item())
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            
            # Validation phase
            val_metrics = self.evaluate_model(model, val_dataloader)
            
            # Record metrics
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_accuracy'].append(val_metrics['accuracy'])
            training_history['val_f1'].append(val_metrics['f1'])
            
            epoch_time = time.time() - epoch_start_time
            training_history['epoch_times'].append(epoch_time)
            
            # Save best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                best_model_state = copy.deepcopy(model.state_dict())
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            print("-" * 50)
        
        # Load best model state
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return {
            'model': model,
            'history': training_history,
            'best_val_accuracy': best_val_accuracy
        }
    
    def evaluate_model(self, model, dataloader):
        """
        Evaluate model on given dataloader
        
        Args:
            model: Model to evaluate
            dataloader: Data loader for evaluation
        
        Returns:
            dict: Evaluation metrics
        """
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                # Get predictions
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        metrics = self.compute_metrics(all_predictions, all_labels)
        metrics['loss'] = avg_loss
        
        return metrics

def create_data_loaders(train_dataset, val_dataset, batch_size=16):
    """
    Create data loaders for training and validation
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
    
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_dataloader, val_dataloader
