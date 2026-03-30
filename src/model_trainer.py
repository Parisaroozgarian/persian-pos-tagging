import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer, 
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
        # ✅ FIX 1: Auto-detect GPU instead of hardcoding CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ✅ FIX 2: Removed torch.set_num_threads(1) — it was crippling performance
        
    def create_model(self, num_labels, label2id, id2label):
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label
        )
        return model.to(self.device)
    
    def apply_freezing_strategy(self, model, strategy, layer_indices=None):
        total_layers = len(model.distilbert.transformer.layer)
        frozen_layers = []
        
        for param in model.parameters():
            param.requires_grad = True
        
        if strategy == 'none':
            pass
        elif strategy == 'early':
            freeze_count = total_layers // 2
            for i in range(freeze_count):
                for param in model.distilbert.transformer.layer[i].parameters():
                    param.requires_grad = False
                frozen_layers.append(i)
        elif strategy == 'late':
            freeze_start = total_layers // 2
            for i in range(freeze_start, total_layers):
                for param in model.distilbert.transformer.layer[i].parameters():
                    param.requires_grad = False
                frozen_layers.append(i)
        elif strategy == 'alternating':
            for i in range(0, total_layers, 2):
                for param in model.distilbert.transformer.layer[i].parameters():
                    param.requires_grad = False
                frozen_layers.append(i)
        elif strategy == 'custom' and layer_indices:
            for i in layer_indices:
                if 0 <= i < total_layers:
                    for param in model.distilbert.transformer.layer[i].parameters():
                        param.requires_grad = False
                    frozen_layers.append(i)
        
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
        # ✅ FIX 3: Only optimize parameters that require gradients
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate
        )
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'epoch_times': []
        }
        
        best_val_accuracy = 0
        best_model_state = None

        # ✅ FIX 4: Push live metrics into session_state so UI can poll them
        st.session_state['live_metrics'] = {
            'device': str(self.device),
            'epoch': 0,
            'total_epochs': epochs,
            'batch': 0,
            'total_batches': len(train_dataloader),
            'current_loss': None,
            'history': training_history,
        }
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_train_loss = 0
            model.train()
            
            for batch_idx, batch in enumerate(train_dataloader):
                input_ids      = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels         = batch['labels'].to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                # ✅ FIX 5: Gradient clipping for stable training
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # ✅ FIX 6: Update session_state every batch so UI shows live loss
                st.session_state['live_metrics'].update({
                    'epoch': epoch + 1,
                    'batch': batch_idx + 1,
                    'current_loss': round(loss.item(), 4),
                })

                if progress_callback:
                    progress_callback(epoch, batch_idx, len(train_dataloader), loss.item())
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            val_metrics    = self.evaluate_model(model, val_dataloader)
            epoch_time     = time.time() - epoch_start_time

            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_accuracy'].append(val_metrics['accuracy'])
            training_history['val_f1'].append(val_metrics['f1'])
            training_history['epoch_times'].append(epoch_time)

            # ✅ FIX 7: Push epoch results so UI shows accuracy/F1 after each epoch
            st.session_state['live_metrics']['history'] = training_history
            
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                best_model_state  = copy.deepcopy(model.state_dict())
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Loss: {avg_train_loss:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"Device: {self.device} | "
                  f"Time: {epoch_time:.1f}s")
        
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return {
            'model': model,
            'history': training_history,
            'best_val_accuracy': best_val_accuracy
        }
    
    def evaluate_model(self, model, dataloader):
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids      = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels         = batch['labels'].to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        metrics  = self.compute_metrics(all_predictions, all_labels)
        metrics['loss'] = avg_loss
        
        return metrics


def create_data_loaders(train_dataset, val_dataset, batch_size=16):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader