import streamlit as st
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from collections import Counter
import numpy as np

class PersianPOSDataset(Dataset):
    """Custom dataset for Persian POS tagging"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
    def __len__(self):
        return len(self.labels)

@st.cache_data
def load_persian_ud_dataset(subset_size=None):
    """
    Load Persian Universal Dependencies dataset
    
    Args:
        subset_size: Optional integer to limit dataset size for faster training
    
    Returns:
        dict: Dictionary containing train and validation datasets
    """
    try:
        # Load the Persian-Seraji UD dataset
        dataset = load_dataset("universalDependencies/universal_dependencies", "fa_seraji", trust_remote_code=True)
        
        train_data = dataset['train']
        test_data = dataset['test']
        
        # If subset_size is specified, take a random subset
        if subset_size and subset_size < len(train_data):
            indices = np.random.choice(len(train_data), subset_size, replace=False)
            train_data = train_data.select(indices)
            
            # Proportionally reduce test set
            test_subset_size = min(len(test_data), subset_size // 4)
            test_indices = np.random.choice(len(test_data), test_subset_size, replace=False)
            test_data = test_data.select(test_indices)
        
        return {
            'train': train_data,
            'validation': test_data
        }
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def extract_pos_data(dataset_split):
    """
    Extract tokens and POS tags from UD dataset format
    
    Args:
        dataset_split: HuggingFace dataset split
    
    Returns:
        tuple: (sentences, pos_tags) where each is a list of lists
    """
    sentences = []
    pos_tags = []
    
    for example in dataset_split:
        tokens = example['tokens']
        upos = example['upos']
        
        # Filter out empty tokens and corresponding tags
        filtered_tokens = []
        filtered_pos = []
        
        for token, pos in zip(tokens, upos):
            if token.strip():  # Non-empty token
                filtered_tokens.append(token)
                filtered_pos.append(pos)
        
        if filtered_tokens:  # Only add non-empty sentences
            sentences.append(filtered_tokens)
            pos_tags.append(filtered_pos)
    
    return sentences, pos_tags

def create_label_mapping(pos_tags):
    """
    Create mapping between POS tags and indices
    
    Args:
        pos_tags: List of lists containing POS tags
    
    Returns:
        tuple: (label2id, id2label) dictionaries
    """
    all_tags = set()
    for tags in pos_tags:
        all_tags.update(tags)
    
    sorted_tags = sorted(list(all_tags))
    label2id = {tag: idx for idx, tag in enumerate(sorted_tags)}
    id2label = {idx: tag for tag, idx in label2id.items()}
    
    return label2id, id2label

def tokenize_and_align_labels(sentences, pos_tags, tokenizer, label2id, max_length=128):
    """
    Tokenize sentences and align POS labels with subword tokens
    
    Args:
        sentences: List of token lists
        pos_tags: List of POS tag lists
        tokenizer: HuggingFace tokenizer
        label2id: Mapping from labels to IDs
        max_length: Maximum sequence length
    
    Returns:
        tuple: (tokenized_inputs, aligned_labels)
    """
    tokenized_inputs = []
    aligned_labels = []
    
    for sentence, tags in zip(sentences, pos_tags):
        # Tokenize the sentence
        encoding = tokenizer(
            sentence,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True
        )
        
        # Align labels with subword tokens
        labels = []
        word_ids = encoding.word_ids()
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get label -100 (ignored in loss)
                labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subword token of a word gets the actual label
                if word_idx < len(tags):
                    labels.append(label2id[tags[word_idx]])
                else:
                    labels.append(-100)
            else:
                # Other subword tokens get label -100 (ignored in loss)
                labels.append(-100)
            
            previous_word_idx = word_idx
        
        # Remove offset_mapping as it's not needed for training
        del encoding['offset_mapping']
        
        tokenized_inputs.append(encoding)
        aligned_labels.append(labels)
    
    return tokenized_inputs, aligned_labels

def prepare_datasets(dataset_dict, tokenizer_name="distilbert-base-multilingual-cased", max_length=128, subset_size=None):
    """
    Prepare datasets for training
    
    Args:
        dataset_dict: Dictionary with train/validation splits
        tokenizer_name: Name of the tokenizer to use
        max_length: Maximum sequence length
        subset_size: Optional size limit for dataset
    
    Returns:
        dict: Prepared datasets and metadata
    """
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Extract data from both splits
    train_sentences, train_pos = extract_pos_data(dataset_dict['train'])
    val_sentences, val_pos = extract_pos_data(dataset_dict['validation'])
    
    # Create label mapping from training data
    label2id, id2label = create_label_mapping(train_pos + val_pos)
    
    # Tokenize and align labels
    train_encodings, train_labels = tokenize_and_align_labels(
        train_sentences, train_pos, tokenizer, label2id, max_length
    )
    val_encodings, val_labels = tokenize_and_align_labels(
        val_sentences, val_pos, tokenizer, label2id, max_length
    )
    
    # Convert to batch format
    train_batch = {
        'input_ids': [enc['input_ids'] for enc in train_encodings],
        'attention_mask': [enc['attention_mask'] for enc in train_encodings]
    }
    val_batch = {
        'input_ids': [enc['input_ids'] for enc in val_encodings],
        'attention_mask': [enc['attention_mask'] for enc in val_encodings]
    }
    
    # Create datasets
    train_dataset = PersianPOSDataset(train_batch, train_labels)
    val_dataset = PersianPOSDataset(val_batch, val_labels)
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'tokenizer': tokenizer,
        'label2id': label2id,
        'id2label': id2label,
        'num_labels': len(label2id),
        'train_sentences': train_sentences,
        'val_sentences': val_sentences,
        'train_pos': train_pos,
        'val_pos': val_pos
    }

def get_dataset_statistics(data_dict):
    """
    Calculate dataset statistics
    
    Args:
        data_dict: Prepared dataset dictionary
    
    Returns:
        dict: Statistics about the dataset
    """
    train_pos_flat = [tag for tags in data_dict['train_pos'] for tag in tags]
    val_pos_flat = [tag for tags in data_dict['val_pos'] for tag in tags]
    
    train_pos_counts = Counter(train_pos_flat)
    val_pos_counts = Counter(val_pos_flat)
    
    stats = {
        'train_size': len(data_dict['train_dataset']),
        'val_size': len(data_dict['val_dataset']),
        'num_labels': data_dict['num_labels'],
        'train_pos_distribution': dict(train_pos_counts),
        'val_pos_distribution': dict(val_pos_counts),
        'total_train_tokens': len(train_pos_flat),
        'total_val_tokens': len(val_pos_flat),
        'avg_sentence_length': np.mean([len(sent) for sent in data_dict['train_sentences']]),
        'max_sentence_length': max([len(sent) for sent in data_dict['train_sentences']]),
        'min_sentence_length': min([len(sent) for sent in data_dict['train_sentences']])
    }
    
    return stats
