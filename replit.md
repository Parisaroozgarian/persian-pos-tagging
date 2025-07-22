# Persian PoS Tagging - Layer Freezing Research

## Overview

This repository contains a Streamlit-based research application that explores the impact of partial layer freezing when fine-tuning distilled multilingual transformer models (specifically DistilmBERT) for Part-of-Speech tagging in Persian. The application provides an interactive interface for conducting experiments with different freezing strategies and analyzing their effects on model performance.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with multi-page architecture
- **Structure**: Main app.py entry point with separate pages for different functionality
- **Pages**:
  - Data Exploration (pages/1_Data_Exploration.py)
  - Model Training (pages/2_Model_Training.py) 
  - Results Analysis (pages/3_Results_Analysis.py)
- **State Management**: Uses Streamlit session state to persist data and results across pages

### Backend Architecture
- **ML Framework**: PyTorch with HuggingFace Transformers
- **Model Architecture**: DistilmBERT-based token classification for sequence labeling
- **Training Pipeline**: Custom ModelTrainer class with freezing strategy support
- **Data Processing**: Custom dataset classes and preprocessing pipelines

### Key Design Patterns
- **Modular Design**: Functionality separated into focused modules in src/ directory
- **Strategy Pattern**: Implemented for different layer freezing approaches
- **Caching**: Streamlit caching decorators for expensive operations like data loading

## Key Components

### Data Management (`src/data_loader.py`)
- **Purpose**: Handles loading and preprocessing of Persian Universal Dependencies dataset
- **Key Features**:
  - Dataset subsetting for faster experimentation
  - Custom PyTorch Dataset class for POS tagging
  - Tokenization with configurable tokenizers
  - Statistical analysis of dataset properties

### Model Training (`src/model_trainer.py`)
- **Purpose**: Manages model training with various freezing strategies
- **Key Features**:
  - Model instantiation and configuration
  - Layer freezing implementation for different strategies
  - Training loop with metrics tracking
  - Support for different optimization strategies

### Freezing Strategies (`src/freezing_strategies.py`)
- **Purpose**: Implements different approaches to layer freezing
- **Available Strategies**:
  - No freezing (baseline)
  - Early layer freezing (preserve basic linguistic features)
  - Late layer freezing (allow task-specific adaptation)
  - Alternating layer freezing (maintain information flow)
  - Custom layer selection
- **Rationale**: Each strategy tests different hypotheses about which layers contain transferable vs. task-specific knowledge

### Evaluation (`src/evaluation.py`)
- **Purpose**: Comprehensive analysis and comparison of training results
- **Key Features**:
  - Performance comparison across strategies
  - Training efficiency analysis
  - Statistical significance testing
  - Result export functionality

### Visualization (`src/visualization.py`)
- **Purpose**: Interactive charts and visualizations for data exploration and results
- **Technologies**: Plotly for interactive charts, seamless Streamlit integration
- **Chart Types**: Bar charts, line plots, radar charts, confusion matrices

## Data Flow

1. **Data Loading**: Persian UD dataset loaded from HuggingFace datasets
2. **Preprocessing**: Tokenization using multilingual tokenizers, label alignment
3. **Dataset Creation**: Custom PyTorch datasets with configurable parameters
4. **Model Training**: 
   - Model instantiation with freezing strategy applied
   - Training loop with real-time metrics tracking
   - Results storage in session state
5. **Evaluation**: Performance comparison and statistical analysis
6. **Visualization**: Interactive charts for results exploration

## External Dependencies

### Core ML Libraries
- **PyTorch**: Deep learning framework for model training
- **Transformers**: HuggingFace library for pre-trained models
- **Datasets**: HuggingFace library for dataset loading
- **scikit-learn**: Metrics and evaluation utilities

### Web Framework
- **Streamlit**: Web application framework for interactive UI

### Visualization
- **Plotly**: Interactive charting library
- **Pandas/NumPy**: Data manipulation and analysis

### Dataset
- **Universal Dependencies**: Persian-Seraji dataset for POS tagging
- **Source**: HuggingFace datasets hub

## Deployment Strategy

### Local Development
- **Environment**: Python virtual environment with requirements.txt
- **Data Storage**: Session state for temporary results, no persistent database
- **Compute**: Local CPU/GPU depending on availability

### Potential Cloud Deployment
- **Platform**: Streamlit Cloud, Heroku, or similar PaaS
- **Considerations**: 
  - Model size and memory requirements
  - Training time limitations on free tiers
  - Dataset download and caching strategies

### Resource Management
- **Memory Optimization**: Dataset subsetting options for limited resources
- **Training Efficiency**: Configurable hyperparameters to balance speed vs. performance
- **Caching**: Streamlit caching for expensive operations like data loading

## Research Methodology

The application implements a systematic approach to studying layer freezing:

1. **Baseline Establishment**: Full fine-tuning without freezing
2. **Hypothesis Testing**: Different freezing strategies based on transformer architecture understanding
3. **Comparative Analysis**: Statistical comparison of performance and efficiency
4. **Interpretability**: Visualization of which strategies work best and why

The modular design allows researchers to easily extend the system with new freezing strategies or evaluation metrics.