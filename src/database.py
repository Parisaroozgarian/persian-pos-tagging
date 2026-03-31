"""
Database models and operations for Persian PoS Tagging research application
"""

import os
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from datetime import datetime
import json
import uuid

Base = declarative_base()

class Dataset(Base):
    """Store dataset configurations and statistics"""
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    subset_size = Column(Integer)
    tokenizer_name = Column(String(100))
    max_length = Column(Integer)
    train_size = Column(Integer)
    val_size = Column(Integer)
    num_labels = Column(Integer)
    avg_sentence_length = Column(Float)
    statistics = Column(JSON)  # Store detailed stats as JSON
    created_at = Column(DateTime, default=func.now())
    
class Experiment(Base):
    """Store experiment configurations and metadata"""
    __tablename__ = 'experiments'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    description = Column(Text)
    dataset_id = Column(Integer)
    model_name = Column(String(100))
    
    # Training hyperparameters
    epochs = Column(Integer)
    batch_size = Column(Integer)
    learning_rate = Column(Float)
    
    # Experiment metadata
    strategies_tested = Column(JSON)  # List of freezing strategies
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    status = Column(String(20), default='created')  # created, running, completed, failed
    
class TrainingResult(Base):
    """Store training results for each strategy"""
    __tablename__ = 'training_results'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(String(36), nullable=False)
    strategy = Column(String(50), nullable=False)
    
    # Performance metrics
    best_val_accuracy = Column(Float)
    final_f1_score = Column(Float)
    final_val_loss = Column(Float)
    
    # Training details
    total_training_time = Column(Float)
    avg_epoch_time = Column(Float)
    
    # Freezing information
    frozen_layers = Column(JSON)
    trainable_params = Column(Integer)
    total_params = Column(Integer)
    frozen_percentage = Column(Float)
    
    # Training history
    training_history = Column(JSON)  # Store complete training curves
    
    created_at = Column(DateTime, default=func.now())

class DatabaseManager:
    """Manage database connections and operations"""
    
    def __init__(self):
        self.engine = None
        self.Session = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database connection"""
        try:
            database_url = os.getenv('DATABASE_URL', 'sqlite:///research.db')
            self.engine = create_engine(database_url)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            
        except Exception as e:
            st.error(f"Failed to initialize database: {str(e)}")
    
    def get_session(self):
        """Get database session"""
        if self.Session:
            return self.Session()
        return None
    
    def save_dataset(self, dataset_config, statistics):
        """Save dataset configuration and statistics"""
        session = self.get_session()
        if not session:
            return None
        
        try:
            dataset = Dataset(
                name=f"Persian_UD_{dataset_config.get('subset_size', 'full')}",
                description="Persian Universal Dependencies dataset for PoS tagging",
                subset_size=dataset_config.get('subset_size'),
                tokenizer_name=dataset_config.get('tokenizer_name'),
                max_length=dataset_config.get('max_length'),
                train_size=statistics.get('train_size'),
                val_size=statistics.get('val_size'),
                num_labels=statistics.get('num_labels'),
                avg_sentence_length=statistics.get('avg_sentence_length'),
                statistics=statistics
            )
            
            session.add(dataset)
            session.commit()
            dataset_id = dataset.id
            session.close()
            
            return dataset_id
            
        except Exception as e:
            session.rollback()
            session.close()
            st.error(f"Failed to save dataset: {str(e)}")
            return None
    
    def create_experiment(self, name, description, dataset_id, config):
        """Create new experiment record"""
        session = self.get_session()
        if not session:
            return None
        
        try:
            experiment = Experiment(
                name=name,
                description=description,
                dataset_id=dataset_id,
                model_name=config.get('model_name'),
                epochs=config.get('epochs'),
                batch_size=config.get('batch_size'),
                learning_rate=config.get('learning_rate'),
                strategies_tested=config.get('strategies', []),
                status='created'
            )
            
            session.add(experiment)
            session.commit()
            experiment_id = experiment.experiment_id
            session.close()
            
            return experiment_id
            
        except Exception as e:
            session.rollback()
            session.close()
            st.error(f"Failed to create experiment: {str(e)}")
            return None
    
    def update_experiment_status(self, experiment_id, status, completed_at=None):
        """Update experiment status"""
        session = self.get_session()
        if not session:
            return False
        
        try:
            experiment = session.query(Experiment).filter_by(experiment_id=experiment_id).first()
            if experiment:
                experiment.status = status
                if completed_at:
                    experiment.completed_at = completed_at
                session.commit()
            
            session.close()
            return True
            
        except Exception as e:
            session.rollback()
            session.close()
            st.error(f"Failed to update experiment: {str(e)}")
            return False
    
    def save_training_result(self, experiment_id, strategy, results, freezing_info):
        """Save training results for a strategy"""
        session = self.get_session()
        if not session:
            return None
        
        try:
            result = TrainingResult(
                experiment_id=experiment_id,
                strategy=strategy,
                best_val_accuracy=results.get('best_val_accuracy'),
                final_f1_score=results['history']['val_f1'][-1] if results['history']['val_f1'] else None,
                final_val_loss=results['history']['val_loss'][-1] if results['history']['val_loss'] else None,
                total_training_time=results.get('training_time'),
                avg_epoch_time=sum(results['history']['epoch_times']) / len(results['history']['epoch_times']) if results['history']['epoch_times'] else None,
                frozen_layers=freezing_info.get('frozen_layers', []),
                trainable_params=freezing_info.get('trainable_params'),
                total_params=freezing_info.get('total_params'),
                frozen_percentage=freezing_info.get('frozen_percentage'),
                training_history=results['history']
            )
            
            session.add(result)
            session.commit()
            result_id = result.id
            session.close()
            
            return result_id
            
        except Exception as e:
            session.rollback()
            session.close()
            st.error(f"Failed to save training result: {str(e)}")
            return None
    
    def get_experiments(self, limit=50):
        """Get list of experiments"""
        session = self.get_session()
        if not session:
            return []
        
        try:
            experiments = session.query(Experiment).order_by(Experiment.created_at.desc()).limit(limit).all()
            result = []
            
            for exp in experiments:
                result.append({
                    'id': exp.experiment_id,
                    'name': exp.name,
                    'description': exp.description,
                    'model_name': exp.model_name,
                    'strategies': exp.strategies_tested,
                    'status': exp.status,
                    'created_at': exp.created_at,
                    'completed_at': exp.completed_at
                })
            
            session.close()
            return result
            
        except Exception as e:
            session.close()
            st.error(f"Failed to get experiments: {str(e)}")
            return []
    
    def get_experiment_results(self, experiment_id):
        """Get results for specific experiment"""
        session = self.get_session()
        if not session:
            return {}
        
        try:
            results = session.query(TrainingResult).filter_by(experiment_id=experiment_id).all()
            experiment_results = {}
            
            for result in results:
                experiment_results[result.strategy] = {
                    'best_val_accuracy': result.best_val_accuracy,
                    'final_f1_score': result.final_f1_score,
                    'final_val_loss': result.final_val_loss,
                    'training_time': result.total_training_time,
                    'avg_epoch_time': result.avg_epoch_time,
                    'history': result.training_history,
                    'freezing_info': {
                        'frozen_layers': result.frozen_layers,
                        'trainable_params': result.trainable_params,
                        'total_params': result.total_params,
                        'frozen_percentage': result.frozen_percentage
                    }
                }
            
            session.close()
            return experiment_results
            
        except Exception as e:
            session.close()
            st.error(f"Failed to get experiment results: {str(e)}")
            return {}
    
    def get_dataset_stats(self):
        """Get dataset usage statistics"""
        session = self.get_session()
        if not session:
            return {}
        
        try:
            total_datasets = session.query(Dataset).count()
            total_experiments = session.query(Experiment).count()
            completed_experiments = session.query(Experiment).filter_by(status='completed').count()
            
            # Get most recent dataset
            recent_dataset = session.query(Dataset).order_by(Dataset.created_at.desc()).first()
            
            session.close()
            
            return {
                'total_datasets': total_datasets,
                'total_experiments': total_experiments,
                'completed_experiments': completed_experiments,
                'recent_dataset': recent_dataset.name if recent_dataset else None
            }
            
        except Exception as e:
            session.close()
            st.error(f"Failed to get dataset stats: {str(e)}")
            return {}

# Global database manager instance
@st.cache_resource
def get_database_manager():
    """Get cached database manager instance"""
    return DatabaseManager()