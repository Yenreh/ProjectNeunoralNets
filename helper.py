"""
Helper functions for Neural Networks Project - Multi-Layer Perceptron
Proyecto Redes Neuronales 2025-II
Universidad Del Valle

This module contains reusable utility functions for:
- Data loading and preprocessing
- Model training and evaluation
- Results saving and visualization
- Experiment tracking
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class DataLoader:
    """Handles data loading and preprocessing for the neural network models."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader with data directory path.
        
        Args:
            data_dir (str): Path to the directory containing CSV files
        """
        self.data_dir = data_dir
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        
    def load_csv_data(self, file_name: str) -> pd.DataFrame:
        """
        Load CSV file from data directory.
        
        Args:
            file_name (str): Name of the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        file_path = os.path.join(self.data_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} samples from {file_name}")
            return df
        except Exception as e:
            raise Exception(f"Error loading {file_name}: {str(e)}")
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load train, validation, and test datasets.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, test data
        """
        train_df = self.load_csv_data("train.csv")
        val_df = self.load_csv_data("validation.csv") 
        test_df = self.load_csv_data("test.csv")
        
        return train_df, val_df, test_df
    
    def preprocess_text_data(self, 
                           train_df: pd.DataFrame, 
                           val_df: pd.DataFrame, 
                           test_df: pd.DataFrame,
                           text_column: str = 'review_body',
                           target_column: str = 'stars',
                           max_words: int = 10000,
                           max_length: int = 100) -> Dict:
        """
        Preprocess text data for neural network training.
        
        Args:
            train_df, val_df, test_df: DataFrames with text data
            text_column (str): Name of text column
            target_column (str): Name of target column
            max_words (int): Maximum vocabulary size
            max_length (int): Maximum sequence length
            
        Returns:
            Dict: Preprocessed data with X_train, y_train, etc.
        """
        print("Preprocessing text data...")
        
        # Combine all text for tokenizer fitting
        all_texts = pd.concat([
            train_df[text_column], 
            val_df[text_column], 
            test_df[text_column]
        ]).fillna("")
        
        # Initialize and fit tokenizer
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(all_texts)
        
        # Convert texts to sequences
        X_train = self.tokenizer.texts_to_sequences(train_df[text_column].fillna(""))
        X_val = self.tokenizer.texts_to_sequences(val_df[text_column].fillna(""))
        X_test = self.tokenizer.texts_to_sequences(test_df[text_column].fillna(""))
        
        # Pad sequences
        X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
        X_val = pad_sequences(X_val, maxlen=max_length, padding='post', truncating='post')
        X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')
        
        # Process target variable
        y_train = self.label_encoder.fit_transform(train_df[target_column])
        y_val = self.label_encoder.transform(val_df[target_column])
        y_test = self.label_encoder.transform(test_df[target_column])
        
        # Convert to categorical for multi-class classification
        num_classes = len(self.label_encoder.classes_)
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
        
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Number of classes: {num_classes}")
        print(f"Sequence length: {max_length}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        
        return {
            'X_train': X_train, 'y_train': y_train_cat,
            'X_val': X_val, 'y_val': y_val_cat, 
            'X_test': X_test, 'y_test': y_test_cat,
            'num_classes': num_classes,
            'vocab_size': len(self.tokenizer.word_index) + 1,
            'max_length': max_length
        }

class ModelTrainer:
    """Handles model training with callbacks and monitoring."""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize ModelTrainer.
        
        Args:
            model_dir (str): Directory to save trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
    def train_model(self, 
                    model: tf.keras.Model,
                    X_train: np.ndarray,
                    y_train: np.ndarray, 
                    X_val: np.ndarray,
                    y_val: np.ndarray,
                    epochs: int = 50,
                    batch_size: int = 32,
                    patience: int = 10,
                    model_name: str = "mlp_model") -> Dict:
        """
        Train a Keras model with early stopping.
        
        Args:
            model: Compiled Keras model
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs (int): Maximum number of epochs
            batch_size (int): Batch size for training
            patience (int): Early stopping patience
            model_name (str): Name for saving the model
            
        Returns:
            Dict: Training history and results
        """
        print(f"Training {model_name}...")
        print(f"Model parameters: {model.count_params():,}")
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        start_time = datetime.now()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save the trained model
        model_path = os.path.join(self.model_dir, f"{model_name}.h5")
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        return {
            'history': history.history,
            'epochs_trained': len(history.history['loss']),
            'training_time': training_time,
            'model_path': model_path,
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else None
        }

class ResultsManager:
    """Handles saving and loading experiment results."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize ResultsManager.
        
        Args:
            output_dir (str): Directory to save experiment results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results_file = os.path.join(output_dir, "experiment_history.json")
    
    def save_experiment_results(self, 
                              experiment_data: Dict,
                              experiment_id: Optional[int] = None) -> int:
        """
        Save experiment results to JSON file.
        
        Args:
            experiment_data (Dict): Complete experiment data
            experiment_id (int, optional): Specific experiment ID
            
        Returns:
            int: Experiment ID assigned
        """
        # Load existing history or create new
        history_data = self.load_experiment_history()
        
        # Assign experiment ID
        if experiment_id is None:
            existing_ids = [exp.get('experiment_id', 0) for exp in history_data.get('experiments', [])]
            experiment_id = max(existing_ids) + 1 if existing_ids else 1
        
        experiment_data['experiment_id'] = experiment_id
        experiment_data['timestamp'] = datetime.now().isoformat()
        
        # Add to history
        if 'experiments' not in history_data:
            history_data['experiments'] = []
        
        history_data['experiments'].append(experiment_data)
        history_data['last_updated'] = datetime.now().isoformat()
        
        # Save to file
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        print(f"Experiment {experiment_id} results saved to {self.results_file}")
        return experiment_id
    
    def load_experiment_history(self) -> Dict:
        """
        Load experiment history from JSON file.
        
        Returns:
            Dict: Complete experiment history
        """
        if not os.path.exists(self.results_file):
            return {"experiments": [], "created": datetime.now().isoformat()}
        
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading experiment history: {e}")
            return {"experiments": [], "created": datetime.now().isoformat()}
    
    def display_experiment_history(self):
        """Display formatted experiment history."""
        history_data = self.load_experiment_history()
        experiments = history_data.get("experiments", [])
        
        if not experiments:
            print("No experiments found in history.")
            return
        
        print(f"\nEXPERIMENT HISTORY ({len(experiments)} experiments)")
        print("=" * 100)
        
        # Summary table with language column
        print(f"\n{'ID':<3} {'Model':<12} {'Lang':<6} {'Accuracy':<10} {'Loss':<10} {'Epochs':<8} {'Time (s)':<10} {'Samples':<10}")
        print("-" * 95)
        
        for exp in experiments:
            training = exp.get('training_results', {})
            config = exp.get('configuration', {})
            dataset = exp.get('dataset_info', {})
            language = config.get('language_filter', 'multi')
            if language is None:
                language = 'multi'
            
            print(f"{exp.get('experiment_id', 0):<3} "
                  f"{config.get('model_type', 'Unknown')[:11]:<12} "
                  f"{language:<6} "
                  f"{training.get('final_val_accuracy', 0):<10.4f} "
                  f"{training.get('final_val_loss', 0):<10.4f} "
                  f"{training.get('epochs_trained', 0):<8} "
                  f"{training.get('training_time', 0):<10.1f} "
                  f"{dataset.get('train_samples', 0):<10,}")
        
        # Best experiment by language
        if experiments:
            # Group by language and find best for each
            lang_groups = {}
            for exp in experiments:
                config = exp.get('configuration', {})
                lang = config.get('language_filter', 'multi')
                if lang is None:
                    lang = 'multi'
                if lang not in lang_groups:
                    lang_groups[lang] = []
                lang_groups[lang].append(exp)
            
            print(f"\nBEST EXPERIMENTS BY LANGUAGE:")
            print("-" * 50)
            for lang, lang_experiments in lang_groups.items():
                best_exp = max(lang_experiments, 
                              key=lambda x: x.get('training_results', {}).get('final_val_accuracy', 0))
                accuracy = best_exp.get('training_results', {}).get('final_val_accuracy', 0)
                exp_id = best_exp.get('experiment_id', 0)
                samples = best_exp.get('dataset_info', {}).get('train_samples', 0)
                print(f"{lang:<6}: ID #{exp_id} - Accuracy: {accuracy:.4f} ({samples:,} samples)")
            
            # Overall best
            overall_best = max(experiments, 
                              key=lambda x: x.get('training_results', {}).get('final_val_accuracy', 0))
            best_accuracy = overall_best.get('training_results', {}).get('final_val_accuracy', 0)
            best_lang = overall_best.get('configuration', {}).get('language_filter', 'multi')
            if best_lang is None:
                best_lang = 'multi'
            print(f"\nOVERALL BEST: ID #{overall_best.get('experiment_id')} ({best_lang}) - Accuracy: {best_accuracy:.4f}")

class Visualizer:
    """Handles visualization of training results and model performance."""
    
    @staticmethod
    def plot_training_history(history: Dict, 
                            model_name: str = "Model",
                            save_path: Optional[str] = None):
        """
        Plot training and validation metrics.
        
        Args:
            history (Dict): Training history from Keras
            model_name (str): Name for plot title
            save_path (str, optional): Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        if 'accuracy' in history and 'val_accuracy' in history:
            axes[0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
            axes[0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
            axes[0].set_title(f'{model_name} - Accuracy')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[1].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title(f'{model_name} - Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            class_names: List[str],
                            model_name: str = "Model",
                            save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels  
            class_names (List[str]): Names of classes
            model_name (str): Name for plot title
            save_path (str, optional): Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()

def evaluate_model(model: tf.keras.Model,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  class_names: List[str] = None) -> Dict:
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained Keras model
        X_test, y_test: Test data
        class_names: Names of classes for classification report
        
    Returns:
        Dict: Evaluation metrics
    """
    print("Evaluating model on test data...")
    
    # Get predictions
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
    
    # Calculate metrics
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Classification report
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'classification_report': report,
        'y_true': y_true,
        'y_pred': y_pred
    }

def get_gpu_info():
    """Get GPU information for experiment tracking."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_info = {
                'gpu_available': True,
                'gpu_count': len(gpus),
                'gpu_names': [tf.config.experimental.get_device_details(gpu)['device_name'] for gpu in gpus]
            }
        else:
            gpu_info = {'gpu_available': False, 'gpu_count': 0, 'gpu_names': []}
        return gpu_info
    except Exception as e:
        return {'gpu_available': False, 'error': str(e)}

# Utility function for experiment setup
def setup_experiment_environment(seed: int = 42):
    """
    Setup reproducible environment for experiments.
    
    Args:
        seed (int): Random seed for reproducibility
    """
    # Set seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Configure GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU configured: {len(gpus)} GPU(s) available")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU available, using CPU")
    
    return get_gpu_info()