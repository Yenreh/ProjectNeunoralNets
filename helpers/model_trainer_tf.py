"""
Model Training Module for TensorFlow/Keras
Proyecto Redes Neuronales 2025-II - Universidad Del Valle

Maneja el entrenamiento de modelos con TensorFlow/Keras.
Compatible con la Parte 1 del proyecto.
"""

import os
import numpy as np
from datetime import datetime
from typing import Dict
import tensorflow as tf


class ModelTrainer:
    """Maneja el entrenamiento de modelos TensorFlow con callbacks y monitoreo."""
    
    def __init__(self, model_dir: str = "models/project_part_1"):
        """
        Inicializa ModelTrainer.
        
        Args:
            model_dir (str): Directorio para guardar modelos entrenados
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
        Entrena un modelo de Keras con early stopping.
        
        Args:
            model: Modelo de Keras compilado
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            epochs (int): Número máximo de épocas
            batch_size (int): Tamaño del batch para entrenamiento
            patience (int): Paciencia para early stopping
            model_name (str): Nombre para guardar el modelo
            
        Returns:
            Dict: Historial y resultados del entrenamiento
        """
        print(f"Entrenando {model_name}...")
        print(f"Parámetros del modelo: {model.count_params():,}")
        
        # Asegurar que los datos sean float32
        if X_train.dtype != np.float32:
            X_train = X_train.astype(np.float32)
            X_val = X_val.astype(np.float32)
    
        print(f"Forma de entrada: {X_train.shape}")
        print(f"Tipo de datos: {X_train.dtype}")
        
        # Definir callbacks
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
        
        # Entrenar el modelo
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
        
        # Guardar el modelo entrenado
        model_path = os.path.join(self.model_dir, f"{model_name}.h5")
        model.save(model_path)
        print(f"Modelo guardado en: {model_path}")
        
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
