"""
Funciones auxiliares para el Proyecto de Redes Neuronales - Perceptrón Multicapa
Proyecto Redes Neuronales 2025-II
Universidad Del Valle

Este módulo contiene funciones utilitarias reutilizables para:
- Carga y preprocesamiento de datos
- Entrenamiento y evaluación de modelos
- Guardado de resultados y visualización
- Seguimiento de experimentos
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle

# Estilos de visualización
plt.style.use('default')
sns.set_palette("husl")

class DataLoader:
    """Maneja la carga y preprocesamiento de datos para los modelos de redes neuronales."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Inicializa DataLoader con la ruta del directorio de datos.
        
        Args:
            data_dir (str): Ruta al directorio que contiene los archivos CSV
        """
        self.data_dir = data_dir
        self.tokenizer = None
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        
    def load_csv_data(self, file_name: str) -> pd.DataFrame:
        """
        Carga archivo CSV desde el directorio de datos.
        
        Args:
            file_name (str): Nombre del archivo CSV
            
        Returns:
            pd.DataFrame: Datos cargados
        """
        file_path = os.path.join(self.data_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo de datos no encontrado: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            print(f"Datos cargados exitosamente: {file_name}")
            return df
        except Exception as e:
            raise Exception(f"Error cargando {file_name}: {str(e)}")
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Carga los conjuntos de datos de entrenamiento, validación y prueba.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Datos de entrenamiento, validación y prueba
        """
        train_df = self.load_csv_data("train.csv")
        val_df = self.load_csv_data("validation.csv") 
        test_df = self.load_csv_data("test.csv")
        
        return train_df, val_df, test_df
    
    def preprocess_text_data_embedding(self, 
                           train_df: pd.DataFrame, 
                           val_df: pd.DataFrame, 
                           test_df: pd.DataFrame,
                           text_column: str = 'review_body',
                           title_column: str = None,
                           target_column: str = 'stars',
                           max_words: int = None,
                           max_length: int = None,
                           use_title_and_body: bool = False) -> Dict:
        """
        Preprocesa datos de texto para el entrenamiento de redes neuronales usando embeddings.
        
        Args:
            train_df, val_df, test_df: DataFrames con datos de texto
            text_column (str): Nombre de la columna de texto (cuerpo)
            title_column (str): Nombre de la columna de título (opcional)
            target_column (str): Nombre de la columna objetivo
            max_words (int): Tamaño máximo del vocabulario
            max_length (int): Longitud máxima de secuencia
            use_title_and_body (bool): Si True, combina título y cuerpo
            
        Returns:
            Dict: Datos preprocesados con X_train, y_train, etc. (X_* son arrays densos)
        """
        print("Preprocesando datos de texto...")
        
        def combine_title_and_body(df, text_col, title_col, use_combined):
            """Combina título y cuerpo de texto si está habilitado."""
            if use_combined and title_col and title_col in df.columns:
                # Combinar título y cuerpo con un separador
                combined = (df[title_col].fillna("").astype(str) + " " + 
                           df[text_col].fillna("").astype(str))
                return combined
            else:
                return df[text_col].fillna("")
        
        # Combinar todo el texto para ajustar el tokenizer
        train_combined = combine_title_and_body(train_df, text_column, title_column, use_title_and_body)
        val_combined = combine_title_and_body(val_df, text_column, title_column, use_title_and_body)
        test_combined = combine_title_and_body(test_df, text_column, title_column, use_title_and_body)
        
        all_texts = pd.concat([train_combined, val_combined, test_combined])
        
        # Inicializar y ajustar tokenizer
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(all_texts)
        
        # Convertir textos a secuencias
        X_train = self.tokenizer.texts_to_sequences(train_combined)
        X_val = self.tokenizer.texts_to_sequences(val_combined)
        X_test = self.tokenizer.texts_to_sequences(test_combined)
        
        # Aplicar padding a las secuencias
        X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
        X_val = pad_sequences(X_val, maxlen=max_length, padding='post', truncating='post')
        X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')
        
        # Procesar variable objetivo
        y_train = self.label_encoder.fit_transform(train_df[target_column])
        y_val = self.label_encoder.transform(val_df[target_column])
        y_test = self.label_encoder.transform(test_df[target_column])
        
        # Convertir a categórico para clasificación multiclase
        num_classes = len(self.label_encoder.classes_)
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
        
        print(f"Tamaño del vocabulario: {len(self.tokenizer.word_index)}")
        print(f"Número de clases: {num_classes}")
        print(f"Longitud de secuencia: {max_length}")
        print(f"Texto combinado: {'Sí (título + cuerpo)' if use_title_and_body else 'No (solo cuerpo)'}")
        print(f"Muestras de entrenamiento: {X_train.shape[0]}")
        print(f"Muestras de validación: {X_val.shape[0]}")
        print(f"Muestras de prueba: {X_test.shape[0]}")
        
        return {
            'X_train': X_train, 'y_train': y_train_cat,
            'X_val': X_val, 'y_val': y_val_cat, 
            'X_test': X_test, 'y_test': y_test_cat,
            'num_classes': num_classes,
            'vocab_size': len(self.tokenizer.word_index) + 1
        }

    def preprocess_text_data_bow(self, 
                               train_df: pd.DataFrame, 
                               val_df: pd.DataFrame, 
                               test_df: pd.DataFrame,
                               text_column: str = 'review_body',
                               title_column: str = None,
                               target_column: str = 'stars',
                               max_features: int = 5000,
                               min_df: int = 3,
                               max_df: float = 0.85,
                               use_title_and_body: bool = False) -> Dict:
        """
        Preprocesa datos de texto usando BoW/TF-IDF con matrices dispersas.
        Se usan matrices dispersas para reducir significativamente el uso de memoria.
        
        Args:
            train_df, val_df, test_df: DataFrames con datos de texto
            text_column (str): Nombre de la columna de texto (cuerpo)
            title_column (str): Nombre de la columna de título (opcional)
            target_column (str): Nombre de la columna objetivo
            max_features (int): Número máximo de características (5000 por defecto para optimización)
            min_df (int): Frecuencia mínima de documento para incluir término
            max_df (float): Frecuencia máxima de documento (ratio) para filtrar términos comunes
            use_title_and_body (bool): Si True, combina título y cuerpo
            
        Returns:
            Dict: Datos preprocesados con X_train, y_train, etc. (X_* son matrices dispersas)
        """
        print("Preprocesando datos de texto con BoW...")
        print(f"Configuración: max_features={max_features}, min_df={min_df}, max_df={max_df}")
        
        def combine_title_and_body(df, text_col, title_col, use_combined):
            """Combina título y cuerpo de texto si está habilitado."""
            if use_combined and title_col and title_col in df.columns:
                # Combinar título y cuerpo con un separador
                combined = (df[title_col].fillna("").astype(str) + " " + 
                           df[text_col].fillna("").astype(str))
                print(f"Combinando {title_col} + {text_col}")
                return combined
            else:
                print(f"Usando solo {text_col}")
                return df[text_col].fillna("")
        
        # Combinar texto para cada conjunto
        train_combined = combine_title_and_body(train_df, text_column, title_column, use_title_and_body)
        val_combined = combine_title_and_body(val_df, text_column, title_column, use_title_and_body)
        test_combined = combine_title_and_body(test_df, text_column, title_column, use_title_and_body)
        
        # Inicializar vectorizador TF-IDF optimizado
        self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                stop_words='english', 
                lowercase=True,
                strip_accents='unicode',
                ngram_range=(1, 2)
            )

        # Ajustar vocabulario solo en datos de entrenamiento
        print("Ajustando vocabulario en datos de entrenamiento...")
        X_train = self.vectorizer.fit_transform(train_combined)
        
        # Transformar conjuntos de validación y prueba
        X_val = self.vectorizer.transform(val_combined)
        X_test = self.vectorizer.transform(test_combined)
        
        # Procesar variable objetivo
        y_train = self.label_encoder.fit_transform(train_df[target_column])
        y_val = self.label_encoder.transform(val_df[target_column])
        y_test = self.label_encoder.transform(test_df[target_column])
        
        # Convertir a categórico para clasificación multiclase
        num_classes = len(self.label_encoder.classes_)
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
        
        # Estadísticas sparsity
        sparsity_train = 1.0 - (X_train.nnz / (X_train.shape[0] * X_train.shape[1]))

        
        print(f"Vocabulario final: {X_train.shape[1]} características")
        print(f"Número de clases: {num_classes}")
        print(f"Sparsity de entrenamiento: {sparsity_train:.2%}")
        print(f"Texto combinado: {'Sí (título + cuerpo)' if use_title_and_body else 'No (solo cuerpo)'}")
        print(f"Muestras de entrenamiento: {X_train.shape[0]}")
        print(f"Muestras de validación: {X_val.shape[0]}")
        print(f"Muestras de prueba: {X_test.shape[0]}")
        
        # Convertir matrices dispersas a densas para TensorFlow
        print(f"\n Optimización para TensorFlow: Convirtiendo matrices dispersas a densas...")
        X_train_dense = X_train.toarray().astype(np.float32)
        X_val_dense = X_val.toarray().astype(np.float32)
        X_test_dense = X_test.toarray().astype(np.float32)
    
        
        return {
            'X_train': X_train_dense, 'y_train': y_train_cat,
            'X_val': X_val_dense, 'y_val': y_val_cat, 
            'X_test': X_test_dense, 'y_test': y_test_cat,
            'num_classes': num_classes,
            'vocab_size': X_train.shape[1],  # Para BoW es el número de características
            'sparsity': sparsity_train,
        }

class ModelTrainer:
    """Maneja el entrenamiento de modelos con callbacks y monitoreo."""
    
    def __init__(self, model_dir: str = "models"):
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
        
        # Asegurar que los datos sean float32 para mejor rendimiento
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
    
    

class ResultsManager:
    """Maneja el guardado y carga de resultados de experimentos."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Inicializa ResultsManager.
        
        Args:
            output_dir (str): Directorio para guardar resultados de experimentos
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results_file = os.path.join(output_dir, "experiment_history.json")
    
    def save_experiment(self, experiment_data: Dict, experiment_name: str) -> str:
        """
        Guarda resultados del experimento en archivo JSON.
        
        Args:
            experiment_data (Dict): Diccionario conteniendo resultados del experimento
            experiment_name (str): Nombre del experimento
            
        Returns:
            str: Ruta al archivo guardado
        """
        # Añadir timestamp al experimento
        experiment_data['timestamp'] = datetime.now().isoformat()
        experiment_data['experiment_name'] = experiment_name
        
        # Cargar historial existente
        history_data = self.load_experiment_history()
        
        # Añadir nuevo experimento
        history_data["experiments"].append(experiment_data)
        
        # Guardar historial actualizado
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        print(f"Resultados del experimento guardados en: {self.results_file}")
        return self.results_file
    
    def save_experiment_results(self, experiment_data: Dict) -> int:
        """
        Guarda resultados del experimento y retorna el ID del experimento.
        
        Args:
            experiment_data (Dict): Diccionario conteniendo resultados del experimento
            
        Returns:
            int: ID del experimento guardado
        """
        # Añadir timestamp al experimento
        experiment_data['timestamp'] = datetime.now().isoformat()
        
        # Cargar historial existente
        history_data = self.load_experiment_history()
        
        # Generar ID del experimento
        experiment_id = len(history_data["experiments"]) + 1
        experiment_data['experiment_id'] = experiment_id
        
        # Añadir nuevo experimento
        history_data["experiments"].append(experiment_data)
        
        # Guardar historial actualizado
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        print(f"Experiment {experiment_id} results saved to {self.results_file}")
        return experiment_id
    
    def load_experiment_history(self) -> Dict:
        """
        Carga historial de experimentos desde archivo JSON.
        
        Returns:
            Dict: Historial completo de experimentos
        """
        if not os.path.exists(self.results_file):
            return {"experiments": [], "created": datetime.now().isoformat()}
        
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error cargando historial de experimentos: {e}")
            return {"experiments": [], "created": datetime.now().isoformat()}
    
    def display_experiment_history(self):
        """Muestra historial de experimentos formateado."""
        history_data = self.load_experiment_history()
        experiments = history_data.get("experiments", [])
        
        if not experiments:
            print("No se encontraron experimentos en el historial.")
            return
        
        print(f"\nHISTORIAL DE EXPERIMENTOS ({len(experiments)} experimentos)")
        print("=" * 100)
        
        # Summary table with language column
        print(f"\n{'ID':<3} {'Modelo':<12} {'Lang':<6} {'Precisión':<10} {'Pérdida':<10} {'Épocas':<8} {'Tiempo (s)':<10} {'Muestras':<10}")
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
    """Maneja la visualización de resultados de entrenamiento y rendimiento del modelo."""
    
    @staticmethod
    def plot_training_history(history: Dict, 
                            model_name: str = "Model",
                            save_path: Optional[str] = None):
        """
        Grafica métricas de entrenamiento y validación.
        
        Args:
            history (Dict): Historial de entrenamiento de Keras
            model_name (str): Nombre para el título del gráfico
            save_path (str, optional): Ruta para guardar el gráfico
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Graficar precisión
        if 'accuracy' in history and 'val_accuracy' in history:
            axes[0].plot(history['accuracy'], label='Precisión Entrenamiento', linewidth=2)
            axes[0].plot(history['val_accuracy'], label='Precisión Validación', linewidth=2)
            axes[0].set_title(f'{model_name} - Precisión')
            axes[0].set_xlabel('Época')
            axes[0].set_ylabel('Precisión')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Graficar pérdida
        axes[1].plot(history['loss'], label='Pérdida Entrenamiento', linewidth=2)
        axes[1].plot(history['val_loss'], label='Pérdida Validación', linewidth=2)
        axes[1].set_title(f'{model_name} - Pérdida')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Pérdida')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico de historial de entrenamiento guardado en: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            class_names: List[str],
                            model_name: str = "Model",
                            save_path: Optional[str] = None):
        """
        Grafica matriz de confusión.
        
        Args:
            y_true (np.ndarray): Etiquetas verdaderas
            y_pred (np.ndarray): Etiquetas predichas  
            class_names (List[str]): Nombres de las clases
            model_name (str): Nombre para el título del gráfico
            save_path (str, optional): Ruta para guardar el gráfico
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} - Matriz de Confusión')
        plt.xlabel('Etiqueta Predicha')
        plt.ylabel('Etiqueta Verdadera')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Matriz de confusión guardada en: {save_path}")
        
        plt.show()

def evaluate_model(model: tf.keras.Model,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  class_names: List[str] = None) -> Dict:
    """
    Evalúa el rendimiento del modelo en datos de prueba.
    
    Args:
        model: Modelo de Keras entrenado
        X_test, y_test: Datos de prueba
        class_names: Nombres de clases para el reporte de clasificación
        
    Returns:
        Dict: Métricas de evaluación
    """
    print("Evaluando modelo en datos de prueba...")
    
    # Obtener predicciones
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
    
    # Calcular métricas
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Reporte de clasificación
    if class_names is None:
        class_names = [f"Clase_{i}" for i in range(len(np.unique(y_true)))]
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    print(f"Precisión de Prueba: {test_accuracy:.4f}")
    print(f"Pérdida de Prueba: {test_loss:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'classification_report': report,
        'y_true': y_true,
        'y_pred': y_pred
    }

def get_gpu_info():
    """Obtiene información de GPU para seguimiento de experimentos."""
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

# Función de utilidad para configuración de experimentos
def setup_experiment_environment(seed: int = 42):
    """
    Configura entorno reproducible para experimentos.
    
    Args:
        seed (int): Semilla aleatoria para reproducibilidad
    """
    # Establecer semillas para reproducibilidad
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Configurar GPU si está disponible
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU configurada: {len(gpus)} GPU(s) disponibles")
        except RuntimeError as e:
            print(f"Error de configuración GPU: {e}")
    else:
        print("No hay GPU disponible, usando CPU")
    
    return get_gpu_info()

def save_model_components(model_name: str, 
                         model: tf.keras.Model, 
                         vectorizer=None, 
                         tokenizer=None,
                         label_encoder=None,
                         model_dir: str = "models"):
    """
    Guarda modelo y todos sus componentes de preprocesamiento para facilitar la carga posterior.
    
    Args:
        model_name (str): Nombre base para los archivos guardados
        model (tf.keras.Model): Modelo de Keras entrenado
        vectorizer: Vectorizador TF-IDF para modelos BoW (opcional)
        tokenizer: Tokenizer de Keras para modelos de embedding (opcional)
        label_encoder: Encoder de etiquetas para las clases
        model_dir (str): Directorio donde guardar los componentes
    """
    try:
        # Crear directorio si no existe
        os.makedirs(model_dir, exist_ok=True)
        
        # Guardar modelo Keras
        model_path = os.path.join(model_dir, f"{model_name}.h5")
        model.save(model_path)
        print(f"Modelo guardado: {model_path}")
        
        # Guardar vectorizador TF-IDF si existe (para modelos BoW)
        if vectorizer is not None:
            vectorizer_path = os.path.join(model_dir, f"{model_name}_vectorizer.pkl")
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            print(f"Vectorizador TF-IDF guardado: {vectorizer_path}")
        
        # Guardar tokenizer si existe (para modelos de embedding)
        if tokenizer is not None:
            tokenizer_path = os.path.join(model_dir, f"{model_name}_tokenizer.pkl")
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(tokenizer, f)
            print(f"Tokenizer guardado: {tokenizer_path}")
        
        # Guardar label encoder
        if label_encoder is not None:
            label_encoder_path = os.path.join(model_dir, f"{model_name}_label_encoder.pkl")
            with open(label_encoder_path, 'wb') as f:
                pickle.dump(label_encoder, f)
            print(f"Label encoder guardado: {label_encoder_path}")
        
        print(f"Todos los componentes del modelo '{model_name}' guardados correctamente en {model_dir}/")
        
    except Exception as e:
        print(f"Error guardando componentes del modelo {model_name}: {e}")
        raise

def load_model_components(model_name: str, model_dir: str = "models") -> Dict:
    """
    Carga modelo y todos sus componentes de preprocesamiento.
    
    Args:
        model_name (str): Nombre base de los archivos del modelo
        model_dir (str): Directorio donde están los componentes
        
    Returns:
        Dict: Diccionario con modelo y componentes cargados
    """
    try:
        components = {}
        
        # Cargar modelo Keras
        model_path = os.path.join(model_dir, f"{model_name}.h5")
        if os.path.exists(model_path):
            components['model'] = tf.keras.models.load_model(model_path)
            print(f"Modelo cargado: {model_path}")
        else:
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        # Cargar vectorizador TF-IDF si existe
        vectorizer_path = os.path.join(model_dir, f"{model_name}_vectorizer.pkl")
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                components['vectorizer'] = pickle.load(f)
            print(f"Vectorizador TF-IDF cargado: {vectorizer_path}")
        
        # Cargar tokenizer si existe
        tokenizer_path = os.path.join(model_dir, f"{model_name}_tokenizer.pkl")
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                components['tokenizer'] = pickle.load(f)
            print(f"Tokenizer cargado: {tokenizer_path}")
        
        # Cargar label encoder
        label_encoder_path = os.path.join(model_dir, f"{model_name}_label_encoder.pkl")
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'rb') as f:
                components['label_encoder'] = pickle.load(f)
            print(f"Label encoder cargado: {label_encoder_path}")
        
        # Calcular información adicional si es posible
        if 'vectorizer' in components and hasattr(components['vectorizer'], 'vocabulary_'):
            components['vocab_size'] = len(components['vectorizer'].vocabulary_)
        
        if 'tokenizer' in components and hasattr(components['tokenizer'], 'word_index'):
            components['vocab_size'] = len(components['tokenizer'].word_index) + 1
        
        if 'label_encoder' in components and hasattr(components['label_encoder'], 'classes_'):
            components['num_classes'] = len(components['label_encoder'].classes_)
        
        print(f"Componentes del modelo '{model_name}' cargados correctamente")
        return components
        
    except Exception as e:
        print(f"Error cargando componentes del modelo {model_name}: {e}")
        return None