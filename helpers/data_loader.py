"""
Data Loading and Preprocessing Module
Proyecto Redes Neuronales 2025-II - Universidad Del Valle

Maneja la carga y preprocesamiento de datos para modelos de TensorFlow.
Compatible con la Parte 1 del proyecto (TensorFlow).
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


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
        
        Args:
            train_df, val_df, test_df: DataFrames con datos de texto
            text_column (str): Nombre de la columna de texto (cuerpo)
            title_column (str): Nombre de la columna de título (opcional)
            target_column (str): Nombre de la columna objetivo
            max_features (int): Número máximo de características
            min_df (int): Frecuencia mínima de documento
            max_df (float): Frecuencia máxima de documento
            use_title_and_body (bool): Si True, combina título y cuerpo
            
        Returns:
            Dict: Datos preprocesados con X_train, y_train, etc. (arrays densos)
        """
        print("Preprocesando datos de texto con BoW...")
        print(f"Configuración: max_features={max_features}, min_df={min_df}, max_df={max_df}")
        
        def combine_title_and_body(df, text_col, title_col, use_combined):
            """Combina título y cuerpo de texto si está habilitado."""
            if use_combined and title_col and title_col in df.columns:
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
        
        # Inicializar vectorizador TF-IDF
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
        print(f"\nOptimización para TensorFlow: Convirtiendo matrices dispersas a densas...")
        X_train_dense = X_train.toarray().astype(np.float32)
        X_val_dense = X_val.toarray().astype(np.float32)
        X_test_dense = X_test.toarray().astype(np.float32)
        
        return {
            'X_train': X_train_dense, 'y_train': y_train_cat,
            'X_val': X_val_dense, 'y_val': y_val_cat, 
            'X_test': X_test_dense, 'y_test': y_test_cat,
            'num_classes': num_classes,
            'vocab_size': X_train.shape[1],
            'sparsity': sparsity_train,
        }
