"""
Data Loading and Preprocessing Module for PyTorch
Proyecto Redes Neuronales 2025-II - Universidad Del Valle

Maneja la carga y preprocesamiento de datos para modelos de PyTorch.
Usa Keras Preprocessing (igual que TensorFlow) para tokenización.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Importar Keras Preprocessing (igual que TensorFlow)
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


class DataLoaderTorch:
    """Maneja la carga y preprocesamiento de datos para PyTorch."""

    def __init__(self, data_dir: str = "data"):
        """
        Inicializa DataLoaderTorch.

        Args:
            data_dir (str): Ruta al directorio que contiene los archivos CSV
        """
        self.data_dir = data_dir
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        self.vocab = None
        self.tokenizer = None

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
            Tuple: (train_df, val_df, test_df)
        """
        train_df = self.load_csv_data("train.csv")
        val_df = self.load_csv_data("validation.csv")
        test_df = self.load_csv_data("test.csv")

        return train_df, val_df, test_df

    def preprocess_text_data_bow(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_column: str = "review_body",
        title_column: str = None,
        target_column: str = "stars",
        max_features: int = 5000,
        min_df: int = 3,
        max_df: float = 0.85,
        use_title_and_body: bool = False,
        batch_size: int = 32,
    ) -> Dict:
        """
        Preprocesa datos de texto usando BoW/TF-IDF para PyTorch.

        Args:
            train_df, val_df, test_df: DataFrames con datos de texto
            text_column (str): Columna de texto (cuerpo)
            title_column (str): Columna de título (opcional)
            target_column (str): Columna objetivo
            max_features (int): Máximo número de características
            min_df (int): Frecuencia mínima de documento
            max_df (float): Frecuencia máxima de documento
            use_title_and_body (bool): Combinar título y cuerpo
            batch_size (int): Tamaño del batch para DataLoaders

        Returns:
            Dict: DataLoaders y metadatos
        """
        print("Preprocesando datos con BoW para PyTorch...")
        print(
            f"Configuración: max_features={max_features}, min_df={min_df}, max_df={max_df}"
        )

        def combine_title_and_body(df, text_col, title_col, use_combined):
            """Combina título y cuerpo de texto."""
            if use_combined and title_col and title_col in df.columns:
                combined = (
                    df[title_col].fillna("").astype(str)
                    + " "
                    + df[text_col].fillna("").astype(str)
                )
                return combined
            else:
                return df[text_col].fillna("")

        # Combinar textos
        train_combined = combine_title_and_body(
            train_df, text_column, title_column, use_title_and_body
        )
        val_combined = combine_title_and_body(
            val_df, text_column, title_column, use_title_and_body
        )
        test_combined = combine_title_and_body(
            test_df, text_column, title_column, use_title_and_body
        )

        # Vectorizar con TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words="english",
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
        )

        print("Ajustando vocabulario...")
        X_train = self.vectorizer.fit_transform(train_combined)
        X_val = self.vectorizer.transform(val_combined)
        X_test = self.vectorizer.transform(test_combined)

        # Procesar etiquetas
        y_train = self.label_encoder.fit_transform(train_df[target_column])
        y_val = self.label_encoder.transform(val_df[target_column])
        y_test = self.label_encoder.transform(test_df[target_column])

        num_classes = len(self.label_encoder.classes_)
        vocab_size = X_train.shape[1]
        sparsity = 1.0 - (X_train.nnz / (X_train.shape[0] * X_train.shape[1]))

        print(f"Vocabulario: {vocab_size} características")
        print(f"Número de clases: {num_classes}")
        print(f"Sparsity: {sparsity:.2%}")
        print(
            f"Muestras - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}"
        )

        # Convertir a tensores PyTorch
        X_train_dense = torch.FloatTensor(X_train.toarray())
        X_val_dense = torch.FloatTensor(X_val.toarray())
        X_test_dense = torch.FloatTensor(X_test.toarray())

        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)
        y_test_tensor = torch.LongTensor(y_test)

        # Crear DataLoaders
        train_dataset = TensorDataset(X_train_dense, y_train_tensor)
        val_dataset = TensorDataset(X_val_dense, y_val_tensor)
        test_dataset = TensorDataset(X_test_dense, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "X_train": X_train_dense,
            "X_val": X_val_dense,
            "X_test": X_test_dense,
            "y_train": y_train_tensor,
            "y_val": y_val_tensor,
            "y_test": y_test_tensor,
            "num_classes": num_classes,
            "vocab_size": vocab_size,
            "sparsity": sparsity,
        }

    def preprocess_text_data_embedding(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_column: str = "review_body",
        title_column: str = None,
        target_column: str = "stars",
        max_words: int = 10000,
        max_length: int = 200,
        use_title_and_body: bool = False,
        batch_size: int = 32,
    ) -> Dict:
        """
        Preprocesa datos de texto para embeddings en PyTorch.
        Usa Keras Preprocessing (igual que TensorFlow).

        Args:
            train_df, val_df, test_df: DataFrames con datos
            text_column (str): Columna de texto
            title_column (str): Columna de título (opcional)
            target_column (str): Columna objetivo
            max_words (int): Tamaño máximo del vocabulario
            max_length (int): Longitud máxima de secuencia
            use_title_and_body (bool): Combinar título y cuerpo
            batch_size (int): Tamaño del batch

        Returns:
            Dict: DataLoaders y metadatos
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

        # Combinar textos
        train_combined = combine_title_and_body(train_df, text_column, title_column, use_title_and_body)
        val_combined = combine_title_and_body(val_df, text_column, title_column, use_title_and_body)
        test_combined = combine_title_and_body(test_df, text_column, title_column, use_title_and_body)

        # Combinar TODOS los textos para ajustar tokenizer (EXACTAMENTE como TensorFlow)
        all_texts = pd.concat([train_combined, val_combined, test_combined])

        # Inicializar y ajustar tokenizer (igual que TensorFlow)
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

        num_classes = len(self.label_encoder.classes_)
        

        actual_vocab = len(self.tokenizer.word_index) + 1
        vocab_size = min(max_words, actual_vocab) if max_words else actual_vocab

        print(f"Tamaño del vocabulario real: {len(self.tokenizer.word_index)}")
        print(f"Vocab size usado (limitado): {vocab_size}")
        print(f"Número de clases: {num_classes}")
        print(f"Longitud de secuencia: {max_length}")
        print(f"Texto combinado: {'Sí (título + cuerpo)' if use_title_and_body else 'No (solo cuerpo)'}")
        print(f"Muestras de entrenamiento: {X_train.shape[0]}")
        print(f"Muestras de validación: {X_val.shape[0]}")
        print(f"Muestras de prueba: {X_test.shape[0]}")

        # Convertir a tensores PyTorch
        X_train_tensor = torch.LongTensor(X_train)
        X_val_tensor = torch.LongTensor(X_val)
        X_test_tensor = torch.LongTensor(X_test)

        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)
        y_test_tensor = torch.LongTensor(y_test)

        # Crear DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "X_train": X_train_tensor,
            "X_val": X_val_tensor,
            "X_test": X_test_tensor,
            "y_train": y_train_tensor,
            "y_val": y_val_tensor,
            "y_test": y_test_tensor,
            "num_classes": num_classes,
            "vocab_size": vocab_size,
            "max_length": max_length,
        }
