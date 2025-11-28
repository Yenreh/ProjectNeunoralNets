"""
Model Architectures for PyTorch
Proyecto Redes Neuronales 2025-II - Universidad Del Valle

Definiciones de arquitecturas de modelos para la Parte 2 (PyTorch).
"""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Perceptrón Multicapa para clasificación de texto.
    Compatible con representaciones BoW/TF-IDF y embeddings.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_layers: list, 
                 num_classes: int, 
                 dropout_rate: float = 0.3,
                 activation: str = 'relu'):
        """
        Inicializa el modelo MLP.
        
        Args:
            input_dim (int): Dimensión de entrada (vocab_size para BoW o embedding_dim)
            hidden_layers (list): Lista con número de neuronas por capa oculta
            num_classes (int): Número de clases de salida
            dropout_rate (float): Tasa de dropout para regularización
            activation (str): Función de activación ('relu', 'tanh', 'gelu')
        """
        super(MLPClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        
        # Seleccionar función de activación
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Activación no soportada: {activation}")
        
        # Construir capas
        layers = []
        prev_dim = input_dim
        
        # Capas ocultas
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Capa de salida (sin activación, CrossEntropyLoss lo maneja)
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # Red secuencial
        self.network = nn.Sequential(*layers)
        
        # Referencias para compatibilidad con guardado
        self.fc1 = nn.Linear(input_dim, hidden_layers[0])
        self.fc_out = nn.Linear(hidden_layers[-1], num_classes)
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializa pesos de las capas lineales."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [batch_size, input_dim]
        
        Returns:
            Tensor de salida [batch_size, num_classes]
        """
        return self.network(x)
    
    def get_config(self):
        """Retorna configuración del modelo para guardado."""
        return {
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'num_classes': self.num_classes
        }


class MLPWithEmbedding(nn.Module):
    """
    MLP con capa de embedding para procesamiento de secuencias.
    Útil para trabajar con representaciones de texto tokenizadas.
    """
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_layers: list,
                 num_classes: int,
                 dropout_rate: float = 0.3,
                 padding_idx: int = 0,
                 use_masked_pooling: bool = False):
        """
        Inicializa MLP con embedding.
        
        Args:
            vocab_size (int): Tamaño del vocabulario
            embedding_dim (int): Dimensión del embedding
            hidden_layers (list): Capas ocultas del MLP
            num_classes (int): Número de clases
            dropout_rate (float): Tasa de dropout
            padding_idx (int): Índice de padding en vocabulario
            use_masked_pooling (bool): Usar mean pooling con máscara para ignorar padding.
                                       Por defecto False para coincidir con TensorFlow's
                                       GlobalAveragePooling1D que incluye padding en el promedio.
        """
        super(MLPWithEmbedding, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.padding_idx = padding_idx
        self.use_masked_pooling = use_masked_pooling
        
        # Capa de embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        # MLP después del embedding (usa mean pooling)
        layers = []
        prev_dim = embedding_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
        # Referencias para compatibilidad
        self.fc1 = nn.Linear(embedding_dim, hidden_layers[0])
        self.fc_out = nn.Linear(hidden_layers[-1], num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Inicializa pesos.
        Usa inicialización similar a TensorFlow para mejor compatibilidad.
        """
        # Embedding: TensorFlow uses uniform initialization [-0.05, 0.05]
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
        # Keep padding_idx as zeros
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)
        
        # Dense layers: glorot/xavier uniform (same as TensorFlow default)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass con mean pooling mejorado.
        
        Args:
            x: Tensor de índices [batch_size, seq_length]
        
        Returns:
            Tensor de salida [batch_size, num_classes]
        """
        # Embedding: [batch_size, seq_length, embedding_dim]
        embedded = self.embedding(x)
        
        if self.use_masked_pooling:
            # Mean pooling con máscara (ignora padding)
            # Crear máscara: 1 para tokens reales, 0 para padding
            mask = (x != self.padding_idx).unsqueeze(-1).float()  # [batch, seq_len, 1]
            
            # Aplicar máscara a embeddings
            masked_embedded = embedded * mask
            
            # Calcular mean solo sobre tokens reales
            sum_embeddings = masked_embedded.sum(dim=1)  # [batch, emb_dim]
            sum_mask = mask.sum(dim=1).clamp(min=1)  # [batch, 1], evitar división por 0
            pooled = sum_embeddings / sum_mask  # [batch, emb_dim]
        else:
            # Mean pooling simple (incluye padding)
            pooled = embedded.mean(dim=1)
        
        # MLP: [batch_size, num_classes]
        output = self.mlp(pooled)
        
        return output
    
    def get_config(self):
        """Retorna configuración del modelo."""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_layers': self.hidden_layers,
            'num_classes': self.num_classes,
            'use_masked_pooling': self.use_masked_pooling
        }


class ImprovedMLP(nn.Module):
    """
    MLP mejorado con BatchNorm y técnicas avanzadas de regularización.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_layers: list,
                 num_classes: int,
                 dropout_rate: float = 0.3,
                 use_batch_norm: bool = True):
        """
        Inicializa MLP mejorado.
        
        Args:
            input_dim (int): Dimensión de entrada
            hidden_layers (list): Capas ocultas
            num_classes (int): Número de clases
            dropout_rate (float): Tasa de dropout
            use_batch_norm (bool): Usar Batch Normalization
        """
        super(ImprovedMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_layers):
            # Capa lineal
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (antes de activación)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activación
            layers.append(nn.ReLU())
            
            # Dropout
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Capa de salida
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Referencias para compatibilidad
        self.fc1 = nn.Linear(input_dim, hidden_layers[0])
        self.fc_out = nn.Linear(hidden_layers[-1], num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializa pesos con He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        return self.network(x)
    
    def get_config(self):
        """Retorna configuración del modelo."""
        return {
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'num_classes': self.num_classes,
            'use_batch_norm': self.use_batch_norm
        }


# Función helper para crear modelos fácilmente
def create_mlp_model(input_dim, hidden_layers, num_classes, 
                    model_type='standard', **kwargs):
    """
    Factory function para crear modelos MLP.
    
    Args:
        input_dim (int): Dimensión de entrada
        hidden_layers (list): Capas ocultas
        num_classes (int): Número de clases
        model_type (str): 'standard', 'improved', 'embedding'
        **kwargs: Argumentos adicionales para el modelo
    
    Returns:
        nn.Module: Modelo instanciado
    """
    if model_type == 'standard':
        return MLPClassifier(input_dim, hidden_layers, num_classes, **kwargs)
    elif model_type == 'improved':
        return ImprovedMLP(input_dim, hidden_layers, num_classes, **kwargs)
    elif model_type == 'embedding':
        return MLPWithEmbedding(input_dim, hidden_layers[0], 
                               hidden_layers[1:], num_classes, **kwargs)
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")
