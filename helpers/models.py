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
                                       Por defecto False para coincidir con GlobalAveragePooling1D
                                       de TensorFlow que incluye padding en el promedio.
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
        Inicializa pesos EXACTAMENTE como TensorFlow para compatibilidad.
        """
        # Embedding: TensorFlow Embedding usa uniform [-0.05, 0.05] por defecto
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
        # Keep padding_idx as zeros (TensorFlow también hace esto con mask_zero)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)
        
        # Dense layers: glorot/xavier uniform (TensorFlow default para Dense)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                # Xavier/Glorot uniform: igual que TensorFlow
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # TensorFlow inicializa bias en 0
                    nn.init.constant_(m.bias, 0)
    
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


class SimpleRNNClassifier(nn.Module):
    """RNN simple (Elman RNN) para clasificacion de texto."""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_size: int,
                 num_classes: int,
                 num_layers: int = 1,
                 dropout_rate: float = 0.3,
                 padding_idx: int = 0,
                 bidirectional: bool = False):
        super(SimpleRNNClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(rnn_output_size, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded)
        
        # Average pooling con mascara
        mask = (x != self.padding_idx).float().unsqueeze(-1)
        masked_output = rnn_out * mask
        sum_output = masked_output.sum(dim=1)
        seq_lengths = mask.sum(dim=1)
        avg_output = sum_output / (seq_lengths + 1e-8)
        
        dropped = self.dropout(avg_output)
        output = self.fc(dropped)
        return output
    
    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_classes': self.num_classes,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional
        }


class LSTMClassifier(nn.Module):
    """LSTM para clasificacion de texto."""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_size: int,
                 num_classes: int,
                 num_layers: int = 1,
                 dropout_rate: float = 0.3,
                 recurrent_dropout: float = 0.0,
                 padding_idx: int = 0,
                 bidirectional: bool = False):
        super(LSTMClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        if self.bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            last_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            last_hidden = hidden[-1, :, :]
        
        dropped = self.dropout(last_hidden)
        output = self.fc(dropped)
        return output
    
    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_classes': self.num_classes,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional
        }


class GRUClassifier(nn.Module):
    """GRU para clasificacion de texto."""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_size: int,
                 num_classes: int,
                 num_layers: int = 1,
                 dropout_rate: float = 0.3,
                 recurrent_dropout: float = 0.0,
                 padding_idx: int = 0,
                 bidirectional: bool = False):
        super(GRUClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(gru_output_size, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        
        if self.bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            last_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            last_hidden = hidden[-1, :, :]
        
        dropped = self.dropout(last_hidden)
        output = self.fc(dropped)
        return output
    
    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_classes': self.num_classes,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional
        }