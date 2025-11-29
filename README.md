# Proyecto Redes Neuronales

**Universidad Del Valle - Redes Neuronales 2025-II**  
**Autor:** Herney Eduardo Quintero Trochez  

## Descripción

Sistema de clasificación de texto para análisis de sentimientos en reviews de Amazon (1-5 estrellas) usando diferentes arquitecturas de redes neuronales: MLP y RNN.

## Entregas del Proyecto

### 📦 Entrega 1: Perceptrón Multicapa (MLP)
Implementación de modelos MLP con BoW y Embeddings usando TensorFlow/Keras y PyTorch.

**Notebooks:**
- `mlp_tf_bow.ipynb` - MLP con Bag of Words (TensorFlow)
- `mlp_tf_embedding.ipynb` - MLP con Embeddings (TensorFlow)
- `mlp_pytorch_bow.ipynb` - MLP con Bag of Words (PyTorch)
- `mlp_pytorch_embedding.ipynb` - MLP con Embeddings (PyTorch)

### 🔄 Entrega 2: Redes Neuronales Recurrentes (RNN)
Implementación de modelos RNN sin memoria y con memoria usando PyTorch.

**Notebooks:**
- `rnn_pytorch.ipynb` - SimpleRNN (RNN sin memoria)
- `lstm_pytorch.ipynb` - LSTM (RNN con memoria a largo plazo)
- `gru_pytorch.ipynb` - GRU (RNN con memoria eficiente)
- `rnn_comparison.ipynb` - Comparación y análisis de modelos RNN

📄 **Documentación completa:** Ver [ENTREGA2_README.md](ENTREGA2_README.md)

## Estructura del Proyecto

```
ProjectNeunoralNets/
├── data/                          # Datasets CSV
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── models/                        # Modelos entrenados
│   ├── project_part_1/           # Modelos MLP (Entrega 1)
│   └── project_part_2/           # Modelos RNN (Entrega 2)
│
├── output/                        # Resultados y gráficos
│   ├── project_part_1/           # Resultados Entrega 1
│   └── project_part_2/           # Resultados Entrega 2
│
├── helpers/                       # Módulos de ayuda
│   ├── models.py                 # Arquitecturas (MLP, RNN, LSTM, GRU)
│   ├── model_trainer_torch.py    # Entrenamiento PyTorch
│   ├── model_trainer_tf.py       # Entrenamiento TensorFlow
│   ├── data_loader_torch.py      # Carga de datos PyTorch
│   ├── data_loader.py            # Carga de datos TensorFlow
│   ├── results_manager.py        # Gestión de resultados
│   ├── visualizer.py             # Visualizaciones
│   └── utils.py                  # Utilidades
│
├── controllers/                   # Flask app (Web UI)
│   ├── routes.py
│   ├── prediction_controller.py
│   └── ...
│
├── static/                        # Assets web
├── templates/                     # Templates HTML
│
├── Notebooks:
├── mlp_tf_bow.ipynb              # MLP BoW (TensorFlow)
├── mlp_tf_embedding.ipynb        # MLP Embedding (TensorFlow)
├── mlp_pytorch_bow.ipynb         # MLP BoW (PyTorch)
├── mlp_pytorch_embedding.ipynb   # MLP Embedding (PyTorch)
├── rnn_pytorch.ipynb             # SimpleRNN (Entrega 2)
├── lstm_pytorch.ipynb            # LSTM (Entrega 2)
├── gru_pytorch.ipynb             # GRU (Entrega 2)
├── rnn_comparison.ipynb          # Comparación RNN
│
├── app.py                         # Flask app principal
├── requirements.txt               # Dependencias Python
├── environment.yml                # Ambiente Conda
├── README.md                      # Este archivo
└── ENTREGA2_README.md            # Documentación Entrega 2
```

## Instalación y Uso

### 1. Crear ambiente (recomendado)
```bash
# Opción 1: Conda
conda env create -f environment.yml
conda activate rn_project

# Opción 2: pip + venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Ejecutar Notebooks

**Para Entrega 1 (MLP):**
```bash
# TensorFlow
jupyter notebook mlp_tf_bow.ipynb
jupyter notebook mlp_tf_embedding.ipynb

# PyTorch
jupyter notebook mlp_pytorch_bow.ipynb
jupyter notebook mlp_pytorch_embedding.ipynb
```

**Para Entrega 2 (RNN):**
```bash
# Entrenar modelos
jupyter notebook rnn_pytorch.ipynb    # SimpleRNN
jupyter notebook lstm_pytorch.ipynb   # LSTM
jupyter notebook gru_pytorch.ipynb    # GRU

# Comparar resultados
jupyter notebook rnn_comparison.ipynb
```

### 3. Ejecutar Web App (opcional)
```bash
python app.py
# Abrir http://localhost:5000
```

### 4. Datos requeridos
Los archivos CSV en `data/` deben tener:
- `review_body`: Texto del review  
- `review_title`: Título del review (opcional)
- `stars`: Clasificación (1-5 estrellas)
- `language`: Idioma del review

## Características Principales

### Modelos Implementados

#### Entrega 1: MLP (Perceptrón Multicapa)
- **TensorFlow/Keras:**
  - MLP con Bag of Words (TF-IDF)
  - MLP con Embeddings + GlobalAveragePooling
- **PyTorch:**
  - MLP con Bag of Words
  - MLP con Embeddings + Mean Pooling

#### Entrega 2: RNN (Redes Recurrentes)
- **SimpleRNN:** RNN sin memoria (baseline)
  - Elman RNN básica
  - Problemas con gradiente desvaneciente
  
- **LSTM:** Long Short-Term Memory
  - 3 compuertas (input, forget, output)
  - Cell state para memoria a largo plazo
  - Bidireccional con 2 capas apiladas
  
- **GRU:** Gated Recurrent Unit
  - 2 compuertas (update, reset)
  - Más eficiente que LSTM (~25% menos parámetros)
  - Rendimiento similar a LSTM

### Funcionalidades
- ✅ Preprocesamiento automático de texto (tokenización, padding)
- ✅ Filtrado por idioma configurable
- ✅ Soporte CPU y GPU automático
- ✅ Tracking de experimentos con historial JSON
- ✅ Visualización automática (curvas de aprendizaje, matrices de confusión)
- ✅ Evaluación completa con métricas detalladas
- ✅ Early stopping y learning rate scheduling
- ✅ Comparación automática de modelos

### Archivos Generados
```
models/
├── project_part_1/               # Modelos MLP
│   ├── MLP_BoW_TF.h5
│   ├── MLP_Embedding_TF.h5
│   ├── MLP_BoW_Torch.pth
│   └── MLP_Embedding_Torch.pth
└── project_part_2/               # Modelos RNN
    ├── SimpleRNN_Torch.pth
    ├── LSTM_Torch.pth
    └── GRU_Torch.pth

output/
├── project_part_1/               # Resultados Entrega 1
│   ├── *_results.json
│   └── *.png
└── project_part_2/               # Resultados Entrega 2
    ├── *_results.json
    ├── comparison_report.txt
    └── *.png
```

## Comparación de Arquitecturas

| Modelo | Tipo | Memoria | Parámetros | Velocidad | Uso Recomendado |
|--------|------|---------|------------|-----------|-----------------|
| **MLP BoW** | Feedforward | No | Bajo | Muy rápido | Baseline simple |
| **MLP Embedding** | Feedforward | No | Medio | Rápido | Texto sin secuencia |
| **SimpleRNN** | Recurrente | Corto plazo | Bajo | Rápido | Baseline RNN |
| **GRU** | Recurrente | Largo plazo | Medio | Medio | Default para RNN |
| **LSTM** | Recurrente | Largo plazo | Alto | Más lento | Tareas complejas |

## Tecnologías Utilizadas

- **Python 3.10+**
- **Deep Learning:**
  - PyTorch 2.x
  - TensorFlow 2.x / Keras
- **Procesamiento:**
  - NumPy
  - Pandas
  - scikit-learn
- **Visualización:**
  - Matplotlib
  - Seaborn
- **Web (opcional):**
  - Flask
  - Bootstrap

## Dataset

**Amazon Reviews Multi-language**
- Reviews de productos en múltiples idiomas
- 5 clases: 1-5 estrellas
- Splits: train (80%), validation (10%), test (10%)
- Fuente: https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi

## Licencia

Este proyecto es parte del curso de Redes Neuronales de la Universidad Del Valle y tiene propósitos educativos.

## Contacto

Herney Eduardo Quintero Trochez  
Universidad Del Valle  
Escuela de Ingeniería de Sistemas y Computación