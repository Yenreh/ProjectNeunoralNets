# Proyecto Redes Neuronales

**Universidad Del Valle - Redes Neuronales 2025-II**  
**Autor:** Herney Eduardo Quintero Trochez

## Descripción

Sistema de clasificación de texto para análisis de sentimientos en reviews de Amazon (1-5 estrellas) usando diferentes arquitecturas de redes neuronales

## Notebooks Principales

### Entrega 1: Perceptrón Multicapa (MLP)

- `mlp_tf_bow.ipynb` - MLP con Bag of Words (TensorFlow) (Legacy)
- `mlp_tf_embedding.ipynb` - MLP con Embeddings (TensorFlow) (Legacy)
- `mlp_pytorch_bow.ipynb` - MLP con Bag of Words (PyTorch)
- `mlp_pytorch_embedding.ipynb` - MLP con Embeddings (PyTorch)

### Entrega 2: Redes Neuronales Recurrentes (RNN)

- `rnn_pytorch.ipynb` - SimpleRNN (RNN sin memoria)
- `lstm_pytorch.ipynb` - LSTM (RNN con memoria a largo plazo)
- `gru_pytorch.ipynb` - GRU (RNN con memoria eficiente)
- `transformer_pytorch.ipynb` - Transformer (Atención multi-cabeza)
- `transformer_pytorch_pretrain.ipynb` - Transformer con modelo preentrenado
- `rnn_comparison.ipynb` - Comparación y análisis de modelos RNN

## Instalación

### Opción 1: Conda (Recomendado)

**Con soporte GPU (CUDA):**

```bash
# Crear ambiente desde archivo
conda env create -f environment-gpu.yml

# Activar ambiente
conda activate rn_project
```

**Solo CPU (sin GPU - para demo/inferencia):**

```bash
# Crear ambiente desde archivo
conda env create -f environment-cpu.yml

# Activar ambiente
conda activate rn_project
```

### Opción 2: pip + venv

**Con soporte GPU (CUDA):**

```bash
# Crear ambiente virtual
python -m venv venv

# Activar ambiente
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias GPU
pip install -r requirements-gpu.txt
```

**Solo CPU (sin GPU - para demo/inferencia):**

```bash
# Crear ambiente virtual
python -m venv venv

# Activar ambiente
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias CPU
pip install -r requirements-cpu.txt
```

## Ejecución

### Ejecutar Notebooks

```bash
# Iniciar Jupyter
jupyter notebook

# O JupyterLab
jupyter lab
```

Luego abrir cualquiera de los notebooks listados arriba.

### Ejecutar Demo Web (app.py)

```bash
# Ejecutar con configuración por defecto (puerto 5000)
python app.py

# Ejecutar en puerto personalizado
python app.py --port 8080

# Ejecutar en host específico
python app.py --host 127.0.0.1 --port 5000
```

La aplicación estará disponible en `http://localhost:5000` (o el puerto especificado).

## Requisitos de Datos

### Carpeta `data/`

La carpeta `data/` debe contener los siguientes archivos CSV:

- `train.csv` - Datos de entrenamiento
- `validation.csv` - Datos de validación
- `test.csv` - Datos de prueba

**Columnas requeridas en los CSV:**

- `review_body` (str): Texto del review
- `review_title` (str): Título del review (opcional)
- `stars` (int): Clasificación de 1-5 estrellas
- `language` (str): Idioma del review (ej: 'es', 'en')

**Fuente del dataset:** [Amazon Reviews Multi-language](https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi)

## Requisitos de Modelos (para app.py)

### Carpeta `models/`

La aplicación web carga modelos desde las siguientes carpetas:

- `models/project_part_1/` - Modelos MLP (Entrega 1)
- `models/project_part_2/` - Modelos RNN (Entrega 2)
- `models/project_part_3/` - Modelos LSTM/GRU (Entrega 2)
- `models/project_part_4/` - Modelos Transformer (Entrega 2)

**Archivos requeridos por modelo:**

Cada modelo necesita 3 archivos con el mismo prefijo:

1. **Archivo del modelo:**
   - `.pth` (PyTorch) o `.h5` (TensorFlow/Keras)
2. **Label encoder:**
   - `_label_encoder.pkl`
3. **Preprocesador de texto:**
   - `_tokenizer.pkl` (para modelos con embeddings)
   - `_vectorizer.pkl` (para modelos con Bag of Words)

**Ejemplo:**

```
models/project_part_1/
├── MLP_BoW_Torch_20251130_135449.pth
├── MLP_BoW_Torch_20251130_135449_label_encoder.pkl
└── MLP_BoW_Torch_20251130_135449_vectorizer.pkl
```

Los modelos se generan automáticamente al ejecutar los notebooks de entrenamiento.

## Estructura del Proyecto

```
ProjectNeunoralNets/
├── data/                          # Datasets CSV (train, validation, test)
│   └── glove/                     # Embeddings GloVe
├── models/                        # Modelos entrenados
│   ├── project_part_1/           # Modelos MLP
│   └── project_part_2/           # Modelos RNN
│   └── project_part_3/           # Modelos LSTM/GRU
│   └── project_part_4/           # Modelos Transformer
├── output/                        # Resultados
├── helpers/                       # Módulos de ayuda
├── controllers/                   # Controladores Flask
├── static/                        # Assets web
├── templates/                     # Templates HTML
├── *.ipynb                        # Notebooks de entrenamiento
├── app.py                         # Aplicación web Flask
├── requirements.txt               # Dependencias pip (GPU)
├── requirements-cpu.txt           # Dependencias pip (CPU)
├── environment.yml                # Ambiente Conda (GPU)
└── environment-cpu.yml            # Ambiente Conda (CPU)
```

## Modelos Implementados

### MLP (Perceptrón Multicapa)

- **TensorFlow/Keras:** MLP con BoW y Embeddings
- **PyTorch:** MLP con BoW y Embeddings

### RNN (Redes Recurrentes)

- **SimpleRNN:** RNN básica sin memoria
- **LSTM:** Long Short-Term Memory con compuertas
- **GRU:** Gated Recurrent Unit (más eficiente)

### Transformer

- **Transformer:** Redes Neuronales Recurrentes con atención multi-cabeza

## Tecnologías

- Python 3.10+
- PyTorch 2.x / TensorFlow 2.x
- NumPy, Pandas, scikit-learn
- Matplotlib, Seaborn
- Flask, Bootstrap
