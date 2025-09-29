# Proyecto Redes Neuronales - Perceptrón Multicapa

**Universidad Del Valle - Redes Neuronales 2025-II**  
**Autor:** Herney Eduardo Quintero Trochez  

## Descripción

Sistema de clasificación de texto usando Perceptrón Multicapa (MLP) para análisis de sentimientos en reviews de Amazon (1-5 estrellas).

## Estructura del Proyecto

```
ProjectNeunoralNets/
├── data/                          # Datasets CSV
├── models/                        # Modelos entrenados
├── output/                        # Resultados y gráficos
├── helper.py                      # Funciones utilitarias
├── multi_layer_perceptron.ipynb  # Notebook principal
├── requirements.txt               # Dependencias
└── README.md
```

## Instalación y Uso

### 1. Instalar dependencias
- Python 3.10(recomendado usar Conda para el versionamiento)
```bash
pip install -r requirements.txt
```

### 2. Ejecutar el notebook
```bash
jupyter notebook multi_layer_perceptron.ipynb
```

### 3. Datos requeridos
Los archivos CSV en `data/` deben tener:
- `review_body`: Texto del review  
- `stars`: Clasificación (1-5 estrellas)
- `language`: Idioma del review

## Características Principales

### Modelo MLP
- **Arquitectura**: Embedding + GlobalAveragePooling + Dense layers + Dropout
- **Configuración**: Parámetros modificables desde el notebook
- **Entrenamiento**: Early stopping y learning rate scheduling
- **Soporte**: CPU y GPU automático

### Funcionalidades
- Preprocesamiento automático de texto (tokenización, padding)
- Filtrado por idioma configurable
- Tracking de experimentos con historial JSON
- Visualización automática (gráficos de entrenamiento, matriz de confusión)
- Evaluación completa con múltiples métricas

### Archivos Generados
- `models/`: Modelos entrenados (.h5)
- `output/experiment_history.json`: Historial de experimentos
- `output/`: Gráficos y visualizaciones (.png)

