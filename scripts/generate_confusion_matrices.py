#!/usr/bin/env python3
"""
Script para generar matrices de confusion reales cargando los modelos entrenados.
Reutiliza la logica de carga de modelos del proyecto.

Autor: Herney Eduardo Quintero Trochez
Fecha: Diciembre 2025
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Agregar directorio padre al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.utils import load_model_components_tf, load_model_components_torch
from helpers.data_loader import DataLoader
from controllers.gpu_config import configure_gpu

import torch
from sklearn.metrics import confusion_matrix, classification_report

# Configuracion de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# Directorios
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "report" / "figures"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Clases
CLASS_NAMES = ['1 Estrella', '2 Estrellas', '3 Estrellas', '4 Estrellas', '5 Estrellas']


def load_experiment_history(json_path):
    """Carga historial de experimentos."""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_model_category(model_type):
    """Determina categoria base del modelo."""
    model_type_lower = model_type.lower()
    if 'transformer' in model_type_lower:
        return 'Transformer'
    elif 'gru' in model_type_lower:
        return 'GRU'
    elif 'lstm' in model_type_lower:
        return 'LSTM'
    elif 'simplernn' in model_type_lower or 'simple_rnn' in model_type_lower:
        return 'SimpleRNN'
    elif 'mlp' in model_type_lower:
        if 'bow' in model_type_lower:
            return 'MLP_BoW'
        elif 'embedding' in model_type_lower:
            return 'MLP_Embedding'
        return 'MLP'
    return model_type


def get_display_name(model_type, experiment_name=''):
    """Genera nombre legible basado en experiment_name."""
    # Usar experiment_name como base si esta disponible
    if experiment_name:
        # Detectar variantes especiales
        if 'GloVe' in experiment_name or 'Frozen' in experiment_name:
            base = model_type.replace('_Torch', '').replace('_', ' ')
            if 'GloVe' in experiment_name:
                base += ' + GloVe'
            if 'Frozen' in experiment_name:
                base += ' (Frozen)'
            return base.strip()
    
    # Fallback: usar model_type
    name = model_type.replace('_Torch', '').replace('_', ' ')
    
    if 'glove' in model_type.lower():
        if '+ GloVe' not in name:
            name += ' + GloVe'
    
    return name.strip()


def find_model_file(model_name):
    """Busca archivo del modelo en los directorios de modelos."""
    for subdir in ['project_part_1', 'project_part_2', 'project_part_3', 'project_part_4']:
        model_dir = BASE_DIR / 'models' / subdir
        
        # Buscar .pth (PyTorch)
        pth_path = model_dir / f"{model_name}.pth"
        if pth_path.exists():
            return str(model_dir), 'pytorch'
        
        # Buscar .h5 (TensorFlow)
        h5_path = model_dir / f"{model_name}.h5"
        if h5_path.exists():
            return str(model_dir), 'tensorflow'
    
    return None, None


def load_test_data_for_model(model_components, category):
    """Carga y preprocesa datos de test segun el tipo de modelo."""
    data_loader = DataLoader(str(DATA_DIR))
    train_df, val_df, test_df = data_loader.load_all_data()
    
    # Filtrar solo ingles (como en el entrenamiento)
    train_df = train_df[train_df['language'] == 'en']
    val_df = val_df[val_df['language'] == 'en']
    test_df = test_df[test_df['language'] == 'en']
    
    print(f"    Muestras de test (ingles): {len(test_df)}")
    
    # Determinar tipo de preprocesamiento
    has_vectorizer = model_components.get('vectorizer') is not None
    has_tokenizer = model_components.get('tokenizer') is not None
    
    if has_vectorizer:
        # Modelo BoW - usar vectorizador guardado
        print("    Tipo: BoW (usando vectorizador guardado)")
        vectorizer = model_components['vectorizer']
        
        # Combinar titulo y cuerpo
        test_texts = (test_df['review_title'].fillna('').astype(str) + ' ' + 
                     test_df['review_body'].fillna('').astype(str))
        
        X_test = vectorizer.transform(test_texts)
        X_test = X_test.toarray().astype(np.float32)
        
    elif has_tokenizer:
        # Modelo con Embedding - usar tokenizer guardado
        print("    Tipo: Embedding (usando tokenizer guardado)")
        tokenizer = model_components['tokenizer']
        max_length = model_components.get('max_length', 200)
        
        # Combinar titulo y cuerpo
        test_texts = (test_df['review_title'].fillna('').astype(str) + ' ' + 
                     test_df['review_body'].fillna('').astype(str))
        
        # Tokenizar y padding
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        X_test = tokenizer.texts_to_sequences(test_texts)
        X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')
        
    else:
        print("    ADVERTENCIA: No se encontro vectorizer ni tokenizer")
        return None, None
    
    # Etiquetas (0-4)
    y_test = test_df['stars'].values - 1  # Convertir 1-5 a 0-4
    
    return X_test, y_test


def predict_pytorch(model, X_test, device, batch_size=64):
    """Genera predicciones con modelo PyTorch."""
    model.eval()
    all_preds = []
    
    # Convertir a tensor
    X_tensor = torch.LongTensor(X_test)
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
    
    return np.array(all_preds)


def predict_tensorflow(model, X_test, batch_size=64):
    """Genera predicciones con modelo TensorFlow."""
    predictions = model.predict(X_test, batch_size=batch_size, verbose=0)
    return np.argmax(predictions, axis=1)


def plot_confusion_matrix(cm, model_name, accuracy, output_path, normalize=True):
    """Genera y guarda la matriz de confusion."""
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.0%'
        title_suffix = '(Normalizada)'
    else:
        cm_display = cm
        fmt = 'd'
        title_suffix = '(Absoluta)'
    
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Heatmap
    im = ax.imshow(cm_display, cmap='Blues', vmin=0, vmax=1 if normalize else cm.max())
    
    # Etiquetas
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(CLASS_NAMES)
    
    ax.set_xlabel('Prediccion')
    ax.set_ylabel('Clase Real')
    ax.set_title(f'{model_name}\nMatriz de Confusion {title_suffix}\nTest Accuracy: {accuracy:.1%}')
    
    # Anotaciones
    for i in range(5):
        for j in range(5):
            if normalize:
                value = cm_display[i, j]
                text = f'{value:.0%}'
            else:
                value = cm_display[i, j]
                text = f'{int(value)}'
            
            color = 'white' if (normalize and value > 0.5) or (not normalize and value > cm.max()/2) else 'black'
            weight = 'bold' if i == j else 'normal'
            ax.text(j, i, text, ha='center', va='center', 
                   color=color, fontsize=12, fontweight=weight)
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='Proporcion' if normalize else 'Cantidad')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"    [OK] {output_path.name}")


def plot_all_confusion_matrices(all_results, output_path):
    """Genera figura con todas las matrices de confusion."""
    n_models = len(all_results)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (ax, result) in enumerate(zip(axes, all_results)):
        cm = result['confusion_matrix']
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(5))
        ax.set_yticks(np.arange(5))
        ax.set_xticklabels(['1', '2', '3', '4', '5'], fontsize=9)
        ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=9)
        
        ax.set_title(f"{result['display_name']}\nAcc: {result['accuracy']:.1%}", fontsize=10)
        
        if idx % cols == 0:
            ax.set_ylabel('Real')
        if idx >= len(all_results) - cols:
            ax.set_xlabel('Prediccion')
        
        # Anotaciones
        for i in range(5):
            for j in range(5):
                value = cm_norm[i, j]
                color = 'white' if value > 0.5 else 'black'
                ax.text(j, i, f'{value:.0%}', ha='center', va='center',
                       color=color, fontsize=8, fontweight='bold' if i==j else 'normal')
    
    # Ocultar ejes vacios
    for idx in range(len(all_results), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Matrices de Confusion - Mejores Modelos por Categoria', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"    [OK] {output_path.name}")


def main():
    print("=" * 70)
    print("Generador de Matrices de Confusion Reales")
    print("=" * 70)
    
    # Configurar GPU
    print("\nConfigurando GPU...")
    configure_gpu()
    
    # Recopilar todos los experimentos
    experiments_by_category = defaultdict(list)
    
    parts = [
        'output/project_part_1/experiment_history.json',
        'output/project_part_2/experiment_history.json',
        'output/project_part_3/experiment_history.json',
        'output/project_part_4/experiment_history.json',
    ]
    
    print("\nCargando historial de experimentos...")
    for json_rel_path in parts:
        json_path = BASE_DIR / json_rel_path
        if not json_path.exists():
            continue
        
        data = load_experiment_history(str(json_path))
        for exp in data.get('experiments', []):
            config = exp.get('configuration', {})
            model_type = config.get('model_type', 'Unknown')
            category = get_model_category(model_type)
            
            # Extraer nombre del modelo desde model_path
            model_path = exp.get('training_results', {}).get('model_path', '')
            if model_path:
                model_name = Path(model_path).stem
            else:
                model_name = exp.get('experiment_name', '')
            
            experiments_by_category[category].append({
                'experiment_name': exp.get('experiment_name', ''),
                'model_name': model_name,
                'model_type': model_type,
                'category': category,
                'accuracy': exp.get('evaluation_metrics', {}).get('test_accuracy', 0),
                'config': config,
            })
    
    # Seleccionar mejor modelo por categoria
    best_models = {}
    print("\nSeleccionando mejores modelos por categoria:")
    for category, exps in sorted(experiments_by_category.items()):
        best = max(exps, key=lambda x: x['accuracy'])
        best_models[category] = best
        print(f"  [{category}] {best['model_name']} - Acc: {best['accuracy']:.1%}")
    
    # Generar matrices de confusion para cada mejor modelo
    all_results = []
    
    print("\n" + "=" * 70)
    print("Generando Matrices de Confusion")
    print("=" * 70)
    
    for category, model_info in best_models.items():
        model_name = model_info['model_name']
        display_name = get_display_name(model_info['model_type'], model_info['experiment_name'])
        
        print(f"\n>>> {display_name} ({category})")
        print(f"    Modelo: {model_name}")
        
        # Buscar archivo del modelo
        model_dir, framework = find_model_file(model_name)
        if not model_dir:
            print(f"    ERROR: No se encontro archivo del modelo")
            continue
        
        print(f"    Directorio: {model_dir}")
        print(f"    Framework: {framework}")
        
        # Cargar modelo
        try:
            if framework == 'pytorch':
                components = load_model_components_torch(model_name, model_dir)
            else:
                components = load_model_components_tf(model_name, model_dir)
            
            if not components or 'model' not in components:
                print(f"    ERROR: No se pudo cargar el modelo")
                continue
            
            model = components['model']
            print(f"    Modelo cargado exitosamente")
            
        except Exception as e:
            print(f"    ERROR cargando modelo: {e}")
            continue
        
        # Cargar datos de test
        try:
            X_test, y_test = load_test_data_for_model(components, category)
            if X_test is None:
                continue
        except Exception as e:
            print(f"    ERROR cargando datos: {e}")
            continue
        
        # Generar predicciones
        print(f"    Generando predicciones...")
        try:
            if framework == 'pytorch':
                device = components.get('device', torch.device('cpu'))
                y_pred = predict_pytorch(model, X_test, device)
            else:
                y_pred = predict_tensorflow(model, X_test)
        except Exception as e:
            print(f"    ERROR en prediccion: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Calcular metricas
        accuracy = np.mean(y_pred == y_test)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"    Accuracy calculado: {accuracy:.1%}")
        print(f"    Accuracy guardado:  {model_info['accuracy']:.1%}")
        
        # Guardar resultados
        result = {
            'category': category,
            'display_name': display_name,
            'model_name': model_name,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'y_true': y_test,
            'y_pred': y_pred,
        }
        all_results.append(result)
        
        # Generar matriz de confusion individual
        safe_name = category.replace(' ', '_').replace('+', '_')
        output_file = OUTPUT_DIR / f"best_{safe_name}_confusion_real.png"
        plot_confusion_matrix(cm, display_name, accuracy, output_file, normalize=True)
        
        # Limpiar memoria
        del model
        del components
        if framework == 'pytorch':
            torch.cuda.empty_cache()
    
    # Generar figura comparativa con todas las matrices
    if len(all_results) >= 2:
        print("\n" + "=" * 70)
        print("Generando Figura Comparativa")
        print("=" * 70)
        output_file = OUTPUT_DIR / "all_confusion_matrices.png"
        plot_all_confusion_matrices(all_results, output_file)
    
    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"\n{'Categoria':<20} {'Modelo':<30} {'Accuracy'}")
    print("-" * 70)
    for r in sorted(all_results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{r['category']:<20} {r['display_name']:<30} {r['accuracy']:.1%}")
    
    print(f"\nFiguras guardadas en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
