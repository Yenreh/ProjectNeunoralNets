#!/usr/bin/env python3
"""
Script para generar graficos profesionales para el informe del proyecto.
Version 2: Selecciona solo el mejor modelo de cada tipo y genera matrices de confusion.

Autor: Herney Eduardo Quintero Trochez
Fecha: Diciembre 2025
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Configuracion de estilo para graficos profesionales
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# Paleta de colores profesional
COLORS = {
    'train': '#2E86AB',
    'val': '#E94F37',
    'test': '#2A9D8F',
    'primary': '#1B4965',
    'secondary': '#E76F51',
    'tertiary': '#264653',
}

# Colores para cada tipo de modelo
MODEL_COLORS = {
    'MLP': '#3498db',
    'SimpleRNN': '#9b59b6',
    'LSTM': '#e74c3c',
    'GRU': '#2ecc71',
    'Transformer': '#f39c12',
}

# Directorio base del proyecto
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "report" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Clases del dataset
CLASS_NAMES = ['1', '2', '3', '4', '5']
CLASS_LABELS = ['1 Estrella', '2 Estrellas', '3 Estrellas', '4 Estrellas', '5 Estrellas']


def load_experiment_history(json_path):
    """Carga el historial de experimentos desde un archivo JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_model_category(model_type):
    """Determina la categoria base del modelo."""
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
    """Genera un nombre legible para mostrar basado en experiment_name."""
    # Usar experiment_name como base si esta disponible
    if experiment_name:
        # Simplificar el nombre removiendo timestamp completo
        parts = experiment_name.split('_')
        # Mantener la parte significativa del nombre
        if 'GloVe' in experiment_name or 'Frozen' in experiment_name:
            # Mantener indicacion especial
            base = model_type.replace('_Torch', '').replace('_', ' ')
            if 'GloVe' in experiment_name:
                base += ' + GloVe'
            if 'Frozen' in experiment_name:
                base += ' (Frozen)'
            return base.strip()
    
    # Fallback: usar model_type
    name = model_type.replace('_Torch', '').replace('_', ' ')
    
    # Detectar variantes especiales desde model_type
    if 'glove' in model_type.lower():
        if '+ GloVe' not in name:
            name += ' + GloVe'
    
    return name.strip()


def plot_training_curves(history, model_name, output_path, category):
    """Genera graficos de curvas de entrenamiento."""
    train_loss_key = 'loss' if 'loss' in history else 'train_loss'
    train_acc_key = 'accuracy' if 'accuracy' in history else 'train_accuracy'
    
    train_loss = history.get(train_loss_key, [])
    train_acc = history.get(train_acc_key, [])
    val_loss = history.get('val_loss', [])
    val_acc = history.get('val_accuracy', [])
    
    if not train_loss:
        return
    
    epochs = range(1, len(train_loss) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    color = MODEL_COLORS.get(category.split('_')[0], '#2E86AB')
    
    # Grafico de Loss
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, '-', color=color, linewidth=2, 
             label='Entrenamiento', marker='o', markersize=4, alpha=0.8)
    ax1.plot(epochs, val_loss, '--', color=color, linewidth=2, 
             label='Validacion', marker='s', markersize=4, alpha=0.6)
    ax1.set_xlabel('Epoca')
    ax1.set_ylabel('Perdida (Loss)')
    ax1.set_title(f'{model_name} - Perdida')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Marcar mejor epoca
    if val_loss:
        best_epoch = np.argmin(val_loss) + 1
        min_val_loss = min(val_loss)
        ax1.axvline(x=best_epoch, color='gray', linestyle=':', alpha=0.7)
        ax1.annotate(f'Mejor: {min_val_loss:.3f}', 
                    xy=(best_epoch, min_val_loss),
                    xytext=(5, 10), textcoords='offset points',
                    fontsize=9, color='gray')
    
    # Grafico de Accuracy
    ax2 = axes[1]
    ax2.plot(epochs, train_acc, '-', color=color, linewidth=2, 
             label='Entrenamiento', marker='o', markersize=4, alpha=0.8)
    ax2.plot(epochs, val_acc, '--', color=color, linewidth=2, 
             label='Validacion', marker='s', markersize=4, alpha=0.6)
    ax2.set_xlabel('Epoca')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.3, 0.75])
    
    # Marcar mejor accuracy
    if val_acc:
        best_epoch_acc = np.argmax(val_acc) + 1
        max_val_acc = max(val_acc)
        ax2.axvline(x=best_epoch_acc, color='gray', linestyle=':', alpha=0.7)
        ax2.annotate(f'Mejor: {max_val_acc:.2%}', 
                    xy=(best_epoch_acc, max_val_acc),
                    xytext=(5, -15), textcoords='offset points',
                    fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"    [OK] {output_path.name}")


def plot_class_metrics(classification_report, model_name, output_path):
    """Genera grafico de barras con metricas por clase."""
    classes = []
    precisions = []
    recalls = []
    f1_scores = []
    
    class_keys = ['1★', '2★', '3★', '4★', '5★']
    alt_keys = ['1', '2', '3', '4', '5']
    
    for key, alt in zip(class_keys, alt_keys):
        if key in classification_report:
            classes.append(key.replace('★', ' Est.'))
            precisions.append(classification_report[key].get('precision', 0))
            recalls.append(classification_report[key].get('recall', 0))
            f1_scores.append(classification_report[key].get('f1-score', 0))
        elif alt in classification_report:
            classes.append(f'{alt} Est.')
            precisions.append(classification_report[alt].get('precision', 0))
            recalls.append(classification_report[alt].get('recall', 0))
            f1_scores.append(classification_report[alt].get('f1-score', 0))
    
    if not classes:
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(classes))
    width = 0.25
    
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#3498db', alpha=0.85)
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#e74c3c', alpha=0.85)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#2ecc71', alpha=0.85)
    
    ax.set_xlabel('Clase')
    ax.set_ylabel('Puntuacion')
    ax.set_title(f'{model_name} - Metricas por Clase')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Valores sobre barras
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"    [OK] {output_path.name}")


def plot_model_type_comparison(experiments, model_type, output_path):
    """Compara diferentes ejecuciones del mismo tipo de modelo."""
    if len(experiments) < 2:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Grafico 1: Comparacion de accuracy
    ax1 = axes[0]
    names = []
    accuracies = []
    f1_scores = []
    
    for i, exp in enumerate(experiments):
        config = exp.get('configuration', {})
        eval_metrics = exp.get('evaluation_metrics', {})
        
        # Crear nombre descriptivo
        name = f"Exp {i+1}"
        details = []
        if 'hidden_size' in config:
            details.append(f"h={config['hidden_size']}")
        if 'num_layers' in config:
            details.append(f"L={config['num_layers']}")
        if 'num_heads' in config:
            details.append(f"heads={config['num_heads']}")
        if 'embedding_dim' in config:
            details.append(f"emb={config['embedding_dim']}")
        if details:
            name = ', '.join(details)
        
        names.append(name)
        accuracies.append(eval_metrics.get('test_accuracy', 0))
        f1_scores.append(eval_metrics.get('f1_macro', 0))
    
    x = np.arange(len(names))
    width = 0.35
    
    color = MODEL_COLORS.get(model_type.split('_')[0], '#2E86AB')
    
    bars1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', color=color, alpha=0.9)
    bars2 = ax1.bar(x + width/2, f1_scores, width, label='F1-Macro', color=color, alpha=0.5)
    
    ax1.set_ylabel('Puntuacion')
    ax1.set_title(f'{model_type} - Comparacion de Configuraciones')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim([0.4, 0.7])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Marcar mejor
    best_idx = np.argmax(accuracies)
    ax1.annotate('Mejor', xy=(best_idx - width/2, accuracies[best_idx]),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=10, fontweight='bold', color='green')
    
    # Grafico 2: Curvas de entrenamiento superpuestas
    ax2 = axes[1]
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(experiments)))
    
    for i, (exp, color_i) in enumerate(zip(experiments, cmap)):
        history = exp.get('training_results', {}).get('history', {})
        val_acc = history.get('val_accuracy', [])
        if val_acc:
            epochs = range(1, len(val_acc) + 1)
            label = f'Exp {i+1} (max: {max(val_acc):.2%})'
            ax2.plot(epochs, val_acc, '-', color=color_i, linewidth=2, 
                    label=label, alpha=0.8)
    
    ax2.set_xlabel('Epoca')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title(f'{model_type} - Evolucion del Entrenamiento')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_ylim([0.3, 0.7])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"    [OK] {output_path.name}")


def plot_global_comparison(best_models, output_path):
    """Genera comparacion global de los mejores modelos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Ordenar por accuracy
    best_models = sorted(best_models, key=lambda x: x['accuracy'], reverse=True)
    
    names = [m['display_name'] for m in best_models]
    accuracies = [m['accuracy'] for m in best_models]
    f1_scores = [m['f1_macro'] for m in best_models]
    categories = [m['category'] for m in best_models]
    
    # Asignar colores segun categoria
    colors = [MODEL_COLORS.get(cat.split('_')[0], '#2E86AB') for cat in categories]
    
    # Grafico 1: Barras horizontales de Accuracy y F1
    ax1 = axes[0]
    y = np.arange(len(names))
    height = 0.35
    
    bars1 = ax1.barh(y - height/2, accuracies, height, label='Accuracy', color=colors, alpha=0.9)
    bars2 = ax1.barh(y + height/2, f1_scores, height, label='F1-Macro', color=colors, alpha=0.5)
    
    ax1.set_xlabel('Puntuacion')
    ax1.set_title('Comparacion de Mejores Modelos')
    ax1.set_yticks(y)
    ax1.set_yticklabels(names)
    ax1.legend(loc='lower right')
    ax1.set_xlim([0.45, 0.7])
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Anotaciones
    for bar, acc in zip(bars1, accuracies):
        ax1.text(acc + 0.005, bar.get_y() + bar.get_height()/2,
                f'{acc:.2%}', va='center', fontsize=9, fontweight='bold')
    
    # Grafico 2: Scatter plot accuracy vs tiempo
    ax2 = axes[1]
    times = [m['time'] / 60 for m in best_models]  # minutos
    
    for i, m in enumerate(best_models):
        ax2.scatter(times[i], accuracies[i], s=200, c=[colors[i]], 
                   alpha=0.8, edgecolors='black', linewidths=1)
        ax2.annotate(m['display_name'], 
                    xy=(times[i], accuracies[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9)
    
    ax2.set_xlabel('Tiempo de Entrenamiento (minutos)')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Eficiencia: Accuracy vs Tiempo')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"    [OK] {output_path.name}")


def plot_class_heatmap(best_models, output_path):
    """Genera heatmap de F1-score por clase para los mejores modelos."""
    names = []
    class_f1_matrix = []
    
    for m in best_models:
        if m['class_f1']:
            names.append(m['display_name'])
            f1_values = []
            for c in CLASS_NAMES:
                key_star = f'{c}★'
                if key_star in m['class_f1']:
                    f1_values.append(m['class_f1'][key_star])
                elif c in m['class_f1']:
                    f1_values.append(m['class_f1'][c])
                else:
                    f1_values.append(0)
            class_f1_matrix.append(f1_values)
    
    if not names:
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    data = np.array(class_f1_matrix)
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.45, vmax=0.8)
    
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(CLASS_LABELS, rotation=45, ha='right')
    ax.set_yticklabels(names)
    
    ax.set_xlabel('Clase')
    ax.set_ylabel('Modelo')
    ax.set_title('F1-Score por Clase - Mejores Modelos')
    
    # Anotaciones
    for i in range(len(names)):
        for j in range(len(CLASS_NAMES)):
            color = 'white' if data[i, j] < 0.55 else 'black'
            ax.text(j, i, f'{data[i, j]:.2f}',
                   ha='center', va='center', color=color, fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='F1-Score', shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"    [OK] {output_path.name}")


def plot_training_time_comparison(best_models, output_path):
    """Genera comparacion de tiempos de entrenamiento."""
    # Ordenar por tiempo
    models_sorted = sorted(best_models, key=lambda x: x['time'])
    
    names = [m['display_name'] for m in models_sorted]
    times = [m['time'] / 60 for m in models_sorted]  # minutos
    categories = [m['category'] for m in models_sorted]
    colors = [MODEL_COLORS.get(cat.split('_')[0], '#2E86AB') for cat in categories]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(names, times, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Tiempo de Entrenamiento (minutos)')
    ax.set_title('Tiempo de Entrenamiento - Mejores Modelos')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Anotaciones
    for bar, time in zip(bars, times):
        if time >= 60:
            label = f'{time/60:.1f}h'
        else:
            label = f'{time:.1f}min'
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
               label, va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"    [OK] {output_path.name}")


def extract_experiment_info(experiment):
    """Extrae informacion relevante de un experimento."""
    exp_name = experiment.get('experiment_name', '')
    config = experiment.get('configuration', {})
    eval_metrics = experiment.get('evaluation_metrics', {})
    training_results = experiment.get('training_results', {})
    
    model_type = config.get('model_type', 'Unknown')
    category = get_model_category(model_type)
    display_name = get_display_name(model_type, exp_name)
    
    # Obtener model_name desde model_path o experiment_name
    model_path = training_results.get('model_path', '')
    if model_path:
        model_name = Path(model_path).stem
    else:
        model_name = exp_name
    
    # Extraer F1 por clase
    class_f1 = {}
    classification_report = eval_metrics.get('classification_report', {})
    for key, value in classification_report.items():
        if key not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(value, dict):
            class_f1[key] = value.get('f1-score', 0)
    
    return {
        'experiment_name': exp_name,
        'model_name': model_name,
        'model_type': model_type,
        'category': category,
        'display_name': display_name,
        'accuracy': eval_metrics.get('test_accuracy', 0),
        'f1_macro': eval_metrics.get('f1_macro', 0),
        'time': training_results.get('training_time', 0),
        'epochs': training_results.get('epochs_trained', 0),
        'class_f1': class_f1,
        'classification_report': classification_report,
        'history': training_results.get('history', {}),
        'config': config,
    }


def main():
    """Funcion principal."""
    print("=" * 70)
    print("Generador de Figuras v2 - Mejores Modelos por Categoria")
    print("=" * 70)
    
    # Estructura para almacenar experimentos por categoria
    experiments_by_category = defaultdict(list)
    all_experiments = []
    
    # Cargar todos los experimentos
    parts = [
        ('project_part_1', 'output/project_part_1/experiment_history.json'),
        ('project_part_2', 'output/project_part_2/experiment_history.json'),
        ('project_part_3', 'output/project_part_3/experiment_history.json'),
        ('project_part_4', 'output/project_part_4/experiment_history.json'),
    ]
    
    for part_name, json_rel_path in parts:
        json_path = BASE_DIR / json_rel_path
        
        if not json_path.exists():
            print(f"\n[SKIP] No encontrado: {json_path}")
            continue
        
        print(f"\nCargando: {part_name}")
        data = load_experiment_history(json_path)
        
        for exp in data.get('experiments', []):
            info = extract_experiment_info(exp)
            info['part'] = part_name
            experiments_by_category[info['category']].append(info)
            all_experiments.append(info)
    
    # Seleccionar el mejor de cada categoria
    best_models = []
    
    print("\n" + "=" * 70)
    print("Seleccion de Mejores Modelos por Categoria")
    print("=" * 70)
    
    for category, exps in sorted(experiments_by_category.items()):
        # Ordenar por accuracy descendente
        exps_sorted = sorted(exps, key=lambda x: x['accuracy'], reverse=True)
        best = exps_sorted[0]
        best_models.append(best)
        
        print(f"\n[{category}] Mejor: {best['display_name']}")
        print(f"    Accuracy: {best['accuracy']:.1%}, F1-Macro: {best['f1_macro']:.3f}")
        print(f"    Tiempo: {best['time']/60:.1f} min, Epocas: {best['epochs']}")
    
    # Generar figuras para cada mejor modelo
    print("\n" + "=" * 70)
    print("Generando Figuras de Mejores Modelos")
    print("=" * 70)
    
    for model in best_models:
        category = model['category']
        print(f"\n>>> {model['display_name']} ({category})")
        
        # Nombre de archivo seguro
        safe_name = category.replace(' ', '_').replace('+', '_')
        
        # Curvas de entrenamiento
        if model['history']:
            output_file = OUTPUT_DIR / f"best_{safe_name}_training.png"
            plot_training_curves(model['history'], model['display_name'], 
                               output_file, category)
        
        # Metricas por clase
        if model['classification_report']:
            output_file = OUTPUT_DIR / f"best_{safe_name}_class_metrics.png"
            plot_class_metrics(model['classification_report'], 
                              model['display_name'], output_file)
    
    # Generar comparativas por tipo de modelo
    print("\n" + "=" * 70)
    print("Generando Comparativas por Tipo de Modelo")
    print("=" * 70)
    
    for category, exps in experiments_by_category.items():
        if len(exps) >= 2:
            print(f"\n>>> Comparativa: {category} ({len(exps)} experimentos)")
            safe_name = category.replace(' ', '_').replace('+', '_')
            output_file = OUTPUT_DIR / f"comparison_{safe_name}_variants.png"
            
            # Reconstruir experimentos completos para esta comparativa
            plot_model_type_comparison(
                [{'configuration': e['config'], 
                  'evaluation_metrics': {'test_accuracy': e['accuracy'], 'f1_macro': e['f1_macro']},
                  'training_results': {'history': e['history']}} 
                 for e in exps],
                category, output_file
            )
    
    # Generar graficos comparativos globales
    print("\n" + "=" * 70)
    print("Generando Graficos Comparativos Globales")
    print("=" * 70)
    
    # Comparacion global de mejores modelos
    print("\n>>> Comparacion global de mejores modelos")
    plot_global_comparison(best_models, OUTPUT_DIR / "global_best_comparison.png")
    
    # Heatmap de rendimiento por clase
    print("\n>>> Heatmap de F1-Score por clase")
    plot_class_heatmap(best_models, OUTPUT_DIR / "global_class_heatmap.png")
    
    # Comparacion de tiempos
    print("\n>>> Comparacion de tiempos de entrenamiento")
    plot_training_time_comparison(best_models, OUTPUT_DIR / "global_training_time.png")
    
    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN DE MEJORES MODELOS")
    print("=" * 70)
    print(f"\n{'Categoria':<20} {'Modelo':<25} {'Accuracy':<12} {'F1-Macro':<12} {'Tiempo':<12}")
    print("-" * 80)
    
    for m in sorted(best_models, key=lambda x: x['accuracy'], reverse=True):
        time_str = f"{m['time']/60:.1f}min" if m['time'] < 3600 else f"{m['time']/3600:.1f}h"
        print(f"{m['category']:<20} {m['display_name']:<25} {m['accuracy']:.1%}        {m['f1_macro']:.3f}        {time_str}")
    
    print(f"\n{'=' * 70}")
    print(f"Figuras guardadas en: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
