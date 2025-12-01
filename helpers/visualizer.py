"""
Visualization Module
Proyecto Redes Neuronales 2025-II - Universidad Del Valle

Maneja la visualización de resultados de entrenamiento y rendimiento del modelo.
Compatible con TensorFlow y PyTorch.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix

# Estilos de visualización
plt.style.use('default')
sns.set_palette("husl")


class Visualizer:
    """Maneja la visualización de resultados de entrenamiento y rendimiento del modelo."""
    
    @staticmethod
    def plot_training_history(history: Dict, 
                            model_name: str = "Model",
                            save_path: Optional[str] = None,
                            framework: str = "tensorflow"):
        """
        Grafica métricas de entrenamiento y validación.
        
        Args:
            history (Dict): Historial de entrenamiento
            model_name (str): Nombre para el título del gráfico
            save_path (str, optional): Ruta para guardar el gráfico
            framework (str): 'tensorflow' o 'pytorch' para adaptar keys
        """
        # Determinar las keys según el framework
        if framework == "pytorch":
            train_acc_key = 'train_accuracy'
            val_acc_key = 'val_accuracy'
            train_loss_key = 'train_loss'
            val_loss_key = 'val_loss'
            train_f1_key = 'train_f1_macro'
            val_f1_key = 'val_f1_macro'
        else:  # tensorflow
            train_acc_key = 'accuracy'
            val_acc_key = 'val_accuracy'
            train_loss_key = 'loss'
            val_loss_key = 'val_loss'
            train_f1_key = 'train_f1_macro'
            val_f1_key = 'val_f1_macro'
        
        # Verificar si tenemos métricas F1
        has_f1 = train_f1_key in history and val_f1_key in history
        
        # Crear subplots según si tenemos F1 o no
        if has_f1:
            fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            axes = [axes[0], axes[1]]  # Para compatibilidad con código siguiente
        
        # Graficar precisión
        if train_acc_key in history and val_acc_key in history:
            axes[0].plot(history[train_acc_key], label='Precisión Entrenamiento', linewidth=2)
            axes[0].plot(history[val_acc_key], label='Precisión Validación', linewidth=2)
            axes[0].set_title(f'{model_name} - Precisión')
            axes[0].set_xlabel('Época')
            axes[0].set_ylabel('Precisión')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Graficar pérdida
        axes[1].plot(history[train_loss_key], label='Pérdida Entrenamiento', linewidth=2)
        axes[1].plot(history[val_loss_key], label='Pérdida Validación', linewidth=2)
        axes[1].set_title(f'{model_name} - Pérdida')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Pérdida')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Graficar F1 Macro si está disponible
        if has_f1:
            axes[2].plot(history[train_f1_key], label='F1-Macro Entrenamiento', linewidth=2)
            axes[2].plot(history[val_f1_key], label='F1-Macro Validación', linewidth=2)
            axes[2].set_title(f'{model_name} - F1-Macro')
            axes[2].set_xlabel('Época')
            axes[2].set_ylabel('F1-Macro')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Matriz de confusión guardada en: {save_path}")
        
        plt.show()
