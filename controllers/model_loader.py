"""
Model Loader Module

Handles loading, caching, and management of TensorFlow and PyTorch models.
"""

import os
import gc
import tensorflow as tf
import torch
from helpers.results_manager import ResultsManager
from helpers.utils import load_model_components_tf, load_model_components_torch

# Global variables for model management
loaded_models = {}
current_loaded_model = None

# Configuration
MODEL_DIR = "models/project_part_1"
OUTPUT_DIR = "output"


def get_model_framework(model_name, model_dir):
    """Determina si un modelo es TensorFlow o PyTorch basándose en la extensión del archivo"""
    if os.path.exists(os.path.join(model_dir, f"{model_name}.h5")):
        return "tensorflow"
    elif os.path.exists(os.path.join(model_dir, f"{model_name}.pth")):
        return "pytorch"
    return None


def load_available_models(subdir="project_part_1"):
    """Carga la lista de modelos disponibles desde el directorio models/<subdir>"""
    models = []
    model_dir = os.path.join("models", subdir)
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if (file.endswith(".h5") or file.endswith(".pth")) and not file.startswith(
                "backup"
            ):
                model_name = file.replace(".h5", "").replace(".pth", "")
                framework = get_model_framework(model_name, model_dir)
                models.append({"name": model_name, "framework": framework})
    return models


def load_model_stats():
    """Carga las estadísticas de los modelos desde output/project_part_*/experiment_history.json"""
    stats = {}

    # Buscar en todos los subdirectorios de output (project_part_1, project_part_2, etc.)
    if os.path.exists(OUTPUT_DIR):
        for project_dir in os.listdir(OUTPUT_DIR):
            project_path = os.path.join(OUTPUT_DIR, project_dir)
            if os.path.isdir(project_path) and project_dir.startswith("project_part_"):
                try:
                    results_manager = ResultsManager(
                        OUTPUT_DIR, project_part=project_dir
                    )
                    history_data = results_manager.load_experiment_history()

                    for exp in history_data.get("experiments", []):
                        model_path = exp.get("training_results", {}).get(
                            "model_path", ""
                        )
                        if model_path:
                            # Extraer nombre del modelo y eliminar extensiones .h5 y .pth
                            model_name = (
                                os.path.basename(model_path)
                                .replace(".h5", "")
                                .replace(".pth", "")
                            )
                            stats[model_name] = {
                                "experiment_id": exp.get("experiment_id", 0),
                                "accuracy": exp.get("training_results", {}).get(
                                    "final_val_accuracy", 0
                                ),
                                "loss": exp.get("training_results", {}).get(
                                    "final_val_loss", 0
                                ),
                                "f1_score": exp.get("evaluation_metrics", {}).get(
                                    "f1_weighted", 0
                                ),
                                "epochs": exp.get("training_results", {}).get(
                                    "epochs_trained", 0
                                ),
                                "training_time": exp.get("training_results", {}).get(
                                    "training_time", 0
                                ),
                                "parameters": exp.get("configuration", {}).get(
                                    "total_parameters", 0
                                ),
                                "language": exp.get("configuration", {}).get(
                                    "language_filter", "multi"
                                ),
                                "architecture": exp.get("configuration", {}).get(
                                    "hidden_layers", []
                                ),
                                "model_type": exp.get("configuration", {}).get(
                                    "model_type", "Unknown"
                                ),
                                "vocab_size": exp.get("dataset_info", {}).get(
                                    "vocab_size", 0
                                ),
                                "train_samples": exp.get("dataset_info", {}).get(
                                    "train_samples", 0
                                ),
                                "project_part": project_dir,
                            }
                except Exception as e:
                    print(f"Error cargando estadísticas de {project_dir}: {e}")
                    continue

    return stats


def clear_loaded_models():
    """Limpia todos los modelos de memoria y libera recursos"""
    global loaded_models, current_loaded_model

    for model_name, model_data in loaded_models.items():
        if "model" in model_data and model_data["model"] is not None:
            try:
                # Limpiar memoria del modelo de TensorFlow
                del model_data["model"]
                tf.keras.backend.clear_session()
                print(f"Modelo {model_name} liberado de memoria")
            except Exception as e:
                print(f"Error liberando modelo {model_name}: {e}")

    loaded_models.clear()
    current_loaded_model = None

    # Forzar recolección de basura
    gc.collect()


def load_model_and_preprocessor(model_name):
    """Carga un modelo específico y sus componentes de preprocesamiento (TensorFlow o PyTorch)"""
    global current_loaded_model

    # Si ya hay un modelo cargado y es diferente, limpiar memoria
    if current_loaded_model and current_loaded_model != model_name:
        print(f"Cambiando de modelo {current_loaded_model} a {model_name}")
        clear_loaded_models()

    if model_name in loaded_models:
        current_loaded_model = model_name
        return loaded_models[model_name]

    try:
        # Buscar el modelo en todos los subdirectorios posibles
        model_dir = None
        framework = None
        
        for subdir in ["project_part_1", "project_part_2", "project_part_3"]:
            test_dir = os.path.join("models", subdir)
            test_framework = get_model_framework(model_name, test_dir)
            if test_framework:
                model_dir = test_dir
                framework = test_framework
                break
        
        if not model_dir or not framework:
            print(f"No se encontró el modelo {model_name} en ningún subdirectorio")
            return None

        if framework == "tensorflow":
            # Cargar modelo TensorFlow
            components = load_model_components_tf(model_name, model_dir)
        elif framework == "pytorch":
            # Cargar modelo PyTorch
            components = load_model_components_torch(model_name, model_dir)
        else:
            print(f"No se pudo determinar el framework para {model_name}")
            return None

        if components and "model" in components:
            # Determinar device para PyTorch
            device = None
            if framework == "pytorch":
                device = next(components["model"].parameters()).device
                print(f"  Device: {device}")

            # Usar componentes guardados (método eficiente)
            model_data = {
                "model": components["model"],
                "vectorizer": components.get("vectorizer"),
                "tokenizer": components.get("tokenizer"),
                "label_encoder": components.get("label_encoder"),
                "vocab_size": components.get("vocab_size", 5000),
                "num_classes": components.get("num_classes", 5),
                "max_length": components.get("max_length", 100),
                "framework": framework,
                "device": device,  # Guardar device para PyTorch
            }

            loaded_models[model_name] = model_data
            current_loaded_model = model_name
            print(f"Modelo {model_name} ({framework}) cargado exitosamente")
            return model_data

        else:
            print(f"No se pudieron cargar los componentes para {model_name}")
            return None

    except Exception as e:
        print(f"Error cargando modelo {model_name}: {e}")
        import traceback

        traceback.print_exc()
        return None


def cleanup_on_exit():
    """Función de limpieza que se ejecuta al cerrar la aplicación"""
    print("Cerrando aplicación... Liberando memoria de modelos")
    clear_loaded_models()


def get_current_loaded_model():
    """Retorna el nombre del modelo actualmente cargado"""
    return current_loaded_model


def get_loaded_models():
    """Retorna el diccionario de modelos cargados"""
    return loaded_models
