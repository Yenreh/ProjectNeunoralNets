"""
Utility Functions Module
Proyecto Redes Neuronales 2025-II - Universidad Del Valle

Funciones utilitarias generales para configuración de experimentos,
evaluación de modelos y manejo de componentes.
Compatible con TensorFlow y PyTorch.
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Optional


def evaluate_model_tf(
    model, X_test: np.ndarray, y_test: np.ndarray, class_names: List[str] = None
) -> Dict:
    """
    Evalúa el rendimiento del modelo TensorFlow en datos de prueba.

    Args:
        model: Modelo de Keras entrenado
        X_test, y_test: Datos de prueba
        class_names: Nombres de clases para el reporte de clasificación

    Returns:
        Dict: Métricas de evaluación
    """
    from sklearn.metrics import classification_report

    print("Evaluando modelo en datos de prueba...")

    # Obtener predicciones
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test

    # Calcular métricas
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Reporte de clasificación
    if class_names is None:
        class_names = [f"Clase_{i}" for i in range(len(np.unique(y_true)))]

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    print(f"Precisión de Prueba: {test_accuracy:.4f}")
    print(f"Pérdida de Prueba: {test_loss:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return {
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def evaluate_model_torch(
    model, test_loader, criterion, device, class_names: List[str] = None
) -> Dict:
    """
    Evalúa el rendimiento del modelo PyTorch en datos de prueba.

    Args:
        model: Modelo PyTorch entrenado
        test_loader: DataLoader de prueba
        criterion: Función de pérdida
        device: Dispositivo (cuda/cpu)
        class_names: Nombres de clases para el reporte de clasificación

    Returns:
        Dict: Métricas de evaluación
    """
    import torch
    from sklearn.metrics import classification_report

    print("Evaluando modelo en datos de prueba...")

    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Reporte de clasificación
    if class_names is None:
        class_names = [f"Clase_{i}" for i in range(len(np.unique(y_true)))]

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    print(f"Precisión de Prueba: {test_accuracy:.4f}")
    print(f"Pérdida de Prueba: {test_loss:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return {
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def get_gpu_info_tf():
    """Obtiene información de GPU para TensorFlow."""
    import tensorflow as tf

    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            gpu_info = {
                "gpu_available": True,
                "gpu_count": len(gpus),
                "gpu_names": [
                    tf.config.experimental.get_device_details(gpu)["device_name"]
                    for gpu in gpus
                ],
                "framework": "tensorflow",
            }
        else:
            gpu_info = {
                "gpu_available": False,
                "gpu_count": 0,
                "gpu_names": [],
                "framework": "tensorflow",
            }
        return gpu_info
    except Exception as e:
        return {"gpu_available": False, "error": str(e), "framework": "tensorflow"}


def get_gpu_info_torch():
    """Obtiene información de GPU para PyTorch."""
    import torch

    try:
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_available": True,
                "gpu_count": torch.cuda.device_count(),
                "gpu_names": [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ],
                "cuda_version": torch.version.cuda,
                "framework": "pytorch",
            }
        else:
            gpu_info = {
                "gpu_available": False,
                "gpu_count": 0,
                "gpu_names": [],
                "framework": "pytorch",
            }
        return gpu_info
    except Exception as e:
        return {"gpu_available": False, "error": str(e), "framework": "pytorch"}


def setup_experiment_environment_tf(seed: int = 42):
    """
    Configura entorno reproducible para experimentos con TensorFlow.

    Args:
        seed (int): Semilla aleatoria para reproducibilidad
    """
    import tensorflow as tf

    # Establecer semillas
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Configurar GPU si está disponible
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU configurada: {len(gpus)} GPU(s) disponibles")
        except RuntimeError as e:
            print(f"Error de configuración GPU: {e}")
    else:
        print("No hay GPU disponible, usando CPU")

    return get_gpu_info_tf()


def setup_experiment_environment_torch(seed: int = 42):
    """
    Configura entorno reproducible para experimentos con PyTorch.

    Args:
        seed (int): Semilla aleatoria para reproducibilidad
    """
    import torch
    import random

    # Establecer semillas
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"GPU configurada: {torch.cuda.device_count()} GPU(s) disponibles")
        print(f"Usando: {torch.cuda.get_device_name(0)}")
    else:
        print("No hay GPU disponible, usando CPU")

    return get_gpu_info_torch()


def save_model_components_tf(
    model_name: str,
    model,
    vectorizer=None,
    tokenizer=None,
    label_encoder=None,
    model_dir: str = "models/project_part_1",
):
    """
    Guarda modelo TensorFlow y componentes de preprocesamiento.

    Args:
        model_name (str): Nombre base para los archivos
        model: Modelo de Keras entrenado
        vectorizer: Vectorizador TF-IDF (opcional)
        tokenizer: Tokenizer de Keras (opcional)
        label_encoder: Encoder de etiquetas
        model_dir (str): Directorio para guardar
    """
    try:
        os.makedirs(model_dir, exist_ok=True)

        # Guardar modelo Keras
        model_path = os.path.join(model_dir, f"{model_name}.h5")
        model.save(model_path)
        print(f"Modelo guardado: {model_path}")

        # Guardar vectorizador TF-IDF
        if vectorizer is not None:
            vectorizer_path = os.path.join(model_dir, f"{model_name}_vectorizer.pkl")
            with open(vectorizer_path, "wb") as f:
                pickle.dump(vectorizer, f)
            print(f"Vectorizador TF-IDF guardado: {vectorizer_path}")

        # Guardar tokenizer
        if tokenizer is not None:
            tokenizer_path = os.path.join(model_dir, f"{model_name}_tokenizer.pkl")
            with open(tokenizer_path, "wb") as f:
                pickle.dump(tokenizer, f)
            print(f"Tokenizer guardado: {tokenizer_path}")

        # Guardar label encoder
        if label_encoder is not None:
            label_encoder_path = os.path.join(
                model_dir, f"{model_name}_label_encoder.pkl"
            )
            with open(label_encoder_path, "wb") as f:
                pickle.dump(label_encoder, f)
            print(f"Label encoder guardado: {label_encoder_path}")

        print(f"Componentes guardados en {model_dir}/")

    except Exception as e:
        print(f"Error guardando componentes: {e}")
        raise


def save_model_components_torch(
    model_name: str,
    model,
    tokenizer=None,
    vectorizer=None,
    label_encoder=None,
    vocab=None,
    max_length=None,
    model_dir: str = "models/project_part_1",
):
    """
    Guarda modelo PyTorch y componentes de preprocesamiento.

    Args:
        model_name (str): Nombre base para los archivos
        model: Modelo PyTorch entrenado
        tokenizer: Tokenizer de Keras (opcional, para modelos embedding)
        vectorizer: Vectorizador TF-IDF (opcional, para modelos BoW)
        label_encoder: Encoder de etiquetas
        vocab: Vocabulario personalizado (opcional)
        max_length: Longitud máxima de secuencias (para modelos embedding)
        model_dir (str): Directorio para guardar
    """
    import torch

    try:
        os.makedirs(model_dir, exist_ok=True)

        # Obtener configuración del modelo
        model_config = {}
        if hasattr(model, "get_config"):
            model_config = model.get_config()
        else:
            # Fallback para compatibilidad
            if hasattr(model, "fc1"):
                model_config["input_dim"] = model.fc1.in_features
            if hasattr(model, "fc_out"):
                model_config["num_classes"] = model.fc_out.out_features
            if hasattr(model, "hidden_layers"):
                model_config["hidden_layers"] = model.hidden_layers
            if hasattr(model, "embedding_dim"):
                model_config["embedding_dim"] = model.embedding_dim
            if hasattr(model, "embedding"):
                # Para modelos con embedding, guardar vocab_size
                model_config["vocab_size"] = model.embedding.num_embeddings

        # Agregar información adicional para modelos embedding
        if tokenizer is not None:
            model_config["vocab_size"] = len(tokenizer.word_index) + 1
            if max_length is not None:
                model_config["max_length"] = max_length
        elif vocab is not None:
            model_config["vocab_size"] = len(vocab)
            if max_length is not None:
                model_config["max_length"] = max_length

        # Guardar modelo PyTorch
        model_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save(
            {"model_state_dict": model.state_dict(), "model_config": model_config},
            model_path,
        )
        print(f"Modelo guardado: {model_path}")
        print(f"  - Configuración guardada: {model_config}")

        # Guardar tokenizer (para modelos embedding)
        if tokenizer is not None:
            tokenizer_path = os.path.join(model_dir, f"{model_name}_tokenizer.pkl")
            with open(tokenizer_path, "wb") as f:
                pickle.dump(tokenizer, f)
            print(f"Tokenizer guardado: {tokenizer_path}")

        # Guardar vectorizador (para modelos BoW)
        if vectorizer is not None:
            vectorizer_path = os.path.join(model_dir, f"{model_name}_vectorizer.pkl")
            with open(vectorizer_path, "wb") as f:
                pickle.dump(vectorizer, f)
            print(f"Vectorizador guardado: {vectorizer_path}")

        # Guardar vocabulario
        if vocab is not None:
            vocab_path = os.path.join(model_dir, f"{model_name}_vocab.pkl")
            with open(vocab_path, "wb") as f:
                pickle.dump(vocab, f)
            print(f"Vocabulario guardado: {vocab_path}")

        # Guardar label encoder
        if label_encoder is not None:
            label_encoder_path = os.path.join(
                model_dir, f"{model_name}_label_encoder.pkl"
            )
            with open(label_encoder_path, "wb") as f:
                pickle.dump(label_encoder, f)
            print(f"Label encoder guardado: {label_encoder_path}")

        print(f"Componentes guardados en {model_dir}/")

    except Exception as e:
        print(f"Error guardando componentes: {e}")
        raise


def load_model_components_tf(
    model_name: str, model_dir: str = "models/project_part_1"
) -> Dict:
    """Carga modelo TensorFlow y componentes."""
    import tensorflow as tf

    try:
        components = {}

        # Cargar modelo
        model_path = os.path.join(model_dir, f"{model_name}.h5")
        if os.path.exists(model_path):
            components["model"] = tf.keras.models.load_model(model_path)
            print(f"Modelo cargado: {model_path}")

        # Cargar vectorizador
        vectorizer_path = os.path.join(model_dir, f"{model_name}_vectorizer.pkl")
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, "rb") as f:
                components["vectorizer"] = pickle.load(f)
            print(f"Vectorizador cargado")

        # Cargar tokenizer
        tokenizer_path = os.path.join(model_dir, f"{model_name}_tokenizer.pkl")
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, "rb") as f:
                components["tokenizer"] = pickle.load(f)
            print(f"Tokenizer cargado")

        # Cargar label encoder
        label_encoder_path = os.path.join(model_dir, f"{model_name}_label_encoder.pkl")
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, "rb") as f:
                components["label_encoder"] = pickle.load(f)
            print(f"Label encoder cargado")

        return components

    except Exception as e:
        print(f"Error cargando componentes: {e}")
        return None


def load_model_components_torch(
    model_name: str, model_dir: str = "models/project_part_1"
) -> Dict:
    """
    Carga modelo PyTorch y componentes sin requerir la clase del modelo.

    Args:
        model_name (str): Nombre del modelo
        model_dir (str): Directorio donde está el modelo

    Returns:
        Dict: Diccionario con componentes cargados
    """
    import torch
    from helpers.models import MLPClassifier, MLPWithEmbedding

    try:
        components = {}

        # Determinar dispositivo (GPU si está disponible)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cargar modelo
        model_path = os.path.join(model_dir, f"{model_name}.pth")
        if not os.path.exists(model_path):
            print(f"Archivo del modelo no encontrado: {model_path}")
            return None

        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint.get("model_config", {})

        # Detectar tipo de modelo basado en archivos auxiliares
        tokenizer_path = os.path.join(model_dir, f"{model_name}_tokenizer.pkl")
        vocab_path = os.path.join(model_dir, f"{model_name}_vocab.pkl")
        vectorizer_path = os.path.join(model_dir, f"{model_name}_vectorizer.pkl")

        has_tokenizer = os.path.exists(tokenizer_path)
        has_vocab = os.path.exists(vocab_path)
        has_vectorizer = os.path.exists(vectorizer_path)

        # Determinar tipo de modelo y crear instancia
        if has_tokenizer or has_vocab or "vocab_size" in model_config:
            # Modelo con Embedding
            print(f"Detectado modelo de Embedding")

            # Cargar tokenizer o vocab para obtener vocab_size
            if has_tokenizer:
                with open(tokenizer_path, "rb") as f:
                    tokenizer = pickle.load(f)
                    components["tokenizer"] = tokenizer
                    vocab_size = len(tokenizer.word_index) + 1
                    components["max_length"] = 100  # Default, se puede ajustar
                    print(f"Tokenizer cargado, vocab_size: {vocab_size}")
            elif has_vocab:
                with open(vocab_path, "rb") as f:
                    vocab = pickle.load(f)
                    components["vocab"] = vocab
                    vocab_size = len(vocab)
                    print(f"Vocabulario cargado, vocab_size: {vocab_size}")
            else:
                # Fallback: usar vocab_size del checkpoint
                vocab_size = model_config.get("vocab_size", 5000)
                print(
                    f"⚠️  Tokenizer/vocab no encontrado, usando vocab_size del checkpoint: {vocab_size}"
                )
                print(
                    f"   Nota: Las predicciones pueden no funcionar correctamente sin tokenizer"
                )

                # Crear un tokenizer dummy para compatibilidad
                # El usuario necesitará proporcionar el tokenizer correcto o re-entrenar
                components["tokenizer"] = None
                components["max_length"] = model_config.get("max_length", 100)
                components["vocab_size"] = vocab_size

            # Crear modelo de embedding
            # Intentar obtener configuración del checkpoint
            embedding_dim = model_config.get("embedding_dim", 300)
            hidden_layers = model_config.get("hidden_layers", [256, 128, 64])
            num_classes = model_config.get("num_classes", 5)

            model = MLPWithEmbedding(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_layers=hidden_layers,
                num_classes=num_classes,
                dropout_rate=0.3,
            )

        elif has_vectorizer:
            # Modelo BoW
            print(f"Detectado modelo BoW")

            # Cargar vectorizador
            with open(vectorizer_path, "rb") as f:
                vectorizer = pickle.load(f)
                components["vectorizer"] = vectorizer
                input_dim = len(vectorizer.vocabulary_)
                print(f"Vectorizador cargado, input_dim: {input_dim}")

            # Crear modelo BoW
            hidden_layers = model_config.get("hidden_layers", [256, 128, 64])
            num_classes = model_config.get("num_classes", 5)

            model = MLPClassifier(
                input_dim=input_dim,
                hidden_layers=hidden_layers,
                num_classes=num_classes,
                dropout_rate=0.3,
            )
        else:
            print(
                f"No se pudo determinar tipo de modelo (falta vectorizer/tokenizer/vocab)"
            )
            return None

        # Cargar pesos del modelo
        model.load_state_dict(checkpoint["model_state_dict"])

        # Mover modelo a GPU si está disponible
        model = model.to(device)
        model.eval()

        components["model"] = model
        components["device"] = device
        print(f"Modelo cargado exitosamente en dispositivo: {device}")

        # Cargar label encoder
        label_encoder_path = os.path.join(model_dir, f"{model_name}_label_encoder.pkl")
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, "rb") as f:
                components["label_encoder"] = pickle.load(f)
            print(f"Label encoder cargado")
            components["num_classes"] = len(components["label_encoder"].classes_)

        # Agregar información adicional
        if "vocab_size" not in components:
            components["vocab_size"] = (
                vocab_size
                if (has_tokenizer or has_vocab or "vocab_size" in model_config)
                else input_dim
            )

        return components

    except Exception as e:
        print(f"Error cargando componentes PyTorch: {e}")
        import traceback

        traceback.print_exc()
        return None


# Compatibilidad con nombres anteriores
evaluate_model = evaluate_model_tf
get_gpu_info = get_gpu_info_tf
setup_experiment_environment = setup_experiment_environment_tf
save_model_components = save_model_components_tf
load_model_components = load_model_components_tf
