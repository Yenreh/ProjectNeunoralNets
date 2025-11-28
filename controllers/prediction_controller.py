"""
Prediction Controller Module

Handles prediction logic for both TensorFlow and PyTorch models.
Supports both BoW and Embedding architectures.
"""

import numpy as np
import tensorflow as tf
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences


def make_prediction(model_data, text_input):
    """
    Hace una predicción usando el modelo cargado.

    Args:
        model_data: Diccionario con el modelo y sus componentes
        text_input: Texto de entrada para clasificar

    Returns:
        dict: Resultado de la predicción con estrellas, confianza y probabilidades
        str: Mensaje de error si ocurre algún problema
    """
    try:
        framework = model_data.get("framework", "tensorflow")
        device = model_data.get("device", None)  # Device para PyTorch

        # Preprocesar texto según el tipo de modelo
        if model_data.get("tokenizer") is not None:
            # Modelo de embedding - usar tokenizer
            sequences = model_data["tokenizer"].texts_to_sequences([text_input])
            max_length = model_data.get("max_length", 100)

            if framework == "pytorch":
                # PyTorch: crear tensor y moverlo al device correcto
                padded = pad_sequences(
                    sequences, maxlen=max_length, padding="post", truncating="post"
                )
                X_input = torch.tensor(padded, dtype=torch.long)
                # CRITICAL: Mover tensor al mismo device que el modelo
                if device is not None:
                    X_input = X_input.to(device)
            else:
                # TensorFlow: pad sequences
                X_input = tf.keras.preprocessing.sequence.pad_sequences(
                    sequences, maxlen=max_length, padding="post", truncating="post"
                )

        elif model_data.get("vectorizer") is not None:
            # Modelo BoW - usar vectorizer
            X_input = model_data["vectorizer"].transform([text_input])

            if framework == "pytorch":
                # PyTorch: convertir a tensor y moverlo al device correcto
                X_input = torch.tensor(X_input.toarray(), dtype=torch.float32)
                # CRITICAL: Mover tensor al mismo device que el modelo
                if device is not None:
                    X_input = X_input.to(device)
            else:
                # TensorFlow: convertir a array denso
                X_input = X_input.toarray().astype(np.float32)
        else:
            return None, "No se encontró vectorizer ni tokenizer válido"

        # Hacer predicción según el framework
        if framework == "pytorch":
            model = model_data["model"]
            model.eval()
            with torch.no_grad():
                prediction = model(X_input)
                prediction = torch.softmax(prediction, dim=1).cpu().numpy()
        else:
            # TensorFlow
            prediction = model_data["model"].predict(X_input, verbose=0)

        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][predicted_class])

        # Convertir a estrellas (1-5)
        stars = int(model_data["label_encoder"].classes_[predicted_class])

        # Obtener probabilidades por clase
        class_probabilities = {}
        for i, class_label in enumerate(model_data["label_encoder"].classes_):
            class_probabilities[f"{class_label} estrellas"] = float(prediction[0][i])

        result = {
            "predicted_stars": stars,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "text_preview": (
                text_input[:100] + "..." if len(text_input) > 100 else text_input
            ),
            "model_type": (
                "embedding" if model_data["tokenizer"] is not None else "bow"
            ),
            "framework": framework,
        }

        return result, None

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"Error en predicción: {str(e)}"
