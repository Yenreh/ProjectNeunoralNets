"""
Data Handler Module

Manages validation data loading and random sample generation.
"""

from helpers.data_loader import DataLoader

# Global cache for validation data
validation_data = None
DATA_DIR = "data"


def load_validation_samples():
    """Carga muestras del conjunto de validación para pruebas aleatorias"""
    try:
        # Usar DataLoader de TensorFlow (funciona para ambos casos)
        data_loader = DataLoader(DATA_DIR)
        _, val_df, _ = data_loader.load_all_data()
        return val_df.sample(min(100, len(val_df)))  # Máximo 100 muestras
    except Exception as e:
        print(f"Error cargando datos de validación: {e}")
        return None


def get_random_sample_data():
    """Obtiene una muestra aleatoria en inglés del conjunto de validación"""
    global validation_data

    if validation_data is None:
        validation_data = load_validation_samples()

    if validation_data is None or len(validation_data) == 0:
        return None, "No hay datos de validación disponibles"

    # Filtrar solo muestras en inglés
    english_samples = validation_data[validation_data["language"] == "en"]
    if len(english_samples) == 0:
        return None, "No hay muestras en inglés disponibles"

    # Seleccionar muestra aleatoria
    sample = english_samples.sample(1).iloc[0]

    # Construir texto como en el entrenamiento
    title = sample.get("review_title", "")
    body = sample.get("review_body", "")
    combined_text = f"{title} {body}".strip()

    return {
        "text": combined_text,
        "true_stars": int(sample.get("stars", 0)),
        "title": title,
        "body": body,
    }, None
