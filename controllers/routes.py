"""
Routes Module

Defines all Flask routes for the demo application.
"""

from flask import render_template, request, jsonify
from .model_loader import (
    load_available_models,
    load_model_stats,
    load_model_and_preprocessor,
    clear_loaded_models,
    get_model_framework,
    get_current_loaded_model,
    get_loaded_models,
)
from .data_handler import get_random_sample_data
from .prediction_controller import make_prediction


def register_routes(app):
    """Registra todas las rutas en la aplicación Flask"""

    @app.route("/")
    def index():
        """Página principal - selección de tipo de modelo"""
        return render_template("index.html")

    @app.route("/mlp")
    def mlp_models():
        """Página de modelos MLP - Parte 1"""
        models = load_available_models(subdir="project_part_1")
        stats = load_model_stats()
        return render_template("mlp_models.html", models=models, stats=stats)

    @app.route("/rnn")
    def rnn_models():
        """Página de modelos RNN sin memoria - Parte 2"""
        models = load_available_models(subdir="project_part_2")
        stats = load_model_stats()
        return render_template("rnn_models.html", models=models, stats=stats)

    @app.route("/lstm")
    def lstm_models():
        """Página de modelos RNN con memoria - Parte 3"""
        models = load_available_models(subdir="project_part_3")
        stats = load_model_stats()
        return render_template("lstm_models.html", models=models, stats=stats)

    @app.route("/transformer")
    def transformer_models():
        """Página de modelos Transformer - Parte 4"""
        models = load_available_models(subdir="project_part_4")
        stats = load_model_stats()
        return render_template("transformer_models.html", models=models, stats=stats)

    @app.route("/model/<model_name>")
    def model_interface(model_name):
        """Interfaz de prueba para un modelo específico"""
        # Buscar modelo en todas las partes del proyecto
        all_models = []
        model_subdir = None
        for part in ["project_part_1", "project_part_2", "project_part_3", "project_part_4"]:
            models = load_available_models(subdir=part)
            for m in models:
                all_models.append(m["name"])
                if m["name"] == model_name:
                    model_subdir = part

        if model_name not in all_models:
            return "Modelo no encontrado", 404

        stats = load_model_stats()
        model_stats = stats.get(model_name, {})

        # Obtener framework del modelo usando el subdirectorio correcto
        import os
        model_dir = os.path.join("models", model_subdir) if model_subdir else "models/project_part_1"
        framework = get_model_framework(model_name, model_dir)
        model_stats["framework"] = framework

        # Si no hay estadísticas, proporcionar valores por defecto
        if not model_stats or "accuracy" not in model_stats:
            model_stats = {
                "framework": framework,
                "accuracy": 0.0,
                "loss": 0.0,
                "f1_score": 0.0,
                "epochs": 0,
                "training_time": 0,
                "parameters": 0,
                "language": "unknown",
                "architecture": [],
                "model_type": "Unknown",
                "vocab_size": 0,
                "train_samples": 0,
            }

        return render_template(
            "model_test.html", model_name=model_name, stats=model_stats
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        """Endpoint para hacer predicciones (soporta TensorFlow y PyTorch)"""
        data = request.json
        model_name = data.get("model_name")
        text_input = data.get("text_input", "").strip()

        if not model_name or not text_input:
            return jsonify({"error": "Modelo y texto son requeridos"}), 400

        # Cargar modelo y preprocessor
        model_data = load_model_and_preprocessor(model_name)
        if not model_data:
            return jsonify({"error": "Error cargando el modelo"}), 500

        # Hacer predicción
        result, error = make_prediction(model_data, text_input)

        if error:
            return jsonify({"error": error}), 500

        return jsonify(result)

    @app.route("/random_sample/<model_name>")
    def get_random_sample(model_name):
        """Obtiene una muestra aleatoria en inglés del conjunto de validación"""
        sample_data, error = get_random_sample_data()

        if error:
            return jsonify({"error": error}), 500

        return jsonify(sample_data)

    @app.route("/api/models")
    def api_models():
        """API endpoint para obtener lista de modelos con estadísticas y framework"""
        # Combinar modelos de todas las partes del proyecto
        all_models = []
        stats = load_model_stats()
        
        for part in ["project_part_1", "project_part_2", "project_part_3", "project_part_4"]:
            models = load_available_models(subdir=part)
            for model in models:
                model_name = model["name"]
                model_info = {
                    "name": model_name,
                    "framework": model["framework"],
                    "part": part,
                    "stats": stats.get(model_name, {}),
                }
                all_models.append(model_info)

        return jsonify(all_models)

    @app.route("/api/clear_memory", methods=["POST"])
    def clear_memory():
        """API endpoint para liberar memoria de modelos cargados"""
        try:
            clear_loaded_models()
            return jsonify(
                {"success": True, "message": "Memoria liberada exitosamente"}
            )
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/memory_status")
    def memory_status():
        """API endpoint para obtener el estado actual de memoria"""
        current_loaded_model = get_current_loaded_model()
        loaded_models = get_loaded_models()

        return jsonify(
            {
                "current_loaded_model": current_loaded_model,
                "loaded_models_count": len(loaded_models),
                "loaded_model_names": list(loaded_models.keys()),
            }
        )
