from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import os
import gc
import numpy as np
import tensorflow as tf
from helper import DataLoader
from helper import ResultsManager
from helper import load_model_components
from helper import setup_experiment_environment

app = Flask(__name__)

# Configuración global
MODEL_DIR = "models"
OUTPUT_DIR = "output"  
DATA_DIR = "data"

# Variables globales para modelos cargados
loaded_models = {}
loaded_stats = {}
validation_data = None
current_loaded_model = None  # Rastrea el modelo actualmente cargado
setup_experiment_environment(42)
def load_available_models():
    """Carga la lista de modelos disponibles desde el directorio models"""
    models = []
    if os.path.exists(MODEL_DIR):
        for file in os.listdir(MODEL_DIR):
            if file.endswith('.h5'):
                model_name = file.replace('.h5', '')
                models.append(model_name)
    return models

def load_model_stats():
    """Carga las estadísticas de los modelos desde output/experiment_history.json"""
    results_manager = ResultsManager(OUTPUT_DIR)
    history_data = results_manager.load_experiment_history()
    stats = {}
    
    for exp in history_data.get("experiments", []):
        model_path = exp.get('training_results', {}).get('model_path', '')
        if model_path:
            model_name = os.path.basename(model_path).replace('.h5', '')
            stats[model_name] = {
                'experiment_id': exp.get('experiment_id', 0),
                'accuracy': exp.get('training_results', {}).get('final_val_accuracy', 0),
                'loss': exp.get('training_results', {}).get('final_val_loss', 0),
                'f1_score': exp.get('evaluation_metrics', {}).get('f1_weighted', 0),
                'epochs': exp.get('training_results', {}).get('epochs_trained', 0),
                'training_time': exp.get('training_results', {}).get('training_time', 0),
                'parameters': exp.get('configuration', {}).get('total_parameters', 0),
                'language': exp.get('configuration', {}).get('language_filter', 'multi'),
                'architecture': exp.get('configuration', {}).get('hidden_layers', []),
                'model_type': exp.get('configuration', {}).get('model_type', 'Unknown'),
                'vocab_size': exp.get('dataset_info', {}).get('vocab_size', 0),
                'train_samples': exp.get('dataset_info', {}).get('train_samples', 0)
            }
    return stats

def load_validation_samples():
    """Carga muestras del conjunto de validación para pruebas aleatorias"""
    try:
        data_loader = DataLoader(DATA_DIR)
        _, val_df, _ = data_loader.load_all_data()
        return val_df.sample(min(100, len(val_df)))  # Máximo 100 muestras
    except Exception as e:
        print(f"Error cargando datos de validación: {e}")
        return None

def clear_loaded_models():
    """Limpia todos los modelos de memoria y libera recursos"""
    global loaded_models, current_loaded_model
    
    for model_name, model_data in loaded_models.items():
        if 'model' in model_data and model_data['model'] is not None:
            try:
                # Limpiar memoria del modelo de TensorFlow
                del model_data['model']
                tf.keras.backend.clear_session()
                print(f"Modelo {model_name} liberado de memoria")
            except Exception as e:
                print(f"Error liberando modelo {model_name}: {e}")
    
    loaded_models.clear()
    current_loaded_model = None
    
    # Forzar recolección de basura
    gc.collect()

def load_model_and_preprocessor(model_name):
    """Carga un modelo específico y sus componentes de preprocesamiento"""
    global current_loaded_model
    
    # Si ya hay un modelo cargado y es diferente, limpiar memoria
    if current_loaded_model and current_loaded_model != model_name:
        print(f"Cambiando de modelo {current_loaded_model} a {model_name}")
        clear_loaded_models()
    
    if model_name in loaded_models:
        current_loaded_model = model_name
        return loaded_models[model_name]
    
    try:
        # Intentar cargar componentes guardados usando la función del helper
        components = load_model_components(model_name, MODEL_DIR)
        
        if components and 'model' in components:
            # Usar componentes guardados (método eficiente)
            model_data = {
                'model': components['model'],
                'vectorizer': components.get('vectorizer'),
                'tokenizer': components.get('tokenizer'),
                'label_encoder': components.get('label_encoder'),
                'vocab_size': components.get('vocab_size', 5000),
                'num_classes': components.get('num_classes', 5),
                'max_length': components.get('max_length', 100)  # Longitud por defecto para embedding
            }
            
            loaded_models[model_name] = model_data
            current_loaded_model = model_name
            return model_data
        
        else:
            # Fallback: recrear componentes (método legacy con optimización)
            print(f"Componentes no encontrados para {model_name}, usando método legacy optimizado...")
            
            model_path = os.path.join(MODEL_DIR, f"{model_name}.h5")
            model = tf.keras.models.load_model(model_path)
            
            # Crear vectorizador básico sin procesar todo el dataset
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import LabelEncoder
            
            vectorizer = TfidfVectorizer(
                max_features=5000,
                min_df=3,
                max_df=0.85,
                stop_words='english'
            )
            
            # Crear label_encoder básico (asumiendo clasificación 1-5 estrellas)
            label_encoder = LabelEncoder()
            label_encoder.fit([1, 2, 3, 4, 5])
            
            print(f"Usando configuración por defecto. Considera volver a entrenar y guardar con save_model_components().")
            
            model_data = {
                'model': model,
                'vectorizer': vectorizer,
                'tokenizer': None,
                'label_encoder': label_encoder,
                'vocab_size': 5000,
                'num_classes': 5,
                'max_length': None  # BoW no necesita max_length
            }
            
            loaded_models[model_name] = model_data
            current_loaded_model = model_name
            return model_data
        
    except Exception as e:
        print(f"Error cargando modelo {model_name}: {e}")
        return None

@app.route('/')
def index():
    """Página principal con selección de modelos"""
    models = load_available_models()
    stats = load_model_stats()
    return render_template('index.html', models=models, stats=stats)

@app.route('/model/<model_name>')
def model_interface(model_name):
    """Interfaz de prueba para un modelo específico"""
    if model_name not in load_available_models():
        return "Modelo no encontrado", 404
    
    stats = load_model_stats()
    model_stats = stats.get(model_name, {})
    
    return render_template('model_test.html', 
                         model_name=model_name, 
                         stats=model_stats)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para hacer predicciones"""
    data = request.json
    model_name = data.get('model_name')
    text_input = data.get('text_input', '').strip()
    
    if not model_name or not text_input:
        return jsonify({'error': 'Modelo y texto son requeridos'}), 400
    
    # Cargar modelo y preprocessor
    model_data = load_model_and_preprocessor(model_name)
    if not model_data:
        return jsonify({'error': 'Error cargando el modelo'}), 500
    
    try:
        # Determinar tipo de modelo y preprocesar texto apropiadamente
        if model_data['tokenizer'] is not None:
            # Modelo de embedding - usar tokenizer
            # Tokenizar el texto
            sequences = model_data['tokenizer'].texts_to_sequences([text_input])
            max_length = model_data.get('max_length', 100)  # Usar el valor guardado o 100 por defecto
            X_input = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
            
        elif model_data['vectorizer'] is not None:
            # Modelo BoW - usar vectorizer
            X_input = model_data['vectorizer'].transform([text_input])
            X_input = X_input.toarray().astype(np.float32)
            
        else:
            return jsonify({'error': 'No se encontró vectorizer ni tokenizer válido'}), 500
        
        # Hacer predicción
        prediction = model_data['model'].predict(X_input, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][predicted_class])
        
        # Convertir a estrellas (1-5)
        stars = int(model_data['label_encoder'].classes_[predicted_class])
        
        # Obtener probabilidades por clase
        class_probabilities = {}
        for i, class_label in enumerate(model_data['label_encoder'].classes_):
            class_probabilities[f"{class_label} estrellas"] = float(prediction[0][i])
        
        return jsonify({
            'predicted_stars': stars,
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'text_preview': text_input[:100] + '...' if len(text_input) > 100 else text_input,
            'model_type': 'embedding' if model_data['tokenizer'] is not None else 'bow'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error en predicción: {str(e)}'}), 500

@app.route('/random_sample/<model_name>')
def get_random_sample(model_name):
    """Obtiene una muestra aleatoria en inglés del conjunto de validación"""
    global validation_data

    if validation_data is None:
        validation_data = load_validation_samples()

    if validation_data is None or len(validation_data) == 0:
        return jsonify({'error': 'No hay datos de validación disponibles'}), 500

    # Filtrar solo muestras en inglés
    english_samples = validation_data[validation_data['language'] == 'en']
    if len(english_samples) == 0:
        return jsonify({'error': 'No hay muestras en inglés disponibles'}), 500

    # Seleccionar muestra aleatoria
    sample = english_samples.sample(1).iloc[0]

    # Construir texto como en el entrenamiento
    title = sample.get('review_title', '')
    body = sample.get('review_body', '')
    combined_text = f"{title} {body}".strip()

    return jsonify({
        'text': combined_text,
        'true_stars': int(sample.get('stars', 0)),
        'title': title,
        'body': body
    })

@app.route('/api/models')
def api_models():
    """API endpoint para obtener lista de modelos con estadísticas"""
    models = load_available_models()
    stats = load_model_stats()
    
    model_list = []
    for model in models:
        model_info = {
            'name': model,
            'stats': stats.get(model, {})
        }
        model_list.append(model_info)
    
    return jsonify(model_list)

@app.route('/api/clear_memory', methods=['POST'])
def clear_memory():
    """API endpoint para liberar memoria de modelos cargados"""
    try:
        clear_loaded_models()
        return jsonify({'success': True, 'message': 'Memoria liberada exitosamente'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/memory_status')
def memory_status():
    """API endpoint para obtener el estado actual de memoria"""
    global current_loaded_model, loaded_models
    
    return jsonify({
        'current_loaded_model': current_loaded_model,
        'loaded_models_count': len(loaded_models),
        'loaded_model_names': list(loaded_models.keys())
    })

def cleanup_on_exit():
    """Función de limpieza que se ejecuta al cerrar la aplicación"""
    print("Cerrando aplicación... Liberando memoria de modelos")
    clear_loaded_models()

if __name__ == '__main__':
    import atexit
    atexit.register(cleanup_on_exit)
    
    print("Iniciando aplicación Flask...")
    print(f"Modelos disponibles: {load_available_models()}")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nCerrando aplicación por interrupción del usuario...")
        cleanup_on_exit()