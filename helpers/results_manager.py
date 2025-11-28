"""
Results Management Module
Proyecto Redes Neuronales 2025-II - Universidad Del Valle

Maneja el guardado y carga de resultados de experimentos.
Soporta tanto TensorFlow como PyTorch.
"""

import os
import json
from datetime import datetime
from typing import Dict


class ResultsManager:
    """Maneja el guardado y carga de resultados de experimentos."""
    
    def __init__(self, output_dir: str = "output", project_part: str = "project_part_1"):
        """
        Inicializa ResultsManager.
        
        Args:
            output_dir (str): Directorio base para resultados
            project_part (str): Subdirectorio del proyecto (project_part_1 o project_part_2)
        """
        self.output_dir = os.path.join(output_dir, project_part)
        os.makedirs(self.output_dir, exist_ok=True)
        self.results_file = os.path.join(self.output_dir, "experiment_history.json")
    
    def save_experiment(self, experiment_data: Dict, experiment_name: str) -> str:
        """
        Guarda resultados del experimento en archivo JSON.
        
        Args:
            experiment_data (Dict): Diccionario conteniendo resultados del experimento
            experiment_name (str): Nombre del experimento
            
        Returns:
            str: Ruta al archivo guardado
        """
        # Añadir timestamp al experimento
        experiment_data['timestamp'] = datetime.now().isoformat()
        experiment_data['experiment_name'] = experiment_name
        
        # Cargar historial existente
        history_data = self.load_experiment_history()
        
        # Añadir nuevo experimento
        history_data["experiments"].append(experiment_data)
        
        # Guardar historial actualizado
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        print(f"Resultados del experimento guardados en: {self.results_file}")
        return self.results_file
    
    def save_experiment_results(self, experiment_data: Dict) -> int:
        """
        Guarda resultados del experimento y retorna el ID del experimento.
        
        Args:
            experiment_data (Dict): Diccionario conteniendo resultados del experimento
            
        Returns:
            int: ID del experimento guardado
        """
        # Añadir timestamp al experimento
        experiment_data['timestamp'] = datetime.now().isoformat()
        
        # Cargar historial existente
        history_data = self.load_experiment_history()
        
        # Generar ID del experimento
        experiment_id = len(history_data["experiments"]) + 1
        experiment_data['experiment_id'] = experiment_id
        
        # Añadir nuevo experimento
        history_data["experiments"].append(experiment_data)
        
        # Guardar historial actualizado
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        print(f"Experiment {experiment_id} results saved to {self.results_file}")
        return experiment_id
    
    def load_experiment_history(self) -> Dict:
        """
        Carga historial de experimentos desde archivo JSON.
        
        Returns:
            Dict: Historial completo de experimentos
        """
        if not os.path.exists(self.results_file):
            return {"experiments": [], "created": datetime.now().isoformat()}
        
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error cargando historial de experimentos: {e}")
            return {"experiments": [], "created": datetime.now().isoformat()}
    
    def display_experiment_history(self):
        """Muestra historial de experimentos formateado."""
        history_data = self.load_experiment_history()
        experiments = history_data.get("experiments", [])
        
        if not experiments:
            print("No se encontraron experimentos en el historial.")
            return
        
        print(f"\nHISTORIAL DE EXPERIMENTOS ({len(experiments)} experimentos)")
        print("=" * 100)
        
        # Summary table
        print(f"\n{'ID':<3} {'Modelo':<15} {'Lang':<6} {'Precisión':<10} {'Pérdida':<10} {'Épocas':<8} {'Tiempo (s)':<10} {'Muestras':<10}")
        print("-" * 95)
        
        for exp in experiments:
            training = exp.get('training_results', {})
            config = exp.get('configuration', {})
            dataset = exp.get('dataset_info', {})
            language = config.get('language_filter', 'multi')
            if language is None:
                language = 'multi'
            
            # Get accuracy from the correct key depending on framework
            val_accuracy = training.get('final_val_accuracy') or training.get('best_val_accuracy', 0)
            val_loss = training.get('final_val_loss') or training.get('best_val_loss', 0)
            
            print(f"{exp.get('experiment_id', 0):<3} "
                  f"{config.get('model_type', 'Unknown')[:14]:<15} "
                  f"{language:<6} "
                  f"{val_accuracy:<10.4f} "
                  f"{val_loss:<10.4f} "
                  f"{training.get('epochs_trained', 0):<8} "
                  f"{training.get('training_time', 0):<10.1f} "
                  f"{dataset.get('train_samples', 0):<10,}")
        
        # Best experiment by language
        if experiments:
            lang_groups = {}
            for exp in experiments:
                config = exp.get('configuration', {})
                lang = config.get('language_filter', 'multi')
                if lang is None:
                    lang = 'multi'
                if lang not in lang_groups:
                    lang_groups[lang] = []
                lang_groups[lang].append(exp)
            
            print(f"\nBEST EXPERIMENTS BY LANGUAGE:")
            print("-" * 50)
            for lang, lang_experiments in lang_groups.items():
                best_exp = max(lang_experiments, 
                              key=lambda x: x.get('training_results', {}).get('final_val_accuracy') or 
                                          x.get('training_results', {}).get('best_val_accuracy', 0))
                training_res = best_exp.get('training_results', {})
                accuracy = training_res.get('final_val_accuracy') or training_res.get('best_val_accuracy', 0)
                exp_id = best_exp.get('experiment_id', 0)
                samples = best_exp.get('dataset_info', {}).get('train_samples', 0)
                print(f"{lang:<6}: ID #{exp_id} - Accuracy: {accuracy:.4f} ({samples:,} samples)")
            
            # Overall best
            overall_best = max(experiments, 
                              key=lambda x: x.get('training_results', {}).get('final_val_accuracy') or 
                                          x.get('training_results', {}).get('best_val_accuracy', 0))
            training_res = overall_best.get('training_results', {})
            best_accuracy = training_res.get('final_val_accuracy') or training_res.get('best_val_accuracy', 0)
            best_lang = overall_best.get('configuration', {}).get('language_filter', 'multi')
            if best_lang is None:
                best_lang = 'multi'
            print(f"\nOVERALL BEST: ID #{overall_best.get('experiment_id')} ({best_lang}) - Accuracy: {best_accuracy:.4f}")
