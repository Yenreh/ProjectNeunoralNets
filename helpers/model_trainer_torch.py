"""
Model Training Module for PyTorch
Proyecto Redes Neuronales 2025-II - Universidad Del Valle

Maneja el entrenamiento de modelos con PyTorch.
Compatible con la Parte 2 del proyecto.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from tqdm import tqdm


class ModelTrainerTorch:
    """Maneja el entrenamiento de modelos PyTorch con callbacks y monitoreo."""
    
    def __init__(self, model_dir: str = "models/project_part_2", device: str = None):
        """
        Inicializa ModelTrainerTorch.
        
        Args:
            model_dir (str): Directorio para guardar modelos
            device (str): Dispositivo ('cuda' o 'cpu'). Si None, se detecta automáticamente
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Usando dispositivo: {self.device}")
    
    def train_model(self,
                   model: nn.Module,
                   train_loader,
                   val_loader,
                   criterion,
                   optimizer,
                   epochs: int = 50,
                   patience: int = 10,
                   model_name: str = "mlp_model",
                   scheduler=None,
                   clip_grad_norm: float = None,
                   verbose: bool = True) -> Dict:
        """
        Entrena un modelo PyTorch con early stopping.
        
        Args:
            model: Modelo PyTorch
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            criterion: Función de pérdida
            optimizer: Optimizador
            epochs (int): Número máximo de épocas
            patience (int): Paciencia para early stopping
            model_name (str): Nombre para guardar el modelo
            scheduler: Learning rate scheduler (opcional)
            clip_grad_norm (float): Valor máximo para gradient clipping (None para desactivar)
            verbose (bool): Mostrar progreso
            
        Returns:
            Dict: Historial y resultados del entrenamiento
        """
        print(f"\nEntrenando {model_name}...")
        
        # Contar parámetros
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parámetros totales: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,}")
        
        if clip_grad_norm is not None:
            print(f"Gradient clipping activado: max_norm={clip_grad_norm}")
        
        # Mover modelo al dispositivo
        model = model.to(self.device)
        
        # Inicializar seguimiento
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        epochs_without_improvement = 0
        best_model_state = None
        
        start_time = datetime.now()
        
        # Loop de entrenamiento
        for epoch in range(epochs):
            # Entrenamiento
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            if verbose:
                train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            else:
                train_pbar = train_loader
            
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping (útil para RNN)
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                
                optimizer.step()
                
                # Estadísticas
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if verbose and hasattr(train_pbar, 'set_postfix'):
                    train_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100 * train_correct / train_total:.2f}%'
                    })
            
            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total
            
            # Validación
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                if verbose:
                    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                else:
                    val_pbar = val_loader
                
                for inputs, labels in val_pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    if verbose and hasattr(val_pbar, 'set_postfix'):
                        val_pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'acc': f'{100 * val_correct / val_total:.2f}%'
                        })
            
            val_loss /= len(val_loader)
            val_accuracy = val_correct / val_total
            
            # Guardar en historial
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Imprimir resumen de época
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Learning rate scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                epochs_without_improvement = 0
                best_model_state = model.state_dict().copy()
                print(f"  ✓ Mejor modelo guardado (Val Loss: {val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                print(f"  Épocas sin mejora: {epochs_without_improvement}/{patience}")
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping en época {epoch+1}")
                break
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Restaurar mejor modelo
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print("\nMejor modelo restaurado")
        
        # Guardar modelo
        model_path = os.path.join(self.model_dir, f"{model_name}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_accuracy': best_val_accuracy,
            'history': history,
        }, model_path)
        print(f"Modelo guardado en: {model_path}")
        
        return {
            'history': history,
            'epochs_trained': epoch + 1,
            'training_time': training_time,
            'model_path': model_path,
            'best_val_loss': best_val_loss,
            'best_val_accuracy': best_val_accuracy,
            'final_train_loss': history['train_loss'][-1],
            'final_train_accuracy': history['train_accuracy'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
        }
    
    def evaluate_model(self,
                      model: nn.Module,
                      test_loader,
                      criterion) -> Dict:
        """
        Evalúa el modelo en el conjunto de prueba.
        
        Args:
            model: Modelo PyTorch
            test_loader: DataLoader de prueba
            criterion: Función de pérdida
            
        Returns:
            Dict: Métricas de evaluación
        """
        print("\nEvaluando modelo en conjunto de prueba...")
        
        model = model.to(self.device)
        model.eval()
        
        test_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluando"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels)
        }
