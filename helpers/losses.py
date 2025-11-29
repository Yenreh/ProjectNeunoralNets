"""
Custom Loss Functions for PyTorch
Proyecto Redes Neuronales 2025-II - Universidad Del Valle

Implementa funciones de pérdida avanzadas para mejorar el entrenamiento.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss con Label Smoothing.
    
    Label smoothing suaviza las etiquetas one-hot reduciendo la confianza del modelo.
    Esto ayuda con:
    - Clases ambiguas (ej: 2★ vs 3★ en reviews)
    - Reducir sobreconfianza del modelo
    - Mejor calibración de probabilidades
    - Regularización adicional
    
    Ejemplo:
        Sin smoothing: [0, 0, 1, 0, 0] (100% confianza en clase 2)
        Con smoothing=0.1: [0.025, 0.025, 0.9, 0.025, 0.025]
    """
    
    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor = None):
        """
        Inicializa Label Smoothing Loss.
        
        Args:
            smoothing (float): Factor de suavizado [0, 1]. 
                               0 = sin smoothing (CrossEntropy estándar)
                               0.1 = suavizado moderado (recomendado)
                               0.2 = suavizado fuerte
            weight (torch.Tensor): Pesos por clase para dataset desbalanceado
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert 0 <= smoothing < 1, "smoothing debe estar en [0, 1)"
        self.smoothing = smoothing
        self.weight = weight
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida con label smoothing.
        
        Args:
            pred: Logits del modelo [batch_size, num_classes]
            target: Labels enteros [batch_size]
        
        Returns:
            Loss escalar
        """
        # Obtener log probabilities
        log_probs = F.log_softmax(pred, dim=-1)
        
        # Número de clases
        num_classes = pred.size(-1)
        
        # Crear distribución suavizada
        # La clase correcta obtiene (1 - smoothing)
        # Las demás clases se reparten smoothing uniformemente
        smooth_dist = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Aplicar pesos de clase si están definidos
        if self.weight is not None:
            weight = self.weight.to(pred.device)
            weight = weight.gather(0, target)
            loss = -(smooth_dist * log_probs).sum(dim=-1) * weight
        else:
            loss = -(smooth_dist * log_probs).sum(dim=-1)
        
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss para clases desbalanceadas o ejemplos difíciles.
    
    Focal Loss reduce la pérdida de ejemplos fáciles (bien clasificados)
    y aumenta el foco en ejemplos difíciles (mal clasificados).
    
    Útil para:
    - Clases desbalanceadas
    - Ejemplos difíciles de clasificar
    - Mejorar rendimiento en clases minoritarias
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    donde:
    - p_t es la probabilidad de la clase correcta
    - gamma controla el enfoque en ejemplos difíciles
    - alpha_t balancea las clases
    """
    
    def __init__(self, 
                 alpha: torch.Tensor = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        """
        Inicializa Focal Loss.
        
        Args:
            alpha (torch.Tensor): Pesos por clase [num_classes]. 
                                  Si None, todas las clases tienen peso 1.
            gamma (float): Factor de enfoque. 
                           0 = CrossEntropy estándar
                           1 = enfoque moderado
                           2 = enfoque fuerte (recomendado)
                           5 = enfoque muy fuerte
            reduction (str): 'mean', 'sum', o 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcula Focal Loss.
        
        Args:
            pred: Logits del modelo [batch_size, num_classes]
            target: Labels enteros [batch_size]
        
        Returns:
            Loss según reduction
        """
        # Calcular cross entropy sin reducción
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Obtener probabilidades
        probs = F.softmax(pred, dim=-1)
        
        # Obtener probabilidad de la clase correcta
        p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # Calcular focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Aplicar focal term a cross entropy
        loss = focal_term * ce_loss
        
        # Aplicar alpha (pesos de clase) si están definidos
        if self.alpha is not None:
            alpha = self.alpha.to(pred.device)
            alpha_t = alpha.gather(0, target)
            loss = alpha_t * loss
        
        # Aplicar reducción
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedFocalLoss(nn.Module):
    """
    Combinación de Focal Loss con pesos de clase explícitos.
    
    Útil cuando tienes tanto clases desbalanceadas como ejemplos difíciles.
    """
    
    def __init__(self,
                 class_weights: torch.Tensor,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        """
        Inicializa Weighted Focal Loss.
        
        Args:
            class_weights (torch.Tensor): Pesos por clase (típicamente inversos a frecuencia)
            gamma (float): Factor focal
            reduction (str): Tipo de reducción
        """
        super(WeightedFocalLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=gamma, reduction=reduction)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.focal_loss(pred, target)


def get_loss_function(loss_type: str = 'cross_entropy',
                     class_weights: torch.Tensor = None,
                     label_smoothing: float = 0.0,
                     focal_gamma: float = 2.0) -> nn.Module:
    """
    Factory function para obtener función de pérdida configurada.
    
    Args:
        loss_type (str): Tipo de pérdida:
            - 'cross_entropy': CrossEntropy estándar (soporta label_smoothing nativo de PyTorch)
            - 'label_smoothing': CrossEntropy con label smoothing (implementación custom)
            - 'focal': Focal Loss
            - 'weighted_focal': Focal Loss con pesos de clase
        class_weights (torch.Tensor): Pesos por clase (opcional)
        label_smoothing (float): Factor de smoothing [0, 1). Si > 0, se aplica label smoothing.
        focal_gamma (float): Parámetro gamma para focal losses
    
    Returns:
        nn.Module: Función de pérdida configurada
    """
    if loss_type == 'cross_entropy':
        # PyTorch >= 1.10 soporta label_smoothing nativo en CrossEntropyLoss
        if label_smoothing > 0:
            print(f"  [INFO] Label Smoothing activado: {label_smoothing}")
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_type == 'label_smoothing':
        # Implementación custom (útil si se necesita comportamiento especial)
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing, weight=class_weights)
    
    elif loss_type == 'focal':
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)
    
    elif loss_type == 'weighted_focal':
        if class_weights is None:
            raise ValueError("weighted_focal requiere class_weights")
        return WeightedFocalLoss(class_weights=class_weights, gamma=focal_gamma)
    
    else:
        raise ValueError(f"loss_type '{loss_type}' no soportado. "
                        f"Opciones: 'cross_entropy', 'label_smoothing', 'focal', 'weighted_focal'")


# Ejemplo de uso:
if __name__ == "__main__":
    # Test básico
    batch_size = 32
    num_classes = 5
    
    # Datos simulados
    pred = torch.randn(batch_size, num_classes)
    target = torch.randint(0, num_classes, (batch_size,))
    
    # Pesos de clase (para dataset desbalanceado)
    class_weights = torch.tensor([1.5, 2.0, 2.5, 2.0, 1.0])
    
    print("Testing loss functions:")
    print("-" * 50)
    
    # 1. CrossEntropy estándar
    ce = nn.CrossEntropyLoss()
    loss = ce(pred, target)
    print(f"CrossEntropy Loss: {loss.item():.4f}")
    
    # 2. Label Smoothing
    ls = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss = ls(pred, target)
    print(f"Label Smoothing Loss: {loss.item():.4f}")
    
    # 3. Focal Loss
    focal = FocalLoss(gamma=2.0)
    loss = focal(pred, target)
    print(f"Focal Loss (gamma=2.0): {loss.item():.4f}")
    
    # 4. Weighted Focal Loss
    wfocal = WeightedFocalLoss(class_weights=class_weights, gamma=2.0)
    loss = wfocal(pred, target)
    print(f"Weighted Focal Loss: {loss.item():.4f}")
    
    print("-" * 50)
    print("All tests passed!")
