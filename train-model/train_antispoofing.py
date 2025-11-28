#!/usr/bin/env python3
"""
Script de Entrenamiento para Modelo de Anti-Spoofing
Entrena un modelo EfficientNet-B2 para distinguir entre selfies reales y falsos.

Autor: Sistema Anti-Spoofing
Fecha: 2025-11-27
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configuración de logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AntiSpoofingModel(nn.Module):
    """Modelo EfficientNet-B2 para detección de anti-spoofing."""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # Backbone: EfficientNet-B2 (recomendado por balance precisión/velocidad)
        self.backbone = models.efficientnet_b2(pretrained=pretrained)
        
        # Reemplazar clasificador final
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def get_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Define transformaciones de datos para entrenamiento y validación.
    
    Returns:
        Tuple de (train_transform, val_transform)
    """
    # Normalización ImageNet (estándar para modelos pre-entrenados)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Transformaciones de entrenamiento (con data augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Transformaciones de validación (sin augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform


def get_data_loaders(
    dataset_root: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Carga los datasets y crea los data loaders.
    
    Args:
        dataset_root: Ruta raíz del dataset (debe contener Train/, Validation/, Test/)
        batch_size: Tamaño del batch
        num_workers: Número de workers para carga de datos
    
    Returns:
        Tuple de (train_loader, val_loader, test_loader)
    """
    train_transform, val_transform = get_data_transforms()
    
    # Rutas de los datasets
    train_dir = os.path.join(dataset_root, 'Train')
    val_dir = os.path.join(dataset_root, 'Validation')
    test_dir = os.path.join(dataset_root, 'Test')
    
    # Verificar que existan las carpetas
    for dir_path, name in [(train_dir, 'Train'), (val_dir, 'Validation'), (test_dir, 'Test')]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"❌ No se encontró la carpeta {name} en {dir_path}")
    
    # Cargar datasets
    logger.info(f"📂 Cargando datasets desde {dataset_root}")
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)
    test_dataset = ImageFolder(test_dir, transform=val_transform)
    
    # Verificar balance de clases
    train_real_count = sum(1 for _, label in train_dataset.samples if label == 0)
    train_fake_count = len(train_dataset.samples) - train_real_count
    logger.info(f"📊 Train: {train_real_count} Real, {train_fake_count} Fake")
    
    val_real_count = sum(1 for _, label in val_dataset.samples if label == 0)
    val_fake_count = len(val_dataset.samples) - val_real_count
    logger.info(f"📊 Validation: {val_real_count} Real, {val_fake_count} Fake")
    
    # Calcular pesos para balancear clases si hay desbalance
    class_counts = [train_real_count, train_fake_count]
    total_samples = sum(class_counts)
    class_weights = [total_samples / count for count in class_counts]
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Crear data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"✅ Datasets cargados: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    return train_loader, val_loader, test_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """Entrena el modelo por una época."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Estadísticas
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Log cada 50 batches
        if (batch_idx + 1) % 50 == 0:
            logger.debug(
                f"  Batch {batch_idx+1}/{len(train_loader)}: "
                f"Loss={loss.item():.4f}, Acc={100*correct/total:.2f}%"
            )
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Valida el modelo."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evalúa el modelo en el conjunto de test y retorna métricas completas."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo de Anti-Spoofing')
    parser.add_argument(
        '--dataset-root',
        type=str,
        default='../Dataset',
        help='Ruta raíz del dataset (default: ../Dataset)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Tamaño del batch (default: 32)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Número de épocas (default: 50)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate inicial (default: 0.001)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=15,
        help='Paciencia para early stopping (default: 15)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./models',
        help='Directorio para guardar modelos (default: ./models)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Número de workers para carga de datos (default: 4)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='antispoofing_efficientnet_b2',
        help='Nombre del modelo (default: antispoofing_efficientnet_b2)'
    )
    
    args = parser.parse_args()
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Usando dispositivo: {device}")
    if torch.cuda.is_available():
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Cargar datos
    try:
        train_loader, val_loader, test_loader = get_data_loaders(
            args.dataset_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Inicializar modelo
    logger.info("🏗️  Inicializando modelo EfficientNet-B2...")
    model = AntiSpoofingModel(num_classes=2, pretrained=True)
    model.to(device)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"   Parámetros totales: {total_params:,}")
    logger.info(f"   Parámetros entrenables: {trainable_params:,}")
    
    # Optimizador y scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0001
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()
    
    # Entrenamiento
    logger.info("🚀 Iniciando entrenamiento...")
    logger.info(f"   Épocas: {args.epochs}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Learning rate: {args.learning_rate}")
    logger.info(f"   Early stopping patience: {args.patience}")
    
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    training_history = []
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Entrenar
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validar
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Actualizar learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Guardar historial
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr
        })
        
        # Log
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s): "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, "
            f"LR={current_lr:.6f}"
        )
        
        # Early stopping y guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            no_improve = 0
            
            # Guardar mejor modelo
            model_path = os.path.join(args.output_dir, f'{args.model_name}_best.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f"   ✅ Nuevo mejor modelo guardado (Val Acc: {val_acc:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"⏹️  Early stopping en epoch {epoch+1} (sin mejora por {args.patience} épocas)")
                break
    
    total_time = time.time() - start_time
    logger.info(f"⏱️  Tiempo total de entrenamiento: {total_time/60:.1f} minutos")
    logger.info(f"🏆 Mejor modelo en epoch {best_epoch} con Val Acc: {best_val_acc:.2f}%")
    
    # Cargar mejor modelo para evaluación
    logger.info("📊 Evaluando mejor modelo en conjunto de test...")
    best_model_path = os.path.join(args.output_dir, f'{args.model_name}_best.pth')
    model.load_state_dict(torch.load(best_model_path))
    
    # Evaluar en test
    test_metrics = evaluate_model(model, test_loader, device)
    
    logger.info("=" * 60)
    logger.info("📈 RESULTADOS FINALES (Test Set):")
    logger.info(f"   Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    logger.info(f"   Precision: {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
    logger.info(f"   Recall:    {test_metrics['recall']:.4f} ({test_metrics['recall']*100:.2f}%)")
    logger.info(f"   F1-Score:  {test_metrics['f1_score']:.4f} ({test_metrics['f1_score']*100:.2f}%)")
    logger.info("")
    logger.info("📊 Matriz de Confusión:")
    logger.info(f"   [[TN={test_metrics['confusion_matrix'][0,0]}, FP={test_metrics['confusion_matrix'][0,1]}],")
    logger.info(f"    [FN={test_metrics['confusion_matrix'][1,0]}, TP={test_metrics['confusion_matrix'][1,1]}]]")
    logger.info("=" * 60)
    
    # Exportar modelo a TorchScript para producción
    logger.info("💾 Exportando modelo a TorchScript...")
    model.eval()
    example_input = torch.rand(1, 3, 224, 224).to(device)
    traced_model = torch.jit.trace(model, example_input)
    
    torchscript_path = os.path.join(args.output_dir, f'{args.model_name}.pt')
    traced_model.save(torchscript_path)
    logger.info(f"   ✅ Modelo exportado a: {torchscript_path}")
    
    # Guardar modelo completo también (útil para continuar entrenamiento)
    full_model_path = os.path.join(args.output_dir, f'{args.model_name}_full.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': best_epoch,
        'val_acc': best_val_acc,
        'test_metrics': test_metrics,
        'training_history': training_history
    }, full_model_path)
    logger.info(f"   ✅ Modelo completo guardado en: {full_model_path}")
    
    logger.info("✅ Entrenamiento completado exitosamente!")


if __name__ == '__main__':
    main()

